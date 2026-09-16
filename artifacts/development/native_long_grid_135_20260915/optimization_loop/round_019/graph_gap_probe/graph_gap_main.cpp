// R19 standalone integration entrypoint. Build/run remains parent-gated.
#define NOMINMAX
#include <windows.h>
#include <tlhelp32.h>
#include <bcrypt.h>
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cuda.h"
#include <cuda_runtime_api.h>
#include "prepared_identity.h" // Generated only by reviewed build_probe.py before compilation.
#include "frozen_module_guard.h"
#include "graph_gap_probe.cpp"
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

static_assert(sizeof(void *) == 8, "frozen Windows x64 ABI only");

static long long host_qpc() {
    LARGE_INTEGER value{};
    if (!QueryPerformanceCounter(&value)) throw std::runtime_error("QPC unavailable");
    return value.QuadPart;
}

static long long qpf() {
    LARGE_INTEGER value{};
    if (!QueryPerformanceFrequency(&value) || value.QuadPart <= 0) throw std::runtime_error("QPC frequency unavailable");
    return value.QuadPart;
}

static std::string quote(const std::string &text) {
    std::ostringstream out;
    out << '"';
    for (const unsigned char ch : text) {
        switch (ch) {
        case '"': out << "\\\""; break;
        case '\\': out << "\\\\"; break;
        case '\n': out << "\\n"; break;
        case '\r': out << "\\r"; break;
        case '\t': out << "\\t"; break;
        default:
            if (ch < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(ch) << std::dec;
            else out << ch;
        }
    }
    out << '"';
    return out.str();
}

static std::wstring wide(const std::string &text) {
    const int count = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, text.data(), int(text.size()), nullptr, 0);
    if (count <= 0) throw std::runtime_error("invalid UTF-8 path");
    std::wstring output(count, 0);
    MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, text.data(), int(text.size()), output.data(), count);
    return output;
}

static std::string utf8(const wchar_t *text) {
    const int count = WideCharToMultiByte(CP_UTF8, 0, text, -1, nullptr, 0, nullptr, nullptr);
    if (count <= 0) throw std::runtime_error("module path conversion failed");
    std::string output(count, 0);
    WideCharToMultiByte(CP_UTF8, 0, text, -1, output.data(), count, nullptr, nullptr);
    output.pop_back();
    return output;
}

static std::string utc() {
    SYSTEMTIME time{};
    GetSystemTime(&time);
    char output[64]{};
    sprintf_s(output, "%04u-%02u-%02uT%02u:%02u:%02u.%03uZ", time.wYear, time.wMonth, time.wDay,
              time.wHour, time.wMinute, time.wSecond, time.wMilliseconds);
    return output;
}

static std::string file_hash(const std::string &path) {
    std::ifstream file(std::filesystem::path(wide(path)), std::ios::binary);
    if (!file) throw std::runtime_error("unreadable identity file: " + path);
    BCRYPT_ALG_HANDLE algorithm = nullptr;
    BCRYPT_HASH_HANDLE hash = nullptr;
    if (BCryptOpenAlgorithmProvider(&algorithm, BCRYPT_SHA256_ALGORITHM, nullptr, 0) < 0) {
        throw std::runtime_error("SHA256 provider failure");
    }
    ULONG object_size = 0, received = 0;
    if (BCryptGetProperty(algorithm, BCRYPT_OBJECT_LENGTH, reinterpret_cast<PUCHAR>(&object_size), sizeof(object_size), &received, 0) < 0) {
        BCryptCloseAlgorithmProvider(algorithm, 0);
        throw std::runtime_error("SHA256 properties failure");
    }
    std::vector<unsigned char> object(object_size), buffer(1 << 20);
    unsigned char digest[32]{};
    if (BCryptCreateHash(algorithm, &hash, object.data(), object_size, nullptr, 0, 0) < 0) {
        BCryptCloseAlgorithmProvider(algorithm, 0);
        throw std::runtime_error("SHA256 create failure");
    }
    while (file) {
        file.read(reinterpret_cast<char *>(buffer.data()), buffer.size());
        const auto count = file.gcount();
        if (count && BCryptHashData(hash, buffer.data(), ULONG(count), 0) < 0) {
            BCryptDestroyHash(hash); BCryptCloseAlgorithmProvider(algorithm, 0);
            throw std::runtime_error("SHA256 update failure");
        }
    }
    const bool okay = file.eof() && BCryptFinishHash(hash, digest, sizeof(digest), 0) >= 0;
    BCryptDestroyHash(hash); BCryptCloseAlgorithmProvider(algorithm, 0);
    if (!okay) throw std::runtime_error("SHA256 finish/read failure");
    std::ostringstream output;
    for (const auto byte : digest) output << std::hex << std::setw(2) << std::setfill('0') << int(byte);
    return output.str();
}

static void verify_files() {
    for (const auto &entry : frozen_files) {
        if (file_hash(entry.path) != entry.sha256) throw std::runtime_error(std::string("frozen source/runtime changed: ") + entry.path);
    }
}

struct NativeModuleApi {
    using Handle = HMODULE;
    void require_absolute_path(const char *path) const {
        const std::string text(path);
        if (text.size() < 3 || !std::isalpha(static_cast<unsigned char>(text[0])) || text[1] != ':' || (text[2] != '\\' && text[2] != '/')) {
            throw std::runtime_error("DLL path not absolute");
        }
    }
    HMODULE load_absolute(const char *path) const {
        const auto handle = LoadLibraryExW(wide(path).c_str(), nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
        if (!handle) throw std::runtime_error(std::string("absolute DLL preload failed: ") + path);
        return handle;
    }
    void verify_handle(HMODULE handle, const char *path, const char *hash) const {
        wchar_t module_path[32768]{};
        const DWORD count = GetModuleFileNameW(handle, module_path, 32768);
        if (!count || count >= 32768 || _stricmp(utf8(module_path).c_str(), path) != 0 || file_hash(utf8(module_path)) != hash) {
            throw std::runtime_error(std::string("loaded DLL identity mismatch: ") + path);
        }
    }
    void release(HMODULE handle) const noexcept { FreeLibrary(handle); }
};

static void verify_loaded_modules() {
    NativeModuleApi api;
    for (const auto &entry : frozen_modules) {
        const std::string path(entry.path);
        const std::string name = path.substr(path.find_last_of("/\\") + 1);
        const auto handle = GetModuleHandleW(wide(name).c_str());
        if (!handle) throw std::runtime_error("missing frozen module: " + name);
        api.verify_handle(handle, entry.path, entry.hash);
    }
}

static std::string modules_json() {
    const HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPMODULE | TH32CS_SNAPMODULE32, GetCurrentProcessId());
    if (snapshot == INVALID_HANDLE_VALUE) throw std::runtime_error("module snapshot unavailable");
    MODULEENTRY32W entry{};
    entry.dwSize = sizeof(entry);
    std::vector<std::string> modules;
    try {
        if (!Module32FirstW(snapshot, &entry)) throw std::runtime_error("empty module snapshot");
        do {
            const std::string name = utf8(entry.szModule);
            const std::string path = utf8(entry.szExePath);
            std::string lower = name;
            std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char ch) { return char(std::tolower(ch)); });
            if (lower.find("ggml") != std::string::npos || lower.find("cuda") != std::string::npos || lower.find("cublas") != std::string::npos || lower.find("nvtx") != std::string::npos || lower == "graph-gap-probe.exe") {
                modules.push_back("{\"name\":" + quote(name) + ",\"path\":" + quote(path) + ",\"sha256\":" + quote(file_hash(path)) + "}");
            }
        } while (Module32NextW(snapshot, &entry));
    } catch (...) {
        CloseHandle(snapshot);
        throw;
    }
    CloseHandle(snapshot);
    std::sort(modules.begin(), modules.end());
    std::ostringstream output;
    output << '[';
    for (std::size_t index = 0; index < modules.size(); ++index) {
        if (index) output << ',';
        output << modules[index];
    }
    return output << ']', output.str();
}

static std::string environment_json(bool enforce) {
    std::ostringstream output;
    output << '{';
    bool first = true;
    for (const auto &entry : frozen_environment) {
        const char *actual = std::getenv(entry.name);
        if (enforce && ((entry.value == nullptr) != (actual == nullptr) || (entry.value && actual && std::string(entry.value) != actual))) {
            throw std::runtime_error(std::string("runtime environment mismatch: ") + entry.name);
        }
        if (!first) output << ',';
        first = false;
        output << quote(entry.name) << ':' << (actual ? quote(actual) : "null");
    }
    return output << '}', output.str();
}

class Jsonl {
public:
    explicit Jsonl(const std::string &path) : handle_(CreateFileW(wide(path).c_str(), GENERIC_WRITE, FILE_SHARE_READ, nullptr, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr)) {
        if (handle_ == INVALID_HANDLE_VALUE) throw std::runtime_error("raw output must be a new writable path; refusing overwrite");
    }
    ~Jsonl() { close(); }
    void line(const std::string &line) {
        if (handle_ == INVALID_HANDLE_VALUE) throw std::runtime_error("raw output already closed");
        const std::string payload = line + "\n";
        std::size_t offset = 0;
        while (offset < payload.size()) {
            DWORD written = 0;
            const auto count = DWORD(std::min<std::size_t>(payload.size() - offset, 1 << 20));
            if (!WriteFile(handle_, payload.data() + offset, count, &written, nullptr) || written != count) throw std::runtime_error("raw output write failed");
            offset += written;
        }
    }
    void durable() { if (!FlushFileBuffers(handle_)) throw std::runtime_error("raw output flush failed"); }
    void close() noexcept { if (handle_ != INVALID_HANDLE_VALUE) { CloseHandle(handle_); handle_ = INVALID_HANDLE_VALUE; } }
private:
    HANDLE handle_ = INVALID_HANDLE_VALUE;
};

static void write_new_text(const std::string &path, const std::string &text) {
    Jsonl output(path);
    output.line(text); // Jsonl appends the one required newline.
    output.durable();
}

struct Resources {
    ggml_backend_t backend = nullptr;
    ggml_context *context = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ~Resources() {
        if (buffer) ggml_backend_buffer_free(buffer);
        if (context) ggml_free(context);
        if (backend) ggml_backend_free(backend);
    }
};

static void cuda_ok(cudaError_t status, const char *operation) {
    if (status != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

static std::string hardware_json() {
    cudaDeviceProp properties{};
    int driver = 0, runtime = 0;
    cuda_ok(cudaGetDeviceProperties(&properties, 0), "device properties");
    cuda_ok(cudaDriverGetVersion(&driver), "driver version");
    cuda_ok(cudaRuntimeGetVersion(&runtime), "runtime version");
    char pci[64]{};
    cuda_ok(cudaDeviceGetPCIBusId(pci, sizeof(pci), 0), "PCI id");
    std::ostringstream uuid;
    uuid << "GPU-";
    for (int index = 0; index < 16; ++index) {
        if (index == 4 || index == 6 || index == 8 || index == 10) uuid << '-';
        uuid << std::hex << std::setw(2) << std::setfill('0') << int(static_cast<unsigned char>(properties.uuid.bytes[index]));
    }
    if (uuid.str() != expected_gpu_uuid || std::string(properties.name) != expected_gpu_name || properties.major != expected_cc_major || properties.minor != expected_cc_minor) {
        throw std::runtime_error("actual device does not match frozen native device");
    }
    std::ostringstream output;
    output << "{\"name\":" << quote(properties.name) << ",\"uuid\":" << quote(uuid.str())
           << ",\"cc_major\":" << properties.major << ",\"cc_minor\":" << properties.minor
           << ",\"SMs\":" << properties.multiProcessorCount << ",\"total_memory_bytes\":" << properties.totalGlobalMem
           << ",\"L2_bytes\":" << properties.l2CacheSize << ",\"PCI_bus_id\":" << quote(pci)
           << ",\"cuda_driver_api_version\":" << driver << ",\"cuda_runtime_version\":" << runtime << "}";
    return output.str();
}

static std::string scheduling_json() {
    DWORD_PTR process_affinity = 0, system_affinity = 0;
    if (!GetProcessAffinityMask(GetCurrentProcess(), &process_affinity, &system_affinity)) throw std::runtime_error("affinity unavailable");
    const DWORD priority = GetPriorityClass(GetCurrentProcess());
    const int thread_priority = GetThreadPriority(GetCurrentThread());
    if (priority != NORMAL_PRIORITY_CLASS || thread_priority != THREAD_PRIORITY_NORMAL) throw std::runtime_error("unexpected process/thread priority");
    std::ostringstream output;
    output << "{\"process_affinity\":" << process_affinity << ",\"system_affinity\":" << system_affinity
           << ",\"process_priority_class\":" << priority << ",\"caller_thread_priority\":" << thread_priority
           << ",\"caller_thread_id\":" << GetCurrentThreadId() << ",\"CPU_worker_pool_created\":false}";
    return output.str();
}

struct Options {
    bool run = false;
    bool identity_check = false;
    std::string config;
    std::string arm;
    std::string pair_id;
    std::string output;
};

static Options parse_options(int argc, char **argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--run") { options.run = true; continue; }
        if (argument == "--identity-check" || argument == "--check-only") { options.identity_check = true; continue; }
        if (index + 1 == argc) throw std::runtime_error("missing option value");
        const std::string value = argv[++index];
        if (argument == "--config") options.config = value;
        else if (argument == "--arm") options.arm = value;
        else if (argument == "--pair-id") options.pair_id = value;
        else if (argument == "--output") options.output = value;
        else throw std::runtime_error("unknown option " + argument);
    }
    if (options.run == options.identity_check) throw std::runtime_error("select exactly one of --identity-check or --run");
    if (options.run && (options.config.empty() || options.arm.empty() || options.pair_id.empty() || options.output.empty())) {
        throw std::runtime_error("--run requires --config, --arm control|buffered, --pair-id and new --output");
    }
    if (options.arm != "" && options.arm != "control" && options.arm != "buffered") throw std::runtime_error("arm outside frozen A/B set");
    return options;
}

static const ProbeConfig *find_config(const std::string &id) {
    for (const auto &config : frozen_configs) if (id == config.id) return &config;
    return nullptr;
}

static std::string argv_json(int argc, char **argv) {
    std::ostringstream output;
    output << '[';
    for (int index = 0; index < argc; ++index) {
        if (index) output << ',';
        output << quote(argv[index]);
    }
    return output << ']', output.str();
}

static std::string freeze_json() {
    std::ostringstream output;
    output << "{\"protocol_sha256\":" << quote(protocol_sha256)
           << ",\"frozen_file_set_sha256\":" << quote(frozen_file_set_sha256)
           << ",\"frozen_file_count\":" << (sizeof(frozen_files) / sizeof(frozen_files[0]))
           << ",\"frozen_module_count\":" << (sizeof(frozen_modules) / sizeof(frozen_modules[0]))
           << ",\"source_runtime_equivalence_proven\":false}";
    return output.str();
}

static std::string cache_semantics_json() {
    return "{\"graph_constructed_once\":true,\"backend_buffer_allocated_once\":true,\"input_uploaded_once\":true,\"formal_call_reset\":false,\"cache_sweep\":false,\"arm_difference\":\"validation_and_raw_write_cadence_only\"}";
}

static std::string setup_json(const Options &options, const ProbeConfig &config, const Resources &resources,
                              const std::vector<ggml_tensor *> &stages, const std::string &environment,
                              const std::string &scheduling, const std::string &hardware,
                              const std::string &modules_before, long long init_start, long long init_end,
                              long long build_start, long long build_end, long long allocation_start,
                              long long allocation_end, long long upload_start, long long upload_end) {
    std::ostringstream output;
    output << "{\"record\":\"setup\",\"config\":" << quote(options.config) << ",\"arm\":" << quote(options.arm)
           << ",\"pair_id\":" << quote(options.pair_id) << ",\"elements\":" << config.elements
           << ",\"requested_nodes\":" << config.nodes << ",\"actual_ggml_nodes\":" << stages.size()
           << ",\"observed_device_kernel_count\":null,\"allocated_buffer_bytes\":" << ggml_backend_buffer_get_size(resources.buffer)
           << ",\"tensor_payload_bytes\":" << config.elements * 4LL << ",\"allocated_payload_bytes\":" << (config.nodes + 1LL) * config.elements * 4LL
           << ",\"logical_read_bytes\":" << config.nodes * config.elements * 4LL << ",\"logical_write_bytes\":" << config.nodes * config.elements * 4LL
           << ",\"environment\":" << environment << ",\"scheduling\":" << scheduling << ",\"hardware_actual\":" << hardware
           << ",\"loaded_modules_before\":" << modules_before << ",\"qpc_init_start\":" << init_start << ",\"qpc_init_end\":" << init_end
           << ",\"qpc_graph_build_start\":" << build_start << ",\"qpc_graph_build_end\":" << build_end
           << ",\"qpc_allocation_start\":" << allocation_start << ",\"qpc_allocation_end\":" << allocation_end
           << ",\"qpc_upload_start\":" << upload_start << ",\"qpc_upload_end\":" << upload_end << ",\"graph_nodes\":[";
    for (std::size_t index = 0; index < stages.size(); ++index) {
        if (index) output << ',';
        const auto *tensor = stages[index];
        output << "{\"index\":" << index << ",\"name\":" << quote(ggml_get_name(tensor))
               << ",\"src0\":" << quote(ggml_get_name(tensor->src[0])) << ",\"operator\":\"SCALE\",\"dtype\":\"F32\",\"scale\":0.5,\"ne\":["
               << tensor->ne[0] << ',' << tensor->ne[1] << ',' << tensor->ne[2] << ',' << tensor->ne[3] << "]}";
    }
    return output << "]}", output.str();
}

struct RunSummary {
    graph_gap_probe::ArmResult arm{};
    bool modules_stable = false;
    bool quality_pass = false;
};

static RunSummary run_probe(const Options &options, Jsonl &raw, int argc, char **argv) {
    const auto *config = find_config(options.config);
    if (!config) throw std::runtime_error("configuration outside six frozen combinations");
    const std::string environment = environment_json(true);
    const std::string scheduling = scheduling_json();
    verify_files();
    NativeModuleApi api;
    PinnedFrozenModules<NativeModuleApi> pinned(api);
    pinned.preload(frozen_modules);
    verify_loaded_modules();

    Resources resources;
    const auto init_start = host_qpc();
    resources.backend = ggml_backend_cuda_init(0);
    if (!resources.backend) throw std::runtime_error("CUDA backend initialization failed");
    const auto init_end = host_qpc();
    nvtxMarkA("graph_submit/process_setup");
    const std::string hardware = hardware_json();
    const std::string modules_before = modules_json();

    std::vector<float> input(static_cast<std::size_t>(config->elements));
    for (std::size_t index = 0; index < input.size(); ++index) input[index] = graph_gap_reference_at(index, 0);
    const auto build_start = host_qpc();
    constexpr int capacity = 128;
    ggml_init_params parameters{ggml_tensor_overhead() * std::size_t(capacity) + ggml_graph_overhead_custom(capacity, false), nullptr, true};
    resources.context = ggml_init(parameters);
    if (!resources.context) throw std::runtime_error("GGML context allocation failed");
    auto *input_tensor = ggml_new_tensor_1d(resources.context, GGML_TYPE_F32, config->elements);
    ggml_set_name(input_tensor, "graph_input");
    std::vector<ggml_tensor *> stages;
    ggml_tensor *previous = input_tensor;
    for (int stage = 0; stage < config->nodes; ++stage) {
        auto *node = ggml_scale(resources.context, previous, 0.5f);
        const std::string name = "scale_" + std::to_string(stage + 1);
        ggml_set_name(node, name.c_str());
        if (node == previous || node->src[0] != previous || node->op != GGML_OP_SCALE || node->view_src != nullptr || !ggml_is_contiguous(node) || !ggml_backend_supports_op(resources.backend, node)) {
            throw std::runtime_error("graph is not a supported distinct SCALE dependency chain");
        }
        stages.push_back(node);
        previous = node;
    }
    auto *graph = ggml_new_graph_custom(resources.context, capacity, false);
    ggml_build_forward_expand(graph, stages.back());
    if (ggml_graph_n_nodes(graph) != config->nodes) throw std::runtime_error("actual GGML graph node count mismatch");
    for (int index = 0; index < config->nodes; ++index) {
        if (ggml_graph_node(graph, index) != stages[static_cast<std::size_t>(index)]) throw std::runtime_error("actual GGML graph dependency order mismatch");
    }
    const auto build_end = host_qpc();
    const auto allocation_start = host_qpc();
    resources.buffer = ggml_backend_alloc_ctx_tensors(resources.context, resources.backend);
    if (!resources.buffer) throw std::runtime_error("device buffer allocation failed");
    const auto allocation_end = host_qpc();
    for (std::size_t index = 0; index < stages.size(); ++index) {
        if (stages[index]->data == input_tensor->data) throw std::runtime_error("unexpected input/output alias");
        for (std::size_t earlier = 0; earlier < index; ++earlier) {
            if (stages[index]->data == stages[earlier]->data) throw std::runtime_error("unexpected intermediate alias");
        }
    }
    const auto upload_start = host_qpc();
    ggml_backend_tensor_set(input_tensor, input.data(), 0, input.size() * sizeof(float));
    const auto upload_end = host_qpc();
    raw.line(setup_json(options, *config, resources, stages, environment, scheduling, hardware, modules_before,
                        init_start, init_end, build_start, build_end, allocation_start, allocation_end, upload_start, upload_end));
    raw.durable();

    graph_gap_probe::Harness harness{resources.backend, graph, stages, input.size()};
    graph_gap_probe::ArmContext context{options.config, options.arm, argv_json(argc, argv), freeze_json(), cache_semantics_json()};
    auto emit = [&raw](const std::string &line) { raw.line(line); };
    RunSummary summary;
    summary.arm = options.arm == "control" ? graph_gap_probe::run_control_arm(harness, context, emit)
                                                : graph_gap_probe::run_buffered_arm(harness, context, emit);
    verify_files();
    verify_loaded_modules();
    const std::string modules_after = modules_json();
    summary.modules_stable = modules_before == modules_after;
    summary.quality_pass = summary.arm.graph_status_pass && summary.arm.math_pass && summary.modules_stable;
    raw.line("{\"record\":\"footer\",\"status\":" + quote(summary.quality_pass ? "complete" : "quality_failed")
             + ",\"arm\":" + quote(options.arm) + ",\"graph_calls\":" + std::to_string(summary.arm.graph_calls)
             + ",\"numeric_blocks\":" + std::to_string(summary.arm.numeric_blocks)
             + ",\"checked_values\":" + std::to_string(summary.arm.checked_values)
             + ",\"mismatches\":" + std::to_string(summary.arm.mismatches)
             + ",\"nonfinite\":" + std::to_string(summary.arm.nonfinite)
             + ",\"graph_status_pass\":" + (summary.arm.graph_status_pass ? "true" : "false")
             + ",\"math_pass\":" + (summary.arm.math_pass ? "true" : "false")
             + ",\"loaded_modules_after\":" + modules_after
             + ",\"modules_stable\":" + (summary.modules_stable ? "true" : "false")
             + ",\"calibration_ready\":false,\"trace_confirmation_required\":true,\"qpc_end\":" + std::to_string(host_qpc())
             + ",\"utc_end\":" + quote(utc()) + "}");
    raw.durable();
    return summary;
}

static std::string receipt_json(const std::string &status, const Options &options, const std::string &raw_path, const std::string &error) {
    std::ostringstream output;
    output << "{\"schema\":\"graph-gap-run-receipt/v1\",\"status\":" << quote(status)
           << ",\"raw_path\":" << quote(raw_path)
           << ",\"raw_sha256\":" << quote(file_hash(raw_path))
           << ",\"raw_bytes\":" << std::filesystem::file_size(std::filesystem::path(wide(raw_path)))
           << ",\"config\":" << quote(options.config) << ",\"arm\":" << quote(options.arm)
           << ",\"pair_id\":" << quote(options.pair_id) << ",\"protocol_sha256\":" << quote(protocol_sha256)
           << ",\"error\":" << (error.empty() ? "null" : quote(error))
           << ",\"calibration_ready\":false}";
    return output.str();
}

int main(int argc, char **argv) {
    std::unique_ptr<Jsonl> raw;
    Options options;
    try {
        options = parse_options(argc, argv);
        if (options.identity_check) {
            verify_files();
            environment_json(true);
            NativeModuleApi api;
            PinnedFrozenModules<NativeModuleApi> pinned(api);
            pinned.preload(frozen_modules);
            verify_loaded_modules();
            std::cout << "{\"status\":\"identity_and_loaded_dlls_verified_no_gpu_access\",\"protocol_sha256\":" << quote(protocol_sha256)
                      << ",\"loaded_modules\":" << modules_json() << "}\n";
            return 0;
        }
        raw = std::make_unique<Jsonl>(options.output);
        raw->line("{\"record\":\"header\",\"schema\":\"graph-gap-probe/v1\",\"utc_start\":" + quote(utc())
                  + ",\"qpc_start\":" + std::to_string(host_qpc()) + ",\"qpc_frequency\":" + std::to_string(qpf())
                  + ",\"pid\":" + std::to_string(GetCurrentProcessId()) + ",\"protocol_sha256\":" + quote(protocol_sha256)
                  + ",\"argv\":" + argv_json(argc, argv) + ",\"probe_cuda_events\":false,\"timing_clock\":\"absolute_QPC\",\"trace_clock_subtraction_allowed\":false}");
        raw->durable();
        const auto summary = run_probe(options, *raw, argc, argv);
        raw->close();
        write_new_text(options.output + ".receipt.json", receipt_json(summary.quality_pass ? "complete" : "quality_failed", options, options.output, ""));
        return summary.quality_pass ? 0 : 4;
    } catch (const std::exception &error) {
        const std::string message = error.what();
        if (raw) {
            try {
                raw->line("{\"record\":\"error\",\"status\":\"failed\",\"message\":" + quote(message)
                          + ",\"utc\":" + quote(utc()) + ",\"calibration_ready\":false}");
                raw->durable();
                raw->close();
                write_new_text(options.output + ".receipt.json", receipt_json("failed", options, options.output, message));
            } catch (...) {
                // Preserve original error on stderr even if the sidecar cannot be created.
            }
        }
        std::cerr << "{\"status\":\"failed\",\"message\":" << quote(message) << ",\"calibration_ready\":false}\n";
        return 2;
    }
}