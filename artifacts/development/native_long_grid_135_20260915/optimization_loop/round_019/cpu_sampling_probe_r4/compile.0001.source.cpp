// Prepared source only. This file has not been compiled or executed in this task.
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0602
#endif
#include <windows.h>
#include <powrprof.h>
#include <intrin.h>
#include "llama.h"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
constexpr unsigned WARMUP_CALLS = 16, STEADY_REPEATS = 64, PROCESS_COUNT = 3;
constexpr uint32_t INPUT_SEED = 20260916;
constexpr std::array<int, 3> VOCABS{32768, 131072, 262144};
static_assert(sizeof(void*) == 8 && sizeof(float) == 4 && sizeof(llama_token) == 4);
static_assert(sizeof(llama_token_data) == 12 && offsetof(llama_token_data, id) == 0 && offsetof(llama_token_data, logit) == 4 && offsetof(llama_token_data, p) == 8);
static_assert(offsetof(llama_token_data_array, data) == 0 && offsetof(llama_token_data_array, size) == 8 && offsetof(llama_token_data_array, selected) == 16 && offsetof(llama_token_data_array, sorted) == 24 && sizeof(llama_token_data_array) == 32);

static void require(bool good, const char* reason) { if (!good) throw std::runtime_error(reason); }
static std::string utf8(const std::wstring& value) {
    int bytes = WideCharToMultiByte(CP_UTF8, 0, value.data(), int(value.size()), nullptr, 0, nullptr, nullptr);
    std::string out(bytes, '\0');
    if (bytes) WideCharToMultiByte(CP_UTF8, 0, value.data(), int(value.size()), out.data(), bytes, nullptr, nullptr);
    return out;
}
static std::string quote(const std::string& text) {
    std::string out = "\"";
    for (unsigned char c : text) {
        if (c == '\\' || c == '"') { out += '\\'; out += char(c); }
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else if (c == '\t') out += "\\t";
        else if (c >= 32) out += char(c);
    }
    return out + '"';
}
static std::string hex(uint64_t value) { std::ostringstream out; out << "0x" << std::hex << value; return quote(out.str()); }
static uint32_t float_bits(float value) { uint32_t out; std::memcpy(&out, &value, 4); return out; }
static uint64_t bytes_fnv1a(const void* data, size_t bytes) {
    auto p = static_cast<const uint8_t*>(data); uint64_t out = UINT64_C(14695981039346656037);
    for (size_t i = 0; i < bytes; ++i) { out ^= p[i]; out *= UINT64_C(1099511628211); }
    return out;
}
static uint32_t random_u32(uint32_t& state) { state ^= state << 13; state ^= state >> 17; state ^= state << 5; return state; }
static std::vector<float> make_logits(int n, bool random_pattern) {
    std::vector<float> logits(n);
    // All sizes are powers of two below 2^24. Every value and division is exact
    // binary32; values are finite, distinct, and have a unique maximum.
    for (int i = 0; i < n; ++i) logits[i] = float(i - n/2) / float(n);
    if (random_pattern) {
        uint32_t state = INPUT_SEED;
        for (int i = n - 1; i > 0; --i) std::swap(logits[i], logits[random_u32(state) % uint32_t(i + 1)]);
    }
    return logits;
}

// Verbatim materialization statements from the locked common/sampling.cpp
// full-logits branch. Only function parameters/wrapper and noinline are added.
// This is SOURCE-EQUIVALENT EXTERNAL COMPILATION, not common.dll execution.
__declspec(noinline) static void candidate_loop(const float* logits, int n_vocab,
        std::vector<llama_token_data>& cur, llama_token_data_array& cur_p) {
    cur.resize(n_vocab);
    for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
        cur[token_id] = llama_token_data{token_id, logits[token_id], 0.0f};
    }
    cur_p = { cur.data(), cur.size(), -1, false };
}

static std::wstring loaded_path(HMODULE module) {
    std::wstring path(32768, L'\0');
    DWORD count = GetModuleFileNameW(module, path.data(), DWORD(path.size()));
    require(count > 0 && count < path.size(), "loaded module path unavailable");
    path.resize(count); return path;
}
static std::runtime_error win32_error(const std::string & action, DWORD code = GetLastError()) {
    LPWSTR raw = nullptr;
    const DWORD count = FormatMessageW(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
        nullptr, code, 0, reinterpret_cast<LPWSTR>(&raw), 0, nullptr);
    const std::string detail = count && raw ? utf8(std::wstring(raw, count)) : "FormatMessage unavailable";
    if (raw) LocalFree(raw);
    return std::runtime_error(action + " Win32Error=" + std::to_string(code) + " message=" + detail);
}
struct NativeSampler {
    HMODULE library = nullptr;
    decltype(&llama_sampler_init_top_k) init_top_k = nullptr;
    decltype(&llama_sampler_apply) apply = nullptr;
    decltype(&llama_sampler_free) free_sampler = nullptr;
    llama_sampler* sampler = nullptr;
    std::vector<HMODULE> dependency_handles;
    std::vector<DLL_DIRECTORY_COOKIE> directory_cookies;
    std::string module_evidence;
    template<class T> T symbol(const char* name) {
        FARPROC address = GetProcAddress(library, name);
        require(address != nullptr, "required original sampler export missing");
        HMODULE owner = nullptr;
        require(GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            reinterpret_cast<LPCWSTR>(address), &owner) && owner == library, "sampler export is not owned by selected llama.dll");
        return reinterpret_cast<T>(address);
    }
    void add_directory(const fs::path & path) {
        require(path.is_absolute() && fs::is_directory(path), "frozen DLL directory unavailable");
        const DLL_DIRECTORY_COOKIE cookie = AddDllDirectory(path.c_str());
        if (!cookie) throw win32_error("AddDllDirectory " + utf8(path.wstring()));
        directory_cookies.push_back(cookie);
    }
    HMODULE preload(const fs::path & path, const wchar_t * name) {
        require(path.is_absolute() && path.filename() == name && fs::is_regular_file(path), "frozen dependency path unavailable");
        HMODULE handle = LoadLibraryExW(path.c_str(), nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
        if (!handle) throw win32_error("LoadLibraryExW " + utf8(path.wstring()));
        const std::wstring actual = loaded_path(handle);
        if (_wcsicmp(actual.c_str(), path.lexically_normal().c_str()) != 0) {
            FreeLibrary(handle);
            throw std::runtime_error("loaded dependency path mismatch for " + utf8(path.wstring()));
        }
        dependency_handles.push_back(handle);
        return handle;
    }
    void append_evidence(const wchar_t * name, HMODULE handle) {
        const std::wstring actual = loaded_path(handle);
        if (module_evidence != "[") module_evidence += ',';
        module_evidence += "{\"name\":" + quote(utf8(name)) + ",\"actual_path\":" + quote(utf8(actual)) + "}";
    }
    void release_loader() noexcept {
        for (auto iterator = dependency_handles.rbegin(); iterator != dependency_handles.rend(); ++iterator) FreeLibrary(*iterator);
        dependency_handles.clear();
        for (auto iterator = directory_cookies.rbegin(); iterator != directory_cookies.rend(); ++iterator) RemoveDllDirectory(*iterator);
        directory_cookies.clear();
    }
    explicit NativeSampler(const fs::path& path, const fs::path & runtime_dir, const fs::path & cuda_dir) {
        require(path.is_absolute() && path.filename() == L"llama.dll" && path.parent_path() == runtime_dir, "absolute frozen llama.dll path required");
        try {
            // Two frozen absolute directories only. This does not mutate PATH or
            // broaden process search to arbitrary user directories.
            add_directory(runtime_dir);
            add_directory(cuda_dir);
            module_evidence = "[";
            for (const wchar_t * name : {L"cudart64_12.dll", L"cublasLt64_12.dll", L"cublas64_12.dll"}) {
                append_evidence(name, preload(cuda_dir / name, name));
            }
            for (const wchar_t * name : {L"ggml-base.dll", L"ggml-cpu.dll", L"ggml-cuda.dll", L"ggml.dll"}) {
                append_evidence(name, preload(runtime_dir / name, name));
            }
            library = LoadLibraryExW(path.c_str(), nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
            if (!library) throw win32_error("LoadLibraryExW " + utf8(path.wstring()));
            const std::wstring actual = loaded_path(library);
            if (_wcsicmp(actual.c_str(), path.lexically_normal().c_str()) != 0) throw std::runtime_error("loaded llama.dll path mismatch");
            append_evidence(L"llama.dll", library);
            module_evidence += ']';
            init_top_k = symbol<decltype(init_top_k)>("llama_sampler_init_top_k");
            apply = symbol<decltype(apply)>("llama_sampler_apply");
            free_sampler = symbol<decltype(free_sampler)>("llama_sampler_free");
            sampler = init_top_k(1); require(sampler != nullptr, "original top-k sampler initialization failed");
        } catch (...) {
            if (library) { FreeLibrary(library); library = nullptr; }
            release_loader();
            throw;
        }
    }
    ~NativeSampler() {
        if (sampler) free_sampler(sampler);
        if (library) FreeLibrary(library);
        release_loader();
    }
    NativeSampler(const NativeSampler&) = delete;
};

// The documented PROCESSOR_POWER_INFORMATION layout, locally named to avoid
// SDK revisions that omitted or later added the original typedef.
struct CpuPowerInfoRaw { ULONG number, max_mhz, current_mhz, limit_mhz, max_idle_state, current_idle_state; };
static_assert(sizeof(CpuPowerInfoRaw) == 24);
struct CpuObservation {
    unsigned logical = 0, max_mhz = 0, reported_mhz = 0, limit_mhz = 0;
    uint64_t affinity = 0;
    std::string json() const {
        return "{\"group\":0,\"logical_processor\":" + std::to_string(logical) + ",\"thread_affinity_mask\":" + hex(affinity)
            + ",\"os_reported_current_mhz\":" + std::to_string(reported_mhz) + ",\"os_max_mhz\":" + std::to_string(max_mhz)
            + ",\"os_limit_mhz\":" + std::to_string(limit_mhz) + "}";
    }
};
static std::string cpu_brand() {
    std::array<int, 4> registers{}; __cpuid(registers.data(), int(0x80000000));
    if (uint32_t(registers[0]) < 0x80000004) return "CPUID brand unavailable";
    char text[49]{};
    for (int i = 0; i < 3; ++i) { __cpuid(registers.data(), int(0x80000002 + i)); std::memcpy(text + i*16, registers.data(), 16); }
    return text;
}
static uint32_t cpuid_signature() {
    std::array<int, 4> registers{}; __cpuid(registers.data(), 1);
    return uint32_t(registers[0]);
}
struct ActualCpuIdentity {
    std::string brand; uint32_t signature = 0; unsigned group_count = 0, active_count = 0, group = 0, logical = 0; uint64_t affinity = 0;
    std::string json() const {
        return "{\"cpu_brand\":" + quote(brand) + ",\"cpuid_signature\":" + std::to_string(signature)
            + ",\"active_processor_group_count\":" + std::to_string(group_count)
            + ",\"active_processor_count_group0\":" + std::to_string(active_count)
            + ",\"group\":" + std::to_string(group) + ",\"logical_cpu\":" + std::to_string(logical)
            + ",\"thread_affinity_mask\":" + std::to_string(affinity) + "}";
    }
};
static ActualCpuIdentity actual_cpu_identity(unsigned selected) {
    const unsigned groups = GetActiveProcessorGroupCount();
    require(groups == 1, "this small probe supports one Windows processor group only");
    const DWORD count = GetActiveProcessorCount(0);
    require(selected < count && count <= 64, "selected logical CPU outside supported group");
    GROUP_AFFINITY affinity{}; require(GetThreadGroupAffinity(GetCurrentThread(), &affinity), "thread affinity query failed");
    PROCESSOR_NUMBER current{}; GetCurrentProcessorNumberEx(&current);
    require(affinity.Group == 0 && affinity.Mask == (KAFFINITY(1) << selected) && current.Group == 0 && current.Number == selected, "thread migrated or pinning differs");
    return {cpu_brand(), cpuid_signature(), groups, unsigned(count), 0, selected, uint64_t(affinity.Mask)};
}
static CpuObservation observe_cpu(unsigned selected) {
    const ActualCpuIdentity identity = actual_cpu_identity(selected);
    std::vector<CpuPowerInfoRaw> power(identity.active_count);
    require(CallNtPowerInformation(ProcessorInformation, nullptr, 0, power.data(), ULONG(power.size()*sizeof(CpuPowerInfoRaw))) == 0, "OS processor frequency report unavailable");
    auto entry = std::find_if(power.begin(), power.end(), [selected](const auto& item) { return item.number == selected; });
    require(entry != power.end() && entry->current_mhz > 0 && entry->max_mhz > 0 && entry->limit_mhz > 0, "OS frequency report incomplete");
    return {selected, entry->max_mhz, entry->current_mhz, entry->limit_mhz, identity.affinity};
}
struct ThreadPin {
    GROUP_AFFINITY previous{}; bool active = false;
    explicit ThreadPin(unsigned cpu) {
        require(GetActiveProcessorGroupCount() == 1 && cpu < GetActiveProcessorCount(0) && cpu < 64, "unsupported logical CPU");
        DWORD_PTR process_mask = 0, system_mask = 0;
        require(GetProcessAffinityMask(GetCurrentProcess(), &process_mask, &system_mask) && (process_mask & (DWORD_PTR(1) << cpu)), "requested CPU not allowed by process affinity");
        GROUP_AFFINITY chosen{}; chosen.Group = 0; chosen.Mask = KAFFINITY(1) << cpu;
        require(SetThreadGroupAffinity(GetCurrentThread(), &chosen, &previous), "thread pinning failed"); active = true;
    }
    ~ThreadPin() { if (active) SetThreadGroupAffinity(GetCurrentThread(), &previous, nullptr); }
};

template<class Function> static int64_t timed_ticks(Function&& function) {
    LARGE_INTEGER begin{}, end{};
    _ReadWriteBarrier(); require(QueryPerformanceCounter(&begin), "QPC begin failed"); _ReadWriteBarrier();
    function();
    _ReadWriteBarrier(); require(QueryPerformanceCounter(&end), "QPC end failed"); _ReadWriteBarrier();
    require(end.QuadPart >= begin.QuadPart, "QPC moved backwards");
    return end.QuadPart - begin.QuadPart;
}
static void check_candidate(const std::vector<float>& logits, const std::vector<llama_token_data>& cur, const llama_token_data_array& view) {
    require(cur.size() == logits.size() && view.data == cur.data() && view.size == cur.size() && view.selected == -1 && !view.sorted, "candidate view mismatch");
    for (size_t i = 0; i < cur.size(); ++i) require(cur[i].id == int(i) && float_bits(cur[i].logit) == float_bits(logits[i]) && float_bits(cur[i].p) == 0, "candidate materialization not exact");
}
static void check_topk(const std::vector<float>& logits, const std::vector<llama_token_data>& backing, const llama_token_data_array& view, int best) {
    require(view.data == backing.data() && view.size == 1 && view.sorted && view.selected == -1, "native top-k view mismatch");
    require(view.data[0].id == best && float_bits(view.data[0].logit) == float_bits(logits[best]) && float_bits(view.data[0].p) == 0, "native top-k output not exact");
}
struct StageResult {
    std::string name, implementation, first_use_scope;
    int64_t first_use = 0; std::vector<int64_t> steady;
    CpuObservation before, after;
    std::string json() const {
        std::string ticks = "["; for (size_t i=0;i<steady.size();++i) { if(i) ticks+=','; ticks+=std::to_string(steady[i]); } ticks+=']';
        const bool stable = before.max_mhz == after.max_mhz && before.reported_mhz == after.reported_mhz && before.limit_mhz == after.limit_mhz;
        return "{\"stage\":"+quote(name)+",\"implementation\":"+quote(implementation)+",\"first_use_scope\":"+quote(first_use_scope)
            +",\"first_use_ticks\":"+std::to_string(first_use)+",\"warmup_calls\":"+std::to_string(WARMUP_CALLS)
            +",\"steady_repeats\":"+std::to_string(STEADY_REPEATS)+",\"steady_raw_ticks\":"+ticks
            +",\"steady_regime\":\"warm_reused_buffers\",\"numeric_quality\":\"exact_pass\",\"quality_and_reset_outside_clock_window\":true"
            +",\"reported_frequency_fields_checked\":[\"os_max_mhz\",\"os_reported_current_mhz\",\"os_limit_mhz\"]"
            +",\"reported_frequency_stable\":"+(stable?"true":"false")+",\"timing_usable\":"+(stable?"true":"false")+",\"diagnostic_only\":"+(stable?"false":"true")+",\"frequency_changed_diagnostic_only\":"+(stable?"false":"true")
            +",\"cpu_before\":"+before.json()+",\"cpu_after\":"+after.json()+"}";
    }
};
static std::string run_case(NativeSampler& native, int n, bool random_pattern, unsigned cpu) {
    const auto logits = make_logits(n, random_pattern);
    const int best = int(std::max_element(logits.begin(), logits.end()) - logits.begin());
    require(std::count(logits.begin(), logits.end(), logits[best]) == 1, "synthetic pattern must have a unique exact maximum");
    std::vector<llama_token_data> baseline(n), cur, work(n);
    for (int i=0;i<n;++i) baseline[i] = {i, logits[i], 0.0f};
    llama_token_data_array cur_view{};
    StageResult candidate{"candidate_loop", "source_equivalent_external_compilation", "first_vector_resize_including_allocation_and_first_touch_not_a_cold_cache_claim"};
    candidate.steady.reserve(STEADY_REPEATS); candidate.before = observe_cpu(cpu);
    candidate.first_use = timed_ticks([&] { candidate_loop(logits.data(), n, cur, cur_view); });
    check_candidate(logits, cur, cur_view);
    auto* stable_storage = cur.data(); const auto stable_capacity = cur.capacity();
    for (unsigned repeat=0;repeat<WARMUP_CALLS;++repeat) { candidate_loop(logits.data(), n, cur, cur_view); check_candidate(logits, cur, cur_view); }
    for (unsigned repeat=0;repeat<STEADY_REPEATS;++repeat) {
        const int64_t ticks = timed_ticks([&] { candidate_loop(logits.data(), n, cur, cur_view); });
        require(cur.data() == stable_storage && cur.capacity() == stable_capacity, "steady candidate buffer reallocated");
        check_candidate(logits, cur, cur_view); candidate.steady.push_back(ticks);
    }
    candidate.after = observe_cpu(cpu);
    StageResult topk{"original_dll_topk_apply", "original_llama_dll_export", "first_apply_on_prepared_buffer_excludes_dll_load_sampler_init_allocation_and_reset"};
    topk.steady.reserve(STEADY_REPEATS); topk.before = observe_cpu(cpu);
    llama_token_data_array view{};
    auto reset = [&] { std::memcpy(work.data(), baseline.data(), size_t(n)*sizeof(llama_token_data)); view = {work.data(), size_t(n), -1, false}; _ReadWriteBarrier(); };
    reset(); topk.first_use = timed_ticks([&] { native.apply(native.sampler, &view); }); check_topk(logits, work, view, best);
    for (unsigned repeat=0;repeat<WARMUP_CALLS;++repeat) { reset(); native.apply(native.sampler, &view); check_topk(logits, work, view, best); }
    for (unsigned repeat=0;repeat<STEADY_REPEATS;++repeat) {
        reset(); const int64_t ticks = timed_ticks([&] { native.apply(native.sampler, &view); });
        check_topk(logits, work, view, best); topk.steady.push_back(ticks);
    }
    topk.after = observe_cpu(cpu);
    return "{\"vocabulary_size\":"+std::to_string(n)+",\"pattern\":"+quote(random_pattern?"deterministic_random_permutation":"monotone_ascending")
        +",\"split\":"+quote(n==131072?"holdout":"train")+",\"input_seed\":"+std::to_string(INPUT_SEED)
        +",\"input_bits_fnv1a64\":"+hex(bytes_fnv1a(logits.data(),logits.size()*sizeof(float)))
        +",\"expected_unique_max_token\":"+std::to_string(best)+",\"expected_max_logit_bits\":"+std::to_string(float_bits(logits[best]))
        +",\"candidate_record_bytes\":12,\"top_k\":1,\"native_sampler_reused_stateless\":true,\"native_topk_reset_before_every_call\":true"
        +",\"first_use_not_pooled_with_steady\":true,\"stages\":["+candidate.json()+","+topk.json()+"]}";
}
struct NewResultFile {
    HANDLE handle = INVALID_HANDLE_VALUE;
    explicit NewResultFile(const fs::path& path) { require(path.is_absolute(), "absolute new result path required"); handle = CreateFileW(path.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr); require(handle != INVALID_HANDLE_VALUE, "refuse existing or unwritable result"); }
    void write(const std::string& data) { DWORD done=0; require(data.size()<DWORD(-1) && WriteFile(handle,data.data(),DWORD(data.size()),&done,nullptr) && done==data.size() && FlushFileBuffers(handle), "result write failed"); }
    ~NewResultFile() { if(handle != INVALID_HANDLE_VALUE) CloseHandle(handle); }
};
int wmain(int argc, wchar_t** argv) {
    try {
        fs::path dll, runtime_dir, cuda_dir, output; int process_index=-1, cpu=-1; bool approved=false, identity_only=false;
        for(int i=1;i<argc;++i) {
            const std::wstring option=argv[i];
            if(option==L"--measure-root-approved" || option==L"--identity-root-approved") approved=true;
            else if(option==L"--identity-only") identity_only=true;
            else { require(i+1<argc,"option value missing"); const std::wstring value=argv[++i];
                if(option==L"--native-dll") dll=value; else if(option==L"--native-runtime-dir") runtime_dir=value; else if(option==L"--cuda-dependency-dir") cuda_dir=value; else if(option==L"--output") output=value;
                else if(option==L"--process-index") process_index=std::stoi(value);
                else if(option==L"--cpu-index") cpu=std::stoi(value); else throw std::runtime_error("unknown option");
            }
        }
        if (identity_only) {
            require(approved && cpu>=0 && cpu<64 && !output.empty(), "explicit root identity approval, cpu-index and output required");
            NewResultFile file(output); ThreadPin pin{unsigned(cpu)};
            const ActualCpuIdentity identity = actual_cpu_identity(unsigned(cpu));
            file.write("{\"schema\":\"cpu-sampling-probe-actual-cpu-identity/v1\",\"status\":\"complete\",\"GPU_context_created\":false,\"model_loaded\":false,\"timing_values_produced\":false,\"actual_cpu_identity\":" + identity.json() + "}");
            return 0;
        }
        require(dll.is_absolute(), "relative native DLL path is forbidden");
        require(approved && process_index>=0 && process_index<int(PROCESS_COUNT) && cpu>=0 && cpu<64 && !dll.empty() && runtime_dir.is_absolute() && cuda_dir.is_absolute() && !output.empty(), "explicit root measurement approval, process-index 0..2, cpu-index, native-dll, frozen runtime/cuda directories and output required");
        NewResultFile file(output); std::string completed="[", metadata; int result_code=0;
        try {
            ThreadPin pin{unsigned(cpu)};
            const ActualCpuIdentity identity = actual_cpu_identity(unsigned(cpu));
            LARGE_INTEGER frequency{}; require(QueryPerformanceFrequency(&frequency) && frequency.QuadPart>0,"QPC frequency unavailable");
            NativeSampler native(dll.lexically_normal(), runtime_dir.lexically_normal(), cuda_dir.lexically_normal());
            std::string observer_ticks="[";
            for(unsigned i=0;i<64;++i) { if(i) observer_ticks+=','; observer_ticks+=std::to_string(timed_ticks([] { _ReadWriteBarrier(); })); }
            observer_ticks+=']';
            metadata="\"schema\":\"cpu-sampling-independent-probe/v4\",\"GPU_context_created\":false,\"GPU_context_created_by_probe_calls\":false,\"no_probe_gpu_api_calls\":true,\"dependency_dlls_loaded\":true,\"dependency_loading_is_not_GPU_measured\":true,\"no_dllmain_context_claim\":true,\"model_loaded\":false,\"full_sampler_chain_measured\":false"
                ",\"process_index\":"+std::to_string(process_index)+",\"process_id\":"+std::to_string(GetCurrentProcessId())+",\"thread_id\":"+std::to_string(GetCurrentThreadId())
                +",\"logical_cpu\":"+std::to_string(cpu)+",\"actual_cpu_identity\":"+identity.json()+",\"clock\":\"QueryPerformanceCounter\",\"qpc_frequency\":"+std::to_string(frequency.QuadPart)
                +",\"observer_empty_bracket_ticks\":"+observer_ticks+",\"observer_ticks_subtracted\":false"
                +",\"os_frequency_report_is_not_instantaneous_turbo_measurement\":true,\"source_loop_binary_equivalence_claimed\":false,\"loaded_native_modules\":"+native.module_evidence;
            for(int offset=0;offset<6;++offset) {
                const int case_id=(process_index*2+offset)%6;
                std::string item=run_case(native,VOCABS[case_id/2],bool(case_id%2),unsigned(cpu));
                if(completed!="[") completed+=','; completed+=item;
            }
            completed+=']'; file.write("{"+metadata+",\"status\":\"complete\",\"cases\":"+completed+"}");
        } catch(const std::exception& error) {
            result_code=4; completed+=']'; file.write("{"+(metadata.empty()?"":metadata+",")+"\"status\":\"failed\",\"error\":"+quote(error.what())+",\"completed_cases\":"+completed+"}");
        }
        return result_code;
    } catch(const std::exception& error) { std::cerr<<error.what()<<"\n"; return 2; }
}


