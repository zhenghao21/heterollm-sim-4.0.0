// CUPTI activity boundaries for synthetic typed chains; no model is executed.
// Reuse the exact kernel body and resident GPU gate from the timer experiment.
#define main launch_gap_reference_main
#include "cuda_graph_launch_gap_microbench.cu"
#undef main
#include <cupti.h>
#include <fstream>
#include <mutex>

#define CUPTI_CHECK(call) do { auto status = (call); if (status != CUPTI_SUCCESS) { const char *message{}; cuptiGetResultString(status, &message); throw std::runtime_error(std::string(#call) + ": " + (message ? message : "CUPTI error")); } } while (0)
struct Activity { int kind; U64 start, end, bytes, graph_node; unsigned graph; };
static std::vector<Activity> recorded;
static std::mutex recorded_mutex;
static std::atomic<unsigned long long> dropped{0};
static std::atomic<bool> callback_error{false};
static void CUPTIAPI request_buffer(uint8_t **buffer, size_t *size, size_t *maximum) {
    *size = 8 * 1024 * 1024; *maximum = 0;
    *buffer = static_cast<uint8_t *>(_aligned_malloc(*size, 8));
    if (!*buffer) callback_error.store(true);
}
static void CUPTIAPI complete_buffer(CUcontext context, uint32_t stream, uint8_t *buffer, size_t, size_t valid) {
    CUpti_Activity *base{};
    while (true) {
        const auto status = cuptiActivityGetNextRecord(buffer, valid, &base);
        if (status == CUPTI_ERROR_MAX_LIMIT_REACHED) break;
        if (status != CUPTI_SUCCESS) { callback_error.store(true); break; }
        Activity item{}; bool keep = false;
        if (base->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL || base->kind == CUPTI_ACTIVITY_KIND_KERNEL) {
            const auto *record = reinterpret_cast<const CUpti_ActivityKernel9 *>(base);
            if (record->name && std::string(record->name).find("timed_body") != std::string::npos) {
                item = {0, record->start, record->end, 0, record->graphNodeId, record->graphId}; keep = true;
            }
        } else if (base->kind == CUPTI_ACTIVITY_KIND_MEMCPY) {
            const auto *record = reinterpret_cast<const CUpti_ActivityMemcpy6 *>(base);
            if (record->copyKind == CUPTI_ACTIVITY_MEMCPY_KIND_DTOD) {
                item = {1, record->start, record->end, record->bytes, record->graphNodeId, record->graphId}; keep = true;
            }
        }
        if (keep) { std::lock_guard<std::mutex> lock(recorded_mutex); recorded.push_back(item); }
    }
    size_t count{};
    if (cuptiActivityGetNumDroppedRecords(context, stream, &count) != CUPTI_SUCCESS) callback_error.store(true);
    dropped.fetch_add(count);
    _aligned_free(buffer);
}
struct Node { int kind, dependency; size_t bytes; };
static std::vector<Node> read_nodes(const std::string &path) {
    std::ifstream input(path); std::string magic, label; int count{};
    input >> magic >> label >> count;
    if (magic != "heterollm.synthetic-chain/v1" || count < 2 || count > 8192) throw std::runtime_error("invalid typed chain header");
    std::vector<Node> result;
    for (int i = 0; i < count; ++i) {
        Node node{}; int copy_kind{}; input >> node.kind >> node.dependency >> node.bytes >> copy_kind;
        if (!input || (node.kind != 0 && node.kind != 1) || node.dependency < 0 || node.dependency > 1 ||
            (i == 0 && node.dependency) || (node.kind == 1 && (node.dependency || !node.bytes || copy_kind != 3)))
            throw std::runtime_error("unsupported typed chain node");
        if (node.kind == 0 && (node.bytes || copy_kind)) throw std::runtime_error("kernel cannot declare memcpy bytes");
        result.push_back(node);
    }
    std::string extra; if (input >> extra) throw std::runtime_error("trailing typed chain data");
    return result;
}
static void typed_enqueue(const Config &cfg, const std::vector<Node> &nodes, cudaStream_t stream,
                          U64 *times, unsigned int *sink, void *source, void *destination) {
    for (int i = 0; i < static_cast<int>(nodes.size()); ++i) {
        const auto &node = nodes[i];
        if (node.kind == 1) CHECK(cudaMemcpyAsync(destination, source, node.bytes, cudaMemcpyDeviceToDevice, stream));
        else if (node.dependency) {
            cudaLaunchAttribute attr{}; attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
            attr.val.programmaticStreamSerializationAllowed = 1;
            cudaLaunchConfig_t launch{}; launch.gridDim = dim3(cfg.blocks); launch.blockDim = dim3(cfg.threads);
            launch.stream = stream; launch.attrs = &attr; launch.numAttrs = 1;
            CHECK(cudaLaunchKernelEx(&launch, timed_body<true>, times, sink, i, cfg.blocks, cfg.body_ns));
        } else {
            timed_body<false><<<cfg.blocks, cfg.threads, 0, stream>>>(times, sink, i, cfg.blocks, cfg.body_ns);
            CHECK(cudaPeekAtLastError());
        }
    }
}
struct TypedSample { std::string mode; double event_ns; std::vector<Activity> nodes; };
int main(int argc, char **argv) {
    try {
        Config cfg; bool activity = true; std::string structure;
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i]; if (i + 1 >= argc) throw std::runtime_error("missing value");
            std::string value = argv[++i];
            if (arg == "--structure") structure = value;
            else if (arg == "--repetitions") cfg.repetitions = std::stoi(value);
            else if (arg == "--body-ns") cfg.body_ns = std::stoull(value);
            else if (arg == "--blocks") cfg.blocks = std::stoi(value);
            else if (arg == "--threads") cfg.threads = std::stoi(value);
            else if (arg == "--activity" && (value == "0" || value == "1")) activity = value == "1";
            else throw std::runtime_error("unsupported argument");
        }
        const auto nodes = read_nodes(structure); cfg.nodes = nodes.size();
        if (cfg.repetitions < 3 || cfg.repetitions > 999 || cfg.body_ns > 100000 || cfg.blocks < 1 || cfg.blocks > 1024 || cfg.threads < 1 || cfg.threads > 1024)
            throw std::runtime_error("probe configuration outside bounded domain");
        if (activity) {
            CUPTI_CHECK(cuptiActivityRegisterCallbacks(request_buffer, complete_buffer));
            CUPTI_CHECK(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
            CUPTI_CHECK(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY));
        }
        int device{}, driver{}, runtime{}; cudaDeviceProp prop{};
        CHECK(cudaGetDevice(&device)); CHECK(cudaGetDeviceProperties(&prop, device));
        CHECK(cudaDriverGetVersion(&driver)); CHECK(cudaRuntimeGetVersion(&runtime));
        cudaStream_t stream{}; cudaEvent_t begin{}, end{}; cudaGraph_t graph{}; cudaGraphExec_t steady{};
        U64 *times{}; unsigned int *sink{}, *host_flags{}, *device_flags{}; void *source{}, *destination{};
        size_t copy_size = 1; for (const auto &node : nodes) copy_size = std::max(copy_size, node.bytes);
        CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CHECK(cudaEventCreate(&begin)); CHECK(cudaEventCreate(&end));
        CHECK(cudaMalloc(&times, sizeof(U64) * cfg.nodes * cfg.blocks * 2));
        CHECK(cudaMalloc(&sink, sizeof(unsigned int) * cfg.nodes * cfg.blocks));
        CHECK(cudaMemset(sink, 0, sizeof(unsigned int) * cfg.nodes * cfg.blocks));
        CHECK(cudaMalloc(&source, copy_size)); CHECK(cudaMalloc(&destination, copy_size));
        CHECK(cudaMemset(source, 1, copy_size)); CHECK(cudaMemset(destination, 0, copy_size));
        CHECK(cudaHostAlloc(&host_flags, sizeof(unsigned int) * 96, cudaHostAllocMapped | cudaHostAllocWriteCombined));
        CHECK(cudaHostGetDevicePointer(&device_flags, host_flags, 0));
        CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        typed_enqueue(cfg, nodes, stream, times, sink, source, destination);
        CHECK(cudaStreamEndCapture(stream, &graph));
        size_t graph_nodes{}; CHECK(cudaGraphGetNodes(graph, nullptr, &graph_nodes));
        if (graph_nodes != nodes.size()) throw std::runtime_error("captured typed node count changed");
        CHECK(cudaGraphInstantiate(&steady, graph, nullptr, nullptr, 0));
        CHECK(cudaGraphLaunch(steady, stream)); CHECK(cudaStreamSynchronize(stream));
        std::vector<TypedSample> samples; std::mt19937 random(1729);
        std::vector<std::string> modes{"ordinary", "first_launch", "replay"};
        for (int repeat = -2; repeat < cfg.repetitions; ++repeat) {
            std::shuffle(modes.begin(), modes.end(), random);
            for (const auto &mode : modes) {
                cudaGraphExec_t first{};
                if (mode == "first_launch") CHECK(cudaGraphInstantiate(&first, graph, nullptr, nullptr, 0));
                if (activity) CUPTI_CHECK(cuptiActivityFlushAll(0));
                { std::lock_guard<std::mutex> lock(recorded_mutex); recorded.clear(); }
                host_flags[0] = host_flags[32] = host_flags[64] = 0;
                std::atomic_thread_fence(std::memory_order_seq_cst);
                mapped_gpu_gate<<<1, 1, 0, stream>>>(device_flags); CHECK(cudaPeekAtLastError());
                auto query = cudaStreamQuery(stream); if (query != cudaSuccess && query != cudaErrorNotReady) CHECK(query);
                const auto deadline = Clock::now() + std::chrono::milliseconds(400);
                while (!gpu_gate_ready(host_flags) && Clock::now() < deadline) std::this_thread::yield();
                if (!gpu_gate_ready(host_flags)) { release_gpu_gate(host_flags); cudaStreamSynchronize(stream); throw std::runtime_error("GPU gate did not become resident"); }
                try {
                    CHECK(cudaEventRecord(begin, stream));
                    if (mode == "ordinary") typed_enqueue(cfg, nodes, stream, times, sink, source, destination);
                    else CHECK(cudaGraphLaunch(first ? first : steady, stream));
                    CHECK(cudaEventRecord(end, stream));
                    query = cudaEventQuery(end); if (query != cudaSuccess && query != cudaErrorNotReady) CHECK(query);
                } catch (...) { release_gpu_gate(host_flags); cudaStreamSynchronize(stream); throw; }
                release_gpu_gate(host_flags); CHECK(cudaStreamSynchronize(stream));
                if (host_flags[32]) throw std::runtime_error("GPU gate expired; invalid preloaded sample");
                float elapsed{}; CHECK(cudaEventElapsedTime(&elapsed, begin, end));
                if (activity) CUPTI_CHECK(cuptiActivityFlushAll(0));
                if (callback_error.load() || dropped.load()) throw std::runtime_error("incomplete CUPTI activity record set");
                TypedSample sample{mode, elapsed * 1.0e6, {}};
                if (activity) {
                    { std::lock_guard<std::mutex> lock(recorded_mutex); sample.nodes = recorded; }
                    std::sort(sample.nodes.begin(), sample.nodes.end(), [](const auto &a, const auto &b) { return a.start < b.start; });
                    if (sample.nodes.size() != nodes.size()) throw std::runtime_error("CUPTI typed node inventory mismatch: got " + std::to_string(sample.nodes.size()) + " expected " + std::to_string(nodes.size()));
                    const auto origin = sample.nodes.front().start;
                    for (size_t index = 0; index < nodes.size(); ++index) {
                        auto &record = sample.nodes[index];
                        if (!record.start || record.end < record.start || record.kind != nodes[index].kind || record.bytes != nodes[index].bytes ||
                            (mode == "ordinary" ? record.graph != 0 : record.graph == 0)) throw std::runtime_error("CUPTI activity cannot be paired to ordered typed chain");
                        record.start -= origin; record.end -= origin;
                    }
                }
                if (repeat >= 0) samples.push_back(std::move(sample));
                if (first) CHECK(cudaGraphExecDestroy(first));
            }
        }
        std::cout << "{\"schema\":\"heterollm.cuda-typed-activity/v1\",\"target_llm_latency_used\":false,\"activity_enabled\":" << (activity ? "true" : "false")
                  << ",\"hardware_id\":\"" << prop.name << "\",\"architecture\":\"sm_" << prop.major << prop.minor
                  << "\",\"driver_version\":\"" << driver << "\",\"runtime_version\":\"" << runtime << "\",\"node_count\":" << nodes.size()
                  << ",\"blocks\":" << cfg.blocks << ",\"threads\":" << cfg.threads << ",\"requested_body_ns\":" << cfg.body_ns << ",\"nodes\":[";
        for (size_t index = 0; index < nodes.size(); ++index) { if (index) std::cout << ','; const auto &node = nodes[index]; std::cout << '[' << node.kind << ',' << node.dependency << ',' << node.bytes << ']'; }
        std::cout << "],\"samples\":[";
        for (size_t index = 0; index < samples.size(); ++index) {
            if (index) std::cout << ','; const auto &sample = samples[index];
            std::cout << "{\"mode\":\"" << sample.mode << "\",\"device_event_ns\":" << sample.event_ns << ",\"activities\":[";
            for (size_t i = 0; i < sample.nodes.size(); ++i) { if (i) std::cout << ','; const auto &node = sample.nodes[i]; std::cout << '[' << node.kind << ',' << node.start << ',' << node.end << ',' << node.bytes << ',' << node.graph_node << ']'; }
            std::cout << "]}";
        }
        std::cout << "]}\n";
        CHECK(cudaGraphExecDestroy(steady)); CHECK(cudaGraphDestroy(graph));
        CHECK(cudaEventDestroy(begin)); CHECK(cudaEventDestroy(end)); CHECK(cudaFree(times)); CHECK(cudaFree(sink));
        CHECK(cudaFree(source)); CHECK(cudaFree(destination)); CHECK(cudaFreeHost(host_flags)); CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 2; }
}
