// Diagnostic-only instrumentation for the pinned llama.cpp CUDA backend.
// Included immediately before ggml_cuda_graph_update_required. No LLM timings
// are consumed; snapshots are captured before CUDA Graph lifecycle decisions.
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#include <cuda.h>

static FILE * heterollm_graph_trace_file() {
    static FILE * file = []() -> FILE * {
        const char * path = std::getenv("HETEROLLM_CUDA_GRAPH_TRACE");
        if (!path || !*path) return nullptr;
        FILE * result = std::fopen(path, "wb");
        if (!result) GGML_ABORT("cannot open HETEROLLM_CUDA_GRAPH_TRACE");
        return result;
    }();
    return file;
}

static thread_local uint64_t heterollm_graph_call_id = 0;
static std::mutex heterollm_graph_trace_mutex;

// Capture-frontier introspection does not submit kernels or change the graph.
// Record the source operation/fusion owning each emitted CUDA chain segment;
// GGML nodes and CUDA nodes are deliberately not assumed to be one-to-one.
struct heterollm_graph_frontier {
    cudaGraph_t graph = nullptr;
    std::vector<cudaGraphNode_t> dependencies;
    bool active = false;
};

static heterollm_graph_frontier heterollm_graph_capture_frontier(cudaStream_t stream) {
    heterollm_graph_frontier result;
    if (std::getenv("HETEROLLM_CUDA_GRAPH_CAPTURE_ONLY") == nullptr) return result;
    cudaStreamCaptureStatus status;
    unsigned long long id = 0;
    const cudaGraphNode_t * dependencies = nullptr;
    size_t count = 0;
    CUDA_CHECK(cudaStreamGetCaptureInfo_v2(stream, &status, &id, &result.graph, &dependencies, &count));
    if (status != cudaStreamCaptureStatusActive || !result.graph)
        GGML_ABORT("source dispatch ownership requires active stream capture");
    if (count) result.dependencies.assign(dependencies, dependencies + count);
    result.active = true;
    return result;
}

static void heterollm_graph_trace_string(FILE * file, const char * value) {
    std::fputc('"', file);
    for (const unsigned char * p = reinterpret_cast<const unsigned char *>(value); *p; ++p) {
        if (*p == '"' || *p == '\\') std::fprintf(file, "\\%c", *p);
        else if (*p < 0x20) std::fprintf(file, "\\u%04x", unsigned(*p));
        else std::fputc(*p, file);
    }
    std::fputc('"', file);
}

static void heterollm_graph_trace_tensor(FILE * file, const ggml_tensor * tensor) {
    std::fprintf(file, "{\"name\":");
    heterollm_graph_trace_string(file, tensor->name);
    std::fprintf(file, ",\"op\":");
    heterollm_graph_trace_string(file, ggml_op_name(tensor->op));
    std::fprintf(file, ",\"type\":");
    heterollm_graph_trace_string(file, ggml_type_name(tensor->type));
    std::fprintf(file, ",\"tensor\":\"%p\",\"data\":\"%p\",\"ne\":[", (const void *) tensor, tensor->data);
    for (int j = 0; j < GGML_MAX_DIMS; ++j) std::fprintf(file, "%s%lld", j ? "," : "", (long long) tensor->ne[j]);
    std::fprintf(file, "],\"nb\":[");
    for (int j = 0; j < GGML_MAX_DIMS; ++j) std::fprintf(file, "%s%llu", j ? "," : "", (unsigned long long) tensor->nb[j]);
    std::fprintf(file, "]}");
}

static void heterollm_graph_trace_dispatch(ggml_backend_cuda_context * ctx, ggml_cgraph * cgraph,
        int first, int last, const heterollm_graph_frontier & before) {
    if (!before.active) return;
    const auto after = heterollm_graph_capture_frontier(ctx->stream());
    if (!after.active || after.graph != before.graph || first < 0 || last < first || last >= cgraph->n_nodes)
        GGML_ABORT("source dispatch ownership crossed a capture boundary");
    FILE * file = heterollm_graph_trace_file();
    if (!file) GGML_ABORT("source dispatch ownership requires trace output");
    std::lock_guard<std::mutex> lock(heterollm_graph_trace_mutex);
    std::fprintf(file, "{\"kind\":\"source_dispatch\",\"schema\":\"heterollm.cuda-source-dispatch/v1\","
                 "\"source_revision\":\"d3146f2b56c2db4711ac8391871c9e529d1946d7\",\"capture_only\":true,"
                 "\"call_id\":%llu,\"first_ggml_index\":%d,\"last_ggml_index\":%d,\"before\":[",
                 (unsigned long long) heterollm_graph_call_id, first, last);
    for (size_t j = 0; j < before.dependencies.size(); ++j)
        std::fprintf(file, "%s\"%p\"", j ? "," : "", (void *) before.dependencies[j]);
    std::fprintf(file, "],\"after\":[");
    for (size_t j = 0; j < after.dependencies.size(); ++j)
        std::fprintf(file, "%s\"%p\"", j ? "," : "", (void *) after.dependencies[j]);
    std::fprintf(file, "],\"source_nodes\":[");
    for (int i = first; i <= last; ++i) {
        const auto * node = cgraph->nodes[i];
        if (i != first) std::fputc(',', file);
        std::fprintf(file, "{\"ggml_index\":%d,\"output\":", i);
        heterollm_graph_trace_tensor(file, node);
        std::fprintf(file, ",\"sources\":[");
        bool comma = false;
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            if (!node->src[j]) continue;
            if (comma) std::fputc(',', file);
            comma = true;
            std::fprintf(file, "{\"slot\":%d,\"tensor\":", j);
            heterollm_graph_trace_tensor(file, node->src[j]);
            std::fputc('}', file);
        }
        std::fprintf(file, "]}");
    }
    std::fprintf(file, "]}\n");
    std::fflush(file);
}

static void heterollm_graph_trace_event(const char * event, const void * key) {
    FILE * file = heterollm_graph_trace_file();
    if (!file) return;
    std::lock_guard<std::mutex> lock(heterollm_graph_trace_mutex);
    std::fprintf(file, "{\"kind\":\"event\",\"call_id\":%llu,\"event\":\"%s\",\"graph_key\":\"%p\"}\n",
                 (unsigned long long) heterollm_graph_call_id, event, key);
    std::fflush(file);
}

static void heterollm_graph_trace_snapshot(ggml_backend_cuda_context * ctx, ggml_cgraph * cgraph) {
    FILE * file = heterollm_graph_trace_file();
    if (!file) return;
    static uint64_t next_call = 0;
    static std::unordered_map<std::string, uint64_t> property_ids;
    std::lock_guard<std::mutex> lock(heterollm_graph_trace_mutex);
    heterollm_graph_call_id = ++next_call;
    std::vector<uint64_t> refs;
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        ggml_cuda_graph::node_properties prop = {};
        std::memcpy(&prop.node, cgraph->nodes[i], sizeof(ggml_tensor));
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            if (cgraph->nodes[i]->src[j]) {
                prop.node_src_data_ptrs[j] = cgraph->nodes[i]->src[j]->data;
                std::memcpy(prop.node_src_ne[j], cgraph->nodes[i]->src[j]->ne, sizeof(prop.node_src_ne[j]));
                std::memcpy(prop.node_src_nb[j], cgraph->nodes[i]->src[j]->nb, sizeof(prop.node_src_nb[j]));
            }
        }
        std::string bytes(reinterpret_cast<const char *>(&prop), sizeof(prop));
        auto found = property_ids.find(bytes);
        uint64_t id;
        if (found == property_ids.end()) {
            id = property_ids.size() + 1;
            property_ids.emplace(bytes, id);
            std::fprintf(file, "{\"kind\":\"node_property\",\"id\":%llu,\"bytes\":\"", (unsigned long long) id);
            for (unsigned char byte : bytes) std::fprintf(file, "%02x", byte);
            std::fprintf(file, "\"}\n");
        } else {
            id = found->second;
        }
        refs.push_back(id);
    }
    const char * label = std::getenv("HETEROLLM_CUDA_GRAPH_LABEL");
#ifdef _WIN32
    char label_buffer[256] = {};
    if (GetEnvironmentVariableA("HETEROLLM_CUDA_GRAPH_LABEL", label_buffer, sizeof(label_buffer))) label = label_buffer;
#endif
    // Labels are controlled by our harness (letters, digits, underscore, colon).
    if (!label) label = "native_diagnostic";
    const bool dry_run = std::getenv("HETEROLLM_CUDA_GRAPH_DRY_RUN") != nullptr;
    const bool capture_only = std::getenv("HETEROLLM_CUDA_GRAPH_CAPTURE_ONLY") != nullptr;
    if (dry_run && capture_only) GGML_ABORT("dry-run and capture-only cannot be combined");
    const bool compatible = ggml_cuda_graph_check_compability(cgraph);
    std::fprintf(file, "{\"kind\":\"snapshot\",\"source_revision\":\"d3146f2b56c2db4711ac8391871c9e529d1946d7\","
                 "\"call_id\":%llu,\"label\":\"%s\",\"device\":%d,\"context_id\":\"%p\",\"graph_key\":\"%p\","
                 "\"graph_uid\":%llu,\"n_nodes\":%d,\"compatible\":%s,\"dry_run\":%s,\"capture_only\":%s,\"node_property_refs\":[",
                 (unsigned long long) heterollm_graph_call_id, label, ctx->device, (void *) ctx,
                 ggml_cuda_graph_get_key(cgraph), (unsigned long long) cgraph->uid, cgraph->n_nodes,
                 compatible ? "true" : "false", dry_run ? "true" : "false", capture_only ? "true" : "false");
    for (size_t i = 0; i < refs.size(); ++i) std::fprintf(file, "%s%llu", i ? "," : "", (unsigned long long) refs[i]);
    std::fprintf(file, "]}\n");
    std::fflush(file);
}

// Capture-only compilation enumerates the submitted CUDA graph but never
// instantiates or launches it. This is intentionally separate from live traces.
static void heterollm_graph_trace_cuda_structure(cudaGraph_t graph, const void * key) {
    FILE * file = heterollm_graph_trace_file();
    if (!file) GGML_ABORT("capture-only requires structural trace output");
    size_t count = 0, edge_count = 0;
    CUDA_CHECK(cudaGraphGetNodes(graph, nullptr, &count));
    std::vector<cudaGraphNode_t> nodes(count);
    CUDA_CHECK(cudaGraphGetNodes(graph, nodes.data(), &count));
    CUDA_CHECK(cudaGraphGetEdges_v2(graph, nullptr, nullptr, nullptr, &edge_count));
    std::vector<cudaGraphNode_t> from(edge_count), to(edge_count);
    std::vector<cudaGraphEdgeData> edge_data(edge_count);
    CUDA_CHECK(cudaGraphGetEdges_v2(graph, from.data(), to.data(), edge_data.data(), &edge_count));
    std::lock_guard<std::mutex> lock(heterollm_graph_trace_mutex);
    std::fprintf(file, "{\"kind\":\"cuda_structure\",\"source_revision\":\"d3146f2b56c2db4711ac8391871c9e529d1946d7\","
                 "\"capture_only\":true,\"call_id\":%llu,\"graph_key\":\"%p\",\"nodes\":[",
                 (unsigned long long) heterollm_graph_call_id, key);
    for (size_t i = 0; i < count; ++i) {
        cudaGraphNodeType type;
        CUDA_CHECK(cudaGraphNodeGetType(nodes[i], &type));
        std::fprintf(file, "%s{\"id\":\"%p\",\"type\":%d", i ? "," : "", (void *) nodes[i], int(type));
        if (type == cudaGraphNodeTypeKernel) {
            // cuBLAS can capture driver-API kernels which have no runtime
            // function mapping; runtime GetParams rejects those functions.
#ifdef _WIN32
            static auto get_params = reinterpret_cast<decltype(&cuGraphKernelNodeGetParams)>(
                GetProcAddress(LoadLibraryA("nvcuda.dll"), "cuGraphKernelNodeGetParams"));
            if (!get_params) GGML_ABORT("cannot load CUDA driver graph kernel introspection");
            CUDA_KERNEL_NODE_PARAMS params = {};
            if (get_params(reinterpret_cast<CUgraphNode>(nodes[i]), &params) != CUDA_SUCCESS)
                GGML_ABORT("CUDA driver graph kernel introspection failed");
            cudaKernelNodeAttrValue cooperative = {};
            CUDA_CHECK(cudaGraphKernelNodeGetAttribute(nodes[i], cudaKernelNodeAttributeCooperative, &cooperative));
            std::fprintf(file, ",\"function\":\"%p\",\"grid\":[%u,%u,%u],\"block\":[%u,%u,%u],\"shared_bytes\":%u,\"cooperative\":%d",
                         (void *) params.func, params.gridDimX, params.gridDimY, params.gridDimZ,
                         params.blockDimX, params.blockDimY, params.blockDimZ, params.sharedMemBytes, cooperative.cooperative);
#else
            GGML_ABORT("capture-only driver kernel introspection currently requires Windows");
#endif
        } else if (type == cudaGraphNodeTypeMemcpy) {
#ifdef _WIN32
            static auto get_copy = reinterpret_cast<decltype(&cuGraphMemcpyNodeGetParams)>(
                GetProcAddress(LoadLibraryA("nvcuda.dll"), "cuGraphMemcpyNodeGetParams"));
            if (!get_copy) GGML_ABORT("cannot load CUDA driver graph copy introspection");
            CUDA_MEMCPY3D params = {};
            if (get_copy(reinterpret_cast<CUgraphNode>(nodes[i]), &params) != CUDA_SUCCESS)
                GGML_ABORT("CUDA driver graph copy introspection failed");
            static auto get_pointer = reinterpret_cast<decltype(&cuPointerGetAttribute)>(
                GetProcAddress(LoadLibraryA("nvcuda.dll"), "cuPointerGetAttribute"));
            if (!get_pointer) GGML_ABORT("cannot load CUDA pointer introspection");
            if (params.srcMemoryType != CU_MEMORYTYPE_DEVICE || params.dstMemoryType != CU_MEMORYTYPE_DEVICE)
                GGML_ABORT("capture-only pointer context export currently supports D2D copies");
            CUcontext src_context = nullptr, dst_context = nullptr;
            int src_device_ordinal = -1, dst_device_ordinal = -1;
            if (get_pointer(&src_context, CU_POINTER_ATTRIBUTE_CONTEXT, params.srcDevice) != CUDA_SUCCESS ||
                get_pointer(&dst_context, CU_POINTER_ATTRIBUTE_CONTEXT, params.dstDevice) != CUDA_SUCCESS ||
                get_pointer(&src_device_ordinal, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, params.srcDevice) != CUDA_SUCCESS ||
                get_pointer(&dst_device_ordinal, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, params.dstDevice) != CUDA_SUCCESS)
                GGML_ABORT("capture-only cannot determine copy pointer contexts");
            std::fprintf(file, ",\"copy\":{\"width_bytes\":%llu,\"height\":%llu,\"depth\":%llu,"
                         "\"src_memory_type\":%d,\"dst_memory_type\":%d,\"src_pitch\":%llu,\"dst_pitch\":%llu,"
                         "\"src_device\":\"%llx\",\"dst_device\":\"%llx\",\"src_host\":\"%p\",\"dst_host\":\"%p\","
                         "\"src_context\":\"%p\",\"dst_context\":\"%p\",\"src_device_ordinal\":%d,\"dst_device_ordinal\":%d}",
                         (unsigned long long) params.WidthInBytes, (unsigned long long) params.Height,
                         (unsigned long long) params.Depth, int(params.srcMemoryType), int(params.dstMemoryType),
                         (unsigned long long) params.srcPitch, (unsigned long long) params.dstPitch,
                         (unsigned long long) params.srcDevice, (unsigned long long) params.dstDevice,
                         params.srcHost, params.dstHost, (void *) src_context, (void *) dst_context,
                         src_device_ordinal, dst_device_ordinal);
#else
            GGML_ABORT("capture-only driver copy introspection currently requires Windows");
#endif
        }
        std::fprintf(file, "}");
    }
    std::fprintf(file, "],\"edges\":[");
    for (size_t i = 0; i < edge_count; ++i) std::fprintf(file, "%s[\"%p\",\"%p\"]", i ? "," : "", (void *) from[i], (void *) to[i]);
    std::fprintf(file, "],\"edge_data\":[");
    for (size_t i = 0; i < edge_count; ++i) {
        std::fprintf(file, "%s\"", i ? "," : "");
        const unsigned char * bytes = reinterpret_cast<const unsigned char *>(&edge_data[i]);
        for (size_t j = 0; j < sizeof(cudaGraphEdgeData); ++j) std::fprintf(file, "%02x", bytes[j]);
        std::fprintf(file, "\"");
    }
    std::fprintf(file, "]}\n");
    std::fflush(file);
}
