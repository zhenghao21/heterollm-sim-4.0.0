// R19 focused A/B execution component. This is intentionally a drop-in unit, not a collector copy.
// The parent-owned harness must build the same GGML graph, use the same original DLLs, and emit records.
#define NOMINMAX
#include <windows.h>
#include "ggml.h"
#include "ggml-backend.h"
#include <nvtx3/nvToolsExt.h>
#include "math_reference.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace graph_gap_probe {

constexpr int kFirstCalls = 1;
constexpr int kWarmupCalls = 5;
constexpr int kFormalCalls = 30;
constexpr int kTotalCalls = kFirstCalls + kWarmupCalls + kFormalCalls;

enum class Arm { control, buffered };

struct Check {
    std::size_t checked = 0;
    std::size_t mismatches = 0;
    std::size_t nonfinite = 0;
    long long first_bad_index = -1;
    double max_abs_error = 0.0;
    std::uint64_t raw_fnv1a64 = 1469598103934665603ull;
};

struct CallSample {
    int ordinal = 0;
    int index = 0;
    int graph_status = 0;
    char phase[8]{};
    char label[192]{};
    long long outer_push_start = 0;
    long long outer_push_end = 0;
    long long outer_pop_start = 0;
    long long outer_pop_end = 0;
    long long submit_push_start = 0;
    long long submit_push_end = 0;
    long long submit_pop_start = 0;
    long long submit_pop_end = 0;
    long long sync_push_start = 0;
    long long sync_push_end = 0;
    long long sync_pop_start = 0;
    long long sync_pop_end = 0;
    long long submit_start = 0;
    long long submit_end = 0;
    long long sync_start = 0;
    long long sync_end = 0;
};

struct ValidationBlock {
    const char *name = nullptr;
    int after_ordinal = -1;
    int first_stage = 1;
    std::vector<Check> stages; // Consecutive stages beginning at first_stage; every raw hash retained.
};

struct ArmResult {
    int graph_calls = 0;
    int numeric_blocks = 0;
    std::size_t checked_values = 0;
    std::size_t mismatches = 0;
    std::size_t nonfinite = 0;
    bool graph_status_pass = true;
    bool math_pass = true;
};

static void include_block(ArmResult &result, const ValidationBlock &block) {
    ++result.numeric_blocks;
    for (const auto &check : block.stages) {
        result.checked_values += check.checked;
        result.mismatches += check.mismatches;
        result.nonfinite += check.nonfinite;
        result.math_pass = result.math_pass && check.mismatches == 0;
    }
}

struct Harness {
    ggml_backend_t backend = nullptr;
    ggml_cgraph *graph = nullptr;
    std::vector<ggml_tensor *> stages; // final stage is last; every stage is a separate output.
    std::size_t elements = 0;
};

struct ArmContext {
    std::string config;
    std::string arm;
    std::string actual_argv_json;
    std::string freeze_json;
    std::string cache_allocation_reset_json;
};

static long long qpc() {
    LARGE_INTEGER value{};
    if (!QueryPerformanceCounter(&value)) throw std::runtime_error("QPC unavailable");
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

static std::uint64_t fnv1a64(const void *bytes, std::size_t count) {
    auto hash = 1469598103934665603ull;
    const auto *p = static_cast<const unsigned char *>(bytes);
    for (std::size_t i = 0; i < count; ++i) {
        hash ^= p[i];
        hash *= 1099511628211ull;
    }
    return hash;
}

static Check exact_stage_check(ggml_tensor *tensor, int stage, std::vector<float> &host) {
    ggml_backend_tensor_get(tensor, host.data(), 0, host.size() * sizeof(float));
    Check check;
    check.checked = host.size();
    check.raw_fnv1a64 = fnv1a64(host.data(), host.size() * sizeof(float));
    for (std::size_t i = 0; i < host.size(); ++i) {
        const auto got = host[i];
        const auto reference = graph_gap_reference_at(i, stage);
        if (!std::isfinite(got)) ++check.nonfinite;
        if (!std::isfinite(got) || graph_gap_float_bits(got) != graph_gap_float_bits(reference)) {
            ++check.mismatches;
            if (check.first_bad_index < 0) check.first_bad_index = static_cast<long long>(i);
        }
        if (std::isfinite(got)) check.max_abs_error = std::max(check.max_abs_error, std::abs(double(got) - reference));
    }
    return check;
}

static Check check_final_stage(Harness &harness) {
    if (!harness.backend || !harness.graph || harness.stages.empty() || harness.elements == 0) {
        throw std::runtime_error("incomplete graph-gap harness");
    }
    std::vector<float> host(harness.elements);
    return exact_stage_check(harness.stages.back(), static_cast<int>(harness.stages.size()), host);
}
static ValidationBlock check_all_stages(Harness &harness, const char *name, int after_ordinal) {
    if (!harness.backend || !harness.graph || harness.stages.empty() || harness.elements == 0) {
        throw std::runtime_error("incomplete graph-gap harness");
    }
    ValidationBlock block;
    block.name = name;
    block.after_ordinal = after_ordinal;
    block.stages.reserve(harness.stages.size());
    std::vector<float> host(harness.elements);
    for (std::size_t stage = 0; stage < harness.stages.size(); ++stage) {
        block.stages.push_back(exact_stage_check(harness.stages[stage], static_cast<int>(stage + 1), host));
    }
    return block;
}

static ValidationBlock check_intermediate_stages(Harness &harness, const char *name, int after_ordinal) {
    if (!harness.backend || !harness.graph || harness.stages.empty() || harness.elements == 0) {
        throw std::runtime_error("incomplete graph-gap harness");
    }
    ValidationBlock block;
    block.name = name;
    block.after_ordinal = after_ordinal;
    if (harness.stages.size() <= 1) return block;
    block.stages.reserve(harness.stages.size() - 1);
    std::vector<float> host(harness.elements);
    for (std::size_t stage = 0; stage + 1 < harness.stages.size(); ++stage) {
        block.stages.push_back(exact_stage_check(harness.stages[stage], static_cast<int>(stage + 1), host));
    }
    return block;
}
struct Range {
    long long push_start = 0, push_end = 0, pop_start = 0, pop_end = 0;
    bool open = false;
    explicit Range(const char *label) {
        push_start = qpc();
        nvtxRangePushA(label);
        push_end = qpc();
        open = true;
    }
    void close() {
        if (!open) return;
        pop_start = qpc();
        nvtxRangePop();
        pop_end = qpc();
        open = false;
    }
    ~Range() { if (open) nvtxRangePop(); }
};

static void phase_for(int ordinal, const char *&phase, int &index) {
    if (ordinal == 0) { phase = "first"; index = 0; return; }
    if (ordinal <= kWarmupCalls) { phase = "warmup"; index = ordinal - 1; return; }
    phase = "formal";
    index = ordinal - 1 - kWarmupCalls;
}

static CallSample timed_call(Harness &harness, const ArmContext &context, int ordinal) {
    const char *phase = nullptr;
    int index = -1;
    phase_for(ordinal, phase, index);
    CallSample sample;
    sample.ordinal = ordinal;
    sample.index = index;
    std::snprintf(sample.phase, sizeof(sample.phase), "%s", phase);
    // Keep the original R18 NVTX label grammar. Arm identity is a raw-record guard, not a timing-label change.
    std::snprintf(sample.label, sizeof(sample.label), "graph_submit/%s/%s/%d", context.config.c_str(), phase, index);
    Range outer(sample.label);
    const std::string submit_label = std::string(sample.label) + "/host_submit";
    Range submit(submit_label.c_str());
    sample.submit_start = qpc();
    sample.graph_status = static_cast<int>(ggml_backend_graph_compute_async(harness.backend, harness.graph));
    sample.submit_end = qpc();
    submit.close();
    const std::string sync_label = std::string(sample.label) + "/final_sync";
    Range sync(sync_label.c_str());
    sample.sync_start = qpc();
    ggml_backend_synchronize(harness.backend); // Exactly one final sync for every whole-graph call in either arm.
    sample.sync_end = qpc();
    sync.close();
    outer.close();
    sample.outer_push_start = outer.push_start; sample.outer_push_end = outer.push_end;
    sample.outer_pop_start = outer.pop_start; sample.outer_pop_end = outer.pop_end;
    sample.submit_push_start = submit.push_start; sample.submit_push_end = submit.push_end;
    sample.submit_pop_start = submit.pop_start; sample.submit_pop_end = submit.pop_end;
    sample.sync_push_start = sync.push_start; sample.sync_push_end = sync.push_end;
    sample.sync_pop_start = sync.pop_start; sample.sync_pop_end = sync.pop_end;
    return sample;
}

static std::string hex64(std::uint64_t value) {
    char text[32]{};
    std::snprintf(text, sizeof(text), "%016llx", static_cast<unsigned long long>(value));
    return text;
}

static std::string check_json(const Check &check) {
    std::ostringstream out;
    out << "{\"checked\":" << check.checked
        << ",\"mismatches\":" << check.mismatches
        << ",\"nonfinite\":" << check.nonfinite
        << ",\"first_bad_index\":" << check.first_bad_index
        << ",\"max_abs_error\":" << std::setprecision(17) << check.max_abs_error
        << ",\"raw_fnv1a64\":" << quote(hex64(check.raw_fnv1a64))
        << ",\"bitwise_pass\":" << (check.mismatches == 0 ? "true" : "false") << "}";
    return out.str();
}
static std::string call_json(const ArmContext &context, const CallSample &sample) {
    std::ostringstream out;
    out << "{\"record\":\"graph_call\",\"arm\":" << quote(context.arm)
        << ",\"label\":" << quote(sample.label)
        << ",\"phase\":" << quote(sample.phase)
        << ",\"index\":" << sample.index
        << ",\"graph_status\":" << sample.graph_status
        << ",\"observed_device_kernel_count\":null,\"observed_cuda_graph_launch_count\":null"
        << ",\"qpc_submit_start\":" << sample.submit_start << ",\"qpc_submit_end\":" << sample.submit_end
        << ",\"qpc_sync_start\":" << sample.sync_start << ",\"qpc_sync_end\":" << sample.sync_end
        << ",\"qpc_outer_push_start\":" << sample.outer_push_start << ",\"qpc_outer_push_end\":" << sample.outer_push_end
        << ",\"qpc_outer_pop_start\":" << sample.outer_pop_start << ",\"qpc_outer_pop_end\":" << sample.outer_pop_end
        << ",\"qpc_submit_push_start\":" << sample.submit_push_start << ",\"qpc_submit_push_end\":" << sample.submit_push_end
        << ",\"qpc_submit_pop_start\":" << sample.submit_pop_start << ",\"qpc_submit_pop_end\":" << sample.submit_pop_end
        << ",\"qpc_sync_push_start\":" << sample.sync_push_start << ",\"qpc_sync_push_end\":" << sample.sync_push_end
        << ",\"qpc_sync_pop_start\":" << sample.sync_pop_start << ",\"qpc_sync_pop_end\":" << sample.sync_pop_end << "}";
    return out.str();
}

static std::string validation_json(const ArmContext &context, const ValidationBlock &block) {
    std::ostringstream out;
    out << "{\"record\":\"numeric_block\",\"arm\":" << quote(context.arm)
        << ",\"name\":" << quote(block.name) << ",\"after_ordinal\":" << block.after_ordinal << ",\"stages\":[";
    for (std::size_t i = 0; i < block.stages.size(); ++i) {
        if (i) out << ',';
        out << "{\"stage\":" << block.first_stage + int(i) << ",\"result\":" << check_json(block.stages[i]) << "}";
    }
    out << "]}";
    return out.str();
}

template <class Emit>
static void emit_arm_metadata(const ArmContext &context, const char *write_cadence, Emit emit) {
    std::ostringstream out;
    out << "{\"record\":\"arm_metadata\",\"arm\":" << quote(context.arm)
        << ",\"config\":" << quote(context.config)
        << ",\"write_cadence\":" << quote(write_cadence)
        << ",\"actual_argv\":" << context.actual_argv_json
        << ",\"freeze\":" << context.freeze_json
        << ",\"cache_allocation_reset_semantics\":" << context.cache_allocation_reset_json << "}";
    emit(out.str());
}

// R18-like control: the only intended source-level difference is arm identity in emitted raw records.
template <class Emit>
ArmResult run_control_arm(Harness &harness, const ArmContext &context, Emit emit) {
    if (context.arm != "control") throw std::runtime_error("control context guard");
    ArmResult result;
    emit_arm_metadata(context, "per_call_validate_and_write", emit);
    for (int ordinal = 0; ordinal < kTotalCalls; ++ordinal) {
        const auto sample = timed_call(harness, context, ordinal);
        ++result.graph_calls;
        result.graph_status_pass = result.graph_status_pass && sample.graph_status == GGML_STATUS_SUCCESS;
        emit(call_json(context, sample));
        // R18 copied final-result check after each call, preserving its D2H and JSON cadence.
        ValidationBlock final_record{"per_call_final", ordinal, static_cast<int>(harness.stages.size()), {check_final_stage(harness)}};
        include_block(result, final_record);
        emit(validation_json(context, final_record));
        if (ordinal == 0 || ordinal == kTotalCalls - 1) {
            // R18 then checks only stages 1..nodes-1; final was already read above.
            const auto intermediates = check_intermediate_stages(harness, "first_or_postformal_intermediate_stages", ordinal);
            include_block(result, intermediates);
            emit(validation_json(context, intermediates));
        }
    }
    return result;
}

// No D2H or output write appears inside collect_phase: QPC/NVTX/final-sync semantics stay unchanged per call.
static void collect_phase(Harness &harness, const ArmContext &context, int first, int count,
                          std::array<CallSample, kTotalCalls> &calls, int &completed_calls) {
    for (int ordinal = first; ordinal < first + count; ++ordinal) {
        calls[ordinal] = timed_call(harness, context, ordinal);
        completed_calls = ordinal + 1;
    }
}
template <class Emit, class Flush>
ArmResult run_buffered_arm(Harness &harness, const ArmContext &context, Emit emit, Flush durable_flush) {
    if (context.arm != "buffered") throw std::runtime_error("buffered context guard");
    // All per-call records have fixed capacity before the first measured call.
    std::array<CallSample, kTotalCalls> calls{};
    std::array<ValidationBlock, 3> blocks{};
    std::array<bool, 3> block_ready{};
    int completed_calls = 0;
    try {
        // Persist the only hard-termination boundary before the first timed call. In-memory call records after this point are not promised recoverable on a hard crash.
        emit("{\"record\":\"buffered_started\",\"arm\":" + quote(context.arm) + ",\"config\":" + quote(context.config) + ",\"first_timed_ordinal\":0}");
        durable_flush();
        // No allocation, D2H, or JSON output is allowed inside an individual timing phase.
        collect_phase(harness, context, 0, kFirstCalls, calls, completed_calls);
        blocks[0] = check_all_stages(harness, "first", 0);
        block_ready[0] = true;
        collect_phase(harness, context, kFirstCalls, kWarmupCalls, calls, completed_calls);
        blocks[1] = check_all_stages(harness, "post_warmup", kFirstCalls + kWarmupCalls - 1);
        block_ready[1] = true;
        collect_phase(harness, context, kFirstCalls + kWarmupCalls, kFormalCalls, calls, completed_calls);
        blocks[2] = check_all_stages(harness, "post_formal", kTotalCalls - 1);
        block_ready[2] = true;
    } catch (...) {
        // A catchable error retains every completed in-memory call and completed numeric block.
        emit_arm_metadata(context, "buffered_error_flush", emit);
        for (int ordinal = 0; ordinal < completed_calls; ++ordinal) emit(call_json(context, calls[ordinal]));
        for (std::size_t index = 0; index < blocks.size(); ++index) {
            if (block_ready[index]) emit(validation_json(context, blocks[index]));
        }
        throw;
    }
    ArmResult result;
    for (const auto &sample : calls) {
        ++result.graph_calls;
        result.graph_status_pass = result.graph_status_pass && sample.graph_status == GGML_STATUS_SUCCESS;
    }
    for (const auto &block : blocks) include_block(result, block);
    // Buffered arm emits only after all timed calls and all mandated numeric blocks have completed.
    emit_arm_metadata(context, "buffered_after_postformal", emit);
    for (const auto &sample : calls) emit(call_json(context, sample));
    for (const auto &block : blocks) emit(validation_json(context, block));
    return result;
}

} // namespace graph_gap_probe