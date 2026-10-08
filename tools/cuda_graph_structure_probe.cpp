// Structural workload compiler for a diagnostic llama.cpp build. This harness
// never samples logits or emits latency measurements. The instrumented CUDA
// backend either skips execution (dry mode) or records live structural checks.
// Windows: labels use SetEnvironmentVariableA; the diagnostic DLL reads them
// with GetEnvironmentVariableA, avoiding separate-CRT getenv caches.
#include "llama.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

static void label(const std::string & text) {
#ifdef _WIN32
    if (!SetEnvironmentVariableA("HETEROLLM_CUDA_GRAPH_LABEL", text.c_str())) {
        throw std::runtime_error("cannot set structural invocation label");
    }
#else
    if (setenv("HETEROLLM_CUDA_GRAPH_LABEL", text.c_str(), 1)) {
        throw std::runtime_error("cannot set structural invocation label");
    }
#endif
}

static void decode(llama_context * ctx, const std::vector<llama_token> & tokens,
                   int position, const std::string & tag) {
    label(tag);
    llama_batch batch = llama_batch_init((int) tokens.size(), 0, 1);
    batch.n_tokens = (int) tokens.size();
    for (int i = 0; i < batch.n_tokens; ++i) {
        batch.token[i] = tokens[i];
        batch.pos[i] = position + i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = i + 1 == batch.n_tokens;
    }
    const int rc = llama_decode(ctx, batch);
    llama_batch_free(batch);
    if (rc != 0) {
        throw std::runtime_error("llama_decode failed at " + tag + ": " + std::to_string(rc));
    }
    // Avoid reading or sampling output data in dry mode. Synchronization
    // completes backend copies so allocation lifetimes match between calls.
    llama_synchronize(ctx);
}

int main(int argc, char ** argv) {
    if (argc != 7) {
        std::fprintf(stderr, "usage: cuda_graph_structure_probe model.gguf prompt output context warmups repeats\n");
        return 2;
    }
    try {
        const int prompt = std::stoi(argv[2]);
        const int output = std::stoi(argv[3]);
        const int context = std::stoi(argv[4]);
        const int warmups = std::stoi(argv[5]);
        const int repeats = std::stoi(argv[6]);
        if (prompt < 1 || prompt > 512 || output < 1 || prompt + output > context ||
            warmups < 0 || repeats < 1) {
            throw std::runtime_error("invalid workload; probe supports one <=512-token prompt ubatch");
        }
        ggml_backend_load_all();
        llama_backend_init();
        auto model_params = llama_model_default_params();
        model_params.n_gpu_layers = -1;
        model_params.load_mtp = false;
        llama_model * model = llama_model_load_from_file(argv[1], model_params);
        if (!model) { throw std::runtime_error("model load failed"); }
        if (llama_model_has_encoder(model) || !llama_model_has_decoder(model)) {
            throw std::runtime_error("structural probe only covers decoder-only text models");
        }
        auto params = llama_context_default_params();
        params.n_ctx = context;
        params.n_batch = 512;
        params.n_ubatch = 512;
        params.n_seq_max = 1;
        params.n_rs_seq = 0;
        params.n_outputs_max = 1;
        params.n_outputs_max_per_seq = 1;
        params.n_threads = params.n_threads_batch = 16;
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        params.type_k = params.type_v = GGML_TYPE_F16;
        params.kv_unified = true;
        params.offload_kqv = true;
        params.op_offload = true;
        params.no_perf = true;
        llama_context * ctx = llama_init_from_model(model, params);
        if (!ctx) { throw std::runtime_error("context creation failed"); }
        const llama_vocab * vocab = llama_model_get_vocab(model);
        const int vocab_size = llama_vocab_n_tokens(vocab);
        const llama_token bos = llama_vocab_bos(vocab);
        const llama_token eos = llama_vocab_eos(vocab);
        if (vocab_size <= 0 || bos < 0 || bos >= vocab_size || eos < 0 || eos >= vocab_size) {
            throw std::runtime_error("structural probe requires explicit valid BOS and EOS token IDs");
        }
        const std::vector<llama_token> initial{bos, eos};
        decode(ctx, initial, 0, "model_warmup");
        llama_memory_clear(llama_get_memory(ctx), true);
        llama_synchronize(ctx);
        const llama_token token = 0; // explicit fixed valid token; never model-generated data
        // server-context.cpp:1240 calls common_context_can_seq_rm after common
        // model warmup. That probe evaluates two tokens and can establish the
        // very first CUDA executable, so omitting it changes Graph lifetimes.
        llama_memory_clear(llama_get_memory(ctx), true);
        decode(ctx, std::vector<llama_token>{token, token}, 0, "model_seq_rm_probe");
        if (llama_n_rs_seq(ctx) == 0) {
            (void) llama_memory_seq_rm(llama_get_memory(ctx), 0, 1, -1);
        }
        llama_memory_clear(llama_get_memory(ctx), true);
        llama_synchronize(ctx);
        for (int iteration = 0; iteration < warmups + repeats; ++iteration) {
            if (!llama_memory_seq_rm(llama_get_memory(ctx), 0, -1, -1)) {
                throw std::runtime_error("cannot clear complete sequence");
            }
            const bool warmup = iteration < warmups;
            const std::string prefix = std::string(warmup ? "warmup:" : "measured:") +
                std::to_string(warmup ? iteration : iteration - warmups);
            decode(ctx, std::vector<llama_token>(prompt, token), 0, prefix + ":prefill");
            // The prefill logits yield the first of output tokens. Exactly
            // output-1 subsequent backend decodes produce the remaining ones.
            for (int i = 1; i < output; ++i) {
                decode(ctx, std::vector<llama_token>{token}, prompt + i - 1,
                       prefix + ":decode:" + std::to_string(i));
            }
        }
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        std::printf("{\"status\":\"complete\",\"prompt\":%d,\"output\":%d,\"warmups\":%d,\"repeats\":%d,\"uses_sampled_tokens\":false,\"latency_measurements\":false}\n",
                    prompt, output, warmups, repeats);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "structural probe failed: %s\n", error.what());
        return 1;
    }
}
