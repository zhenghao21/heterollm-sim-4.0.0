#include <cstdio>
#include <cstdlib>
#include <cstdarg>
#include <cstring>
#include "llama-trace-annotations.h"
static int metadata=0,pushes=0,pops=0,formats=0;
struct ggml_tensor { const char * name; int op,type; long long ne[4]; };
const char * ggml_op_name(int) { ++metadata; return "MUL_MAT"; }
const char * ggml_type_name(int) { ++metadata; return "f32"; }
bool ggml_is_contiguous(const ggml_tensor *) { ++metadata; return true; }
bool ggml_is_transposed(const ggml_tensor *) { ++metadata; return false; }
bool ggml_is_permuted(const ggml_tensor *) { ++metadata; return false; }
int count_snprintf(char * p,size_t n,const char * f,...) { ++formats; va_list v; va_start(v,f);int r=vsnprintf(p,n,f,v);va_end(v);return r; }
int nvtxRangePushA(const char *) { ++pushes; return pushes; }
int nvtxRangePop() { ++pops; return pops; }
#define snprintf count_snprintf
class ggml_cuda_nvtx_scope {
public:
    explicit ggml_cuda_nvtx_scope(const ggml_tensor * tensor) : active(llama_trace_annotations_enabled()) {
        if (!active) {
            return;
        }
        char label[512];
        const char * name = tensor && tensor->name ? tensor->name : "unnamed";
        const char * op = tensor ? ggml_op_name(tensor->op) : "unknown";
        const char * type = tensor ? ggml_type_name(tensor->type) : "unknown";
        // Preserve a coarse physical layout class in the semantic marker.
        // Strides are the execution-relevant distinction; do not serialize
        // raw addresses or shape-specific pointers into the trace identity.
        const char * layout = "unknown";
        if (tensor) {
            layout = ggml_is_contiguous(tensor) ? "contiguous"
                   : ggml_is_transposed(tensor) ? "transposed"
                   : ggml_is_permuted(tensor) ? "permuted"
                   : "strided";
        }
        snprintf(label, sizeof(label), "operator:%s:%s|shape=%lldx%lldx%lldx%lld|type=%s|layout=%s",
                 op, name,
                 tensor ? (long long) tensor->ne[0] : 0LL,
                 tensor ? (long long) tensor->ne[1] : 0LL,
                 tensor ? (long long) tensor->ne[2] : 0LL,
                 tensor ? (long long) tensor->ne[3] : 0LL,
                 type, layout);
        nvtxRangePushA(label);
    }
    ~ggml_cuda_nvtx_scope() {
        if (active) {
            nvtxRangePop();
        }
    }
    ggml_cuda_nvtx_scope(const ggml_cuda_nvtx_scope &) = delete;
    ggml_cuda_nvtx_scope & operator=(const ggml_cuda_nvtx_scope &) = delete;
private:
    const bool active;
};
int main() {
 bool enabled=llama_trace_annotations_enabled();
 _putenv_s("LLAMA_TRACE_ANNOTATIONS",enabled?"0":"1");
 if(llama_trace_annotations_enabled()!=enabled)return 7;
 ggml_tensor tensor={"test",0,0,{32,64,1,1}};
 { ggml_cuda_nvtx_scope a(&tensor); {ggml_cuda_nvtx_scope b(nullptr);} }
 if(enabled) return pushes==2 && pops==2 && metadata==3 && formats==2 ? 0 : 8;
 return pushes==0 && pops==0 && metadata==0 && formats==0 ? 0 : 9;
}
