// Standalone host-only getenv propagation check. No CUDA or native GGML links.
#include <cstdlib>
#include <iostream>
int main() {
 const char *names[]={"GGML_CUDA_FORCE_MMQ","GGML_CUDA_FORCE_CUBLAS","CUDA_VISIBLE_DEVICES","LLAMA_TRACE_ANNOTATIONS","GGML_CUDA_DISABLE_FUSION","GGML_CUDA_CUBLAS_COMPUTE_TYPE"};
 bool absent=true; std::cout << "{\"values\":{";
 for(int i=0;i<6;++i) {const char *value=std::getenv(names[i]);if(i)std::cout<<',';std::cout<<'"'<<names[i]<<"\":"<<(value?"\"present\"":"null");absent=absent&&value==nullptr;}
 std::cout << "},\"all_absent\":"<<(absent?"true":"false")<<",\"gpu_access\":false,\"compiler_msvc\":"<<_MSC_VER<<"}\n";
 return absent?0:1;
}
