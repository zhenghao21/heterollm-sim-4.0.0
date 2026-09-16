#include "mmvq_probe_abi.h"
#include <type_traits>
using Convert = void (*)(const float *, void *, int, int, int, int, cudaStream_t);
using Main = void (*)(const void *, int, const void *, float *, int, int, int, int, cudaStream_t);
static_assert(std::is_same<decltype(&heterollm_mmvq_probe_convert_q8_1), Convert>::value, "convert ABI mismatch");
static_assert(std::is_same<decltype(&heterollm_mmvq_probe_main), Main>::value, "main ABI mismatch");
int main() {
    // Static linkage check only: this executable must not be launched.
    volatile auto convert_symbol = &heterollm_mmvq_probe_convert_q8_1;
    volatile auto main_symbol = &heterollm_mmvq_probe_main;
    return (convert_symbol && main_symbol) ? 0 : 1;
}
