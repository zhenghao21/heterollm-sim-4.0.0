#pragma once
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

// Independent closed-form reference. It deliberately does not call GGML or CUDA.
inline int graph_gap_input_numerator(std::size_t i) {
    return static_cast<int>((i * 73 + 19) % 255) - 127;
}
inline float graph_gap_reference_at(std::size_t i, int stage) {
    return static_cast<float>(std::ldexp(static_cast<double>(graph_gap_input_numerator(i)), -8 - stage));
}
inline std::uint32_t graph_gap_float_bits(float x) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &x, sizeof(bits));
    return bits;
}