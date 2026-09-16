#pragma once
// Synthetic Q5_0 / Q8_1 reference. No CUDA API or target inference call occurs here.
#include "ggml.h"
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace check {
constexpr int K = 4096;
constexpr int N = 3072;
constexpr int M = 1;
constexpr int BLOCK = 32;
constexpr int PADDED_K = K;
// Locked Q5_0: QI5_0=4, VDR=2, therefore two integer partial dots per
// 32-element block. Four local float operations plus at most P-1 additions
// in any reduction tree give P+3 roundings on any contribution path.
constexpr int PARTIAL_DOTS_PER_BLOCK = 2;
constexpr int FP32_ROUNDING_STEPS = PARTIAL_DOTS_PER_BLOCK*(K/BLOCK)+3;
constexpr std::uint32_t SEED = 0x51354d31U;
static_assert(sizeof(block_q5_0) == 22 && offsetof(block_q5_0, qh) == 2 && offsetof(block_q5_0, qs) == 6, "locked Q5_0 layout changed");
static_assert(sizeof(block_q8_1) == 36 && offsetof(block_q8_1, qs) == 4, "locked Q8_1 layout changed");
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559, "requires IEEE binary32");
static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559, "requires IEEE binary64");
inline void require(bool value, const std::string & message) { if (!value) throw std::runtime_error(message); }
inline std::uint16_t get_u16(const std::uint8_t * p) { std::uint16_t v; std::memcpy(&v, p, 2); return v; }
inline std::uint32_t get_u32(const std::uint8_t * p) { std::uint32_t v; std::memcpy(&v, p, 4); return v; }
inline void put_u16(std::uint8_t * p, std::uint16_t v) { std::memcpy(p, &v, 2); }
inline float half(const std::uint8_t * p) { return ggml_fp16_to_fp32(get_u16(p)); }
inline double gamma(int n, double unit_roundoff) { return (n * unit_roundoff) / (1.0 - n * unit_roundoff); }
inline std::uint32_t mix(std::uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; return x ^ (x >> 16);
}
inline std::vector<float> make_weights() {
    std::vector<float> result(std::size_t(K) * N);
    for (std::size_t i = 0; i < result.size(); ++i) {
        const auto bits = mix(static_cast<std::uint32_t>(i) ^ SEED);
        const float magnitude = float(1U + ((bits >> 1) % 1023U)) / 512.0f;
        result[i] = (bits & 1U) ? magnitude : -magnitude;
    }
    return result;
}
inline std::vector<float> make_input() {
    std::vector<float> x(K);
    for (int b = 0; b < K/BLOCK; ++b) {
        const float d = std::ldexp(1.0f, -10 + b % 4);
        for (int p = 0; p < 15; ++p) {
            const int q = 1 + (b*17 + p*29) % 120;
            x[b*BLOCK + 2*p] = d * q;
            x[b*BLOCK + 2*p + 1] = -d * q;
        }
        x[b*BLOCK + 30] = d * 127;
        x[b*BLOCK + 31] = -d * (64 + b % 63);
    }
    return x;
}
inline std::vector<std::uint8_t> quantize_weight(const std::vector<float> & w) {
    require(w.size() == std::size_t(K)*N, "wrong weights length");
    const auto row = ggml_row_size(GGML_TYPE_Q5_0, K);
    require(row == K/BLOCK*22, "ggml Q5_0 row size mismatch");
    std::vector<std::uint8_t> q(std::size_t(row)*N);
    const auto written = ggml_quantize_chunk(GGML_TYPE_Q5_0, w.data(), q.data(), 0, N, K, nullptr);
    require(written == q.size(), "ggml_quantize_chunk wrote an unexpected length");
    return q;
}
inline std::vector<std::uint8_t> expected_q8(const std::vector<float> & x) {
    require(x.size() == K, "wrong input length");
    std::vector<std::uint8_t> result((K/BLOCK)*sizeof(block_q8_1));
    for (int b = 0; b < K/BLOCK; ++b) {
        float amax = 0; double sum = 0;
        for (int j = 0; j < BLOCK; ++j) {
            const float v=x[b*BLOCK+j];
            require(std::isnormal(v), "synthetic input must be finite normal and nonzero");
            amax = std::max(amax, std::abs(v)); sum += v;
        }
        const float d=amax/127.0f;
        auto * block = result.data()+b*sizeof(block_q8_1);
        put_u16(block, ggml_fp32_to_fp16(d));
        put_u16(block+2, ggml_fp32_to_fp16(static_cast<float>(sum)));
        require(half(block) == d && double(half(block+2)) == sum, "fixture scale/sum not exactly half representable");
        int qsum=0;
        for (int j=0;j<BLOCK;++j) {
            const float ratio=x[b*BLOCK+j]/d;
            const int q=static_cast<int>(std::round(ratio));
            require(q != 0 && q >= -127 && q <= 127 && ratio == q, "fixture crosses a quantization rounding boundary");
            block[4+j]=static_cast<std::uint8_t>(static_cast<std::int8_t>(q)); qsum+=q;
        }
        // Locked CUDA stores half(sum(original x)), not half(d * sum(q)).
        // Exact dyadic fixtures deliberately make those quantities identical.
        require(sum == double(d)*qsum && sum != 0, "Q5_0 sum-correction fixture invariant failed");
    }
    return result;
}
inline int signed_q8(const std::uint8_t * b,int j) { std::int8_t q; std::memcpy(&q,b+4+j,1); return q; }
inline int unsigned_q5(const std::uint8_t * b,int j) {
    const auto high=get_u32(b+2);
    const auto nibble=(j<16 ? b[6+j]&15 : b[6+j-16]>>4);
    return nibble | (((high>>j)&1U)<<4);
}
struct Reference {
    std::vector<double> dot, bound, unsigned_amplitude, absolute_products;
};
inline Reference reference(const std::vector<std::uint8_t>& w,const std::vector<std::uint8_t>& q8) {
    require(w.size()==std::size_t(N)*(K/BLOCK)*22 && q8.size()==(K/BLOCK)*36, "reference length mismatch");
    const auto * traits=ggml_get_type_traits(GGML_TYPE_Q5_0);
    require(traits && traits->to_float, "missing ggml Q5_0 CPU dequantizer");
    Reference r; r.dot.resize(N); r.bound.resize(N); r.unsigned_amplitude.resize(N); r.absolute_products.resize(N);
    std::vector<float> row(K);
    const double g32=gamma(FP32_ROUNDING_STEPS,std::ldexp(1.0,-24));
    const double g64=gamma(4*K+64,std::ldexp(1.0,-53));
    for (int n=0;n<N;++n) {
        const auto * rw=w.data()+std::size_t(n)*(K/BLOCK)*22;
        traits->to_float(rw,row.data(),K);
        double dot=0, abs_products=0, amplitude=0;
        for(int b=0;b<K/BLOCK;++b) {
            const auto * bw=rw+b*22; const auto * bx=q8.data()+b*36;
            const double dw=half(bw), dx=half(bx), sx=half(bx+2);
            require(std::isnormal(dw) && std::isnormal(dx), "scale under/overflow outside tolerance proof");
            double unsigned_products=0; int qsum=0;
            for(int j=0;j<BLOCK;++j) {
                const int u=unsigned_q5(bw,j), q=signed_q8(bx,j);
                require(double(row[b*BLOCK+j])==(u-16)*dw, "independent packed Q5_0 decode differs from locked CPU dequantizer");
                const double dequant_x=dx*q;
                const double product=double(row[b*BLOCK+j])*dequant_x;
                dot+=product; abs_products+=std::abs(product);
                unsigned_products+=std::abs(double(u)*q); qsum+=q;
            }
            require(sx==dx*qsum, "fixture violates exact CUDA Q5_0 sum-correction equivalence");
            amplitude+=std::abs(dw)*(std::abs(dx)*unsigned_products+16*std::abs(sx));
        }
        require(std::isfinite(dot) && amplitude>0, "degenerate reference");
        // Covers float scaling/subtraction and every possible serial reduction;
        // unsigned representation is used so cancellation cannot shrink the bound.
        const double bound=g32*amplitude/(1-g64)+g64*abs_products/(1-g64);
        r.dot[n]=dot; r.bound[n]=std::nextafter(bound,std::numeric_limits<double>::infinity());
        r.unsigned_amplitude[n]=amplitude; r.absolute_products[n]=abs_products;
    }
    return r;
}
struct Compare { std::size_t tested=0,failed=0; double max_absolute_error=0,max_bound_ratio=0; int worst_index=-1; };
inline Compare compare_output(const std::vector<float>& actual,const Reference& r) {
    require(actual.size()==r.dot.size(), "output length mismatch"); Compare c; c.tested=actual.size();
    for(std::size_t i=0;i<actual.size();++i) {
        const double error=std::isfinite(actual[i]) ? std::abs(double(actual[i])-r.dot[i]) : std::numeric_limits<double>::infinity();
        const double ratio=error/r.bound[i];
        if(ratio>c.max_bound_ratio) { c.max_bound_ratio=ratio;c.worst_index=int(i); }
        c.max_absolute_error=std::max(c.max_absolute_error,error);
        if(!std::isfinite(actual[i]) || error>r.bound[i]) ++c.failed;
    }
    return c;
}
inline std::size_t byte_mismatches(const std::vector<std::uint8_t>& a,const std::vector<std::uint8_t>& b) {
    require(a.size()==b.size(), "conversion byte length mismatch"); std::size_t bad=0;
    for(std::size_t i=0;i<a.size();++i) if(a[i]!=b[i])++bad;
    return bad;
}
} // namespace check
