#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
namespace check {
constexpr int K=2048,N=2048;
inline std::vector<uint8_t> bytes(const char*name,size_t expected){
 std::ifstream f(std::string("F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/artifacts/development/native_long_grid_135_20260915/optimization_loop/round_029/q4k_operator_probe/fixture/")+name,std::ios::binary);if(!f)throw std::runtime_error("fixture missing");
 std::vector<uint8_t> v(expected);f.read(reinterpret_cast<char*>(v.data()),expected);
 if(size_t(f.gcount())!=expected||f.peek()!=EOF)throw std::runtime_error("fixture length changed");return v;}
template<class T>inline std::vector<T> values(const char*name,size_t count){auto b=bytes(name,count*sizeof(T));std::vector<T>v(count);std::memcpy(v.data(),b.data(),b.size());return v;}
inline std::vector<float> make_input(){return values<float>("input.f32.bin",K);}
inline std::vector<uint8_t> packed_weight(){return bytes("weights.q4_k.bin",size_t(K/256)*144*N);}
inline std::vector<uint8_t> expected_q8(const std::vector<float>&){return bytes("expected.q8_1.bin",size_t(K/32)*36);}
struct Reference{std::vector<double>dot,bound;};
inline Reference reference(const std::vector<uint8_t>&,const std::vector<uint8_t>&){return {values<double>("reference.f64.bin",N),values<double>("bounds.f64.bin",N)};}
struct Comparison{size_t tested=0,failed=0;double max_absolute_error=0,max_bound_ratio=0;};
inline Comparison compare_output(const std::vector<float>&actual,const Reference&r){Comparison out;for(size_t i=0;i<actual.size();++i){++out.tested;double e=std::abs(double(actual[i])-r.dot[i]);if(!std::isfinite(actual[i])||e>r.bound[i])++out.failed;out.max_absolute_error=std::max(out.max_absolute_error,e);out.max_bound_ratio=std::max(out.max_bound_ratio,e/r.bound[i]);}return out;}
inline size_t byte_mismatches(const std::vector<uint8_t>&a,const std::vector<uint8_t>&b){size_t n=0;for(size_t i=0;i<a.size();++i)n+=a[i]!=b[i];return n;}
}
