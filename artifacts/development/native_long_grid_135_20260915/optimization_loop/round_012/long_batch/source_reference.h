#pragma once
#include "ggml.h"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <algorithm>
// Matches quantize.cu Q8_1 and vecdotq.cuh Q5_0 unsigned-dot correction.
// Final block sum uses double: not bitwise-equivalent to CUDA warp reduction.
inline double q5_source_reference(const uint8_t *packed,const float *x,int k) {
 double result=0;
 for(int block=0;block<k/32;++block){
  float values[32],sums[32],next[32],maximum=0; int q[32];
  for(int i=0;i<32;++i){values[i]=x[block*32+i];sums[i]=values[i];maximum=std::max(maximum,std::abs(values[i]));}
  float scale=maximum/127.0f;
  for(int i=0;i<32;++i)q[i]=scale==0?0:int(std::round(values[i]/scale));
  for(int offset=16;offset;offset/=2){for(int i=0;i<32;++i)next[i]=sums[i]+sums[i^offset];std::memcpy(sums,next,sizeof(sums));}
  float ds=ggml_fp16_to_fp32(ggml_fp32_to_fp16(scale)),sum=ggml_fp16_to_fp32(ggml_fp32_to_fp16(sums[0]));
  const uint8_t *p=packed+block*22;ggml_fp16_t wd;uint32_t high;std::memcpy(&wd,p,2);std::memcpy(&high,p+2,4);int dot=0;
  for(int i=0;i<32;++i){int w=((p[6+i%16]>>(i<16?0:4))&15)|(((high>>i)&1)<<4);dot+=w*q[i];}
  float product=float(dot)*ds;float corrected=product-16.0f*sum;float value=ggml_fp16_to_fp32(wd)*corrected;result+=double(value);
 }
 return result;
}
