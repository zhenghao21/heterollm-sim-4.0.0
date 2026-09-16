#pragma once
#include "ggml.h"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <stdexcept>
// Source-level value reference only. It does not assert actual dispatch or
// reproduce the GPU parallel float reduction order. See protocol/source evidence.
inline double source_path_reference(const uint8_t *packed,const float *x,int k,ggml_type type,bool mmvq) {
 if(k<=0||k%32||(type!=GGML_TYPE_Q5_0&&type!=GGML_TYPE_Q8_0))throw std::runtime_error("reference outside quant/block scope");
 double result=0;
 for(int block=0;block<k/32;++block){
  float values[32],sums[32],next[32],maximum=0;int q[32];
  for(int i=0;i<32;++i){values[i]=x[block*32+i];sums[i]=values[i];maximum=std::max(maximum,std::abs(values[i]));}
  float scale=0;
  if(mmvq){
   float d=maximum/127.0f;
   for(int i=0;i<32;++i)q[i]=maximum==0?0:int(std::round(values[i]/d));
   scale=ggml_fp16_to_fp32(ggml_fp32_to_fp16(d));
  }else{
   // D4 stores an F32 inverse-reciprocal scale, not half(d).
   float inverse=maximum==0?0:127.0f/maximum;
   for(int i=0;i<32;++i)q[i]=maximum==0?0:int(std::round(values[i]*inverse));
   scale=maximum==0?0:1.0f/inverse;
  }
  size_t block_bytes=type==GGML_TYPE_Q5_0?22:34;
  const uint8_t *p=packed+block*block_bytes;ggml_fp16_t wd;std::memcpy(&wd,p,2);float weight_scale=ggml_fp16_to_fp32(wd);int dot=0;
  if(type==GGML_TYPE_Q5_0){
   uint32_t high;std::memcpy(&high,p+2,4);
   for(int i=0;i<32;++i){int w=((p[6+i%16]>>(i<16?0:4))&15)|(((high>>i)&1)<<4);dot+=(mmvq?w:w-16)*q[i];}
   if(mmvq){
    for(int offset=16;offset;offset/=2){for(int i=0;i<32;++i)next[i]=sums[i]+sums[i^offset];std::memcpy(sums,next,sizeof(sums));}
    float original_sum_half=ggml_fp16_to_fp32(ggml_fp32_to_fp16(sums[0]));
    float product=float(dot)*scale;float corrected=product-16.0f*original_sum_half;float value=weight_scale*corrected;result+=double(value);
   }else{
    // Q5 MMQ loader explicitly subtracts16 before its signed integer dot.
    float scales=weight_scale*scale;float value=scales*float(dot);result+=double(value);
   }
  }else{
   for(int i=0;i<32;++i)dot+=int(static_cast<int8_t>(p[2+i]))*q[i];
   float scales=weight_scale*scale;float value=scales*float(dot);result+=double(value);
  }
 }
 return result;
}
inline double q5_source_reference(const uint8_t *p,const float*x,int k){return source_path_reference(p,x,k,GGML_TYPE_Q5_0,true);}
