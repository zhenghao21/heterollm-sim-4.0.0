#pragma once
#include <cstdint>
#include <limits>
// Contiguous two-dimensional weights/input/output; channel/sample dimensions=1.
// Values are derived from logical shape and packed-block layout, not timings.
struct ContiguousLayout {
    int row_x=0,col_y=0,col_dst=0;
    int channel_x=0,channel_y=0,channel_dst=0;
    int sample_x=0,sample_y=0,sample_dst=0;
};
inline bool contiguous_layout(int k,int n,int m,int weight_block,int q8_stride,ContiguousLayout & out) {
    if(k<=0||n<=0||m<=0||weight_block<=0||k%weight_block||q8_stride<=0||
       int64_t(q8_stride)*32<k)return false;
    const int64_t row=k/weight_block;
    const int64_t cx=row*n,cy=int64_t(q8_stride)*m,cd=int64_t(n)*m;
    if(cx>std::numeric_limits<int>::max()||cy>std::numeric_limits<int>::max()||cd>std::numeric_limits<int>::max())return false;
    out.row_x=int(row);out.col_y=q8_stride;out.col_dst=n;
    out.channel_x=out.sample_x=int(cx);
    out.channel_y=out.sample_y=int(cy);
    out.channel_dst=out.sample_dst=int(cd);
    return true;
}
