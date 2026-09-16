#include "contiguous_layout.h"
#include <iostream>
#include <limits>
int main(){
    ContiguousLayout a,b;
    if(!contiguous_layout(4096,3072,1,32,128,a)||a.row_x!=128||a.col_y!=128||a.col_dst!=3072||
       a.channel_x!=393216||a.channel_y!=128||a.channel_dst!=3072||
       a.sample_x!=393216||a.sample_y!=128||a.sample_dst!=3072)return 1;
    if(!contiguous_layout(2048,1024,4,32,64,b)||b.channel_x!=65536||b.channel_y!=256||b.channel_dst!=4096||
       b.sample_x!=65536||b.sample_y!=256||b.sample_dst!=4096)return 2;
    if(contiguous_layout(4096,3072,1,32,127,b)||contiguous_layout(4095,3072,1,32,128,b)||
       contiguous_layout(4096,0,1,32,128,b)||contiguous_layout(4096,3072,0,32,128,b)||
       contiguous_layout(4096,std::numeric_limits<int>::max(),1,32,128,b))return 3;
    std::cout<<"{\"status\":\"passed\",\"fixed_case_and_other_shape_checked\":true,\"invalid_and_overflow_checked\":true,\"GPU_executed\":false}\n";
    return 0;
}
