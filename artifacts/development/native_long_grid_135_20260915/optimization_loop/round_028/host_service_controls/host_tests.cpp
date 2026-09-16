#include "logic.h"
#include <iostream>
int main(){hs::Stamp a{1,2,0,0,100},b{3,4,0,0,200};hs::valid(a,b);if(hs::visible(a,b)||hs::counters_alone_prove_precision())return 1;
 if(hs::kernels(1,512)!=512||hs::kernels(4,128)!=512||hs::kernels(16,32)!=512)return 2;
 bool rejected=false;try{hs::valid(b,a);}catch(...){rejected=true;}if(!rejected)return 3;
 rejected=false;try{hs::kernels(2,128);}catch(...){rejected=true;}if(!rejected)return 4;
 std::cout<<"{\"status\":\"passed\",\"pure_logic\":true,\"experiment_timing_executed\":false,\"CRT_clock_calls_observed\":null,\"GPU_API_called\":false}\n";return 0;}
