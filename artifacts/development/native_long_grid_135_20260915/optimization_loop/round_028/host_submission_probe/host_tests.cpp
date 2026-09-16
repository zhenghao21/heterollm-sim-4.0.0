// Host-only decoder tests: no CUDA/CUPTI/GGML API call or DLL import.
#include "launch_decode.h"
#include <iostream>
#include <stdexcept>
#include <vector>
struct Args{
    uintptr_t input=0x1100,output=0x2200,nullptr_value=0;
    int cols=4096;int64_t stride=4096,zero=0;float eps=1e-5f;uint3 z{};
    std::array<void*,23> ptrs{};
    Args(){ptrs={&input,&output,&cols,&stride,&stride,&stride,&eps,&nullptr_value,&zero,&zero,&zero,&z,&z,&z,&z,&nullptr_value,&zero,&zero,&zero,&z,&z,&z,&z};}
};
static void expect(bool ok,const char* reason){if(!ok)throw std::runtime_error(reason);}
int main(){
    int passed=0;auto test=[&](bool ok,const char* why){expect(ok,why);++passed;};
    Args a;cudaLaunchAttribute attribute{};attribute.id=cudaLaunchAttributeProgrammaticStreamSerialization;attribute.val.programmaticStreamSerializationAllowed=1;
    cudaLaunchConfig_t config{};config.gridDim=dim3(1,1,1);config.blockDim=dim3(1024,1,1);config.dynamicSmemBytes=128;config.attrs=&attribute;config.numAttrs=1;
    cudaLaunchKernelExC_v11060_params extended{};extended.config=&config;extended.func=reinterpret_cast<const void*>(0x3300);extended.args=a.ptrs.data();
    probe::Launch good;good.decoded=probe::decode(CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernelExC_v11060,"cudaLaunchKernelExC",&extended,good);
    good.exited=true;good.result=0;good.symbol_ok=true;
    test(probe::geometry_ok(good),"exact ExC geometry rejected");
    test(good.input==a.input&&good.output==a.output,"pointer arguments lost");
    config.blockDim.x=1;test(good.block.x==1024,"config was not copied");config.blockDim.x=1024;
    probe::Launch bad;
    test(!probe::decode(430,"cudaLaunchKernelExC",nullptr,bad),"missing params accepted");
    auto null_config=extended;null_config.config=nullptr;test(!probe::decode(430,"cudaLaunchKernelExC",&null_config,bad),"missing config accepted");
    config.numAttrs=0;test(!probe::decode(430,"cudaLaunchKernelExC",&extended,bad),"missing PDL accepted");config.numAttrs=1;
    attribute.val.programmaticStreamSerializationAllowed=0;test(!probe::decode(430,"cudaLaunchKernelExC",&extended,bad),"invalid PDL accepted");attribute.val.programmaticStreamSerializationAllowed=1;
    test(!probe::decode(430,"cudaLaunchKernel",&extended,bad),"API name/ID mismatch accepted");
    test(!probe::decode(999,"cudaLaunchKernelExC_ptsz",&extended,bad),"unknown API accepted");
    a.ptrs[22]=nullptr;test(!probe::norm_arguments(a.ptrs.data(),bad),"truncated args accepted");a.ptrs[22]=&a.z;
    a.nullptr_value=1;test(!probe::norm_arguments(a.ptrs.data(),bad),"fused pointer accepted");a.nullptr_value=0;
    a.z.x=1;test(!probe::norm_arguments(a.ptrs.data(),bad),"nondefault broadcast accepted");a.z.x=0;
    cudaLaunchKernel_v7000_params legacy{};legacy.func=reinterpret_cast<const void*>(0x3300);legacy.gridDim=dim3(1,1,1);legacy.blockDim=dim3(1024,1,1);legacy.sharedMem=128;legacy.args=a.ptrs.data();
    probe::Launch old;old.decoded=probe::decode(CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000,"cudaLaunchKernel",&legacy,old);old.exited=old.symbol_ok=true;old.result=0;
    test(probe::geometry_ok(old),"legacy exact geometry rejected");
    probe::Recorder rec;rec.count=1;rec.launches[0]=good;std::array<uintptr_t,16> ins{},outs{};ins[0]=a.input;outs[0]=a.output;
    test(probe::capture_ok(rec,1,ins,outs),"qualified single node rejected");
    test(!probe::capture_ok(rec,4,ins,outs),"incorrect kernel count accepted");
    rec.launches[0].exited=false;test(!probe::capture_ok(rec,1,ins,outs),"missing EXIT accepted");rec.launches[0]=good;
    rec.launches[0].result=1;test(!probe::capture_ok(rec,1,ins,outs),"failed runtime return accepted");rec.launches[0]=good;
    outs[0]++;test(!probe::capture_ok(rec,1,ins,outs),"foreign output accepted");outs[0]--;
    rec.bad=true;test(!probe::capture_ok(rec,1,ins,outs),"unknown memory/launch event accepted");rec.bad=false;
    for(int n:{4,16}){rec.count=n;for(int i=0;i<n;++i){rec.launches[i]=good;rec.launches[i].input=ins[i]=0x1000+i*0x100;rec.launches[i].output=outs[i]=0x3000+i*0x100;}
        test(probe::capture_ok(rec,n,ins,outs),"multi-node exact capture rejected");rec.launches[n-1].stream=1;
        test(!probe::capture_ok(rec,n,ins,outs),"stream mismatch accepted");}
    std::cout<<"{\"status\":\"passed\",\"tests\":"<<passed<<",\"GPU_API_calls\":0,\"target_DLL_loaded\":false,\"performance_measured\":false}\n";
}
