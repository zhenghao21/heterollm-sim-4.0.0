#pragma once
// Header-only host decoding. No CUDA API calls or device memory reads.
#include <cuda_runtime_api.h>
#include <cupti.h>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include "locked_identity.h"
namespace probe {
constexpr int cols=4096, max_nodes=16;
constexpr float epsilon=1e-5f;
struct Launch {
    bool observed=false, decoded=false, exited=false, attributes_ok=false;
    unsigned api=0, correlation=0, context=0;
    char actual_symbol[512]{},api_name[96]{};
    uintptr_t stream=0, function=0, input=0, output=0;
    dim3 grid{},block{};size_t shared=0;int ncols=0;int64_t row=0,channel=0,sample=0;
    float eps=0;int result=-1;bool symbol_ok=false;
};
template<class T> inline T arg(void** args,int index){T x{};std::memcpy(&x,args[index],sizeof(x));return x;}
inline bool norm_arguments(void** args,Launch& out){
    if(!args)return false;
    for(int i=0;i<23;++i)if(!args[i])return false;
    out.input=arg<uintptr_t>(args,0);out.output=arg<uintptr_t>(args,1);out.ncols=arg<int>(args,2);
    out.row=arg<int64_t>(args,3);out.channel=arg<int64_t>(args,4);out.sample=arg<int64_t>(args,5);out.eps=arg<float>(args,6);
    for(int base:{7,15}){
        if(arg<uintptr_t>(args,base)!=0)return false;
        for(int k=1;k<=3;++k)if(arg<int64_t>(args,base+k)!=0)return false;
        for(int k=4;k<=7;++k){auto value=arg<uint3>(args,base+k);if(value.x||value.y||value.z)return false;}
    }
    return true;
}
inline bool decode(unsigned id,const char* name,const void* params,Launch& out){
    if(!name||!params)return false;void** args=nullptr;
    if(id==CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000 && !std::strcmp(name,"cudaLaunchKernel")){
        auto x=*static_cast<const cudaLaunchKernel_v7000_params*>(params);
        out.grid=x.gridDim;out.block=x.blockDim;out.shared=x.sharedMem;out.stream=reinterpret_cast<uintptr_t>(x.stream);
        out.function=reinterpret_cast<uintptr_t>(x.func);out.attributes_ok=true;args=x.args;
    }else if(id==CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernelExC_v11060 && !std::strcmp(name,"cudaLaunchKernelExC")){
        auto x=*static_cast<const cudaLaunchKernelExC_v11060_params*>(params);if(!x.config)return false;
        cudaLaunchConfig_t c{};std::memcpy(&c,x.config,sizeof(c));
        out.grid=c.gridDim;out.block=c.blockDim;out.shared=c.dynamicSmemBytes;out.stream=reinterpret_cast<uintptr_t>(c.stream);
        out.function=reinterpret_cast<uintptr_t>(x.func);args=x.args;
        if(c.numAttrs!=1||!c.attrs)return false;
        cudaLaunchAttribute attr{};std::memcpy(&attr,c.attrs,sizeof(attr));
        out.attributes_ok=attr.id==cudaLaunchAttributeProgrammaticStreamSerialization&&attr.val.programmaticStreamSerializationAllowed==1;
    }else return false;
    out.observed=true;return out.attributes_ok&&norm_arguments(args,out);
}
inline bool geometry_ok(const Launch& x){
    return x.observed&&x.decoded&&x.attributes_ok&&x.symbol_ok&&x.exited&&x.result==0&&x.function
        &&x.grid.x==1&&x.grid.y==1&&x.grid.z==1&&x.block.x==1024&&x.block.y==1&&x.block.z==1
        &&x.shared==128&&x.ncols==cols&&x.row==cols&&x.channel==cols&&x.sample==cols&&x.eps==epsilon;
}
struct Recorder{
    std::array<Launch,max_nodes> launches{};std::atomic<unsigned> count{0};std::atomic<bool> enabled{false},bad{false};
};
inline void CUPTIAPI callback(void* data,CUpti_CallbackDomain domain,CUpti_CallbackId id,const void* raw){
    auto& r=*static_cast<Recorder*>(data);if(!r.enabled||domain!=CUPTI_CB_DOMAIN_RUNTIME_API||!raw)return;
    const auto& info=*static_cast<const CUpti_CallbackData*>(raw);const char* name=info.functionName;
    if(!name){r.bad=true;return;}
    const bool launch=std::strstr(name,"Launch")!=nullptr;
    if(!launch){if(std::strstr(name,"Memcpy")||std::strstr(name,"Memset")||std::strstr(name,"Malloc")||std::strstr(name,"Free"))r.bad=true;return;}
    if(info.callbackSite==CUPTI_API_ENTER){
        unsigned index=r.count.fetch_add(1);if(index>=r.launches.size()){r.bad=true;return;}
        auto& x=r.launches[index];x.api=id;x.correlation=info.correlationId;x.context=info.contextUid;
        if(std::strlen(name)>=sizeof(x.api_name)){r.bad=true;return;}std::memcpy(x.api_name,name,std::strlen(name)+1);
        if(info.symbolName){if(std::strlen(info.symbolName)>=sizeof(x.actual_symbol)){r.bad=true;return;}std::memcpy(x.actual_symbol,info.symbolName,std::strlen(info.symbolName)+1);}
        x.symbol_ok=info.symbolName&&!std::strcmp(info.symbolName,locked::symbol);
        // Only a pinned symbol admits argument dereferencing; unknown launches are rejected.
        x.decoded=x.symbol_ok&&decode(id,name,info.functionParams,x);
        if(!x.decoded)r.bad=true;
    }else if(info.callbackSite==CUPTI_API_EXIT){
        bool found=false;for(unsigned i=0;i<r.count&&i<r.launches.size();++i){auto& x=r.launches[i];
            if(x.api==id&&x.correlation==info.correlationId&&x.context==info.contextUid){
                if(x.exited||!info.functionReturnValue){r.bad=true;return;}
                std::memcpy(&x.result,info.functionReturnValue,sizeof(cudaError_t));x.exited=true;found=true;break;
            }}if(!found)r.bad=true;
    }
}
inline bool capture_ok(const Recorder& r,int nodes,const std::array<uintptr_t,max_nodes>& inputs,const std::array<uintptr_t,max_nodes>& outputs){
    if(r.bad||r.count!=unsigned(nodes))return false;uintptr_t stream=r.launches[0].stream;
    for(int i=0;i<nodes;++i){const auto& x=r.launches[i];
        if(!geometry_ok(x)||x.stream!=stream||x.input!=inputs[i]||x.output!=outputs[i]||!x.input||!x.output)return false;
    }return true;
}
static_assert(sizeof(void*)==8 && sizeof(cudaError_t)==4 && sizeof(cudaLaunchConfig_t)==56);
static_assert(sizeof(cudaLaunchKernelExC_v11060_params)==24 && sizeof(uint3)==12);
}
