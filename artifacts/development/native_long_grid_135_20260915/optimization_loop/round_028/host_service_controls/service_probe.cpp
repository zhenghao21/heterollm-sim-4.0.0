// Reuse exact predecessor graph, CUPTI decoder, numeric check and DLL identity code.
#define main predecessor_main_not_called
#include "../host_submission_probe/probe.cpp"
#undef main
#include "counters.h"
#include "binding.h"
int main(int argc,char** argv){
 if(argc!=4||!std::getenv("HOST_PROBE_AUTHORIZED_GPU_RUN")||std::strcmp(std::getenv("HOST_PROBE_AUTHORIZED_GPU_RUN"),"1"))return 2;
 const std::string mode=argv[1],topology=argv[2];bool path=mode=="path",chain=topology=="chain";if((!path&&mode!="service")||(!chain&&topology!="fanout"))return 2;
 HANDLE output=INVALID_HANDLE_VALUE;ggml_backend_t backend=nullptr;std::string modules,reason,uuid;std::ostringstream rows;int code=0,completed=0;int64_t f=0;unsigned flags=0;
 try{output=hs::create_output(argv[3]);modules=verify_modules();if(!path)forbid_profiler();require(SetThreadAffinityMask(GetCurrentThread(),1)!=0,"affinity failed");
  cudaDeviceProp property{};cuda_ok(cudaGetDeviceProperties(&property,0));uuid=uuid_string(property.uuid);require(uuid==locked::uuid&&property.major==12&&property.minor==0&&property.multiProcessorCount==84,"device mismatch");backend=ggml_backend_cuda_init(0);require(backend,"backend init failed");cuda_ok(cudaGetDeviceFlags(&flags));f=hs::frequency();
  for(int n:{1,4,16}){Graph g(n,chain);g.allocate(backend);g.initialize();ggml_backend_synchronize(backend);for(int w=0;w<16;++w){g.submit(backend);ggml_backend_synchronize(backend);}g.verify(chain);
   if(path){probe::Recorder rec;std::array<uintptr_t,16> in{},out{};for(int i=0;i<n;++i){in[i]=(uintptr_t)g.nodes[i]->src[0]->data;out[i]=(uintptr_t)g.nodes[i]->data;}
    {Cupti cupti(rec);rec.enabled=true;g.submit(backend);rec.enabled=false;ggml_backend_synchronize(backend);}require(probe::capture_ok(rec,n,in,out),"path unqualified");if(completed++)rows<<',';rows<<"{\"nodes\":"<<n<<",\"qualified\":true,\"executed_graph_nodes\":"<<n<<",\"captured_launch_count\":"<<rec.count<<",\"max_abs_error\":"<<g.verify(chain)<<",\"launches\":"<<capture_json(rec)<<'}';
   }else for(int groups:{32,128,512,2048}){if(completed++)rows<<',';rows<<"{\"nodes\":"<<n<<",\"groups\":"<<groups<<",\"kernels\":"<<hs::kernels(n,groups)<<",\"samples\":[";
    for(int sample=0;sample<31;++sample){if(sample)rows<<',';ggml_backend_synchronize(backend);
     auto a=hs::read();hs::empty_loop(groups);auto b=hs::read();auto empty_before=hs::phase(a,b,f);
     a=hs::read();for(int j=0;j<groups;++j)g.submit(backend);b=hs::read();auto submit=hs::phase(a,b,f);
     a=hs::read();ggml_backend_synchronize(backend);b=hs::read();auto busy_sync=hs::phase(a,b,f);
     a=hs::read();ggml_backend_synchronize(backend);b=hs::read();auto idle_sync=hs::phase(a,b,f);
     a=hs::read();hs::empty_loop(groups);b=hs::read();auto empty_after=hs::phase(a,b,f);
     rows<<"{\"sample\":"<<sample<<",\"empty_submit_loop_before\":"<<empty_before<<",\"submit\":"<<submit<<",\"busy_sync\":"<<busy_sync<<",\"idle_sync\":"<<idle_sync<<",\"empty_submit_loop_after\":"<<empty_after<<",\"max_abs_error\":"<<g.verify(chain)<<'}';}rows<<"]}";}
  }require(verify_modules()==modules,"DLL identity changed");
 }catch(const std::exception& e){reason=e.what();code=1;}
 if(backend){ggml_backend_synchronize(backend);ggml_backend_free(backend);}if(output!=INVALID_HANDLE_VALUE){std::ostringstream s;s<<"{\"schema\":\"host-service-probe/v1\",\"mode\":"<<quote(mode)<<",\"topology\":"<<quote(topology)<<",\"status\":"<<quote(code?"failed":path?"path_qualified":"recorded_ETW_required")<<",\"reason\":"<<quote(reason)<<",\"protocol_sha256\":\""<<CONTROL_PROTOCOL_SHA<<"\",\"pid\":"<<GetCurrentProcessId()<<",\"tid\":"<<GetCurrentThreadId()<<",\"qpc_frequency\":"<<f<<",\"GPU_uuid\":"<<quote(uuid)<<",\"device_flags\":"<<flags<<",\"profiling\":"<<(path?"true":"false")<<",\"completed_cases\":"<<completed<<",\"counter_precision_validated\":false,\"cost_model_applied\":false,\"GPU_event_used\":false,\"target_modules\":"<<(modules.empty()?"[]":modules)<<",\"cases\":["<<(code?"":rows.str())<<"],\"partial_records\":"<<(code?quote(rows.str()):"null")<<"}\n";hs::write(output,s.str());CloseHandle(output);}return code;
}
