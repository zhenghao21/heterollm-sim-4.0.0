#include "counters.h"
#include "binding.h"
#include <iostream>
#include <cstdlib>
#include <cstring>
int main(int argc,char** argv){
 if(argc!=2||!std::getenv("HOST_SERVICE_COUNTER_AUTH")||std::strcmp(std::getenv("HOST_SERVICE_COUNTER_AUTH"),"1"))return 2;
 HANDLE out=INVALID_HANDLE_VALUE;std::ostringstream rows;bool first=true;int code=0;std::string reason;int64_t f=0,start=0,finish=0;
 try{out=hs::create_output(argv[1]);hs::need(SetThreadAffinityMask(GetCurrentThread(),1)!=0,"affinity");f=hs::frequency();start=hs::qpc();
  auto record=[&](const char* kind,double amount,int repeat,auto body){hs::need((hs::qpc()-start)*1000.0/f<20000,"pilot sampling budget exhausted");auto a=hs::read();body();auto b=hs::read();if(!first)rows<<',';first=false;rows<<"{\"kind\":\""<<kind<<"\",\"amount\":"<<amount<<",\"repeat\":"<<repeat<<",\"measurement\":"<<hs::phase(a,b,f)<<'}';hs::need((b.qpc_after-start)*1000.0/f<=20000,"pilot sampling budget overrun");};
  for(int repeat=0;repeat<5;++repeat){for(int g:{0,128,512,2048})record("empty_loop",g,repeat,[&](){hs::empty_loop(g);});
   for(double ms:{.25,1.,4.,16.,64.})record("busy",ms,repeat,[&](){const auto end=hs::qpc()+int64_t(ms*f/1000);do{for(int j=0;j<256;++j)hs::sink=hs::sink*1664525+1013904223;}while(hs::qpc()<end);});
   for(int ms:{1,4,16,64})record("Sleep",ms,repeat,[&](){Sleep(DWORD(ms));});}finish=hs::qpc();hs::need((finish-start)*1000.0/f<=20000,"pilot total sampling budget overrun");
 }catch(const std::exception& e){code=1;reason=e.what();}
 if(out!=INVALID_HANDLE_VALUE){std::ostringstream s;s<<"{\"schema\":\"host-service-counter-pilot/v1\",\"status\":\""<<(code?"failed":"recorded_precision_unproven")<<"\",\"reason\":\""<<reason<<"\",\"protocol_sha256\":\""<<CONTROL_PROTOCOL_SHA<<"\",\"pid\":"<<GetCurrentProcessId()<<",\"tid\":"<<GetCurrentThreadId()<<",\"qpc_frequency\":"<<f<<",\"sampling_start_qpc\":"<<start<<",\"sampling_finish_qpc\":"<<finish<<",\"sampling_budget_ms\":20000,\"GPU_API_called\":false,\"cycles_converted_to_ns\":false,\"counter_precision_validated\":false,\"samples\":["<<rows.str()<<"]}\n";hs::write(out,s.str());CloseHandle(out);}return code;
}
