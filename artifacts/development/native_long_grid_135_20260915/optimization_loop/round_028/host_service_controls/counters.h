#pragma once
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <sstream>
#include <iomanip>
#include <string>
#include "logic.h"
namespace hs {
inline uint64_t ft(FILETIME t){return(uint64_t(t.dwHighDateTime)<<32)|t.dwLowDateTime;}
inline int64_t qpc(){LARGE_INTEGER x{};need(QueryPerformanceCounter(&x)!=0,"QPC failed");return x.QuadPart;}
inline int64_t frequency(){LARGE_INTEGER x{};need(QueryPerformanceFrequency(&x)!=0&&x.QuadPart>0,"QPF failed");return x.QuadPart;}
inline Stamp read(){Stamp x;FILETIME c{},e{},k{},u{};ULONG64 cycles=0;x.qpc_before=qpc();need(GetThreadTimes(GetCurrentThread(),&c,&e,&k,&u)!=0,"GetThreadTimes failed");need(QueryThreadCycleTime(GetCurrentThread(),&cycles)!=0,"QueryThreadCycleTime failed");x.qpc_after=qpc();x.user=ft(u);x.kernel=ft(k);x.cycles=cycles;return x;}
inline std::string stamp_json(Stamp x){std::ostringstream s;s<<"{\"qpc_before\":"<<x.qpc_before<<",\"qpc_after\":"<<x.qpc_after<<",\"user_100ns\":"<<x.user<<",\"kernel_100ns\":"<<x.kernel<<",\"cycles\":"<<x.cycles<<'}';return s.str();}
inline std::string phase(Stamp a,Stamp b,int64_t f){valid(a,b);std::ostringstream s;s<<std::setprecision(17)<<"{\"start\":"<<stamp_json(a)<<",\"end\":"<<stamp_json(b)<<",\"body_start_qpc\":"<<a.qpc_after<<",\"body_end_qpc\":"<<b.qpc_before<<",\"wall_ns\":"<<double(b.qpc_before-a.qpc_after)*1e9/f<<",\"snapshot_read_envelope_qpc_ticks\":"<<(a.qpc_after-a.qpc_before+b.qpc_after-b.qpc_before)<<",\"user_delta_100ns\":"<<(b.user-a.user)<<",\"kernel_delta_100ns\":"<<(b.kernel-a.kernel)<<",\"cycle_delta\":"<<(b.cycles-a.cycles)<<",\"CPU_service_ns\":null,\"counter_precision_validated\":false}";return s.str();}
inline volatile uint64_t sink=1;
__declspec(noinline) inline void empty_loop(int g){for(int i=0;i<g;++i){sink^=uint64_t(i+1);}}
inline HANDLE create_output(const char* path){HANDLE h=CreateFileA(path,GENERIC_WRITE,FILE_SHARE_READ,nullptr,CREATE_NEW,FILE_ATTRIBUTE_NORMAL,nullptr);need(h!=INVALID_HANDLE_VALUE,"output must be new");return h;}
inline void write(HANDLE h,const std::string& s){DWORD n=0;need(WriteFile(h,s.data(),DWORD(s.size()),&n,nullptr)&&n==s.size()&&FlushFileBuffers(h),"write/flush failed");}
}
