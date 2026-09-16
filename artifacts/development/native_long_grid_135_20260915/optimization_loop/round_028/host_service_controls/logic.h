#pragma once
#include <cstdint>
#include <stdexcept>
namespace hs {
struct Stamp {int64_t qpc_before=0,qpc_after=0;uint64_t user=0,kernel=0,cycles=0;};
inline void need(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
inline void valid(Stamp a,Stamp b){need(a.qpc_before<=a.qpc_after&&a.qpc_after<=b.qpc_before&&b.qpc_before<=b.qpc_after,"QPC order");need(b.user>=a.user&&b.kernel>=a.kernel&&b.cycles>=a.cycles,"CPU counter reversed");}
inline bool nodes_ok(int n){return n==1||n==4||n==16;}
inline bool groups_ok(int g){return g==32||g==128||g==512||g==2048;}
inline uint64_t kernels(int n,int g){need(nodes_ok(n)&&groups_ok(g),"case outside preregistered domain");return uint64_t(n)*g;}
inline bool visible(Stamp a,Stamp b){valid(a,b);return b.user>a.user||b.kernel>a.kernel;}
// Visibility is never a precision certificate.
inline bool counters_alone_prove_precision(){return false;}
}
