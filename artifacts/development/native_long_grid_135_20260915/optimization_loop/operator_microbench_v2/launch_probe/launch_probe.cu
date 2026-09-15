#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <bcrypt.h>
#include <cuda_runtime.h>
#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "build_identity.h"
namespace fs = std::filesystem;

static int64_t tick() { LARGE_INTEGER value{}; if (!QueryPerformanceCounter(&value)) throw std::runtime_error("QPC failed"); return value.QuadPart; }
static int64_t frequency() { LARGE_INTEGER value{}; if (!QueryPerformanceFrequency(&value) || value.QuadPart <= 0) throw std::runtime_error("QPF failed"); return value.QuadPart; }
static double ns(int64_t begin, int64_t end, int64_t freq) { return double((long double)(end-begin)*1000000000.0L/(long double)freq); }
static std::string quote(const std::string& s) { std::ostringstream o; o << '"'; for (unsigned char c:s) { if(c=='"'||c=='\\') o << '\\' << c; else if(c=='\n')o<<"\\n";else if(c=='\r')o<<"\\r";else if(c=='\t')o<<"\\t";else if(c<32)o<<"\\u"<<std::hex<<std::setw(4)<<std::setfill('0')<<int(c)<<std::dec;else o<<c; } o << '"'; return o.str(); }
static std::string sha256(const fs::path& path) {
    BCRYPT_ALG_HANDLE alg=nullptr; BCRYPT_HASH_HANDLE hash=nullptr;
    auto check=[](NTSTATUS status){if(status<0)throw std::runtime_error("BCrypt SHA256 failure");};
    try { check(BCryptOpenAlgorithmProvider(&alg,BCRYPT_SHA256_ALGORITHM,nullptr,0));
        DWORD count=0,size=0;check(BCryptGetProperty(alg,BCRYPT_OBJECT_LENGTH,(PUCHAR)&size,sizeof(size),&count,0));
        std::vector<UCHAR> object(size);check(BCryptCreateHash(alg,&hash,object.data(),size,nullptr,0,0));
        std::ifstream f(path,std::ios::binary);if(!f)throw std::runtime_error("cannot read identity file: "+path.string());
        std::array<char,65536> buffer{};while(f){f.read(buffer.data(),buffer.size());auto n=f.gcount();if(n>0)check(BCryptHashData(hash,(PUCHAR)buffer.data(),ULONG(n),0));}
        if(!f.eof())throw std::runtime_error("identity file read failed");
        std::array<UCHAR,32> digest{};check(BCryptFinishHash(hash,digest.data(),DWORD(digest.size()),0));
        BCryptDestroyHash(hash);hash=nullptr;BCryptCloseAlgorithmProvider(alg,0);alg=nullptr;
        std::ostringstream out;out<<std::hex<<std::setfill('0');for(auto v:digest)out<<std::setw(2)<<unsigned(v);return out.str();
    } catch(...) { if(hash)BCryptDestroyHash(hash);if(alg)BCryptCloseAlgorithmProvider(alg,0);throw; }
}
class JsonlWriter { HANDLE handle=INVALID_HANDLE_VALUE; public:
    explicit JsonlWriter(const fs::path& path) { handle=CreateFileW(path.wstring().c_str(),GENERIC_WRITE,FILE_SHARE_READ,nullptr,CREATE_NEW,FILE_ATTRIBUTE_NORMAL,nullptr);if(handle==INVALID_HANDLE_VALUE)throw std::runtime_error("raw output must be a new file: "+path.string()); }
    ~JsonlWriter(){if(handle!=INVALID_HANDLE_VALUE)CloseHandle(handle);}
    void line(const std::string& text){std::string value=text+"\n";DWORD written=0;if(!WriteFile(handle,value.data(),DWORD(value.size()),&written,nullptr)||written!=value.size())throw std::runtime_error("raw JSONL write failed");}
    void flush(){if(!FlushFileBuffers(handle))throw std::runtime_error("raw flush failed");}
};
static void require_cuda(cudaError_t status,const char* what){if(status!=cudaSuccess)throw std::runtime_error(std::string(what)+": "+cudaGetErrorString(status));}
static std::string uuid_text(const cudaUUID_t& uuid){std::ostringstream s;s<<"GPU-"<<std::hex<<std::setfill('0');for(int i=0;i<16;i++){if(i==4||i==6||i==8||i==10)s<<'-';s<<std::setw(2)<<unsigned((unsigned char)uuid.bytes[i]);}return s.str();}

__global__ void empty_kernel() { }
__global__ void low_load_integer_kernel(uint32_t* output,uint32_t salt){unsigned lane=threadIdx.x;uint32_t x=salt^(lane*0x9e3779b9u);
#pragma unroll

    for(int j=0;j<PROBE_INTEGER_ITERATIONS;j++){x=x*1664525u+1013904223u;x^=(x>>13);}output[lane]=x;}
static uint32_t expected_word(unsigned lane,uint32_t salt){uint32_t x=salt^(lane*0x9e3779b9u);for(int j=0;j<PROBE_INTEGER_ITERATIONS;j++){x=x*1664525u+1013904223u;x^=(x>>13);}return x;}
static std::string checksum(const uint32_t* words,size_t count){uint64_t h=14695981039346656037ull;for(size_t i=0;i<count;i++){for(int j=0;j<4;j++){h^=(words[i]>>(j*8))&255u;h*=1099511628211ull;}}std::ostringstream out;out<<std::hex<<std::setw(16)<<std::setfill('0')<<h;return out.str();}
struct Sample {
    int kernel=0,mode=0,burst=0,repeat=0,ordinal=0;bool formal=false;
    int64_t wall_begin=0,wall_end=0,start_event_begin=0,start_event_end=0,end_event_begin=0,end_event_end=0;
    std::array<int64_t,64> enqueue_begin{},enqueue_end{};
    std::array<int64_t,65> sync_begin{},sync_end{};std::array<int,65> sync_status{};int sync_count=0;
    int start_record_status=-1,end_record_status=-1,launch_status=-1,start_query_status=-1,end_query_status=-1,elapsed_status=-1;
    float event_ms=0.0f;bool correct=false;size_t mismatch_count=0;std::string output_checksum;
};
static Sample measure(int kernel,int mode,int burst,int repeat,bool formal,int ordinal,cudaStream_t stream,cudaEvent_t start,cudaEvent_t stop,uint32_t* device,uint32_t* host){
    Sample r;r.kernel=kernel;r.mode=mode;r.burst=burst;r.repeat=repeat;r.formal=formal;r.ordinal=ordinal;
    require_cuda(cudaMemsetAsync(device,0xa5,PROBE_OUTPUT_WORDS*sizeof(uint32_t),stream),"reset output");
    require_cuda(cudaStreamSynchronize(stream),"pre-burst reset fence");
    r.wall_begin=r.start_event_begin=tick();r.start_record_status=int(cudaEventRecord(start,stream));r.start_event_end=tick();
    for(int i=0;i<burst;i++){
        r.enqueue_begin[i]=tick();
        if(kernel==0)empty_kernel<<<1,PROBE_THREADS,0,stream>>>();
        else low_load_integer_kernel<<<1,PROBE_THREADS,0,stream>>>(device+i*PROBE_THREADS,PROBE_LOW_LOAD_SEED+uint32_t(i)*0x9e3779b9u);
        r.enqueue_end[i]=tick();
        if(mode==1){int j=r.sync_count++;r.sync_begin[j]=tick();r.sync_status[j]=int(cudaStreamSynchronize(stream));r.sync_end[j]=tick();}
    }
    r.launch_status=int(cudaPeekAtLastError());
    r.end_event_begin=tick();r.end_record_status=int(cudaEventRecord(stop,stream));r.end_event_end=tick();
    int j=r.sync_count++;r.sync_begin[j]=tick();r.sync_status[j]=int(cudaStreamSynchronize(stream));r.sync_end[j]=tick();r.wall_end=r.sync_end[j];
    r.start_query_status=int(cudaEventQuery(start));r.end_query_status=int(cudaEventQuery(stop));r.elapsed_status=int(cudaEventElapsedTime(&r.event_ms,start,stop));
    bool valid=r.start_record_status==0&&r.end_record_status==0&&r.launch_status==0&&r.start_query_status==0&&r.end_query_status==0&&r.elapsed_status==0;
    for(int i=0;i<r.sync_count;i++)valid=valid&&r.sync_status[i]==0;
    if(valid){require_cuda(cudaMemcpyAsync(host,device,PROBE_OUTPUT_WORDS*sizeof(uint32_t),cudaMemcpyDeviceToHost,stream),"validation copy");require_cuda(cudaStreamSynchronize(stream),"validation fence");
        for(unsigned k=0;k<PROBE_OUTPUT_WORDS;k++){uint32_t expected=0xa5a5a5a5u;if(kernel==1&&k<unsigned(burst*PROBE_THREADS))expected=expected_word(k%PROBE_THREADS,PROBE_LOW_LOAD_SEED+uint32_t(k/PROBE_THREADS)*0x9e3779b9u);r.mismatch_count+=host[k]!=expected;}
        r.correct=r.mismatch_count==0;r.output_checksum=checksum(host,PROBE_OUTPUT_WORDS);
    }
    return r;
}
static std::string sample_json(const Sample& r,int64_t f){
    double enq=0,sync=0;for(int i=0;i<r.burst;i++)enq+=ns(r.enqueue_begin[i],r.enqueue_end[i],f);for(int i=0;i<r.sync_count;i++)sync+=ns(r.sync_begin[i],r.sync_end[i],f);
    std::ostringstream o;o<<std::setprecision(17)<<"{\"type\":\"sample\",\"schema\":\"cuda-launch-sync-raw-sample/v1\",\"phase\":"<<quote(r.formal?"formal":"warmup")<<",\"repeat\":"<<r.repeat<<",\"ordinal\":"<<r.ordinal<<",\"kernel\":"<<quote(r.kernel==0?"empty":"low_load_integer")<<",\"supply_mode\":"<<quote(r.mode==0?"continuous_enqueue":"stream_sync_each")<<",\"burst_length\":"<<r.burst;
    o<<",\"wall_qpc\":["<<r.wall_begin<<','<<r.wall_end<<"],\"start_event_record_qpc\":["<<r.start_event_begin<<','<<r.start_event_end<<"],\"end_event_record_qpc\":["<<r.end_event_begin<<','<<r.end_event_end<<"],\"enqueue_qpc\":[";
    for(int i=0;i<r.burst;i++){if(i)o<<',';o<<'['<<r.enqueue_begin[i]<<','<<r.enqueue_end[i]<<']';}o<<"],\"synchronize_qpc\":[";
    for(int i=0;i<r.sync_count;i++){if(i)o<<',';o<<"{\"begin\":"<<r.sync_begin[i]<<",\"end\":"<<r.sync_end[i]<<",\"cuda_status\":"<<r.sync_status[i]<<'}';}
    o<<"],\"host_enqueue_call_total_ns\":"<<enq<<",\"host_enqueue_envelope_ns\":"<<ns(r.enqueue_begin[0],r.enqueue_end[r.burst-1],f)<<",\"enqueue_envelope_contains_prior_sync\":"<<(r.mode==1?"true":"false")<<",\"host_sync_call_total_ns\":"<<sync<<",\"total_wall_ns\":"<<ns(r.wall_begin,r.wall_end,f)<<",\"cuda_event_span_ms\":";
    if(r.elapsed_status==0)o<<r.event_ms;else o<<"null";o<<",\"cuda_event_span_ns\":";if(r.elapsed_status==0)o<<double(r.event_ms)*1e6;else o<<"null";
    o<<",\"cuda_status\":{\"start_event_record\":"<<r.start_record_status<<",\"end_event_record\":"<<r.end_record_status<<",\"last_launch_error\":"<<r.launch_status<<",\"start_event_query_after_sync\":"<<r.start_query_status<<",\"end_event_query_after_sync\":"<<r.end_query_status<<",\"event_elapsed_time\":"<<r.elapsed_status<<"},\"output_validation\":{\"passed\":"<<(r.correct?"true":"false")<<",\"checked_words\":"<<PROBE_OUTPUT_WORDS<<",\"mismatches\":"<<r.mismatch_count<<",\"checksum_fnv1a64\":"<<quote(r.output_checksum)<<"},\"device_span_is_pure_kernel_sum\":false}";return o.str();
}
int main(int argc,char** argv){
    try{
        fs::path executable=fs::absolute(argv[0]),root=executable.parent_path().parent_path();
        fs::path protocol=root/"protocol.json",source=root/"launch_probe.cu",output;bool approved=false;
        for(int i=1;i<argc;i++){std::string arg=argv[i];if(arg=="--idle-window-confirmed")approved=true;else if((arg=="--protocol"||arg=="--source"||arg=="--output")&&i+1<argc){fs::path value=argv[++i];if(arg=="--protocol")protocol=value;else if(arg=="--source")source=value;else output=value;}else if(arg=="--help"){std::cout<<"launch_probe.exe --idle-window-confirmed --output NEW_JSONL [--protocol FILE --source FILE]\n";return 0;}else throw std::runtime_error("unknown/incomplete argument: "+arg);}
        if(!approved||output.empty())throw std::runtime_error("explicit --idle-window-confirmed and a NEW --output are required; build never runs this executable");
        if(sha256(protocol)!=PROBE_PROTOCOL_SHA256||sha256(source)!=PROBE_SOURCE_SHA256)throw std::runtime_error("protocol/source SHA256 differs from compiled identity; refusing CUDA initialization");
        std::string binary_sha=sha256(executable);JsonlWriter writer(output);int64_t f=frequency();
        require_cuda(cudaSetDeviceFlags(cudaDeviceScheduleAuto),"set process schedule");require_cuda(cudaSetDevice(0),"select device");
        cudaDeviceProp properties{};require_cuda(cudaGetDeviceProperties(&properties,0),"device properties");
        std::string uuid=uuid_text(properties.uuid);if(uuid!=PROBE_EXPECTED_GPU_UUID||properties.major!=12||properties.minor!=0||properties.multiProcessorCount!=84)throw std::runtime_error("actual GPU UUID/architecture/SM differs from compiled protocol");
        int driver=0,runtime=0;unsigned flags=0;require_cuda(cudaDriverGetVersion(&driver),"driver API version");require_cuda(cudaRuntimeGetVersion(&runtime),"runtime version");require_cuda(cudaGetDeviceFlags(&flags),"device flags");
        FILETIME utc{};GetSystemTimePreciseAsFileTime(&utc);uint64_t utc_ticks=(uint64_t(utc.dwHighDateTime)<<32)|utc.dwLowDateTime;
        std::ostringstream header;header<<"{\"type\":\"run_identity\",\"schema\":\"cuda-launch-sync-run/v1\",\"source_sha256\":"<<quote(PROBE_SOURCE_SHA256)<<",\"protocol_sha256\":"<<quote(PROBE_PROTOCOL_SHA256)<<",\"binary_sha256\":"<<quote(binary_sha)<<",\"nvcc_sha256\":"<<quote(PROBE_NVCC_SHA256)<<",\"host_compiler_sha256\":"<<quote(PROBE_CL_SHA256)<<",\"build_script_sha256\":"<<quote(PROBE_BUILD_SCRIPT_SHA256)<<",\"nvcc_version\":"<<quote(PROBE_NVCC_VERSION)<<",\"compiled_cudart_version\":"<<CUDART_VERSION<<",\"compiled_msvc_version\":"<<_MSC_VER<<",\"cuda_driver_api_version\":"<<driver<<",\"cuda_runtime_version\":"<<runtime<<",\"gpu_uuid\":"<<quote(uuid)<<",\"gpu_name\":"<<quote(properties.name)<<",\"compute_capability\":["<<properties.major<<','<<properties.minor<<"],\"sm_count\":"<<properties.multiProcessorCount<<",\"device_flags\":"<<flags<<",\"stream_flags\":"<<cudaStreamNonBlocking<<",\"event_flags\":"<<cudaEventDefault<<",\"qpc_frequency_hz\":"<<f<<",\"qpc_utc_anchor\":"<<tick()<<",\"utc_filetime_100ns\":"<<utc_ticks<<",\"process_id\":"<<GetCurrentProcessId()<<",\"no_llm_dependency\":true,\"no_native_dll_modified\":true}";writer.line(header.str());
        cudaStream_t stream{};cudaEvent_t start{},stop{};uint32_t* device=nullptr;uint32_t* host=nullptr;
        require_cuda(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking),"create stream");require_cuda(cudaEventCreateWithFlags(&start,cudaEventDefault),"create start event");require_cuda(cudaEventCreateWithFlags(&stop,cudaEventDefault),"create end event");require_cuda(cudaMalloc(&device,PROBE_OUTPUT_WORDS*sizeof(uint32_t)),"allocate device buffer");require_cuda(cudaHostAlloc(&host,PROBE_OUTPUT_WORDS*sizeof(uint32_t),cudaHostAllocDefault),"allocate pinned host buffer");
        for(int i=0;i<PROBE_QPC_CONTROL_PAIRS;i++){auto b=tick();auto e=tick();std::ostringstream o;o<<"{\"type\":\"qpc_control\",\"ordinal\":"<<i<<",\"begin\":"<<b<<",\"end\":"<<e<<'}';writer.line(o.str());}
        constexpr int bursts[4]={1,4,16,64};int ordinal=0;bool all_ok=true;
        for(int phase=0;phase<2&&all_ok;phase++){int repeats=phase==0?PROBE_WARMUP_REPEATS:PROBE_FORMAL_REPEATS;
            for(int repeat=0;repeat<repeats&&all_ok;repeat++)for(int position=0;position<16;position++){
                int order=(phase==1&&(repeat%2)==1)?15-position:position;int config=(order+repeat*5)%16;
                Sample r=measure(config/8,(config/4)%2,bursts[config%4],repeat,phase==1,ordinal++,stream,start,stop,device,host);writer.line(sample_json(r,f));
                if(!r.correct){all_ok=false;break;}
            }
        }
        require_cuda(cudaFreeHost(host),"free pinned buffer");require_cuda(cudaFree(device),"free device buffer");require_cuda(cudaEventDestroy(start),"destroy start");require_cuda(cudaEventDestroy(stop),"destroy stop");require_cuda(cudaStreamDestroy(stream),"destroy stream");
        writer.line(std::string("{\"type\":\"run_complete\",\"status\":")+quote(all_ok?"complete":"invalid")+",\"sample_count\":"+std::to_string(ordinal)+",\"all_output_checks_passed\":"+(all_ok?"true":"false")+",\"coefficient_or_profile_written\":false}");writer.flush();return all_ok?0:2;
    }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 2;}
}
