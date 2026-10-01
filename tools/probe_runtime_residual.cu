// Independent CUDA runtime microbenchmark, without model weights or LLM timing.
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA(call) do { cudaError_t s=(call); if(s!=cudaSuccess) { std::fprintf(stderr,"%s: %s\n",#call,cudaGetErrorString(s)); return 2; } } while(0)
__global__ void independent_noop(unsigned int *out) { if(threadIdx.x==0) atomicAdd(out,1u); }

int main() {
    const int warmup=20, formal=40;
    const int counts[]={1,8,32,16}; // final count is the frozen independent holdout.
    cudaDeviceProp device{}; CUDA(cudaGetDeviceProperties(&device,0));
    int driver=0,runtime=0; CUDA(cudaDriverGetVersion(&driver)); CUDA(cudaRuntimeGetVersion(&runtime));
    cudaStream_t stream; CUDA(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    cudaEvent_t begin,end; CUDA(cudaEventCreate(&begin)); CUDA(cudaEventCreate(&end));
    unsigned int *output=nullptr; CUDA(cudaMalloc(&output,sizeof(unsigned int)));
    CUDA(cudaMemset(output,0,sizeof(unsigned int)));
    std::printf("{\"schema\":\"heterollm.cuda-runtime-probe/v1\",\"device\":\"%s\",\"cc\":[%d,%d],\"driver_version\":%d,\"runtime_version\":%d,\"warmup\":%d,\"formal\":%d,\"cases\":[",device.name,device.major,device.minor,driver,runtime,warmup,formal);
    for(int c=0;c<4;++c) {
        int count=counts[c];
        cudaGraph_t graph; cudaGraphExec_t executable;
        CUDA(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
        for(int node=0;node<count;++node) independent_noop<<<1,32,0,stream>>>(output);
        CUDA(cudaStreamEndCapture(stream,&graph));
        CUDA(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0));
        std::vector<double> host[2],gpu[2];
        // Alternate paths per repetition, so drift cannot preferentially affect
        // the direct path or the graph path. Synchronization is outside submit.
        for(int sample=0;sample<warmup+formal;++sample) {
            for(int mode=0;mode<2;++mode) {
                CUDA(cudaEventRecord(begin,stream));
                auto start=std::chrono::steady_clock::now();
                if(mode==0) { for(int node=0;node<count;++node) independent_noop<<<1,32,0,stream>>>(output); CUDA(cudaGetLastError()); }
                else { CUDA(cudaGraphLaunch(executable,stream)); }
                auto finish=std::chrono::steady_clock::now();
                CUDA(cudaEventRecord(end,stream)); CUDA(cudaEventSynchronize(end));
                float milliseconds=0; CUDA(cudaEventElapsedTime(&milliseconds,begin,end));
                if(sample>=warmup) {
                    host[mode].push_back(std::chrono::duration<double,std::nano>(finish-start).count());
                    gpu[mode].push_back(double(milliseconds)*1e6);
                }
            }
        }
        if(c) std::printf(",");
        std::printf("{\"kernel_count\":%d,\"split\":\"%s\",\"ordinary_durations_ns\":[",count,c==3?"holdout":"training");
        for(int i=0;i<formal;++i) std::printf("%s%.1f",i?",":"",host[0][i]);
        std::printf("],\"graph_durations_ns\":[");
        for(int i=0;i<formal;++i) std::printf("%s%.1f",i?",":"",host[1][i]);
        std::printf("],\"ordinary_device_intervals_ns\":[");
        for(int i=0;i<formal;++i) std::printf("%s%.1f",i?",":"",gpu[0][i]);
        std::printf("],\"graph_device_intervals_ns\":[");
        for(int i=0;i<formal;++i) std::printf("%s%.1f",i?",":"",gpu[1][i]);
        std::printf("]}");
        CUDA(cudaGraphExecDestroy(executable)); CUDA(cudaGraphDestroy(graph));
    }
    unsigned int actual=0; CUDA(cudaMemcpy(&actual,output,sizeof(actual),cudaMemcpyDeviceToHost));
    unsigned int expected=2*(warmup+formal)*(1+8+32+16);
    std::printf("],\"correctness\":{\"passed\":%s,\"actual\":%u,\"expected\":%u}}\n",actual==expected?"true":"false",actual,expected);
    CUDA(cudaFree(output)); CUDA(cudaEventDestroy(begin)); CUDA(cudaEventDestroy(end)); CUDA(cudaStreamDestroy(stream));
    return actual==expected?0:3;
}
