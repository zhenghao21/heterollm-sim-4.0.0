// Standalone CUDA Graph runtime benchmark. It contains no model or LLM timing.
// Build: nvcc -O2 -std=c++17 tools/cuda_graph_runtime_microbench.cu -o graphbench.exe
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>
#include <random>
#include <fstream>
#include <unordered_map>

#define CUDA_OK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){std::cerr<<#x<<": "<<cudaGetErrorString(e)<<"\n"; return 2;} } while(0)
using Clock = std::chrono::steady_clock;
__global__ void tiny_op(int *p, int value) { if (threadIdx.x == 0 && blockIdx.x == 0) p[0] += value; }
__global__ void tiny_programmatic_op(int *p, int value) {
  cudaGridDependencySynchronize();
  if (threadIdx.x == 0 && blockIdx.x == 0) p[0] += value;
  cudaTriggerProgrammaticLaunchCompletion();
}
struct SyntheticNode { int type{}, incoming_dependency{}; size_t bytes{}; int copy_kind{}; };
static std::vector<SyntheticNode> structure_nodes;
static void *copy_source{}, *copy_destination{};
static std::vector<SyntheticNode> read_structure(const std::string &path, std::string &topology) {
  std::ifstream file(path); std::string magic; int count=0; file>>magic>>topology>>count;
  if(magic!="heterollm.synthetic-chain/v1" || count<1 || topology.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")!=std::string::npos) throw std::runtime_error("invalid typed structure header");
  std::vector<SyntheticNode> result;
  for(int i=0;i<count;++i) { SyntheticNode node; file>>node.type>>node.incoming_dependency>>node.bytes>>node.copy_kind;
    if(!file || (node.type!=0 && node.type!=1) || node.incoming_dependency<0 || node.incoming_dependency>1
       || (i==0 && node.incoming_dependency!=0) || (node.type==1 && (node.incoming_dependency!=0 || node.copy_kind!=3 || node.bytes==0))) throw std::runtime_error("unsupported synthetic node or dependency");
    result.push_back(node);
  }
  return result;
}

static std::vector<int> parse_sizes(const std::string &s) {
  std::vector<int> out; std::stringstream ss(s); std::string x;
  while (std::getline(ss,x,',')) { int n=std::stoi(x); if(n<1) throw std::runtime_error("sizes must be positive"); out.push_back(n); }
  return out;
}
static double ns(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double,std::nano>(b-a).count(); }
static void json_array(const std::vector<double>& v) { std::cout << "["; for(size_t i=0;i<v.size();++i){if(i)std::cout<<",";std::cout<<v[i];} std::cout<<"]"; }

struct GraphRun {
  cudaGraph_t graph{}; cudaGraphExec_t exec{}; cudaStream_t origin{}; std::vector<cudaStream_t> branches;
  std::vector<cudaEvent_t> events; int n{}; std::string topology;
};

static bool prepare_graph(GraphRun &g) {
  if(cudaStreamCreate(&g.origin)!=cudaSuccess) return false;
  if(g.topology=="chain" || !structure_nodes.empty()) {
    return true;
  }
  const int branch_count=std::min(g.n,4);
  g.branches.resize(branch_count); g.events.resize(branch_count+1);
  for(int i=0;i<branch_count;++i) if(cudaStreamCreate(&g.branches[i])!=cudaSuccess) return false;
  for(auto &e:g.events) if(cudaEventCreateWithFlags(&e,cudaEventDisableTiming)!=cudaSuccess) return false;
  return true;
}

static bool enqueue_body(GraphRun &g, int *device, int value) {
  if(!structure_nodes.empty()) {
    for(int i=0;i<g.n;++i) {
      const auto &node=structure_nodes[i];
      if(node.type==1) {
        if(cudaMemcpyAsync(copy_destination,copy_source,node.bytes,cudaMemcpyDeviceToDevice,g.origin)!=cudaSuccess) return false;
      } else if(node.incoming_dependency==1) {
        cudaLaunchAttribute attribute{}; attribute.id=cudaLaunchAttributeProgrammaticStreamSerialization;
        attribute.val.programmaticStreamSerializationAllowed=1;
        cudaLaunchConfig_t config{}; config.gridDim=dim3(1); config.blockDim=dim3(1);
        config.stream=g.origin; config.attrs=&attribute; config.numAttrs=1;
        if(cudaLaunchKernelEx(&config,tiny_programmatic_op,device+i,value+i)!=cudaSuccess) return false;
      } else tiny_op<<<1,1,0,g.origin>>>(device+i,value+i);
    }
  } else if(g.topology=="chain") {
    for(int i=0;i<g.n;++i) tiny_op<<<1,1,0,g.origin>>>(device+i,value+i);
  } else {
    // A captured fork/join DAG: all nodes wait on the origin, execute on
    // independent streams, then the origin joins every branch before ending.
    const int branch_count=static_cast<int>(g.branches.size());
    if(cudaEventRecord(g.events[0],g.origin)!=cudaSuccess) return false;
    for(int i=0;i<branch_count;++i) if(cudaStreamWaitEvent(g.branches[i],g.events[0],0)!=cudaSuccess) return false;
    for(int i=0;i<g.n;++i) {
      const int branch=i%branch_count;
      tiny_op<<<1,1,0,g.branches[branch]>>>(device+i,value+i);
    }
    for(int i=0;i<branch_count;++i) {
      if(cudaEventRecord(g.events[i+1],g.branches[i])!=cudaSuccess) return false;
      if(cudaStreamWaitEvent(g.origin,g.events[i+1],0)!=cudaSuccess) return false;
    }
  }
  return cudaPeekAtLastError()==cudaSuccess;
}

static bool capture_graph(GraphRun &g, int *device, int value) {
  if(cudaStreamBeginCapture(g.origin,cudaStreamCaptureModeGlobal)!=cudaSuccess) return false;
  if(!enqueue_body(g,device,value)) return false;
  if(cudaStreamEndCapture(g.origin,&g.graph)!=cudaSuccess) return false;
  return true;
}

static bool validate_structure(GraphRun &g) {
  if(structure_nodes.empty()) return true;
  size_t count=0, edge_count=0;
  if(cudaGraphGetNodes(g.graph,nullptr,&count)!=cudaSuccess || count!=structure_nodes.size()) return false;
  std::vector<cudaGraphNode_t> nodes(count);
  if(cudaGraphGetNodes(g.graph,nodes.data(),&count)!=cudaSuccess) return false;
  if(cudaGraphGetEdges_v2(g.graph,nullptr,nullptr,nullptr,&edge_count)!=cudaSuccess || edge_count+1!=count) return false;
  std::vector<cudaGraphNode_t> from(edge_count),to(edge_count); std::vector<cudaGraphEdgeData> edge_data(edge_count);
  if(cudaGraphGetEdges_v2(g.graph,from.data(),to.data(),edge_data.data(),&edge_count)!=cudaSuccess) return false;
  std::unordered_map<cudaGraphNode_t,cudaGraphNode_t> next;
  std::unordered_map<cudaGraphNode_t,int> incoming;
  for(size_t i=0;i<edge_count;++i) {
    if(next.count(from[i]) || incoming.count(to[i]) || edge_data[i].to_port!=0
       || edge_data[i].from_port!=edge_data[i].type || edge_data[i].type>1) return false;
    next[from[i]]=to[i]; incoming[to[i]]=edge_data[i].type;
  }
  cudaGraphNode_t current{};
  for(auto node:nodes) if(!incoming.count(node)) { if(current) return false; current=node; }
  for(size_t i=0;i<count;++i) {
    if(!current) return false;
    cudaGraphNodeType type{}; if(cudaGraphNodeGetType(current,&type)!=cudaSuccess) return false;
    if(static_cast<int>(type)!=structure_nodes[i].type || (i?incoming[current]:0)!=structure_nodes[i].incoming_dependency) return false;
    auto found=next.find(current); current=found==next.end()?nullptr:found->second;
  }
  return current==nullptr;
}

static void destroy_graph(GraphRun &g) {
  if(g.exec) cudaGraphExecDestroy(g.exec); if(g.graph) cudaGraphDestroy(g.graph);
  if(g.origin) cudaStreamDestroy(g.origin);
  for(auto s:g.branches) if(s) cudaStreamDestroy(s);
  for(auto e:g.events) if(e) cudaEventDestroy(e);
}

static bool measure(GraphRun &g, int *device, int reps,
                    std::vector<double>& ordinary_host,std::vector<double>& capture_host,
                    std::vector<double>& instantiate_host,std::vector<double>& update_host,
                    std::vector<double>& first_host,std::vector<double>& replay_host,
                    std::vector<double>& ordinary_device,std::vector<double>& first_device,
                    std::vector<double>& replay_device,std::vector<double>& destroy_exec_host,
                    std::vector<double>& destroy_graph_host,std::vector<double>& update_failure_host) {
  if(reps<2) { std::cerr<<"each visit needs warmup plus at least one sample\n"; return false; }
  cudaStream_t stream{}; cudaEvent_t ev0{},ev1{};
  if(cudaStreamCreate(&stream)!=cudaSuccess || cudaEventCreate(&ev0)!=cudaSuccess || cudaEventCreate(&ev1)!=cudaSuccess) return false;
  GraphRun ordinary; ordinary.n=g.n; ordinary.topology=g.topology;
  if(!prepare_graph(ordinary)) return false;
  for(int r=-1;r<reps-1;++r) {
    cudaMemset(device,0,sizeof(int)*g.n); cudaDeviceSynchronize();
    // Ordinary CPU submission and its separate device-event interval.
    cudaEventRecord(ev0,ordinary.origin); auto a=Clock::now();
    if(!enqueue_body(ordinary,device,1)) return false;
    auto b=Clock::now(); cudaEventRecord(ev1,ordinary.origin); cudaEventSynchronize(ev1);
    float ms=0; cudaEventElapsedTime(&ms,ev0,ev1); if(r>=0){ordinary_host.push_back(ns(a,b)); ordinary_device.push_back(ms*1.0e6);}
    // Stream capture host API work; capture does not execute the graph body.
    GraphRun cg; cg.n=g.n; cg.topology=g.topology; if(!prepare_graph(cg)) return false; a=Clock::now();
    bool ok=capture_graph(cg,device,2); b=Clock::now(); if(!ok) return false; if(r>=0)capture_host.push_back(ns(a,b));
    // Count validation is outside the capture timing boundary.
    size_t count=0;
    if(cudaGraphGetNodes(cg.graph,nullptr,&count)!=cudaSuccess || count!=static_cast<size_t>(g.n)) {
      std::cerr<<"captured node count differs from the reported size: "<<count<<" != "<<g.n<<"\n";
      return false;
    }
    if(!validate_structure(cg)) { std::cerr<<"synthetic captured structure differs from typed template\n"; return false; }
    a=Clock::now(); auto err=cudaGraphInstantiate(&cg.exec,cg.graph,nullptr,nullptr,0); b=Clock::now();
    if(err!=cudaSuccess) return false; if(r>=0)instantiate_host.push_back(ns(a,b));
    // Pinned llama.cpp updates the freshly instantiated executable with the
    // identical captured graph BEFORE its first launch. Measure that event,
    // not a changed-parameter update after first execution.
    cudaGraphNode_t error_node{}; cudaGraphExecUpdateResult update_result{}; a=Clock::now();
    err=cudaGraphExecUpdate(cg.exec,cg.graph,&error_node,&update_result); b=Clock::now();
    if(err!=cudaSuccess || update_result!=cudaGraphExecUpdateSuccess) return false;
    if(r>=0)update_host.push_back(ns(a,b));
    // First graph launch and its device event are separate measurements.
    cudaEventRecord(ev0,stream); a=Clock::now(); err=cudaGraphLaunch(cg.exec,stream); b=Clock::now();
    if(err!=cudaSuccess) return false; if(r>=0)first_host.push_back(ns(a,b)); cudaEventRecord(ev1,stream); cudaEventSynchronize(ev1);
    cudaEventElapsedTime(&ms,ev0,ev1); if(r>=0)first_device.push_back(ms*1.0e6);
    // Steady replay host enqueue time; device event interval remains distinct.
    cudaEventRecord(ev0,stream); a=Clock::now(); err=cudaGraphLaunch(cg.exec,stream); b=Clock::now();
    if(err!=cudaSuccess) return false; if(r>=0)replay_host.push_back(ns(a,b)); cudaEventRecord(ev1,stream); cudaEventSynchronize(ev1);
    cudaEventElapsedTime(&ms,ev0,ev1); if(r>=0)replay_device.push_back(ms*1.0e6);
    // An independently induced topology mismatch measures the failed-update
    // API and error clearing. It is reported separately from same-graph update.
    cudaGraph_t incompatible{}; cudaGraphNode_t extra{};
    if(cudaGraphClone(&incompatible,cg.graph)!=cudaSuccess || cudaGraphAddEmptyNode(&extra,incompatible,nullptr,0)!=cudaSuccess) return false;
    a=Clock::now();
    err=cudaGraphExecUpdate(cg.exec,incompatible,&error_node,&update_result);
    cudaGetLastError(); b=Clock::now();
    if(err!=cudaErrorGraphExecUpdateFailure || update_result!=cudaGraphExecUpdateErrorTopologyChanged) {
      std::cerr<<"synthetic topology mismatch did not produce expected update failure\n"; return false;
    }
    if(r>=0)update_failure_host.push_back(ns(a,b));
    cudaGraphDestroy(incompatible);
    a=Clock::now(); cudaGraphExecDestroy(cg.exec); cg.exec=nullptr; b=Clock::now(); if(r>=0)destroy_exec_host.push_back(ns(a,b));
    a=Clock::now(); cudaGraphDestroy(cg.graph); cg.graph=nullptr; b=Clock::now(); if(r>=0)destroy_graph_host.push_back(ns(a,b));
    destroy_graph(cg);
  }
  destroy_graph(ordinary);
  cudaEventDestroy(ev0); cudaEventDestroy(ev1); cudaStreamDestroy(stream); return true;
}

int main(int argc,char**argv) {
  try {
    std::string train="8,32,128,256,512,1024,2048,4096,8192", holdout="64,192,384,768,1536,3072,6144", topology="chain", structure, previous_structure; int reps=45;
    for(int i=1;i<argc;++i){std::string a=argv[i]; if(a=="--train"&&i+1<argc)train=argv[++i]; else if(a=="--holdout"&&i+1<argc)holdout=argv[++i]; else if(a=="--topology"&&i+1<argc)topology=argv[++i]; else if(a=="--structure"&&i+1<argc)structure=argv[++i]; else if(a=="--previous-structure"&&i+1<argc)previous_structure=argv[++i]; else if(a=="--repetitions"&&i+1<argc)reps=std::stoi(argv[++i]); else {std::cerr<<"usage: graphbench [--train sizes] [--holdout sizes] [--topology chain|fork_join] [--structure template.txt] [--previous-structure old.txt] [--repetitions 45]\n";return 2;}}
    if(!structure.empty()) {
      structure_nodes=read_structure(structure,topology);
      train=std::to_string(structure_nodes.size()); holdout="";
    } else if(topology!="chain"&&topology!="fork_join") throw std::runtime_error("unsupported topology");
    if(reps<3) throw std::runtime_error("at least three measured repetitions required");
    int device_index=0; cudaDeviceProp prop{}; CUDA_OK(cudaGetDevice(&device_index)); CUDA_OK(cudaGetDeviceProperties(&prop,device_index));
    int drv=0,rt=0; CUDA_OK(cudaDriverGetVersion(&drv)); CUDA_OK(cudaRuntimeGetVersion(&rt));
    std::vector<int> sizes=parse_sizes(train); auto hs=parse_sizes(holdout); sizes.insert(sizes.end(),hs.begin(),hs.end());
    auto unique_sizes=sizes; std::sort(unique_sizes.begin(),unique_sizes.end());
    if(sizes.empty() || std::adjacent_find(unique_sizes.begin(),unique_sizes.end())!=unique_sizes.end()) throw std::runtime_error("sizes must be nonempty, disjoint and unique");
    std::vector<SyntheticNode> previous_nodes; std::string previous_topology;
    if(!previous_structure.empty()) previous_nodes=read_structure(previous_structure,previous_topology);
    int maxn=std::max(*std::max_element(sizes.begin(),sizes.end()),static_cast<int>(previous_nodes.size())); int *device=nullptr; CUDA_OK(cudaMalloc(&device,sizeof(int)*maxn));
    size_t max_copy=1; for(const auto &node:structure_nodes) max_copy=std::max(max_copy,node.bytes);
    for(const auto &node:previous_nodes) max_copy=std::max(max_copy,node.bytes);
    CUDA_OK(cudaMalloc(&copy_source,max_copy)); CUDA_OK(cudaMalloc(&copy_destination,max_copy));
    CUDA_OK(cudaMemset(copy_source,0,max_copy)); CUDA_OK(cudaMemset(copy_destination,0,max_copy));
    if(!previous_nodes.empty()) {
      auto new_nodes=structure_nodes; std::vector<double> durations;
      if(previous_nodes.size()==new_nodes.size()) throw std::runtime_error("pair benchmark currently requires guaranteed node-count update failure");
      for(int r=-1;r<reps;++r) {
        CUDA_OK(cudaMemset(device,0,sizeof(int)*maxn)); CUDA_OK(cudaDeviceSynchronize());
        structure_nodes=previous_nodes; GraphRun old; old.n=previous_nodes.size(); old.topology=previous_topology;
        if(!prepare_graph(old) || !capture_graph(old,device,1) || !validate_structure(old)) return 3;
        CUDA_OK(cudaGraphInstantiate(&old.exec,old.graph,nullptr,nullptr,0));
        CUDA_OK(cudaGraphLaunch(old.exec,old.origin)); CUDA_OK(cudaStreamSynchronize(old.origin));
        structure_nodes=new_nodes; GraphRun next; next.n=new_nodes.size(); next.topology=topology;
        if(!prepare_graph(next) || !capture_graph(next,device,2) || !validate_structure(next)) return 3;
        cudaGraphNode_t error_node{}; cudaGraphExecUpdateResult result{};
        auto a=Clock::now(); cudaError_t err=cudaGraphExecUpdate(old.exec,next.graph,&error_node,&result); cudaGetLastError(); auto b=Clock::now();
        if(err!=cudaErrorGraphExecUpdateFailure || result!=cudaGraphExecUpdateErrorTopologyChanged) {
          std::cerr<<"pair update produced unexpected result: "<<cudaGetErrorString(err)<<" result="<<static_cast<int>(result)<<"\n"; return 3;
        }
        if(r>=0)durations.push_back(ns(a,b)); destroy_graph(next); destroy_graph(old);
      }
      std::cout<<"{\"update_failure_ns\":"; json_array(durations); std::cout<<"}\n";
      cudaFree(device); cudaFree(copy_source); cudaFree(copy_destination); return 0;
    }
    std::cout<<"{\"schema\":\"heterollm.cuda-graph-runtime/v2\",\"source_kind\":\"independent_synthetic_runtime_microbenchmark\",\"target_llm_latency_used\":false,\"measurement_boundary\":\"host_wall_and_cuda_event_separate\",\"device\":\""<<prop.name<<"\",\"cc\":\"sm_"<<prop.major<<prop.minor<<"\",\"driver_version\":\""<<drv<<"\",\"runtime_version\":\""<<rt<<"\",\"hardware_id\":\""<<prop.name<<"\",\"architecture\":\"sm_"<<prop.major<<prop.minor<<"\",\"runtime_id\":\"CUDA-"<<rt<<"-driver-"<<drv<<"\",\"cpu_id\":\""<<(std::getenv("PROCESSOR_IDENTIFIER")?std::getenv("PROCESSOR_IDENTIFIER"):"unknown")<<"\",\"os_id\":\"Windows\",\"samples\":[";
    using PhaseMeasurements=std::vector<std::vector<double>>;
    std::vector<PhaseMeasurements> all(sizes.size(),PhaseMeasurements(12));
    std::vector<int> order(sizes.size()); std::iota(order.begin(),order.end(),0);
    std::mt19937 generator(8128);
    // Interleave graph sizes so all train and holdout cases span the run's
    // temperature/load history. Each visit discards one local warmup.
    for(int round=0;round<(reps+2)/3;++round) {
      std::shuffle(order.begin(),order.end(),generator);
      for(int index:order) {
        GraphRun g; g.n=sizes[index]; g.topology=topology; auto &m=all[index];
        const int visit_repetitions=1+std::min(3,reps-round*3);
        if(!measure(g,device,visit_repetitions,m[0],m[1],m[2],m[3],m[4],m[5],m[8],m[9],m[10],m[6],m[7],m[11])) {
          std::cerr<<"measurement failed at "<<topology<<" nodes="<<g.n<<"\n";return 3;
        }
      }
    }
    bool first=true;
    for(size_t index=0;index<sizes.size();++index){int n=sizes[index]; auto &m=all[index];
      if(!first)std::cout<<",";first=false;
      std::cout<<"{\"topology\":\""<<topology<<"\",\"node_count\":"<<n<<",\"timings_ns\":{";
      const char* names[]={"ordinary_submit","capture","instantiate","update","first_launch_submit","replay_submit","destroy_exec","destroy_graph","ordinary_device","first_launch_device","replay_device","update_failure"};
      for(int k=0;k<12;++k){if(k)std::cout<<",";std::cout<<"\""<<names[k]<<"\":";json_array(m[k]);}
      auto tr=parse_sizes(train); std::string split=(std::find(tr.begin(),tr.end(),n)!=tr.end())?"train":"holdout";
      std::cout<<"},\"split\":\""<<split<<"\"}";
    }
    std::cout<<"]}\n"; cudaFree(device); cudaFree(copy_source); cudaFree(copy_destination); return 0;
  } catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 2;}
}
