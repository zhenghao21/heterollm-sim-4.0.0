#include "source_reference.h"
#include <fstream>
#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
int main(int argc,char**argv){
 try{
  if(argc!=6)return 2;int k=std::stoi(argv[3]);ggml_type type=std::string(argv[4])=="Q5_0"?GGML_TYPE_Q5_0:std::string(argv[4])=="Q8_0"?GGML_TYPE_Q8_0:GGML_TYPE_COUNT;bool mmvq=std::string(argv[5])=="MMVQ";
  if(std::string(argv[5])!="MMVQ"&&std::string(argv[5])!="MMQ")return 2;
  if(k<=0||k%32||type==GGML_TYPE_COUNT)return 2;
  std::ifstream w(argv[1],std::ios::binary),xf(argv[2],std::ios::binary);if(!w||!xf)return 3;
  std::vector<uint8_t> packed((std::istreambuf_iterator<char>(w)),{});std::vector<char> bytes((std::istreambuf_iterator<char>(xf)),{});if(bytes.size()%4)return 3;
  std::vector<float> input(bytes.size()/4);std::memcpy(input.data(),bytes.data(),bytes.size());size_t row=static_cast<size_t>(k/32)*(type==GGML_TYPE_Q5_0?22:34);int n,m;
  std::cout<<std::setprecision(17);while(std::cin>>n>>m){if(n<0||m<0||(size_t(n)+1)*row>packed.size()||(size_t(m)+1)*k>input.size())return 4;std::cout<<source_path_reference(packed.data()+n*row,input.data()+m*k,k,type,mmvq)<<'\n';}
  return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 5;}
}
