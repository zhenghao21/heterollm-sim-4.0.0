#include "source_reference.h"
#include <fstream>
#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
int main(int argc,char**argv){if(argc!=4)return 2;int k=std::stoi(argv[3]);std::ifstream w(argv[1],std::ios::binary),x(argv[2],std::ios::binary);std::vector<uint8_t> packed((std::istreambuf_iterator<char>(w)),{});std::vector<float> input(4*k);x.read((char*)input.data(),input.size()*4);int n,m;std::cout<<std::setprecision(17);while(std::cin>>n>>m)std::cout<<q5_source_reference(packed.data()+n*(k/32)*22,input.data()+m*k,k)<<'\n';return 0;}

