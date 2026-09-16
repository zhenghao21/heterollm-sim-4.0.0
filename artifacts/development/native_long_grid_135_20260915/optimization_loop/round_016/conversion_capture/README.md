# 原DLL launch记录器源码就绪（未编译）

本次仅准备 launch_recorder.cpp 与 entry.py。没有重写CUDA kernel、没有修改native DLL、没有编译或启动GPU。entry.py inspect 已通过17项头文件/锁定文件身份检查。

## 生命周期结论

锁定 common.cuh 的 ggml_cuda_pool_alloc 析构会调用 pool.free。ggml-cuda.cu 的 legacy pool 在没有空槽时会cudaFree，VMM pool立即回退pool_used。因此不能仅凭同步/图返回证明中间缓冲依旧有效。本实现**没有D2H复制**，不读取vy指针，不声称捕获q码或scale。输出明确device_buffer_copied=false、quantized_code_observed=false。

## 源码行为

单次 Q8_0 M64 N896 K1024 graph，使用与原样本一致的xorshift seed20260914和生成顺序。原DLL通过公共ggml后端正常执行；记录器不替换kernel或修改参数。CUPTI订阅仅cudaLaunchKernel_v7000，在callback入口复制官方params结构、symbol/correlation/context/stream/grid/block/shared memory。只有符号明确含quantize_mmq_q8_1才按锁定quantize.cu原声明复制11个参数值。callback不调用CUDA、不打印、不写磁盘、不保存functionParams悬空指针；未知launch保留名称而不猜参数。未支持的launch API可能导致记录缺失，绝不能把空记录当捕获成功。

记录器在CUDA backend建立前核验加载的ggml-base.dll、ggml-cuda.dll路径；Python入口前后核验锁定DLL及源文件SHA。CUPTI版本由本机已安装2025.1.1提供；当前driver/runtime兼容性仍未知，订阅失败会记录CUPTI状态并失败关闭。不得和Nsight/ncu/其他CUPTI订阅者一起启动。回调捕获不是时延测量；不输出拟合值或成本。

## root空闲后入口

```text
E:/anaconda/python.exe entry.py inspect
E:/anaconda/python.exe entry.py compile --root-idle-confirmed
E:/anaconda/python.exe entry.py host-api-test --root-idle-confirmed
E:/anaconda/python.exe entry.py run --root-idle-confirmed --root-gpu-authorized --output NEW_OUTPUT.json
```

compile使用本机MSVC/CUDA/CUPTI头文件及import libs，不下载任何工具，独占创建exe/manifest/log。host-api-test只构造主机launch参数数组、验证memcpy后的参数值和大小，不调用CUDA API或后端。GPU执行必须由root另行授权；当前未运行compile/host-api-test/run。

编译尚未验证，真实kernel符号、参数、订阅ABI也尚未观察。当前产物是源码准备，不是实际捕获证据。下一阶段如果需要q=63/64直接证据，仍须解决可证明的buffer生命周期或另立source-built诊断，不可把此记录器输出直接当q值证明。
