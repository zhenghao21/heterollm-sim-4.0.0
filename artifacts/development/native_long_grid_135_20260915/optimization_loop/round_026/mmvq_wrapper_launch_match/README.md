# 固定 shape 的 wrapper 启动与数值资格准备

本目录只推进 Q5_0、M=1、K=4096、N=3072、无 ids/fusion、关闭 CUDA graph 的对照；不收集性能时延，不准入成本模型参数，不重测 LLM。R27 正在运行期间只能构建和做主机测试。动态对照由父代理在 R27 full262 终态屏障和全局闲置检查都通过后串行执行。

## 修正与来源

- `QI5_ERRATUM.json` 独立更正旧 contract 的 QI5_0=8、32 blocks/iteration：实际为 QI5_0=4、64 blocks/iteration；K4096 迭代2次，small_k=false。旧 contract 和已保存原生 run 保持不变。
- 新 shim 复制锁定 mmvq.cu 的前1404行原始字节并调用原私有 dispatcher。只改连续布局的主机参数推导；六个 channel/sample stride 为 393216、128、3072，并由静态 shape 推导。
- 直接复用原 R6 ABI、Q8转换对象、运行支撑对象，以及 ExC recorder 的两份头文件。编译后两个 cubin 必须与原始目标 DLL 的黄金 cubin 字节相同。
- 原 R6 数值合格记录不授予新 shim 数值资格。

## 两个独立 pass

1. 启动观测 pass：seed20260916，沿用 target 的 xorshift 数据顺序与锁定 CPU quantizer，conversion→main 相邻发射；中间无同步、事件、拷贝或主机数值检查；末尾仅一次同步。使用同一 ExC recorder 核对两个内核的完整有效参数，包括六个 stride、PDL 属性、stream/context 和缓冲区关系。
2. 新 shim 数值 pass：关闭 CUPTI 后，用已有独立 CPU reference 生成非零 dyadic 数据与预声明误差界，再相邻执行 conversion→main。末尾取回全部4608字节Q8和3072个输出，保存 actual/expected/reference/bound 二进制；Python 再次读取原始文件独立核算。这个结果仅证明第二套 fixture 的固定 shape 数值资格，不能称为第一套随机数据的数值验证。

比较器只忽略跨进程绝对地址、经过关系检查的 context/correlation 编号，以及 PDL union 的未激活尾部字节。属性值、完整 scalar/layout 参数不允许忽略。

## 可复现入口

以下命令均在本目录调用，Python 使用 `E:/anaconda/python.exe`。

```powershell
E:/anaconda/python.exe build.py check-inputs
E:/anaconda/python.exe build.py build
E:/anaconda/python.exe build.py host-test
E:/anaconda/python.exe run_wrapper_capture.py --check
```

构建尝试使用新的 `build_attempt.NNNN` 目录保存输出和失败记录，成功后只创建一次 `build_manifest.json`。主机测试只执行不含CUDA/GGML依赖的布局程序和Python反例测试，不运行wrapper程序。

只有父代理在当前仿真完全结束后使用：

```powershell
E:/anaconda/python.exe run_wrapper_capture.py --execute
```

动态入口必须验证 R27 full262 终态屏障、构建/主机测试/目标原始采集完整身份，并检查仿真、冻结、native和probe进程。无需桌面GPU进程全部消失，因为本步骤不测时延。输出独占创建到 `wrapper_capture_run.0001`；启动、解析、参数或数值失败均保留 rejected finish，不能覆写重试。

通过动态检查后的结论仍限于此固定 shape 的有效启动参数相同与独立数值fixture合格。性能等价、其他shape、并发和模型泛化都未因此得到验证；性能参数准入仍为0。
