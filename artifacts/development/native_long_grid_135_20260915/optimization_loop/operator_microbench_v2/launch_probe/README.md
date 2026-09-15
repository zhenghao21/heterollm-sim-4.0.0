# 独立 CUDA launch / 同步微基准（准备阶段）

此目录只构建独立 executable。当前不执行设备探测、预热或正式计时，也不改任何旧探针、native DLL 或模型配置。

## 固定协议

`protocol.json` 已预先写入并由 `protocol.sha256` 固定。2 个 kernel × 2 个供给模式 × 4 个 burst 长度 = 16 个配置；每配置 10 次 warmup、30 次 formal，共 160 + 480 样本。配置顺序固定轮换，保留全部 warmup 与 formal，不能删离群点或挑选重跑。

空 kernel 没有计算主体；低负载 kernel 以一个 32 线程 block 做 16 步固定整数运算，结果写入各 invocation 独立的小段。每样本在计时后核对整个 8192-byte 输出缓冲区，包括未写部分的 sentinel。

使用预先创建的非阻塞 stream 和两个启用计时的 CUDA event。分配、清零、D2H 拷贝、输出校验、文件写入在测量区间之外。记录每次 API 的 QPC ticks、QPC 频率、事件状态和固定输出检查。

## 区间解释

- `host_enqueue_call_total_ns` 只累加各次 launch API 的不相交区间。
- `host_enqueue_envelope_ns` 是首个 launch 开始至末个 launch 返回；每次同步模式下包含先前同步等待，不能再加同步总数。
- `host_sync_call_total_ns` 是单独记录的同步 API 区间之和；末尾总同步与每次同步各自只记一次。
- `total_wall_ns` 包含开始 event 提交、整个供给过程、结束 event 提交和末尾同步，也包含观测开销。不能把上面包含区间再次加到 wall。
- `cuda_event_span_ns` 由同一 stream 的 event 计算，可能包括主机未及时供给的空隙；它不是 pure kernel sum，也不能与 host wall 相加。
- 另保留 256 对 QPC 控制读数；不自动减去计时器或 observer 开销。

## 构建

使用已有 CUDA 12.8 / MSVC，目标 `sm_120`，静态 CUDA runtime，无 Nsight/CUPTI。构建脚本最多执行两次编译，所有编译尝试追加到同一个 `build/compile.log`。已成功的目录拒绝覆盖重建。

```powershell
E:/anaconda/python.exe ./build_probe.py
```

## 由 root 在空闲窗口执行（现在不要运行）

先确认 R5/R6 仿真和其他 GPU/CPU 微基准已停止占用。然后从此目录显式运行：

```powershell
E:/anaconda/python.exe ./run_probe.py --idle-window-confirmed
```

该命令核对 build/source/protocol/exe/tool SHA，在运行前后读取 `nvidia-smi` 的 UUID、驱动、时钟和状态，不设置时钟。结果创建于新的 `runs/<UTC>/`，不覆盖原数据。binary 自身也要求显式 idle flag，并核对源/协议哈希、GPU UUID/CC/SM。新驱动与协议不一致会在计时前拒绝。

`analyze_results.py` 只读取保存结果，检查完整顺序、全部输出/事件状态和原始 QPC。预设波动门为 MAD/median ≤10%、(P95−P05)/median ≤40%、前后半段中位数相对差 ≤15%；设备跨度 <2µs 标记分辨率受限，QPC 对读开销超过单次 host launch 中位数的20%标记观测受限。全部门都通过也只得到诊断结果。

不能把空 kernel 的某个中位数直接写成 `kernel_launch_ns`。队列供给、WDDM/线程调度、参数与真实算子复杂度、同步模式都会影响可迁移性。需要另行预注册独立 kernel / stream / 供给模式的传递验证；此阶段的 `coefficient_candidate` 固定为 null，不保证成本模型改善。

参考语义：NVIDIA CUDA Runtime API 的 event elapsed-time 说明；Microsoft QueryPerformanceCounter / QueryPerformanceFrequency 文档。本目录通过原始边界保留可复核性，不使用 LLM 时延或模型 shape 选系数。
