# GGUF 预设来源清单

2026-10-08：22 个条目全部保留。原有 5 个本机 GGUF 记录保留；本次新增 17 份完整目录，其中 15 个可运行、2 个 Qwen3 MoE 架构待适配。下面列出本次新增来源。

固定仓库版本与文件身份来自发布者；HTTP Range 已读取完整元数据和张量目录，未下载全部权重。发布者的文件哈希未在本机重新计算，不能当作本机验证过的完整权重哈希。

实际构图以文件张量目录为准，来源一致和运行成功不代表 native 预测精度已验证。

## Llama 3.1 405B Instruct (KV16)

- 预设 ID：`llama3_1-405b`
- 仓库：[bullerwins/Meta-Llama-3.1-405B-Instruct-GGUF](https://huggingface.co/bullerwins/Meta-Llama-3.1-405B-Instruct-GGUF/tree/ad731614180e6fd90426ce87b5ef44f64c897aad)
- 固定版本：`ad731614180e6fd90426ce87b5ef44f64c897aad`
- 变体：Instruct (16 KV heads)
- 文件格式：`MOSTLY_Q4_K_M`
- 状态：可运行

该固定版本 GGUF 实际包含 16 个 KV 头；与旧配置条目的 8 KV 头版本不同，按文件的元数据和完整张量形状构图。 [官方变体定义](https://github.com/meta-llama/llama-models/blob/main/models/sku_list.py)。

| 文件 | 大小（bytes） | 发布者 LFS SHA-256 |
|---|---:|---|
| [Meta-Llama-3.1-405B-Instruct-Q4_K_M-00001-of-00006.gguf](https://huggingface.co/bullerwins/Meta-Llama-3.1-405B-Instruct-GGUF/resolve/ad731614180e6fd90426ce87b5ef44f64c897aad/Meta-Llama-3.1-405B-Instruct-Q4_K_M-00001-of-00006.gguf) | 44535283264 | `0876d57b6a24fed0e1be5944e41ba4072dfa6b4b7ea47255186beffe060070c0` |
| [Meta-Llama-3.1-405B-Instruct-Q4_K_M-00002-of-00006.gguf](https://huggingface.co/bullerwins/Meta-Llama-3.1-405B-Instruct-GGUF/resolve/ad731614180e6fd90426ce87b5ef44f64c897aad/Meta-Llama-3.1-405B-Instruct-Q4_K_M-00002-of-00006.gguf) | 44867531424 | `f431750967da1b01074edc04e041f6e25d14fe9578305df7d6c52f6c770ac749` |
| [Meta-Llama-3.1-405B-Instruct-Q4_K_M-00003-of-00006.gguf](https://huggingface.co/bullerwins/Meta-Llama-3.1-405B-Instruct-GGUF/resolve/ad731614180e6fd90426ce87b5ef44f64c897aad/Meta-Llama-3.1-405B-Instruct-Q4_K_M-00003-of-00006.gguf) | 44527726976 | `b989638f2d4bb12ee2f86c2dc93c2786ca26a59d6d134abe8f88d9c32c234ba4` |
| [Meta-Llama-3.1-405B-Instruct-Q4_K_M-00004-of-00006.gguf](https://huggingface.co/bullerwins/Meta-Llama-3.1-405B-Instruct-GGUF/resolve/ad731614180e6fd90426ce87b5ef44f64c897aad/Meta-Llama-3.1-405B-Instruct-Q4_K_M-00004-of-00006.gguf) | 44642546304 | `63feeb9c001047758dab2d06d03da6e778bf563597c5d7734bd10e43c09990fb` |
| [Meta-Llama-3.1-405B-Instruct-Q4_K_M-00005-of-00006.gguf](https://huggingface.co/bullerwins/Meta-Llama-3.1-405B-Instruct-GGUF/resolve/ad731614180e6fd90426ce87b5ef44f64c897aad/Meta-Llama-3.1-405B-Instruct-Q4_K_M-00005-of-00006.gguf) | 44981498400 | `ce9e363f8fcb99c2f311855c4f53688616bea9697b8d5c563230da7a3454af98` |
| [Meta-Llama-3.1-405B-Instruct-Q4_K_M-00006-of-00006.gguf](https://huggingface.co/bullerwins/Meta-Llama-3.1-405B-Instruct-GGUF/resolve/ad731614180e6fd90426ce87b5ef44f64c897aad/Meta-Llama-3.1-405B-Instruct-Q4_K_M-00006-of-00006.gguf) | 22161364480 | `4cf75e1e28bca5feca67bfb548c5c6025aa87ebb5605547e05132171c0b28b92` |

## Llama 3.1 70B Instruct

- 预设 ID：`llama3_1-70b`
- 仓库：[bartowski/Meta-Llama-3.1-70B-Instruct-GGUF](https://huggingface.co/bartowski/Meta-Llama-3.1-70B-Instruct-GGUF/tree/83fb6e83d0a8aada42d499259bc929d922e9a558)
- 固定版本：`83fb6e83d0a8aada42d499259bc929d922e9a558`
- 变体：Instruct
- 文件格式：`MOSTLY_Q4_K_M`
- 状态：可运行

| 文件 | 大小（bytes） | 发布者 LFS SHA-256 |
|---|---:|---|
| [Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf](https://huggingface.co/bartowski/Meta-Llama-3.1-70B-Instruct-GGUF/resolve/83fb6e83d0a8aada42d499259bc929d922e9a558/Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf) | 42520398400 | `f775c87029be95fb41df9e2882e6e938b73121c30ffc235ac6b6b880add49aa5` |

## Llama 3.1 8B Instruct

- 预设 ID：`llama3_1-8b`
- 仓库：[bartowski/Meta-Llama-3.1-8B-Instruct-GGUF](https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/tree/bf5b95e96dac0462e2a09145ec66cae9a3f12067)
- 固定版本：`bf5b95e96dac0462e2a09145ec66cae9a3f12067`
- 变体：Instruct
- 文件格式：`MOSTLY_Q4_K_M`
- 状态：可运行

| 文件 | 大小（bytes） | 发布者 LFS SHA-256 |
|---|---:|---|
| [Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf](https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/bf5b95e96dac0462e2a09145ec66cae9a3f12067/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf) | 4920739232 | `7b064f5842bf9532c91456deda288a1b672397a54fa729aa665952863033557c` |

## Llama 3.2 1B Instruct

- 预设 ID：`llama3_2-1b`
- 仓库：[bartowski/Llama-3.2-1B-Instruct-GGUF](https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/tree/067b946cf014b7c697f3654f621d577a3e3afd1c)
- 固定版本：`067b946cf014b7c697f3654f621d577a3e3afd1c`
- 变体：Instruct
- 文件格式：`MOSTLY_Q4_K_M`
- 状态：可运行

| 文件 | 大小（bytes） | 发布者 LFS SHA-256 |
|---|---:|---|
| [Llama-3.2-1B-Instruct-Q4_K_M.gguf](https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/067b946cf014b7c697f3654f621d577a3e3afd1c/Llama-3.2-1B-Instruct-Q4_K_M.gguf) | 807694464 | `6f85a640a97cf2bf5b8e764087b1e83da0fdb51d7c9fab7d0fece9385611df83` |

## Llama 3.2 3B Instruct

- 预设 ID：`llama3_2-3b`
- 仓库：[bartowski/Llama-3.2-3B-Instruct-GGUF](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/tree/5ab33fa94d1d04e903623ae72c95d1696f09f9e8)
- 固定版本：`5ab33fa94d1d04e903623ae72c95d1696f09f9e8`
- 变体：Instruct
- 文件格式：`MOSTLY_Q4_K_M`
- 状态：可运行

| 文件 | 大小（bytes） | 发布者 LFS SHA-256 |
|---|---:|---|
| [Llama-3.2-3B-Instruct-Q4_K_M.gguf](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/5ab33fa94d1d04e903623ae72c95d1696f09f9e8/Llama-3.2-3B-Instruct-Q4_K_M.gguf) | 2019377696 | `6c1a2b41161032677be168d354123594c0e6e67d2b9227c84f296ad037c728ff` |

## Llama 3.3 70B Instruct

- 预设 ID：`llama3_3-70b`
- 仓库：[bartowski/Llama-3.3-70B-Instruct-GGUF](https://huggingface.co/bartowski/Llama-3.3-70B-Instruct-GGUF/tree/b6c5c9f176f3279204034e1d16d393105e95cb88)
- 固定版本：`b6c5c9f176f3279204034e1d16d393105e95cb88`
- 变体：Instruct
- 文件格式：`MOSTLY_Q4_K_M`
- 状态：可运行

| 文件 | 大小（bytes） | 发布者 LFS SHA-256 |
|---|---:|---|
| [Llama-3.3-70B-Instruct-Q4_K_M.gguf](https://huggingface.co/bartowski/Llama-3.3-70B-Instruct-GGUF/resolve/b6c5c9f176f3279204034e1d16d393105e95cb88/Llama-3.3-70B-Instruct-Q4_K_M.gguf) | 42520398816 | `32df3baccb556f9840059b2528b2dee4d3d516b24afdfb9d0c56ff5f63e3a664` |

## Qwen2.5-0.5B Instruct

- 预设 ID：`qwen2_5-0_5b`
- 仓库：[Qwen/Qwen2.5-0.5B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/tree/9217f5db79a29953eb74d5343926648285ec7e67)
- 固定版本：`9217f5db79a29953eb74d5343926648285ec7e67`
- 变体：Instruct
- 文件格式：`MOSTLY_Q4_K_M`
- 状态：可运行

| 文件 | 大小（bytes） | 发布者 LFS SHA-256 |
|---|---:|---|
| [qwen2.5-0.5b-instruct-q4_k_m.gguf](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/9217f5db79a29953eb74d5343926648285ec7e67/qwen2.5-0.5b-instruct-q4_k_m.gguf) | 491400032 | `74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db` |

## Qwen2.5-14B Instruct

- 预设 ID：`qwen2_5-14b`
- 仓库：[bartowski/Qwen2.5-14B-Instruct-GGUF](https://huggingface.co/bartowski/Qwen2.5-14B-Instruct-GGUF/tree/05244aa5d871c661c80082a15d3bce44714d068d)
- 固定版本：`05244aa5d871c661c80082a15d3bce44714d068d`
- 变体：Instruct
- 文件格式：`MOSTLY_Q4_K_M`
- 状态：可运行

| 文件 | 大小（bytes） | 发布者 LFS SHA-256 |
|---|---:|---|
| [Qwen2.5-14B-Instruct-Q4_K_M.gguf](https://huggingface.co/bartowski/Qwen2.5-14B-Instruct-GGUF/resolve/05244aa5d871c661c80082a15d3bce44714d068d/Qwen2.5-14B-Instruct-Q4_K_M.gguf) | 8988110976 | `e47ad95dad6ff848b431053b375adb5d39321290ea2c638682577dafca87c008` |

## Qwen2.5-1.5B Instruct

- 预设 ID：`qwen2_5-1_5b`
- 仓库：[Qwen/Qwen2.5-1.5B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/tree/91cad51170dc346986eccefdc2dd33a9da36ead9)
- 固定版本：`91cad51170dc346986eccefdc2dd33a9da36ead9`
- 变体：Instruct
- 文件格式：`MOSTLY_Q4_K_M`
- 状态：可运行

| 文件 | 大小（bytes） | 发布者 LFS SHA-256 |
|---|---:|---|
| [qwen2.5-1.5b-instruct-q4_k_m.gguf](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/91cad51170dc346986eccefdc2dd33a9da36ead9/qwen2.5-1.5b-instruct-q4_k_m.gguf) | 1117320736 | `6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e` |

## Qwen2.5-32B Instruct

- 预设 ID：`qwen2_5-32b`
- 仓库：[bartowski/Qwen2.5-32B-Instruct-GGUF](https://huggingface.co/bartowski/Qwen2.5-32B-Instruct-GGUF/tree/2116cbb385b8ce3a4d28cf3bf1cd2039a55821a6)
- 固定版本：`2116cbb385b8ce3a4d28cf3bf1cd2039a55821a6`
- 变体：Instruct
- 文件格式：`MOSTLY_Q4_K_M`
- 状态：可运行

| 文件 | 大小（bytes） | 发布者 LFS SHA-256 |
|---|---:|---|
| [Qwen2.5-32B-Instruct-Q4_K_M.gguf](https://huggingface.co/bartowski/Qwen2.5-32B-Instruct-GGUF/resolve/2116cbb385b8ce3a4d28cf3bf1cd2039a55821a6/Qwen2.5-32B-Instruct-Q4_K_M.gguf) | 19851336576 | `2e5f6daea180dbc59f65a40641e94d3973b5dbaa32b3c0acf54647fa874e519e` |

## Qwen2.5-3B Instruct

- 预设 ID：`qwen2_5-3b`
- 仓库：[Qwen/Qwen2.5-3B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/tree/7dabda4d13d513e3e842b20f0d435c732f172cbe)
- 固定版本：`7dabda4d13d513e3e842b20f0d435c732f172cbe`
- 变体：Instruct
- 文件格式：`MOSTLY_Q4_K_M`
- 状态：可运行

| 文件 | 大小（bytes） | 发布者 LFS SHA-256 |
|---|---:|---|
| [qwen2.5-3b-instruct-q4_k_m.gguf](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/7dabda4d13d513e3e842b20f0d435c732f172cbe/qwen2.5-3b-instruct-q4_k_m.gguf) | 2104932768 | `626b4a6678b86442240e33df819e00132d3ba7dddfe1cdc4fbb18e0a9615c62d` |

## Qwen2.5-72B Instruct

- 预设 ID：`qwen2_5-72b`
- 仓库：[bartowski/Qwen2.5-72B-Instruct-GGUF](https://huggingface.co/bartowski/Qwen2.5-72B-Instruct-GGUF/tree/d43fd973131bce821f41e2df3c78c6fe15c5627a)
- 固定版本：`d43fd973131bce821f41e2df3c78c6fe15c5627a`
- 变体：Instruct
- 文件格式：`MOSTLY_Q4_K_M`
- 状态：可运行

| 文件 | 大小（bytes） | 发布者 LFS SHA-256 |
|---|---:|---|
| [Qwen2.5-72B-Instruct-Q4_K_M.gguf](https://huggingface.co/bartowski/Qwen2.5-72B-Instruct-GGUF/resolve/d43fd973131bce821f41e2df3c78c6fe15c5627a/Qwen2.5-72B-Instruct-Q4_K_M.gguf) | 47415715488 | `e4c8fad16946be8cf0bbf67eb8f4e18fc7415a5a6d2854b4cda453edb4082545` |

## Qwen2.5-7B Instruct

- 预设 ID：`qwen2_5-7b`
- 仓库：[bartowski/Qwen2.5-7B-Instruct-GGUF](https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/tree/8911e8a47f92bac19d6f5c64a2e2095bd2f7d031)
- 固定版本：`8911e8a47f92bac19d6f5c64a2e2095bd2f7d031`
- 变体：Instruct
- 文件格式：`MOSTLY_Q4_K_M`
- 状态：可运行

| 文件 | 大小（bytes） | 发布者 LFS SHA-256 |
|---|---:|---|
| [Qwen2.5-7B-Instruct-Q4_K_M.gguf](https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/8911e8a47f92bac19d6f5c64a2e2095bd2f7d031/Qwen2.5-7B-Instruct-Q4_K_M.gguf) | 4683074240 | `65b8fcd92af6b4fefa935c625d1ac27ea29dcb6ee14589c55a8f115ceaaa1423` |

## Qwen3-14B

- 预设 ID：`qwen3-14b`
- 仓库：[Qwen/Qwen3-14B-GGUF](https://huggingface.co/Qwen/Qwen3-14B-GGUF/tree/530227a7d994db8eca5ab5ced2fb692b614357fd)
- 固定版本：`530227a7d994db8eca5ab5ced2fb692b614357fd`
- 变体：Qwen3 (thinking/non-thinking)
- 文件格式：`MOSTLY_Q4_K_M`
- 状态：可运行

| 文件 | 大小（bytes） | 发布者 LFS SHA-256 |
|---|---:|---|
| [Qwen3-14B-Q4_K_M.gguf](https://huggingface.co/Qwen/Qwen3-14B-GGUF/resolve/530227a7d994db8eca5ab5ced2fb692b614357fd/Qwen3-14B-Q4_K_M.gguf) | 9001752960 | `500a8806e85ee9c83f3ae08420295592451379b4f8cf2d0f41c15dffeb6b81f0` |

## Qwen3-235B-A22B

- 预设 ID：`qwen3-235b-a22b`
- 仓库：[Qwen/Qwen3-235B-A22B-GGUF](https://huggingface.co/Qwen/Qwen3-235B-A22B-GGUF/tree/211e807ecd6d37446a747409108783e39a04e80f)
- 固定版本：`211e807ecd6d37446a747409108783e39a04e80f`
- 变体：Qwen3 (thinking/non-thinking)
- 文件格式：`MOSTLY_Q4_K_M`
- 状态：已有 GGUF，架构待适配；禁止运行和编辑

| 文件 | 大小（bytes） | 发布者 LFS SHA-256 |
|---|---:|---|
| [Q4_K_M/Qwen3-235B-A22B-Q4_K_M-00001-of-00005.gguf](https://huggingface.co/Qwen/Qwen3-235B-A22B-GGUF/resolve/211e807ecd6d37446a747409108783e39a04e80f/Q4_K_M/Qwen3-235B-A22B-Q4_K_M-00001-of-00005.gguf) | 29742283872 | `ce666f562eb8eefda48eec0aa93680ba40ca5727d0ba9918f3fb1771141375e6` |
| [Q4_K_M/Qwen3-235B-A22B-Q4_K_M-00002-of-00005.gguf](https://huggingface.co/Qwen/Qwen3-235B-A22B-GGUF/resolve/211e807ecd6d37446a747409108783e39a04e80f/Q4_K_M/Qwen3-235B-A22B-Q4_K_M-00002-of-00005.gguf) | 29974106496 | `ac73c457aac5993d02444063c976062c70be609c03ca6584d0c3be0632e79ce6` |
| [Q4_K_M/Qwen3-235B-A22B-Q4_K_M-00003-of-00005.gguf](https://huggingface.co/Qwen/Qwen3-235B-A22B-GGUF/resolve/211e807ecd6d37446a747409108783e39a04e80f/Q4_K_M/Qwen3-235B-A22B-Q4_K_M-00003-of-00005.gguf) | 29933980640 | `6357074e14a748446c2b0dcc64a1a7c64b1c124fbfb9cb48114147645d3310fe` |
| [Q4_K_M/Qwen3-235B-A22B-Q4_K_M-00004-of-00005.gguf](https://huggingface.co/Qwen/Qwen3-235B-A22B-GGUF/resolve/211e807ecd6d37446a747409108783e39a04e80f/Q4_K_M/Qwen3-235B-A22B-Q4_K_M-00004-of-00005.gguf) | 29936094304 | `2eb47718eaeeebee3726cd2ca4ac11b5768faded5d7adddec5b754aeb7c387dd` |
| [Q4_K_M/Qwen3-235B-A22B-Q4_K_M-00005-of-00005.gguf](https://huggingface.co/Qwen/Qwen3-235B-A22B-GGUF/resolve/211e807ecd6d37446a747409108783e39a04e80f/Q4_K_M/Qwen3-235B-A22B-Q4_K_M-00005-of-00005.gguf) | 22567608864 | `dbe2c7547c0d397d2ef16e08218a586383771dc12e9678cfd6fa8cc23e94e8f8` |

## Qwen3-30B-A3B

- 预设 ID：`qwen3-30b-a3b`
- 仓库：[Qwen/Qwen3-30B-A3B-GGUF](https://huggingface.co/Qwen/Qwen3-30B-A3B-GGUF/tree/e4d4bafdfb96a411a163846265362aceb0b9c63a)
- 固定版本：`e4d4bafdfb96a411a163846265362aceb0b9c63a`
- 变体：Qwen3 (thinking/non-thinking)
- 文件格式：`MOSTLY_Q4_K_M`
- 状态：已有 GGUF，架构待适配；禁止运行和编辑

| 文件 | 大小（bytes） | 发布者 LFS SHA-256 |
|---|---:|---|
| [Qwen3-30B-A3B-Q4_K_M.gguf](https://huggingface.co/Qwen/Qwen3-30B-A3B-GGUF/resolve/e4d4bafdfb96a411a163846265362aceb0b9c63a/Qwen3-30B-A3B-Q4_K_M.gguf) | 18556685824 | `0d003f6662faee786ed5da3e31b29c978de5ae5d275c8794c606a7f3c01aa8f5` |

## Qwen3-32B

- 预设 ID：`qwen3-32b`
- 仓库：[Qwen/Qwen3-32B-GGUF](https://huggingface.co/Qwen/Qwen3-32B-GGUF/tree/938a7432affaec9157f883a87164e2646ae17555)
- 固定版本：`938a7432affaec9157f883a87164e2646ae17555`
- 变体：Qwen3 (thinking/non-thinking)
- 文件格式：`MOSTLY_Q4_K_M`
- 状态：可运行

| 文件 | 大小（bytes） | 发布者 LFS SHA-256 |
|---|---:|---|
| [Qwen3-32B-Q4_K_M.gguf](https://huggingface.co/Qwen/Qwen3-32B-GGUF/resolve/938a7432affaec9157f883a87164e2646ae17555/Qwen3-32B-Q4_K_M.gguf) | 19762149024 | `efd971561896866f0e910cce52761ca77b1b138090c7f15fe284676d57d1f689` |

## 405B 来源调查说明

最初的 `bartowski/Meta-Llama-3.1-405B-Instruct-GGUF` 候选返回 HTTP 401。匿名响应无法确定是仓库不存在、私有或访问策略所致，因此没有将该候选标记可用。改用上表 bullerwins 的公开固定版本，已读取并验证全部 6 片。

另一个 Base 候选以 `.gguf.part1of5` 等字节切分文件发布；这与 GGUF 原生分片不同，本次没有将其当作原生分片接入，也没有用第一段冒充完整 GGUF。
