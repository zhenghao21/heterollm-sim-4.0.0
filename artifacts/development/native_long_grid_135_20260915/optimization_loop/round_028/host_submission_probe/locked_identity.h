#pragma once
namespace locked {
struct Module{const char* name;const char* path;const char* sha;};
inline constexpr Module modules[]={
 {"ggml-base.dll","F:\\codex_project\\37_LLMsim\\heterollm-sim-4.0.0\\source\\llama.cpp-native-thread-control\\build-native-thread-control\\bin\\ggml-base.dll","ebb357e640e217aef6a33e646b5d9a8e715635f72e478d85f2aa112b5362e1ca"},
 {"ggml-cuda.dll","F:\\codex_project\\37_LLMsim\\heterollm-sim-4.0.0\\source\\llama.cpp-native-thread-control\\build-native-thread-control\\bin\\ggml-cuda.dll","8a7275a273c225639a94c6cd544d3a891760f4599bfbe5f6ab1b4d15f556a297"},
 {"cudart64_12.dll","E:\\cuda\\bin\\cudart64_12.dll","c2c9a9c22a9bcba90e261825968836787b331038047a26770cffb7a583c28344"},
};
inline constexpr const char* symbol="_Z12rms_norm_f32ILi1024ELb0ELb0EEvPKfPfixxxfS1_xxx5uint3S3_S3_S3_S1_xxxS3_S3_S3_S3_";
inline constexpr const char* uuid="GPU-83b80720-113c-3f3d-c624-f1dc642b3f8b";
inline constexpr const char* cupti_path="F:\\codex_project\\37_LLMsim\\tools\\nsys_cli\\target-windows-x64\\cupti64_134.dll";
inline constexpr const char* cupti_sha="f9281a2c73b0379a48b6afdab438665b176260cf3973fd7d59131bc989142337";
}
