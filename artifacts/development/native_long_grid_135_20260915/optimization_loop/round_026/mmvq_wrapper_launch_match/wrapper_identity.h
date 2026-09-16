#pragma once
namespace locked_identity {
struct Module {const char *name;const char *path;const char *sha256;bool required;};
inline constexpr Module modules[]={
{"ggml-base.dll","F:\\codex_project\\37_LLMsim\\heterollm-sim-4.0.0\\source\\llama.cpp-native-thread-control\\build-native-thread-control\\bin\\ggml-base.dll","ebb357e640e217aef6a33e646b5d9a8e715635f72e478d85f2aa112b5362e1ca",true},
{"ggml-cpu.dll","F:\\codex_project\\37_LLMsim\\heterollm-sim-4.0.0\\source\\llama.cpp-native-thread-control\\build-native-thread-control\\bin\\ggml-cpu.dll","6ead72f75befab54b1afdff993c8bbd780b83cb307f21f6a75ad21cdb33c4232",false},
{"ggml.dll","F:\\codex_project\\37_LLMsim\\heterollm-sim-4.0.0\\source\\llama.cpp-native-thread-control\\build-native-thread-control\\bin\\ggml.dll","ee18c7ee09d675bf2dedbff201ea9c0b841025f57d4860d49d792389181865fc",false},
{"cudart64_12.dll","E:\\cuda\\bin\\cudart64_12.dll","c2c9a9c22a9bcba90e261825968836787b331038047a26770cffb7a583c28344",true},
};
inline constexpr const char *cupti_sha256="f9281a2c73b0379a48b6afdab438665b176260cf3973fd7d59131bc989142337";
}
