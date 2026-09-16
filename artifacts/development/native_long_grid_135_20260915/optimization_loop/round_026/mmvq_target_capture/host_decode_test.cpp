// Standalone host tests: CUDA/CUPTI headers only; no CUDA, CUPTI, GGML DLL imports.
#define NOMINMAX
#include <algorithm>
#include "launch_decode.h"
#include <functional>
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace mmvq_capture;
struct Fixture {
    ConversionArgs c{};
    MainArgs m{};
    TensorPointers tensors{0x11000, 0x22000, 0x33000};
    Fixture() {
        c.x = tensors.input; c.vy = 0x44000;
        c.ne00 = c.s01 = c.s02 = c.s03 = c.ne0 = kK; c.ne1 = kM; c.ne2 = {1, 0, 1};
        m.vx = tensors.weights; m.vy = c.vy; m.dst = tensors.output; m.ncols_x = kK;
        m.stride_row_x = m.stride_col_y = m.stride_channel_y = m.stride_sample_y = 128;
        m.stride_col_dst = m.stride_channel_dst = m.stride_sample_dst = kN;
        m.stride_channel_x = m.stride_sample_x = 393216;
        m.channel_ratio = m.sample_ratio = {1, 0, 1};
    }
    std::array<void *, 9> conversion_arguments() {
        return {&c.x, &c.vy, &c.ne00, &c.s01, &c.s02, &c.s03, &c.ne0, &c.ne1, &c.ne2};
    }
    std::array<void *, 19> main_arguments() {
        return {&m.vx, &m.vy, &m.ids, &m.fusion, &m.dst, &m.ncols_x, &m.nchannels_y,
                &m.stride_row_x, &m.stride_col_y, &m.stride_col_dst, &m.channel_ratio,
                &m.stride_channel_x, &m.stride_channel_y, &m.stride_channel_dst,
                &m.sample_ratio, &m.stride_sample_x, &m.stride_sample_y,
                &m.stride_sample_dst, &m.ids_stride};
    }
    void emit(Recorder & rec, bool conversion, const char * custom_symbol = nullptr,
              bool null_args = false, bool with_exit = true) {
        auto ca = conversion_arguments(); auto ma = main_arguments();
        cudaLaunchKernel_v7000_params launch{};
        launch.func = reinterpret_cast<const void *>(uintptr_t(conversion ? 0xaa00 : 0xbb00));
        launch.args = null_args ? nullptr : (conversion ? ca.data() : ma.data());
        launch.gridDim = conversion ? dim3(16,1,1) : dim3(kN,1,1);
        launch.blockDim = conversion ? dim3(256,1,1) : dim3(32,4,1);
        launch.stream = reinterpret_cast<cudaStream_t>(uintptr_t(0x77));
        CUpti_CallbackData info{};
        info.callbackSite = CUPTI_API_ENTER; info.functionParams = &launch;
        info.functionName = "cudaLaunchKernel";
        info.symbolName = custom_symbol ? custom_symbol : (conversion ? kConversionSymbol : kMainSymbol);
        info.correlationId = conversion ? 41 : 42; info.contextUid = 9;
        callback(&rec, CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000, &info);
        if (with_exit) {
            int status = 0;
            info.callbackSite = CUPTI_API_EXIT; info.functionReturnValue = &status;
            callback(&rec, CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000, &info);
        }
    }
    void pair(Recorder & rec) { rec.reset(); emit(rec, true); emit(rec, false); }
};
static void require(bool result, const char * message) { if (!result) throw std::runtime_error(message); }
static bool accepted(const Recorder & rec, const Fixture & f) { std::string reason; return validate(rec, f.tensors, reason); }
int main() {
    try {
        static_assert(GGML_TYPE_Q5_0 == 6);
        std::vector<std::string> tests;
        auto passed = [&](const char * name) { tests.emplace_back(name); };
        Fixture f; Recorder rec; f.pair(rec);
        require(accepted(rec, f), "valid synthetic callback pair rejected"); passed("valid_pair_and_all_pointer_links");
        const auto baseline_c = rec.records[0]; const auto baseline_m = rec.records[1];
        f.c.ne00 = 999; f.m.ncols_x = 999;
        require(rec.records[0].conversion.ne00 == kK && rec.records[1].main.ncols_x == kK, "copied payload retained caller pointer");
        passed("payload_copied_before_callback_return");
        f = Fixture{};
        // Distinct sentinel values test every source-ordered field and scalar width.
        f.c.x = 101; f.c.vy = 102; f.c.ne00 = (int64_t(1)<<40)+3;
        f.c.s01 = (int64_t(1)<<40)+4; f.c.s02 = (int64_t(1)<<40)+5; f.c.s03 = (int64_t(1)<<40)+6;
        f.c.ne0 = (int64_t(1)<<40)+7; f.c.ne1 = 0x80000008u; f.c.ne2 = {9,10,11};
        auto ca = f.conversion_arguments(); ConversionArgs c{};
        require(decode_conversion(ca.data(),c) && c.x==101 && c.vy==102 && c.ne00==(int64_t(1)<<40)+3 &&
                c.s01==(int64_t(1)<<40)+4 && c.s02==(int64_t(1)<<40)+5 && c.s03==(int64_t(1)<<40)+6 &&
                c.ne0==(int64_t(1)<<40)+7 && c.ne1==0x80000008u && eq(c.ne2,9,10,11), "9-argument conversion order/width");
        passed("conversion_9_fields_distinct_and_64bit");
        f.m.vx=201; f.m.vy=202; f.m.ids=203;
        f.m.fusion.x_bias=reinterpret_cast<void *>(uintptr_t(204)); f.m.fusion.gate=reinterpret_cast<void *>(uintptr_t(205));
        f.m.fusion.gate_bias=reinterpret_cast<void *>(uintptr_t(206)); f.m.fusion.x_scale=reinterpret_cast<void *>(uintptr_t(207));
        f.m.fusion.gate_scale=reinterpret_cast<void *>(uintptr_t(208)); f.m.fusion.glu_op=GGML_GLU_OP_SWIGLU; f.m.fusion.glu_limit=3.5f;
        f.m.dst=209; f.m.ncols_x=210; f.m.nchannels_y={211,212,213};
        f.m.stride_row_x=214; f.m.stride_col_y=215; f.m.stride_col_dst=216; f.m.channel_ratio={217,218,219};
        f.m.stride_channel_x=220; f.m.stride_channel_y=221; f.m.stride_channel_dst=222;
        f.m.sample_ratio={223,224,225}; f.m.stride_sample_x=226; f.m.stride_sample_y=227; f.m.stride_sample_dst=228; f.m.ids_stride=0xf00000e5u;
        auto ma=f.main_arguments(); MainArgs m{};
        require(decode_main(ma.data(),m) && m.vx==201 && m.vy==202 && m.ids==203 &&
                uintptr_t(m.fusion.x_bias)==204 && uintptr_t(m.fusion.gate)==205 && uintptr_t(m.fusion.gate_bias)==206 &&
                uintptr_t(m.fusion.x_scale)==207 && uintptr_t(m.fusion.gate_scale)==208 &&
                m.fusion.glu_op==GGML_GLU_OP_SWIGLU && m.fusion.glu_limit==3.5f && m.dst==209 && m.ncols_x==210 &&
                eq(m.nchannels_y,211,212,213) && m.stride_row_x==214 && m.stride_col_y==215 && m.stride_col_dst==216 &&
                eq(m.channel_ratio,217,218,219) && m.stride_channel_x==220 && m.stride_channel_y==221 && m.stride_channel_dst==222 &&
                eq(m.sample_ratio,223,224,225) && m.stride_sample_x==226 && m.stride_sample_y==227 && m.stride_sample_dst==228 &&
                m.ids_stride==0xf00000e5u, "19-argument main order/width/fusion");
        passed("main_19_fields_distinct_and_fusion_layout");
        auto bad_record = [&](const char * name, std::function<void(Recorder &)> change) {
            Fixture valid; valid.pair(rec); change(rec); require(!accepted(rec,valid),name); passed(name);
        };
        bad_record("reject_wrong_conversion_shape",[](auto&r){r.records[0].conversion.ne00++;});
        bad_record("reject_wrong_conversion_ne2_fastdiv",[](auto&r){r.records[0].conversion.ne2.x=0;});
        bad_record("reject_wrong_main_stride",[](auto&r){r.records[1].main.stride_sample_x++;});
        bad_record("reject_wrong_main_channel_fastdiv",[](auto&r){r.records[1].main.channel_ratio.z=2;});
        bad_record("reject_wrong_main_geometry",[](auto&r){r.records[1].grid.x=1;});
        bad_record("reject_dynamic_shared_change",[](auto&r){r.records[1].shared=1;});
        bad_record("reject_conversion_geometry_change",[](auto&r){r.records[0].block.x=128;});
        bad_record("reject_ids_path",[](auto&r){r.records[1].main.ids=1;});
        bad_record("reject_fusion_path",[](auto&r){r.records[1].main.fusion.gate=reinterpret_cast<void*>(uintptr_t(1));});
        bad_record("reject_producer_consumer_pointer_mismatch",[](auto&r){r.records[1].main.vy++;});
        bad_record("reject_graph_input_mismatch",[](auto&r){r.records[0].conversion.x++;});
        bad_record("reject_graph_weight_mismatch",[](auto&r){r.records[1].main.vx++;});
        bad_record("reject_graph_output_mismatch",[](auto&r){r.records[1].main.dst++;});
        bad_record("reject_pointer_aliasing",[](auto&r){r.records[0].conversion.vy=r.records[0].conversion.x;r.records[1].main.vy=r.records[0].conversion.x;});
        bad_record("reject_different_stream",[](auto&r){r.records[1].stream++;});
        bad_record("reject_different_context",[](auto&r){r.records[1].context++;});
        bad_record("reject_duplicate_correlation",[](auto&r){r.records[1].correlation=r.records[0].correlation;});
        bad_record("reject_missing_return",[](auto&r){r.records[1].exit_seen=false;});
        bad_record("reject_failed_return",[](auto&r){r.records[1].return_code=1;});
        bad_record("reject_missing_launch",[](auto&r){r.count=1;});
        bad_record("reject_extra_launch",[](auto&r){r.count=3;});
        bad_record("reject_overflow",[](auto&r){r.overflow=true;});
        f=Fixture{};rec.reset();f.emit(rec,false);f.emit(rec,true);
        require(!accepted(rec,f),"reverse launch order accepted");passed("reject_reversed_order");
        for (const char * symbol : {"_Z17quantize_mmq_q8_1IL18mmq_q8_1_ds_layout0ELb0EEvPKfPKiPvxxxxxiii", "unknown_mmvq", ""}) {
            rec.reset(); f.emit(rec,true,symbol,true); f.emit(rec,false);
            require(!rec.records[0].decoded && !accepted(rec,f),"unknown/missing symbol decoded");
        }
        passed("reject_legacy_mmq_unknown_and_missing_symbol_without_dereference");
        rec.reset(); f.emit(rec,true,nullptr,true); f.emit(rec,false);
        require(!accepted(rec,f) && rec.records[0].malformed_arguments,"null argument array accepted");
        passed("reject_null_argument_array");
        ca=f.conversion_arguments();ca[8]=nullptr;ma=f.main_arguments();ma[18]=nullptr;
        require(!decode_conversion(ca.data(),c) && !decode_main(ma.data(),m),"null tail arguments accepted");passed("reject_null_parameter_slots");
        rec.reset();CUpti_CallbackData info{};info.callbackSite=CUPTI_API_ENTER;info.functionName="cudaLaunchKernelExC";
        info.functionParams=reinterpret_cast<void*>(uintptr_t(1));info.symbolName=kMainSymbol;
        callback(&rec,CUPTI_CB_DOMAIN_RUNTIME_API,0,&info);
        require(rec.count==1 && !rec.records[0].supported_api && !rec.records[0].decoded,"unknown API reinterpreted");
        passed("reject_unsupported_launch_api_without_cast");
        rec.reset();info.functionName="cudaGetLastError";callback(&rec,CUPTI_CB_DOMAIN_RUNTIME_API,0,&info);
        require(rec.count==0 && !rec.malformed_callback,"nonlaunch callback counted");passed("ignore_nonlaunch_callback");
        auto runtime_pair = [&](const char * name) {
            CUpti_CallbackData aux{};aux.functionName=name;aux.correlationId=501;aux.contextUid=9;
            aux.callbackSite=CUPTI_API_ENTER;aux.functionParams=reinterpret_cast<void*>(uintptr_t(1));
            callback(&rec,CUPTI_CB_DOMAIN_RUNTIME_API,777,&aux);
            int status=0;aux.callbackSite=CUPTI_API_EXIT;aux.functionReturnValue=&status;
            callback(&rec,CUPTI_CB_DOMAIN_RUNTIME_API,777,&aux);status=123;
        };
        for(const char *name:{"cudaMemcpyAsync", "cudaMemcpy2DAsync", "cudaMemset", "cudaMemsetAsync"}) {
            f.pair(rec);runtime_pair(name);
            require(rec.memory_api_count==1 && rec.runtime_count==1 && rec.runtime_calls[0].exit_seen &&
                    rec.runtime_calls[0].return_code==0 && !accepted(rec,f), "memory API missing or accepted");
        }
        passed("record_and_reject_graph_copy_and_memset");
        for(const char *name:{"cudaMalloc", "cudaFreeAsync", "cudaStreamSynchronize", "cudaStreamSynchronize_ptsz"}) {
            f.pair(rec);runtime_pair(name);
            require(rec.memory_api_count==0 && rec.runtime_count==1 && accepted(rec,f), "allocation/sync misclassified");
        }
        passed("allocation_and_sync_classified_separately");
        require(rec.runtime_calls[0].return_code==0, "return code lifetime not copied");
        passed("runtime_return_value_deep_copied");
        rec.runtime_calls[0].return_code=7;require(!accepted(rec,f),"failed runtime API accepted");
        passed("reject_failed_runtime_auxiliary");
        rec.reset();f.emit(rec,true,nullptr,false,false);f.emit(rec,false);
        require(!accepted(rec,f),"entry only launch accepted");passed("reject_entry_without_exit");
        rec.reset();info={};info.callbackSite=CUPTI_API_EXIT;info.functionName="cudaLaunchKernel";
        int status=0;info.functionReturnValue=&status;callback(&rec,CUPTI_CB_DOMAIN_RUNTIME_API,CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000,&info);
        require(rec.malformed_callback,"orphan exit accepted");passed("reject_orphan_exit");
        std::cout << "{\"schema\":\"heterollm.mmvq-target-host-tests/v1\",\"status\":\"passed\",\"test_count\":" << tests.size()
                  << ",\"GPU_context_created\":false,\"CUPTI_subscription_created\":false,\"CUDA_API_called\":false,\"GPU_CUPTI_GGML_DLL_imports\":0,\"explicit_DLL_loads\":0,"
                  << "\"actual_GPU_callback_compatibility_validated\":false,\"tests\":[";
        for (size_t i=0;i<tests.size();++i) {if(i)std::cout<<',';std::cout<<'"'<<tests[i]<<'"';}
        std::cout << "]}\n";return 0;
    } catch(const std::exception & e) {std::cerr<<e.what()<<'\n';return 1;}
}
