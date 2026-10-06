"""HeteroLLM system-level inference simulator."""

__version__ = "4.0.0"


def __getattr__(name):
    if name in {
        "AccessRequest", "BatchResult", "DramConfig", "MemoryKind", "NandConfig",
        "Operation", "TransactionResult", "make_ddr_config", "make_lpddr_config",
        "make_hbm_config", "make_gddr_config", "make_ssd_config", "make_hbf_config",
        "parse_physical_memory_config",
    }:
        from .memory_types import (
            AccessRequest, BatchResult, DramConfig, MemoryKind, NandConfig,
            Operation, TransactionResult, make_ddr_config, make_lpddr_config,
            make_hbm_config, make_gddr_config, make_ssd_config, make_hbf_config,
            parse_physical_memory_config,
        )
        return locals()[name]
    if name in {"DramCore", "dram_service"}:
        from .dram_core import DramCore, dram_service
        return {"DramCore": DramCore, "dram_service": dram_service}[name]
    if name in {"NandCore", "nand_service"}:
        from .nand_core import NandCore, nand_service
        return {"NandCore": NandCore, "nand_service": nand_service}[name]
    if name in {
        "AUTHORING_SCHEMA_VERSION",
        "ControllerProfile",
    }:
        from . import schema_v4

        return getattr(schema_v4, name)
    if name in {
        "PlacementDecision",
        "PlacementPolicy",
        "plan_runtime_placement",
    }:
        from . import control_plane_planner

        return getattr(control_plane_planner, name)
    if name in {"ControlPlaneBootstrap", "bootstrap_control_plane"}:
        from . import control_plane

        return getattr(control_plane, name)
    if name in {"ControlPlaneRuntime", "RuntimeRunResult"}:
        from . import runtime

        return getattr(runtime, name)
    if name in {"LlamaCppAdapter", "LlamaCppRuntimeConfig", "RuntimeExecutionPlan", "VLLMAdapter"}:
        from .runtime_adapters import LlamaCppAdapter, LlamaCppRuntimeConfig, RuntimeExecutionPlan, VLLMAdapter

        return {"LlamaCppAdapter": LlamaCppAdapter, "LlamaCppRuntimeConfig": LlamaCppRuntimeConfig,
                "RuntimeExecutionPlan": RuntimeExecutionPlan,
                "VLLMAdapter": VLLMAdapter}[name]
    if name == "apply_llama_runtime_config":
        from .llama_scenario import apply_llama_runtime_config
        return apply_llama_runtime_config
    if name == "diff_scheduler_traces":
        from .llama_trace_diff import diff_scheduler_traces
        return diff_scheduler_traces
    if name in {"NativeCalibrationProfile", "load_native_calibration", "apply_native_calibration"}:
        from .calibration import NativeCalibrationProfile, load_native_calibration, apply_native_calibration
        return {"NativeCalibrationProfile": NativeCalibrationProfile, "load_native_calibration": load_native_calibration, "apply_native_calibration": apply_native_calibration}[name]
    if name in {"GGUFError", "GGUFTensor", "GGUFMetadata", "read_gguf_metadata", "compare_gguf_to_model", "assert_gguf_parity", "build_model_from_gguf", "build_gguf_model_preset", "import_gguf_model_preset", "GGUFArchitectureAdapter", "resolve_gguf_architecture_adapter", "list_gguf_architecture_adapters"}:
        from .gguf_parity import (GGUFError, GGUFTensor, GGUFMetadata,
                                  read_gguf_metadata, compare_gguf_to_model,
                                  assert_gguf_parity, build_model_from_gguf,
                                  build_gguf_model_preset, import_gguf_model_preset)
        from .architecture_adapters import (GGUFArchitectureAdapter,
                                             resolve_gguf_architecture_adapter,
                                             list_gguf_architecture_adapters)
        return {"GGUFError": GGUFError, "GGUFTensor": GGUFTensor,
                "GGUFMetadata": GGUFMetadata, "read_gguf_metadata": read_gguf_metadata,
                "compare_gguf_to_model": compare_gguf_to_model,
                "assert_gguf_parity": assert_gguf_parity,
                "build_model_from_gguf": build_model_from_gguf,
                "build_gguf_model_preset": build_gguf_model_preset,
                "import_gguf_model_preset": import_gguf_model_preset,
                "GGUFArchitectureAdapter": GGUFArchitectureAdapter,
                "resolve_gguf_architecture_adapter": resolve_gguf_architecture_adapter,
                "list_gguf_architecture_adapters": list_gguf_architecture_adapters}[name]
    if name in {
        "MODEL_ARTIFACT_SCHEMA",
        "ModelArtifactError",
        "default_model_artifact_dir",
        "make_model_artifact",
        "save_model_artifact",
        "load_model_artifact",
        "list_model_artifacts",
    }:
        from .model_artifacts import (
            MODEL_ARTIFACT_SCHEMA,
            ModelArtifactError,
            default_model_artifact_dir,
            make_model_artifact,
            save_model_artifact,
            load_model_artifact,
            list_model_artifacts,
        )
        return {
            "MODEL_ARTIFACT_SCHEMA": MODEL_ARTIFACT_SCHEMA,
            "ModelArtifactError": ModelArtifactError,
            "default_model_artifact_dir": default_model_artifact_dir,
            "make_model_artifact": make_model_artifact,
            "save_model_artifact": save_model_artifact,
            "load_model_artifact": load_model_artifact,
            "list_model_artifacts": list_model_artifacts,
        }[name]
    if name in {
        "ControllerKind",
        "ControllerTransactionBatch",
        "HardwareTask",
        "InstructionBatch",
        "InstructionClass",
        "KernelCompletion",
        "RuntimeAction",
        "RuntimeDelta",
        "RuntimePhase",
    }:
        from . import runtime_ir

        return getattr(runtime_ir, name)
    raise AttributeError(name)

__all__ = [
    "AUTHORING_SCHEMA_VERSION",
    "ControllerProfile",
    "ControllerKind",
    "ControllerTransactionBatch",
    "ControlPlaneBootstrap",
    "ControlPlaneRuntime",
    "HardwareTask",
    "InstructionBatch",
    "InstructionClass",
    "KernelCompletion",
    "PlacementDecision",
    "PlacementPolicy",
    "RuntimeAction",
    "RuntimeDelta",
    "RuntimePhase",
    "RuntimeRunResult",
    "LlamaCppAdapter",
    "LlamaCppRuntimeConfig",
    "apply_llama_runtime_config",
    "diff_scheduler_traces",
    "NativeCalibrationProfile",
    "load_native_calibration",
    "apply_native_calibration",
    "GGUFError",
    "GGUFTensor",
    "GGUFMetadata",
    "read_gguf_metadata",
    "compare_gguf_to_model",
    "assert_gguf_parity",
    "build_model_from_gguf",
    "build_gguf_model_preset",
    "import_gguf_model_preset",
    "GGUFArchitectureAdapter",
    "resolve_gguf_architecture_adapter",
    "list_gguf_architecture_adapters",
    "MODEL_ARTIFACT_SCHEMA",
    "ModelArtifactError",
    "default_model_artifact_dir",
    "make_model_artifact",
    "save_model_artifact",
    "load_model_artifact",
    "list_model_artifacts",
    "RuntimeExecutionPlan",
    "VLLMAdapter",
    "AccessRequest",
    "BatchResult",
    "DramConfig",
    "MemoryKind",
    "NandConfig",
    "Operation",
    "TransactionResult",
    "make_ddr_config",
    "make_lpddr_config",
    "make_hbm_config",
    "make_gddr_config",
    "make_ssd_config",
    "make_hbf_config",
    "parse_physical_memory_config",
    "DramCore",
    "NandCore",
    "dram_service",
    "nand_service",
    "__version__",
    "bootstrap_control_plane",
    "plan_runtime_placement",
]
