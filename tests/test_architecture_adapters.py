from heterollm_sim.architecture_adapters import (
    list_gguf_architecture_adapters,
    resolve_gguf_architecture_adapter,
)


def test_llama_and_qwen2_share_the_decoder_adapter():
    llama = resolve_gguf_architecture_adapter("llama")
    qwen2 = resolve_gguf_architecture_adapter("QWEN2")
    qwen3 = resolve_gguf_architecture_adapter("qwen3")
    assert llama is not None
    assert qwen2 is llama
    assert qwen3 is llama
    assert llama.adapter_id == "llama_like_decoder"
    assert not llama.hybrid


def test_unknown_architecture_fails_closed():
    assert resolve_gguf_architecture_adapter("some_future_arch") is None
    assert {item.adapter_id for item in list_gguf_architecture_adapters()} == {
        "llama_like_decoder",
        "qwen35_hybrid_decoder",
    }
