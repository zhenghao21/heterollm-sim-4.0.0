def test_join_tool_has_entrypoint():
    import tools.join_native_simulator as module
    assert callable(module.main)
