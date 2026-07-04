from importlib import import_module


def test_core_modules_import_without_loading_models():
    import_module("modules.main_pipeline")
    import_module("modules.llm_handler")
    import_module("modules.transcribe")
    import_module("modules.vision")
