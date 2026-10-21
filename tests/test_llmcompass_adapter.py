from pathlib import Path


def test_llmcompass_submodule_and_dependency_are_pinned():
    root = Path(__file__).resolve().parents[1]
    modules = (root / ".gitmodules").read_text(encoding="utf-8")
    requirements = (root / "requirements-llmcompass.txt").read_text(
        encoding="utf-8")
    assert 'path = external/LLMCompass' in modules
    assert 'branch = ISCA_AE' in modules
    assert requirements.strip() == "scalesim==2.0.2"
