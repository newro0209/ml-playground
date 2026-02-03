"""smollm2_gguf 테스트."""

from __future__ import annotations

from pathlib import Path

from ml_playground import smollm2_gguf


def test_resolve_llama_cpp_dir_from_env(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LLAMA_CPP_DIR", tmp_path.as_posix())
    resolved = smollm2_gguf.resolve_llama_cpp_dir("")
    assert resolved == tmp_path


def test_build_convert_command(tmp_path: Path) -> None:
    script_path = tmp_path / "convert-hf-to-gguf.py"
    checkpoint_dir = tmp_path / "ckpt"
    output_path = tmp_path / "model.gguf"
    command = smollm2_gguf.build_convert_command(
        script_path=script_path,
        checkpoint_dir=checkpoint_dir,
        output_path=output_path,
        outtype="q8_0",
    )
    joined = " ".join(command)
    assert "convert-hf-to-gguf.py" in joined
    assert "--outtype q8_0" in joined
    assert checkpoint_dir.as_posix() in joined
