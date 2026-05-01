from __future__ import annotations

from merging.runtime.logging import banner, section


def test_runtime_logging_helpers_emit_readable_headers(capsys) -> None:
    banner("Merge Run")
    section("Evaluation")

    output = capsys.readouterr().out
    assert "Merge Run" in output
    assert "Evaluation" in output
    assert "=" * 60 in output
