"""docs/usage.md's Python snippets run, and its request schema is complete. Every python
block is executed in order in one namespace, offline: blocks that need the network
(yfinance) or a file the reader supplies are skipped."""

import re
from pathlib import Path

USAGE = Path(__file__).resolve().parents[1] / "docs" / "usage.md"
NEEDS_OUTSIDE = ("from_tickers(", "estimate_beta_detailed(", '"returns.csv"')


def test_usage_snippets_run_offline(capsys):
    blocks = re.findall(r"```python\n(.*?)```", USAGE.read_text(), re.S)
    runnable = [b for b in blocks if not any(m in b for m in NEEDS_OUTSIDE)]
    assert len(runnable) >= 7, len(runnable)
    namespace = {}
    for i, code in enumerate(runnable):
        try:
            exec(compile(code, f"usage.md block {i}", "exec"), namespace)
        except Exception as exc:  # name the failing snippet
            raise AssertionError(f"docs/usage.md python block {i} failed: {exc!r}\n{code}") from exc
    assert "betas" in namespace and list(namespace["betas"].index) == ["alpha", "MKT", "SMB"]


def test_the_schema_lists_every_request_field_and_query_parameter():
    from riskcanvas.api.app import PortfolioRequest, app

    schema = USAGE.read_text().split("Request body schema", 1)[1].split("## 7.", 1)[0]
    fields = set(PortfolioRequest.model_fields)
    for path in app.openapi()["paths"].values():
        for operation in path.values():
            fields |= {p["name"] for p in operation.get("parameters", [])}
    assert {"weights", "allow_large_returns", "portfolio_value", "seed"} <= fields
    missing = sorted(f for f in fields if f"`{f}`" not in schema)
    assert not missing, missing
