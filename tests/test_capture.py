"""The README screenshot and the social-preview image come from ONE capture, each from the
state it describes.

`make capture` (e2e/capture.mjs) writes both images and docs/capture.json with their
SHA-256; this fails if either image changes without the other being re-captured with it.
The preview, frontend/public/og.png, is what every shared link shows: the state a
frontend-only deploy serves, "Demo data" and its figures. The driver (e2e/smoke.mjs)
checks the demo still shows the recorded figures."""

import hashlib
import json
import struct
from datetime import date
from pathlib import Path

from fastapi.testclient import TestClient

from riskcanvas.api.app import app

ROOT = Path(__file__).resolve().parents[1]


def _manifest():
    return json.loads((ROOT / "docs" / "capture.json").read_text())


def _sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def _png_size(path):
    return struct.unpack(">II", (ROOT / path).read_bytes()[16:24])


def test_both_images_are_the_ones_the_capture_recorded():
    m = _manifest()
    assert _sha(m["screenshot"]["path"]) == m["screenshot"]["sha256"], "docs/screenshot.png changed without `make capture`"
    assert _sha(m["og"]["path"]) == m["og"]["sha256"], "frontend/public/og.png changed without `make capture`"


def test_the_manifest_dates_the_capture_and_pairs_the_images_by_hash_alone():
    m = _manifest()
    assert "commit" not in m
    assert date.fromisoformat(m["captured"])
    assert len(m["screenshot"]["sha256"]) == len(m["og"]["sha256"]) == 64


def test_og_image_is_1200x630_at_2x_and_the_layout_points_at_it():
    assert _png_size("frontend/public/og.png") == (2400, 1260)
    layout = (ROOT / "frontend" / "app" / "layout.tsx").read_text()
    assert 'url: "/og.png"' in layout and "width: 1200" in layout and "height: 630" in layout


def test_the_preview_is_the_state_a_no_backend_deploy_serves():
    m = _manifest()
    assert m["og"]["badge"] == "Demo data"
    assert m["og"]["build"] == "production, no NEXT_PUBLIC_API_URL"
    assert m["screenshot"]["badge"] == "Live engine", "the README screenshot stays the live-engine capture"
    assert m["og"]["metric_cards"].startswith("VAR 95% (1D) ") and m["og"]["metric_cards"] != m["metric_cards"]


def test_the_metadata_describes_every_deploy():
    """The static description cannot know whether a build has an engine, so it may
    not claim one. Test what a visitor reads."""
    layout = (ROOT / "frontend" / "app" / "layout.tsx").read_text()
    assert "powered by a real Python engine" not in layout
    description = layout.split("const DESCRIPTION =", 1)[1].split(";", 1)[0]
    assert "synthetic inputs" in description and "demo" in description
    assert layout.count("description: DESCRIPTION") == 3
    assert "demo data" in layout.split("OG_IMAGES = SITE_URL", 1)[1].split("]", 1)[0]


def _pct(x):
    return f"{x * 100:.2f}%"


def _usd(x, portfolio_value=1_000_000):
    return f"${abs(x * portfolio_value):,.0f}"


def test_the_captured_cards_are_what_the_engine_computes_for_the_captured_request():
    """docs/capture_request.json is the body the dashboard sent; the page's
    formatting of the engine's answer must be the captured cards, so an engine
    change that moves a published number fails here (e2e/smoke.mjs checks the
    dashboard still sends that body)."""
    body = json.loads((ROOT / "docs" / "capture_request.json").read_text())
    out = TestClient(app).post("/portfolio/analyze", json=body)
    assert out.status_code == 200, out.text[:200]
    r, p = out.json()["risk"], out.json()["performance"]
    assert _manifest()["metric_cards"] == (
        f"VAR 95% (1D) {_pct(r['var_95_1d'])} {_usd(r['var_95_1d'])} "
        f"VAR 99% (1D) {_pct(r['var_99_1d'])} {_usd(r['var_99_1d'])} "
        f"CVAR 95% {_pct(r['cvar_95'])} {_usd(r['cvar_95'])} "
        f"MAX DD {_pct(r['max_drawdown'])} "
        f"SHARPE {p['sharpe']:.2f} SORTINO {p['sortino']:.2f} "
        f"CAGR {_pct(p['cagr'])} ANNUAL VOL {_pct(r['annual_vol'])} CALMAR {p['calmar']:.2f} "
        f"SKEWNESS {r['skewness']:.2f} EXCESS KURT. {r['excess_kurtosis']:.2f} "
        f"VAR 95% (10D) {_pct(r['var_95_10d'])}"
    )
