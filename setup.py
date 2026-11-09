from setuptools import setup, find_packages
from pathlib import Path

setup(
    name="riskcanvas",
    version="0.1.0",
    author="Rohan Khiani",
    # Project contact — same address SECURITY.md publishes for reports.
    author_email="rskv.khiani@gmail.com",
    description=(
        "RiskCanvas is a Python library, REST API and demo dashboard for portfolio risk (VaR, "
        "expected shortfall, Monte Carlo and crisis stress tests), built so you can check its "
        "numbers by hand."
    ),
    long_description=(Path(__file__).parent / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/ROROK20/RiskCanvas",
    packages=find_packages(exclude=["tests*"]),
    # 3.11+: the oldest Python CI runs on. CI installs the newest versions
    # these floors admit, not the floors themselves (no job tests minimum
    # versions). The exact pins in requirements.txt need 3.12+ (numpy 2.5.1 /
    # scipy 1.18.0 ship no 3.11 wheels).
    python_requires=">=3.11",
    # Compatible-version FLOORS, not pins — a library that hard-pins is
    # uninstallable next to anything else. The exact, reproducible version set
    # lives in `requirements.txt`, which doubles as a constraints file:
    #   pip install -e ".[dev]" -c requirements.txt
    install_requires=[
        # pandas 2.2 introduced the "ME" month-end alias performance.py uses.
        "yfinance>=0.2.40", "pandas>=2.2.0", "numpy>=1.24.0",
        "scipy>=1.11.0", "fastapi>=0.104.0",
        # starlette (fastapi's ASGI core) floored at 1.3.1 to exclude
        # PYSEC-2026-248 / PYSEC-2026-249; see SECURITY.md.
        "starlette>=1.3.1", "uvicorn>=0.24.0",
        # pydantic 2.9: every fastapi that accepts starlette>=1.3.1 needs it,
        # and the API's FiniteFloat validation relies on it.
        "pydantic>=2.9.0",
    ],
    # httpx: required by fastapi/starlette TestClient (tests/test_api.py)
    extras_require={"dev": ["pytest>=7.4.0", "pytest-cov>=4.1.0", "httpx>=0.24.0"]},
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
