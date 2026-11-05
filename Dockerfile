# RiskCanvas API — FastAPI + numpy/scipy engine. Serves on :8004.
# 3.12: the pinned numpy 2.5.1 / scipy 1.18.0 publish no Python 3.11 wheels.
FROM python:3.12-slim

WORKDIR /app

# System deps kept minimal; numpy/scipy ship manylinux wheels.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Package source + metadata (setup.py reads README.md as long_description).
COPY setup.py README.md ./
COPY riskcanvas ./riskcanvas
RUN pip install --no-cache-dir -e .

# Run as an unprivileged user (python:slim has no non-root user by default).
# Create it, hand it the workdir, then drop privileges before the CMD so a
# compromise of the process cannot write outside /app or run as root.
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8004

# 0.0.0.0 so the container is reachable from the host / other services.
CMD ["uvicorn", "riskcanvas.api.app:app", "--host", "0.0.0.0", "--port", "8004"]
