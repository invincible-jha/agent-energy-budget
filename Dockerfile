FROM python:3.12-slim

LABEL org.opencontainers.image.source="https://github.com/invincible-jha/agent-energy-budget"
LABEL org.opencontainers.image.description="agent-energy-budget: Agent cost control and energy budget SDK"
LABEL org.opencontainers.image.licenses="Apache-2.0"
LABEL org.opencontainers.image.vendor="AumOS"

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/
RUN pip install --no-cache-dir . && rm -rf /root/.cache

RUN useradd -m -s /bin/bash aumos
USER aumos

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["agent-energy-budget"]
CMD ["--help"]
