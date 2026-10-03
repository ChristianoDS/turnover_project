# Imagem para rodar o app Streamlit do Score de Rotatividade.
# Observação: o deploy em produção é feito pelo Streamlit Community Cloud.
# Esta imagem serve para execução local e portabilidade.
FROM python:3.12-slim

# uv: gerenciador de pacotes/projeto.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Instala apenas as dependências primeiro (camada cacheável).
# O README e o pacote são exigidos pelos metadados do projeto (readme + hatch),
# por isso o README entra já aqui; --no-install-project evita instalar o projeto.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project --no-dev

# Copia o restante do código e instala o projeto.
COPY . .
RUN uv sync --frozen --no-dev

EXPOSE 8501

# Healthcheck padrão do Streamlit.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["uv", "run", "--no-dev", "streamlit", "run", "app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
