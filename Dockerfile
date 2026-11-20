FROM python:3.11-slim

# Run as a non-root user
RUN useradd -m -u 1000 user
USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PORT=7860

WORKDIR $HOME/app

# Dependencies first, to keep them in the Docker cache.
# Every dependency ships as a wheel: no compiler is needed.
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# The app only needs its code and the trained artifacts (see .dockerignore)
COPY --chown=user . .

EXPOSE $PORT

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://localhost:{os.environ[\"PORT\"]}/_stcore/health', timeout=4)"

CMD ["sh", "-c", "exec streamlit run app.py --server.port=${PORT} --server.address=0.0.0.0"]
