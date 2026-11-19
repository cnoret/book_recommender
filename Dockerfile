FROM python:3.11-slim

# scikit-surprise is compiled from source: it needs a C compiler.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Run as a non-root user
RUN useradd -m -u 1000 user
USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PORT=7860

WORKDIR $HOME/app

# Dependencies first, to keep them in the Docker cache
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# The app only needs its code and the trained artifacts (see .dockerignore)
COPY --chown=user . .

EXPOSE $PORT

CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
