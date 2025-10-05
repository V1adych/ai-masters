FROM apache/spark:3.5.0

USER root

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget \
    unzip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY process_movielens.py /app/process_movielens.py
COPY entrypoint.sh /app/entrypoint.sh

ENTRYPOINT ["bash", "/app/entrypoint.sh"]
