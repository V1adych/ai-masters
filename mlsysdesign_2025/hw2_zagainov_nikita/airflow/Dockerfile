FROM apache/spark:3.5.0 AS sparkdist
FROM apache/airflow:2.9.2-python3.11

USER root
COPY --from=sparkdist /opt/spark /opt/spark
ENV SPARK_HOME=/opt/spark
ENV PATH=$PATH:/opt/spark/bin

RUN apt-get update && apt-get install -y --no-install-recommends openjdk-17-jre-headless ca-certificates procps && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /requirements.txt
USER airflow

ARG AIRFLOW_VERSION=2.9.2
ARG PYTHON_VERSION=3.11
ENV AIRFLOW_CONSTRAINTS_URL=https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt
RUN pip install --no-cache-dir -r /requirements.txt --constraint ${AIRFLOW_CONSTRAINTS_URL}
