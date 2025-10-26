FROM apache/spark:3.5.0

USER root

COPY requirements.txt /tmp/requirements.txt
RUN apt-get update && apt-get install -y python3-pip curl ca-certificates && \
    rm -rf /var/lib/apt/lists/* && \
    pip3 install --no-cache-dir -r /tmp/requirements.txt && \
    rm -f /tmp/requirements.txt

ENV SPARK_HOME=/opt/spark
RUN curl -fsSL -o ${SPARK_HOME}/jars/hadoop-aws-3.3.4.jar https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.3.4/hadoop-aws-3.3.4.jar && \
    curl -fsSL -o ${SPARK_HOME}/jars/aws-java-sdk-bundle-1.12.262.jar https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.12.262/aws-java-sdk-bundle-1.12.262.jar
