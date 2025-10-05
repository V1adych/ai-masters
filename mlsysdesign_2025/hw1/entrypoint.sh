#!/bin/bash
set -e

echo "Downloading MovieLens dataset..."
cd /tmp
wget -q https://files.grouplens.org/datasets/movielens/ml-latest-small.zip
echo "Download complete"

echo "Extracting dataset..."
unzip -q ml-latest-small.zip

echo "Setting up data directory..."
mkdir -p /data
mv ml-latest-small /data/

echo "Creating output directory..."
mkdir -p /workspace/output
mkdir -p /workspace/temp_output

echo "Running PySpark processing..."
/opt/spark/bin/spark-submit --master local[*] /app/process_movielens.py

CSV_FILE=$(find /workspace/temp_output -name "part-*.csv" -type f | head -1)
mv "$CSV_FILE" /workspace/output/top10genres.csv
rm -rf /workspace/temp_output
echo "Result saved to /workspace/output/top10genres.csv"
