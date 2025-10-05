#!/bin/bash

set -e
mkdir -p output

docker build -t movielens-spark .
docker run --rm -v "$(pwd)/output:/workspace/output" movielens-spark

echo "top10genres.csv contents:"
cat $(pwd)/output/top10genres.csv
