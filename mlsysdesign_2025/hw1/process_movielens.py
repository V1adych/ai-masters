#!/usr/bin/env python3

from pyspark.sql import SparkSession, functions as F, types as T
import sys


def main():
    spark = SparkSession.builder.appName("MovieLens_Top_Genres").getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    print("Loading MovieLens data...")

    ratings_schema = T.StructType(
        [
            T.StructField("userId", T.IntegerType(), False),
            T.StructField("movieId", T.IntegerType(), False),
            T.StructField("rating", T.FloatType(), False),
            T.StructField("timestamp", T.LongType(), False),
        ]
    )

    movies_schema = T.StructType(
        [
            T.StructField("movieId", T.IntegerType(), False),
            T.StructField("title", T.StringType(), False),
            T.StructField("genres", T.StringType(), False),
        ]
    )

    ratings = spark.read.csv("/data/ml-latest-small/ratings.csv", schema=ratings_schema, header=True)
    movies = spark.read.csv("/data/ml-latest-small/movies.csv", schema=movies_schema, header=True)

    result = (
        ratings.join(movies, on="movieId")
        .select(F.explode(F.split(F.col("genres"), "\\|")).alias("genre"))
        .filter(F.col("genre") != "(no genres listed)")  # Filter out invalid genres
        .groupBy("genre")
        .count()
        .orderBy("count", ascending=False)
        .limit(10)
    )

    temp_output = "/workspace/temp_output"
    result.coalesce(1).write.mode("overwrite").option("header", "true").csv(temp_output)

    spark.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
