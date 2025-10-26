import argparse

from pyspark.sql import SparkSession, functions as F, types as T


def build_spark(app_name: str) -> SparkSession:
    return (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--s3-endpoint", required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--raw-prefix", required=True)
    parser.add_argument("--features-prefix", required=True)
    parser.add_argument("--aws-access-key", required=True)
    parser.add_argument("--aws-secret-key", required=True)
    args = parser.parse_args()

    spark = build_spark("features_job")

    # Configure S3A for MinIO access
    hconf = spark.sparkContext._jsc.hadoopConfiguration()
    hconf.set("fs.s3a.endpoint", args.s3_endpoint)
    hconf.set("fs.s3a.access.key", args.aws_access_key)
    hconf.set("fs.s3a.secret.key", args.aws_secret_key)
    hconf.set("fs.s3a.path.style.access", "true")
    hconf.set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")

    ratings_path = f"s3a://{args.bucket}/{args.raw_prefix}ratings.csv"
    movies_path = f"s3a://{args.bucket}/{args.raw_prefix}movies.csv"

    ratings_schema = T.StructType([
        T.StructField("userId", T.IntegerType(), False),
        T.StructField("movieId", T.IntegerType(), False),
        T.StructField("rating", T.FloatType(), False),
        T.StructField("timestamp", T.LongType(), False),
    ])

    movies_schema = T.StructType([
        T.StructField("movieId", T.IntegerType(), False),
        T.StructField("title", T.StringType(), False),
        T.StructField("genres", T.StringType(), False),
    ])

    ratings = spark.read.csv(ratings_path, header=True, schema=ratings_schema)
    movies = spark.read.csv(movies_path, header=True, schema=movies_schema)

    base = ratings.groupBy("userId").agg(
        F.avg("rating").alias("avg_rating"),
        F.countDistinct("movieId").alias("num_movies"),
        F.max("timestamp").alias("last_interaction_ts"),
        F.sort_array(F.collect_set("movieId")).alias("movie_ids"),
    )

    movie_genres = (
        movies
        .select(
            F.col("movieId"),
            F.explode(F.split(F.col("genres"), "\\|")).alias("genre")
        )
        .filter(F.col("genre") != "(no genres listed)")
    )
    
    rated = ratings.select("userId", "movieId").distinct()
    rated_with_genres = rated.join(movie_genres, "movieId", "inner")
    genre_counts = (
        rated_with_genres
        .where(F.col("genre").isNotNull())
        .groupBy("userId", "genre").count()
    )
    genre_total = genre_counts.groupBy("userId").agg(F.sum("count").alias("total"))
    genre_norm = genre_counts.join(genre_total, "userId").select(
        F.col("userId"),
        F.col("genre"),
        (F.col("count") / F.col("total")).alias("weight")
    )

    genre_json = (
        genre_norm
        .groupBy("userId")
        .agg(F.map_from_entries(F.collect_list(F.struct("genre", "weight"))).alias("genre_map"))
        .select(
            F.col("userId"),
            F.to_json("genre_map").alias("genre_profile_json")
        )
    )

    features = (
        base
        .join(genre_json, base.userId == genre_json.userId, "left")
        .drop(genre_json.userId)
        .withColumnRenamed("userId", "user_id")
        .withColumn("movie_ids_json", F.to_json("movie_ids"))
        .withColumn("genre_profile_json", F.coalesce(F.col("genre_profile_json"), F.lit("{}")))
    )

    out_path = f"s3a://{args.bucket}/{args.features_prefix}user_features.parquet"
    (
        features
        .select("user_id", "avg_rating", "num_movies", "last_interaction_ts", "genre_profile_json", "movie_ids_json")
        .write.mode("overwrite").parquet(out_path)
    )

    spark.stop()


if __name__ == "__main__":
    main()


