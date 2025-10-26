from datetime import datetime
import json
import os
import shutil
import tempfile

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator


MINIO_ENDPOINT = os.environ["MINIO_ENDPOINT"]
MINIO_ACCESS_KEY = os.environ["MINIO_ACCESS_KEY"]
MINIO_SECRET_KEY = os.environ["MINIO_SECRET_KEY"]
S3_ENDPOINT_URL = os.environ["S3_ENDPOINT_URL"]
SPARK_MASTER_URL = os.environ["SPARK_MASTER_URL"]

DATASET_URL = "https://files.grouplens.org/datasets/movielens/ml-latest.zip"
BUCKET_NAME = "movielens"
RAW_PREFIX = "raw/"
FEATURES_PREFIX = "features/"


def get_dataset_fn(**context):
    import requests
    import zipfile

    tmp_dir = tempfile.mkdtemp(prefix="movielens_")
    zip_path = os.path.join(tmp_dir, "ml.zip")

    with requests.get(DATASET_URL, stream=True) as r:
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(tmp_dir)

    context['ti'].xcom_push(key='raw_dir', value=tmp_dir)


def put_dataset_fn(**context):
    from minio import Minio

    raw_dir = context['ti'].xcom_pull(key='raw_dir', task_ids='get_dataset')
    client = Minio(MINIO_ENDPOINT.replace("http://", "").replace("https://", ""),
                   access_key=MINIO_ACCESS_KEY, secret_key=MINIO_SECRET_KEY, secure=MINIO_ENDPOINT.startswith("https"))

    found = client.bucket_exists(BUCKET_NAME)
    if not found:
        client.make_bucket(BUCKET_NAME)

    import glob
    base = None
    for path in glob.glob(os.path.join(raw_dir, "ml-latest*")):
        base = path
        break
    assert base is not None, "Extracted directory not found"

    for fname in ["ratings.csv", "movies.csv"]:
        fpath = os.path.join(base, fname)
        client.fput_object(BUCKET_NAME, RAW_PREFIX + fname, fpath)

    shutil.rmtree(raw_dir, ignore_errors=True)


def load_features_fn(**context):
    import tempfile
    import pyarrow.dataset as ds
    from minio import Minio
    import redis

    client = Minio(MINIO_ENDPOINT.replace("http://", "").replace("https://", ""),
                   access_key=MINIO_ACCESS_KEY, secret_key=MINIO_SECRET_KEY, secure=MINIO_ENDPOINT.startswith("https"))
    r = redis.Redis(host=os.environ['REDIS_HOST'], port=int(os.environ['REDIS_PORT']), decode_responses=True)

    tmp = tempfile.mkdtemp(prefix="features_")
    try:
        objects = client.list_objects(BUCKET_NAME, prefix=FEATURES_PREFIX, recursive=True)
        parquet_keys = [o.object_name for o in objects if o.object_name.endswith('.parquet') or '/part-' in o.object_name]
        if not parquet_keys:
            raise RuntimeError('No features parquet found in MinIO')
        for key in parquet_keys:
            data = client.get_object(BUCKET_NAME, key)
            with open(os.path.join(tmp, os.path.basename(key)), 'wb') as f:
                for d in data.stream(32 * 1024):
                    f.write(d)
            data.close()
            data.release_conn()
        dataset = ds.dataset(tmp, format="parquet")
        scanner = ds.Scanner.from_dataset(dataset, columns=[
            "user_id",
            "avg_rating",
            "num_movies",
            "genre_profile_json",
            "last_interaction_ts",
            "movie_ids_json",
        ])
        for batch in scanner.to_batches():
            cols = {name: batch.column(i) for i, name in enumerate(batch.schema.names)}
            for i in range(batch.num_rows):
                user_id = int(cols["user_id"][i].as_py())
                payload = {
                    "user_id": user_id,
                    "avg_rating": float(cols["avg_rating"][i].as_py()),
                    "num_movies": int(cols["num_movies"][i].as_py()),
                    "genre_profile": json.loads(cols["genre_profile_json"][i].as_py()),
                    "last_interaction_ts": int(cols["last_interaction_ts"][i].as_py()),
                    "movie_ids": json.loads(cols["movie_ids_json"][i].as_py()),
                }
                r.set(str(user_id), json.dumps(payload))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


with DAG(
    dag_id="movielens_features",
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
    is_paused_upon_creation=False,
) as dag:

    get_dataset = PythonOperator(
        task_id="get_dataset",
        python_callable=get_dataset_fn,
        provide_context=True,
    )

    put_dataset = PythonOperator(
        task_id="put_dataset",
        python_callable=put_dataset_fn,
        provide_context=True,
    )

    spark_cmd = (
        f"spark-submit "
        f"--master {SPARK_MASTER_URL} "
        f"--packages org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262 "
        f"--conf spark.hadoop.fs.s3a.endpoint={S3_ENDPOINT_URL} "
        f"--conf spark.hadoop.fs.s3a.access.key={MINIO_ACCESS_KEY} "
        f"--conf spark.hadoop.fs.s3a.secret.key={MINIO_SECRET_KEY} "
        f"--conf spark.hadoop.fs.s3a.path.style.access=true "
        f"--conf spark.hadoop.fs.s3a.impl=org.apache.hadoop.fs.s3a.S3AFileSystem "
        f"/opt/spark-apps/features_job.py "
        f"--s3-endpoint={S3_ENDPOINT_URL} "
        f"--bucket={BUCKET_NAME} "
        f"--raw-prefix={RAW_PREFIX} "
        f"--features-prefix={FEATURES_PREFIX} "
        f"--aws-access-key={MINIO_ACCESS_KEY} "
        f"--aws-secret-key={MINIO_SECRET_KEY}"
    )

    features_engineering = BashOperator(
        task_id="features_engineering",
        bash_command=spark_cmd,
    )

    load_features = PythonOperator(
        task_id="load_features",
        python_callable=load_features_fn,
        provide_context=True,
    )

    get_dataset >> put_dataset >> features_engineering >> load_features


