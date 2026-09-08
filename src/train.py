from __future__ import annotations

import io
import json
import os
import time
from dataclasses import asdict, dataclass
from urllib.parse import urlparse

import boto3
import joblib
import mlflow
import mlflow.sklearn
import ray
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class Candidate:
    index: int
    n_estimators: int
    max_depth: int | None
    random_state: int


@dataclass(frozen=True)
class CandidateResult:
    index: int
    n_estimators: int
    max_depth: int | None
    random_state: int
    accuracy: float
    f1_macro: float
    duration_seconds: float


def init_ray() -> None:
    try:
        ray.init(address="auto", ignore_reinit_error=True)
    except ConnectionError:
        ray.init(ignore_reinit_error=True)


def split_dataset() -> tuple:
    iris = load_iris()
    return train_test_split(
        iris.data,
        iris.target,
        test_size=0.25,
        random_state=42,
        stratify=iris.target,
    )


@ray.remote
def train_candidate(candidate: Candidate) -> CandidateResult:
    start = time.time()
    x_train, x_test, y_train, y_test = split_dataset()
    model = RandomForestClassifier(
        n_estimators=candidate.n_estimators,
        max_depth=candidate.max_depth,
        random_state=candidate.random_state,
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    return CandidateResult(
        index=candidate.index,
        n_estimators=candidate.n_estimators,
        max_depth=candidate.max_depth,
        random_state=candidate.random_state,
        accuracy=round(float(accuracy_score(y_test, predictions)), 6),
        f1_macro=round(float(f1_score(y_test, predictions, average="macro")), 6),
        duration_seconds=round(time.time() - start, 6),
    )


def build_candidates() -> list[Candidate]:
    raw = os.getenv("TRAIN_CANDIDATES", "20:2,50:3,100:4,120:none")
    candidates: list[Candidate] = []
    for index, item in enumerate(raw.split(",")):
        estimators_raw, depth_raw = item.split(":", 1)
        depth = None if depth_raw.lower() in {"none", "null", "0"} else int(depth_raw)
        candidates.append(
            Candidate(
                index=index,
                n_estimators=int(estimators_raw),
                max_depth=depth,
                random_state=100 + index,
            )
        )
    return candidates


def fit_final_model(best: CandidateResult) -> RandomForestClassifier:
    x_train, _, y_train, _ = split_dataset()
    model = RandomForestClassifier(
        n_estimators=best.n_estimators,
        max_depth=best.max_depth,
        random_state=best.random_state,
    )
    model.fit(x_train, y_train)
    return model


def s3_client():
    endpoint = os.getenv("MLFLOW_S3_ENDPOINT_URL") or os.getenv("S3_ENDPOINT_URL")
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_DEFAULT_REGION", "ap-northeast-2"),
    )


def ensure_bucket(client, bucket: str) -> None:
    buckets = client.list_buckets().get("Buckets", [])
    if any(item.get("Name") == bucket for item in buckets):
        return
    client.create_bucket(Bucket=bucket)


def upload_model_bundle(model: RandomForestClassifier, payload: dict) -> str:
    bucket = os.getenv("MODEL_BUCKET", "mlops-demo-models")
    run_id = os.getenv("RUN_ID", os.getenv("IMAGE_TAG", "manual"))
    prefix = os.getenv("MODEL_PREFIX", f"runs/{run_id}").strip("/")
    model_key = f"{prefix}/model.joblib"
    metadata_key = f"{prefix}/metadata.json"

    model_bytes = io.BytesIO()
    joblib.dump(model, model_bytes)
    model_bytes.seek(0)

    client = s3_client()
    ensure_bucket(client, bucket)
    client.put_object(Bucket=bucket, Key=model_key, Body=model_bytes.getvalue())
    client.put_object(
        Bucket=bucket,
        Key=metadata_key,
        Body=json.dumps(payload, sort_keys=True, indent=2).encode(),
        ContentType="application/json",
    )
    return f"s3://{bucket}/{model_key}"


def validate_s3_uri(uri: str) -> None:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path:
        raise ValueError(f"invalid s3 uri: {uri}")


def main() -> None:
    init_ray()
    candidates = build_candidates()
    refs = [train_candidate.remote(candidate) for candidate in candidates]
    results = ray.get(refs)
    best = max(results, key=lambda item: (item.accuracy, item.f1_macro, -item.index))
    model = fit_final_model(best)

    run_name = os.getenv("RUN_NAME_PREFIX", "ray-mlops-demo")
    image_tag = os.getenv("IMAGE_TAG", "unknown")
    run_id = os.getenv("RUN_ID", image_tag)
    payload = {
        "run_name": run_name,
        "image_tag": image_tag,
        "run_id": run_id,
        "mlflow_tracking_uri": os.getenv("MLFLOW_TRACKING_URI", ""),
        "s3_endpoint_url": os.getenv("S3_ENDPOINT_URL", ""),
        "candidates": [asdict(result) for result in results],
        "best_candidate": asdict(best),
    }

    model_s3_uri = upload_model_bundle(model, payload)
    validate_s3_uri(model_s3_uri)
    payload["model_s3_uri"] = model_s3_uri

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "ray-mlops-demo"))
    with mlflow.start_run(run_name=f"{run_name}-{image_tag}") as run:
        mlflow.log_params(
            {
                "image_tag": image_tag,
                "run_id": run_id,
                "ray_job": os.getenv("RAY_JOB_NAME", ""),
                "best_n_estimators": best.n_estimators,
                "best_max_depth": "none" if best.max_depth is None else best.max_depth,
                "model_s3_uri": model_s3_uri,
            }
        )
        mlflow.log_metrics(
            {
                "best_accuracy": best.accuracy,
                "best_f1_macro": best.f1_macro,
                "candidate_count": float(len(results)),
                "training_duration_total_seconds": round(
                    sum(result.duration_seconds for result in results), 6
                ),
            }
        )
        for result in results:
            mlflow.log_metric(f"candidate_{result.index}_accuracy", result.accuracy)
            mlflow.log_metric(f"candidate_{result.index}_f1_macro", result.f1_macro)
        mlflow.set_tags(
            {
                "demo.stage": "ray-train",
                "demo.model_s3_uri": model_s3_uri,
                "demo.image_tag": image_tag,
            }
        )
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=[[5.1, 3.5, 1.4, 0.2]],
        )
        mlflow.log_dict(payload, "training-summary.json")
        payload["mlflow_run_id"] = run.info.run_id
        payload["mlflow_model_artifact"] = f"runs:/{run.info.run_id}/model"

    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
