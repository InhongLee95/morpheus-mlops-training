from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse
import io
import json
import os
import threading

import boto3
import joblib


MODEL = None
MODEL_LOCK = threading.Lock()


def json_response(handler: BaseHTTPRequestHandler, status: int, body: dict) -> None:
    data = json.dumps(body, sort_keys=True).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def s3_client():
    endpoint = os.getenv("MLFLOW_S3_ENDPOINT_URL") or os.getenv("S3_ENDPOINT_URL")
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_DEFAULT_REGION", "ap-northeast-2"),
    )


def parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path:
        raise ValueError(f"MODEL_S3_URI must be s3://bucket/key, got {uri!r}")
    return parsed.netloc, parsed.path.lstrip("/")


def load_model():
    global MODEL
    if MODEL is not None:
        return MODEL
    with MODEL_LOCK:
        if MODEL is not None:
            return MODEL
        model_uri = os.environ["MODEL_S3_URI"]
        bucket, key = parse_s3_uri(model_uri)
        obj = s3_client().get_object(Bucket=bucket, Key=key)
        MODEL = joblib.load(io.BytesIO(obj["Body"].read()))
        return MODEL


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        print(fmt % args)

    def do_GET(self) -> None:
        if self.path in {"/healthz", "/v1/models/mlops-demo"}:
            try:
                load_model()
                json_response(
                    self,
                    200,
                    {
                        "name": "mlops-demo",
                        "ready": True,
                        "model_s3_uri": os.getenv("MODEL_S3_URI", ""),
                        "image_tag": os.getenv("IMAGE_TAG", "unknown"),
                    },
                )
            except Exception as exc:
                json_response(self, 503, {"ready": False, "error": str(exc)})
            return

        json_response(
            self,
            200,
            {
                "app": "mlops-demo",
                "message": "KServe custom inference service for the Morpheus MLOps demo",
                "predict_path": "/v1/models/mlops-demo:predict",
            },
        )

    def do_POST(self) -> None:
        if self.path not in {
            "/v1/models/mlops-demo:predict",
            "/v2/models/mlops-demo/infer",
        }:
            json_response(self, 404, {"error": f"unsupported path {self.path}"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode())
            instances = body.get("instances")
            if instances is None and "inputs" in body:
                instances = body["inputs"][0].get("data")
            if not isinstance(instances, list) or not instances:
                raise ValueError("request body must include non-empty instances")

            model = load_model()
            predictions = model.predict(instances).tolist()
            probabilities = (
                model.predict_proba(instances).round(6).tolist()
                if hasattr(model, "predict_proba")
                else []
            )
            json_response(
                self,
                200,
                {
                    "predictions": predictions,
                    "probabilities": probabilities,
                    "model_s3_uri": os.getenv("MODEL_S3_URI", ""),
                    "image_tag": os.getenv("IMAGE_TAG", "unknown"),
                },
            )
        except Exception as exc:
            json_response(self, 400, {"error": str(exc)})


if __name__ == "__main__":
    load_model()
    port = int(os.getenv("PORT", "8080"))
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
