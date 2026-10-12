# morpheus-mlops-training

모델 학습 코드와 서빙 코드를 담은 레포입니다.

`src/train.py`는 Ray로 학습해 MLflow에 모델을 기록하고, `inference/main.py`는 저장된 모델을 불러와 예측 API를 제공합니다.

Dockerfile로 이미지를 빌드해 Harbor에 올리면, 이 이미지로 학습(RayJob)과 서빙(KServe)이 모두 실행됩니다.
