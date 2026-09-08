# morpheus-mlops-training

Demo Ray training repository for the Morpheus MLOps pipeline.

The GitHub Actions workflow builds `harbor.morpheus.test/mlops/ray-mlops-demo`
with both `latest` and the commit short SHA, updates the RayJob chart in
`baynex-hjkim/morpheus-mlops-charts`, and triggers an Argo CD sync for the
`mlops-demo` application.

The workflow is triggered by changes under `src/**`, `inference/**`, `Dockerfile`,
`requirements.txt`, or the workflow file itself.
