#!/usr/bin/env bash
set -Eeuo pipefail

DOMAIN="${DOMAIN:-lih.local}"
MORPH_URL="${MORPH_URL:-}"
API_TOKEN="${API_TOKEN:-}"
DEMO_NS="${DEMO_NS:-mlops-demo}"
ARGOCD_NS="${ARGOCD_NS:-argocd}"
HARBOR_REGISTRY="${HARBOR_REGISTRY:-harbor.${DOMAIN}}"
HARBOR_PROJECT="${HARBOR_PROJECT:-mlops}"
IMAGE_NAME="${IMAGE_NAME:-ray-mlops-demo}"
HARBOR_USERNAME="${HARBOR_USERNAME:-admin}"
HARBOR_PASSWORD="${HARBOR_PASSWORD:-Harbor12345}"
NODE_SSH_USER="${NODE_SSH_USER:-lih}"
NODE_SSH_PASSWORD="${NODE_SSH_PASSWORD:-1234qwer}"
BUILD_VM_NAME="${BUILD_VM_NAME:-mlops-build}"
BUILD_VM_HOST="${BUILD_VM_HOST:-${BUILD_VM_IP:-192.168.1.97}}"
TRAINING_REPO_URL="${TRAINING_REPO_URL:-https://github.com/InhongLee95/morpheus-mlops-training.git}"
ARGOCD_APP="${ARGOCD_APP:-mlops-demo}"
RAY_DASHBOARD_HOST="${RAY_DASHBOARD_HOST:-ray-mlops-demo.${DOMAIN}}"
KSERVE_EXPECTED_HOST="${KSERVE_EXPECTED_HOST:-mlops-demo.${DOMAIN}}"
MLFLOW_HOST_INTERNAL="${MLFLOW_HOST_INTERNAL:-mlflow.mlflow.svc.cluster.local:5000}"
MLFLOW_EXPERIMENT_NAME="${MLFLOW_EXPERIMENT_NAME:-ray-mlops-demo}"
MODEL_BUCKET="${MODEL_BUCKET:-mlops-demo-models}"
S3_ENDPOINT_URL="${S3_ENDPOINT_URL:-http://minio.minio.svc.cluster.local:9000}"
MLFLOW_S3_ENV_SECRET="${MLFLOW_S3_ENV_SECRET:-mlflow-s3-env}"
RUN_ID="${RUN_ID:-wf$(date +%Y%m%d%H%M%S)}"
IMAGE_REPO="${IMAGE_REPO:-${HARBOR_REGISTRY}/${HARBOR_PROJECT}/${IMAGE_NAME}}"
RAYJOB_NAME="${RAYJOB_NAME:-${IMAGE_NAME}-${RUN_ID}}"

log() { echo "[demo-e2e] $*"; }
die() { echo "[demo-e2e][ERROR] $*" >&2; exit 1; }

setup_cli() {
  export PATH="/var/lib/rancher/rke2/bin:/usr/local/bin:/usr/bin:/bin:${PATH}"
  if [[ -z "${KUBECONFIG:-}" && -f /etc/rancher/rke2/rke2.yaml ]]; then
    export KUBECONFIG=/etc/rancher/rke2/rke2.yaml
  fi
  command -v jq >/dev/null 2>&1 || (apt-get update -qq && apt-get install -y -qq jq)
  command -v curl >/dev/null 2>&1 || (apt-get update -qq && apt-get install -y -qq curl)
  command -v sshpass >/dev/null 2>&1 || (apt-get update -qq && apt-get install -y -qq sshpass)
}

require_foundation() {
  kubectl get ns "$DEMO_NS" >/dev/null
  kubectl -n "$DEMO_NS" get secret harbor-pull-secret >/dev/null
  kubectl -n "$DEMO_NS" get secret "$MLFLOW_S3_ENV_SECRET" >/dev/null
  kubectl -n "$ARGOCD_NS" get application.argoproj.io "$ARGOCD_APP" >/dev/null
  kubectl get clusterservingruntime kserve-mlserver >/dev/null
  kubectl -n kserve get cm inferenceservice-config -o jsonpath='{.data.ingress}' | jq -e \
    '.disableIngressCreation == true and .urlScheme == "https"' >/dev/null
}

resolve_build_vm_host() {
  local host
  if [[ -n "$BUILD_VM_HOST" ]]; then
    printf '%s\n' "$BUILD_VM_HOST"
    return 0
  fi
  if [[ -z "$MORPH_URL" || -z "$API_TOKEN" ]]; then
    die "BUILD_VM_HOST is not set and Morpheus API token was not rendered"
  fi
  host=$(curl -kfsS \
    -H "Authorization: BEARER ${API_TOKEN}" \
    "${MORPH_URL%/}/api/servers?max=20&phrase=${BUILD_VM_NAME}" \
    | jq -r --arg name "$BUILD_VM_NAME" '
        .servers[]
        | select(.name == $name or .hostname == $name or .externalName == $name)
        | .sshHost // .externalIp // .internalIp // empty
      ' | head -1)
  [[ -n "$host" && "$host" != "null" ]] || die "failed to resolve build VM sshHost from Morpheus API for ${BUILD_VM_NAME}"
  printf '%s\n' "$host"
}

build_and_push_image() {
  local build_vm_host
  build_vm_host=$(resolve_build_vm_host)
  log "Building ${IMAGE_REPO}:${RUN_ID} on build VM ${BUILD_VM_NAME} (${build_vm_host})"
  sshpass -p "$NODE_SSH_PASSWORD" ssh \
    -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    "${NODE_SSH_USER}@${build_vm_host}" \
    "REGISTRY='${HARBOR_REGISTRY}' PROJECT='${HARBOR_PROJECT}' IMAGE_NAME='${IMAGE_NAME}' IMAGE_TAG='${RUN_ID}' USERNAME='${HARBOR_USERNAME}' PASSWORD='${HARBOR_PASSWORD}' TRAINING_REPO='${TRAINING_REPO_URL}' bash -s" <<'REMOTE'
set -Eeuo pipefail
cert_tmp=$(mktemp)
openssl s_client -showcerts -connect "${REGISTRY}:443" -servername "$REGISTRY" </dev/null 2>/dev/null \
  | awk '/BEGIN CERTIFICATE/,/END CERTIFICATE/ {print}' > "$cert_tmp" || true
if [[ -s "$cert_tmp" ]]; then
  echo "1234qwer" | sudo -S mkdir -p "/etc/docker/certs.d/${REGISTRY}"
  echo "1234qwer" | sudo -S cp "$cert_tmp" "/etc/docker/certs.d/${REGISTRY}/ca.crt"
  echo "1234qwer" | sudo -S cp "$cert_tmp" /usr/local/share/ca-certificates/lih-local-wildcard.crt
  echo "1234qwer" | sudo -S update-ca-certificates >/dev/null || true
  echo "1234qwer" | sudo -S systemctl restart docker
fi
rm -f "$cert_tmp"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
git clone --depth 1 "$TRAINING_REPO" "$tmp/training"
printf '%s\n' "$PASSWORD" | docker login "$REGISTRY" -u "$USERNAME" --password-stdin
docker build \
  -t "${REGISTRY}/${PROJECT}/${IMAGE_NAME}:${IMAGE_TAG}" \
  -t "${REGISTRY}/${PROJECT}/${IMAGE_NAME}:latest" \
  "$tmp/training"
docker push "${REGISTRY}/${PROJECT}/${IMAGE_NAME}:${IMAGE_TAG}"
docker push "${REGISTRY}/${PROJECT}/${IMAGE_NAME}:latest"
REMOTE
}

argocd_server_pod() {
  kubectl -n "$ARGOCD_NS" get pod -l app.kubernetes.io/name=argocd-server -o jsonpath='{.items[0].metadata.name}'
}

patch_argocd_application() {
  local patch_json
  log "Patching Argo CD application ${ARGOCD_APP} for run ${RUN_ID}"
  patch_json=$(jq -n \
    --arg imageRepo "$IMAGE_REPO" \
    --arg imageTag "$RUN_ID" \
    --arg runId "$RUN_ID" \
    --arg modelBucket "$MODEL_BUCKET" \
    '{
      spec: {
        source: {
          helm: {
            parameters: [
              {name:"image.repository", value:$imageRepo},
              {name:"image.tag", value:$imageTag},
              {name:"runId", value:$runId},
              {name:"model.bucket", value:$modelBucket}
            ]
          }
        }
      }
    }')
  kubectl -n "$ARGOCD_NS" patch application.argoproj.io "$ARGOCD_APP" --type merge -p "$patch_json"
}

sync_argocd_application() {
  local pass pod
  pass=$(kubectl -n "$ARGOCD_NS" get secret argocd-initial-admin-secret -o jsonpath='{.data.password}' | base64 -d)
  pod=$(argocd_server_pod)
  kubectl -n "$ARGOCD_NS" exec "$pod" -- argocd login localhost:8080 --username admin --password "$pass" --plaintext
  kubectl -n "$ARGOCD_NS" exec "$pod" -- argocd app sync "$ARGOCD_APP" --server localhost:8080 --plaintext --prune --timeout 900
  kubectl -n "$ARGOCD_NS" exec "$pod" -- argocd app wait "$ARGOCD_APP" --server localhost:8080 --plaintext --sync --operation --timeout 900
}

wait_rayjob() {
  local status deployment message elapsed=0
  log "Waiting for RayJob ${RAYJOB_NAME}"
  while true; do
    status=$(kubectl -n "$DEMO_NS" get rayjob "$RAYJOB_NAME" -o jsonpath='{.status.jobStatus}' 2>/dev/null || true)
    deployment=$(kubectl -n "$DEMO_NS" get rayjob "$RAYJOB_NAME" -o jsonpath='{.status.jobDeploymentStatus}' 2>/dev/null || true)
    message=$(kubectl -n "$DEMO_NS" get rayjob "$RAYJOB_NAME" -o jsonpath='{.status.message}' 2>/dev/null || true)
    log "RayJob status=${status:-pending} deployment=${deployment:-pending} ${message}"
    if [[ "$status" == "SUCCEEDED" ]]; then
      break
    fi
    if [[ "$status" == "FAILED" || "$deployment" == "Failed" ]]; then
      kubectl -n "$DEMO_NS" get rayjob,pod -o wide || true
      kubectl -n "$DEMO_NS" describe rayjob "$RAYJOB_NAME" || true
      kubectl -n "$DEMO_NS" logs "job/${RAYJOB_NAME}" --tail=200 || true
      die "RayJob ${RAYJOB_NAME} failed"
    fi
    if (( elapsed >= 1200 )); then
      kubectl -n "$DEMO_NS" get rayjob,pod -o wide || true
      kubectl -n "$DEMO_NS" describe rayjob "$RAYJOB_NAME" || true
      kubectl -n "$DEMO_NS" logs "job/${RAYJOB_NAME}" --tail=200 || true
      die "RayJob ${RAYJOB_NAME} did not complete within timeout"
    fi
    sleep 10
    elapsed=$((elapsed + 10))
  done
  kubectl -n "$DEMO_NS" logs "job/${RAYJOB_NAME}" --tail=200
}

mlflow_api() {
  local method="$1" path="$2" payload="${3:-}"
  local pod
  pod=$(kubectl -n mlflow get pod \
    -l app.kubernetes.io/instance=mlflow,app.kubernetes.io/name=mlflow \
    -o jsonpath='{.items[0].metadata.name}')
  [[ -n "$pod" ]] || die "MLflow pod not found"
  printf '%s' "$payload" | kubectl -n mlflow exec -i "$pod" -- python3 -c '
import sys
import urllib.request

method = sys.argv[1]
url = sys.argv[2]
body = sys.stdin.read().encode()
headers = {"Content-Type": "application/json"}
req = urllib.request.Request(url, data=(body if body else None), headers=headers, method=method)
with urllib.request.urlopen(req, timeout=30) as response:
    print(response.read().decode())
' "$method" "http://127.0.0.1:5000${path}"
}

verify_mlflow_run() {
  local experiment_id run_json run_status model_s3_uri run_file
  experiment_id=$(mlflow_api GET "/api/2.0/mlflow/experiments/get-by-name?experiment_name=${MLFLOW_EXPERIMENT_NAME}" \
    | jq -r '.experiment.experiment_id // empty')
  [[ -n "$experiment_id" ]] || die "MLflow experiment ${MLFLOW_EXPERIMENT_NAME} not found"

  run_json=$(mlflow_api POST "/api/2.0/mlflow/runs/search" \
    "$(jq -n --arg exp "$experiment_id" --arg run "$RUN_ID" '{experiment_ids:[$exp], filter:"params.run_id = '\''\($run)'\''", max_results:1}')")
  run_file=$(mktemp /tmp/mlops-demo-mlflow-run.XXXXXX.json)
  printf '%s\n' "$run_json" > "$run_file"
  run_status=$(jq -r '.runs[0].info.status // empty' "$run_file")
  model_s3_uri=$(jq -r '.runs[0].data.params[]? | select(.key=="model_s3_uri") | .value' "$run_file")
  rm -f "$run_file"
  [[ "$run_status" == "FINISHED" ]] || die "MLflow run for ${RUN_ID} is not FINISHED: ${run_status}"
  [[ "$model_s3_uri" == "s3://${MODEL_BUCKET}/runs/${RUN_ID}/model.joblib" ]] || die "unexpected model_s3_uri: ${model_s3_uri}"
  log "Verified MLflow run for ${RUN_ID}: ${model_s3_uri}"
}

deploy_kserve() {
  local model_s3_uri="s3://${MODEL_BUCKET}/runs/${RUN_ID}/model.joblib"
  log "Deploying KServe InferenceService with ${model_s3_uri}"
  kubectl -n "$DEMO_NS" delete inferenceservice mlops-demo --ignore-not-found=true
  cat <<EOF | kubectl apply -f -
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: mlops-demo
  namespace: ${DEMO_NS}
  annotations:
    serving.kserve.io/deploymentMode: Standard
spec:
  predictor:
    dnsConfig:
      options:
      - name: ndots
        value: "1"
    minReplicas: 1
    imagePullSecrets:
    - name: harbor-pull-secret
    containers:
    - name: kserve-container
      image: ${IMAGE_REPO}:${RUN_ID}
      imagePullPolicy: IfNotPresent
      command:
      - python
      - -m
      - inference.main
      ports:
      - containerPort: 8080
        protocol: TCP
      envFrom:
      - secretRef:
          name: ${MLFLOW_S3_ENV_SECRET}
      env:
      - name: MODEL_S3_URI
        value: ${model_s3_uri}
      - name: MLFLOW_S3_ENDPOINT_URL
        value: ${S3_ENDPOINT_URL}
      - name: S3_ENDPOINT_URL
        value: ${S3_ENDPOINT_URL}
      - name: AWS_DEFAULT_REGION
        value: ap-northeast-2
      - name: IMAGE_TAG
        value: ${RUN_ID}
      readinessProbe:
        httpGet:
          path: /healthz
          port: 8080
        initialDelaySeconds: 5
        periodSeconds: 5
        timeoutSeconds: 3
      resources:
        requests:
          cpu: "250m"
          memory: "512Mi"
        limits:
          cpu: "1"
          memory: "1Gi"
EOF
  kubectl -n "$DEMO_NS" wait --for=condition=Ready inferenceservice/mlops-demo --timeout=10m || {
    kubectl -n "$DEMO_NS" describe inferenceservice mlops-demo || true
    kubectl -n "$DEMO_NS" get pod,svc -o wide || true
    kubectl -n "$DEMO_NS" logs -l serving.kserve.io/inferenceservice=mlops-demo --tail=200 || true
    die "InferenceService mlops-demo did not become Ready"
  }
}

kserve_status_url() {
  local status_url host scheme elapsed=0
  while true; do
    status_url=$(kubectl -n "$DEMO_NS" get inferenceservice mlops-demo -o jsonpath='{.status.url}' 2>/dev/null || true)
    scheme=$(printf '%s' "$status_url" | sed -E 's#^([a-zA-Z]+)://.*#\1#')
    host=$(printf '%s' "$status_url" | sed -E 's#^[a-zA-Z]+://([^/]+).*#\1#')
    if [[ -n "$status_url" && "$scheme" == "https" && "$host" == "$KSERVE_EXPECTED_HOST" ]]; then
      printf '%s\n' "${status_url%/}"
      return 0
    fi
    if (( elapsed >= 120 )); then
      kubectl -n "$DEMO_NS" get inferenceservice mlops-demo -o yaml || true
      kubectl -n "$DEMO_NS" get httproute -o wide || true
      die "InferenceService status.url did not resolve to https://${KSERVE_EXPECTED_HOST}: ${status_url:-empty}"
    fi
    sleep 5
    elapsed=$((elapsed + 5))
  done
}

predictor_service() {
  local svc
  svc=$(kubectl -n "$DEMO_NS" get svc mlops-demo-predictor -o jsonpath='{.metadata.name}' 2>/dev/null || true)
  if [[ -z "$svc" ]]; then
    svc=$(kubectl -n "$DEMO_NS" get svc -l serving.kserve.io/inferenceservice=mlops-demo -o json \
      | jq -r '.items[] | select(.metadata.name | test("predictor")) | .metadata.name' | head -1)
  fi
  [[ -n "$svc" ]] || die "KServe predictor service not found"
  printf '%s\n' "$svc"
}

apply_kserve_status_route() {
  local status_url host svc port accepted resolved elapsed=0
  status_url=$(kserve_status_url)
  host=$(printf '%s' "$status_url" | sed -E 's#^[a-zA-Z]+://([^/]+).*#\1#')
  svc=$(predictor_service)
  port=$(kubectl -n "$DEMO_NS" get svc "$svc" -o jsonpath='{.spec.ports[0].port}')
  log "Publishing KServe status URL ${status_url} through HTTPRoute/${DEMO_NS}/mlops-demo-kserve"
  cat <<EOF | kubectl apply -f -
apiVersion: gateway.networking.k8s.io/v1
kind: HTTPRoute
metadata:
  name: mlops-demo-kserve
  namespace: ${DEMO_NS}
spec:
  parentRefs:
  - name: nginx-gateway
    namespace: nginx-gateway
    sectionName: https
  hostnames:
  - ${host}
  rules:
  - matches:
    - path:
        type: PathPrefix
        value: /
    backendRefs:
    - name: ${svc}
      port: ${port}
EOF
  while true; do
    accepted=$(kubectl -n "$DEMO_NS" get httproute mlops-demo-kserve -o json \
      | jq -r '.status.parents[]? | select(.parentRef.name=="nginx-gateway" and .parentRef.sectionName=="https") | .conditions[]? | select(.type=="Accepted") | .status' \
      | head -1)
    resolved=$(kubectl -n "$DEMO_NS" get httproute mlops-demo-kserve -o json \
      | jq -r '.status.parents[]? | select(.parentRef.name=="nginx-gateway" and .parentRef.sectionName=="https") | .conditions[]? | select(.type=="ResolvedRefs") | .status' \
      | head -1)
    if [[ "$accepted" == "True" && "$resolved" == "True" ]]; then
      log "HTTPRoute mlops-demo-kserve accepted for ${host}"
      return 0
    fi
    if (( elapsed >= 120 )); then
      kubectl -n "$DEMO_NS" describe httproute mlops-demo-kserve || true
      die "HTTPRoute mlops-demo-kserve was not accepted"
    fi
    sleep 5
    elapsed=$((elapsed + 5))
  done
}

verify_inference() {
  local body response last_error attempt err_file inference_url
  inference_url=$(kserve_status_url)
  log "Using KServe InferenceService URL: ${inference_url}"
  body='{"instances":[[5.1,3.5,1.4,0.2],[6.7,3.0,5.2,2.3]]}'
  err_file=$(mktemp /tmp/mlops-demo-kserve-curl.XXXXXX.err)
  for attempt in $(seq 1 30); do
    if response=$(curl -kfsS \
      -H "Content-Type: application/json" \
      "${inference_url}/v1/models/mlops-demo:predict" \
      -d "$body" 2>"$err_file"); then
      break
    fi
    last_error=$(cat "$err_file" 2>/dev/null || true)
    log "Inference endpoint not ready yet (attempt ${attempt}/30): ${last_error}"
    sleep 5
  done
  rm -f "$err_file"
  [[ -n "${response:-}" ]] || die "Inference endpoint did not respond: ${last_error:-unknown error}"
  echo "$response" | jq .
  echo "$response" | jq -e '.predictions | length == 2' >/dev/null
  echo "$response" | jq -e '.model_s3_uri == "s3://'"${MODEL_BUCKET}"'/runs/'"${RUN_ID}"'/model.joblib"' >/dev/null
  log "Inference verified at ${inference_url}/v1/models/mlops-demo:predict"
}

setup_cli
kubectl wait --for=condition=Ready nodes --all --timeout=10m
require_foundation
build_and_push_image
patch_argocd_application
sync_argocd_application
wait_rayjob
verify_mlflow_run
deploy_kserve
apply_kserve_status_route
verify_inference
log "Demo E2E train, serve, and inference completed for ${RUN_ID}"
