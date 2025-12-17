

### 1. Run Preprocess Job (Fresh Run)
kubectl delete job preprocess --ignore-not-found
kubectl apply -k .k8s-gke/
kubectl wait --for=condition=complete job/preprocess --timeout=300s
kubectl logs job/preprocess --tail=30

### 2. Monitor Training Logs (GRU, LSTM, Prophet)
kubectl logs -l tier=training -f --max-log-requests=5

### 3. Monitor Evaluation & Promotion
kubectl logs -f deployment/eval


### Extended Stress Test (300 users, 5m)
kubectl exec deployment/locust-master -- locust --headless -u 300 -r 50 -t 1m --host=http://inference-http:8000 --only-summary

### Watch HPA Targets & Scaling (run in separate terminal)
kubectl get hpa -w

### Watch Inference HTTP Replicas (run in separate terminal)
kubectl get pods -l app=inference-http -w

### Live CPU Usage During Load Test
kubectl top pods -l app=inference-http

### Check Current HPA Status
kubectl get hpa
kubectl describe hpa inference-http-hpa

### Check Current Pod Count
kubectl get pods -l app=inference-http