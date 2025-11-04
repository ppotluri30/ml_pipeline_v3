{{/*
Expand the name of the chart.
*/}}
{{- define "chart.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "chart.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "chart.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "chart.labels" -}}
helm.sh/chart: {{ include "chart.chart" . }}
{{ include "chart.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "chart.selectorLabels" -}}
app.kubernetes.io/name: {{ include "chart.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "chart.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "chart.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Get MLflow tracking URI - uses external backend if enabled, otherwise in-cluster service
*/}}
{{- define "chart.mlflowTrackingUri" -}}
{{- if .Values.externalBackend.enabled }}
{{- .Values.externalBackend.mlflow.trackingUri }}
{{- else }}
{{- printf "http://mlflow:%v" .Values.mlflow.service.port }}
{{- end }}
{{- end }}

{{/*
Get MinIO endpoint URL - uses external backend if enabled, otherwise in-cluster service
*/}}
{{- define "chart.minioEndpoint" -}}
{{- if .Values.externalBackend.enabled }}
{{- .Values.externalBackend.minio.endpoint }}
{{- else }}
{{- printf "http://minio:%v" .Values.minio.service.apiPort }}
{{- end }}
{{- end }}

{{/*
Get MinIO access key - uses external backend if enabled, otherwise in-cluster config
*/}}
{{- define "chart.minioAccessKey" -}}
{{- if .Values.externalBackend.enabled }}
{{- .Values.externalBackend.minio.accessKey }}
{{- else }}
{{- .Values.minio.auth.accessKey }}
{{- end }}
{{- end }}

{{/*
Get MinIO secret key - uses external backend if enabled, otherwise in-cluster config
*/}}
{{- define "chart.minioSecretKey" -}}
{{- if .Values.externalBackend.enabled }}
{{- .Values.externalBackend.minio.secretKey }}
{{- else }}
{{- .Values.minio.auth.secretKey }}
{{- end }}
{{- end }}

{{/*
Get FastAPI gateway URL - uses external backend if enabled, otherwise in-cluster service
*/}}
{{- define "chart.gatewayUrl" -}}
{{- if .Values.externalBackend.enabled }}
{{- .Values.externalBackend.fastapi.gatewayUrl }}
{{- else }}
{{- printf "http://fastapi-app:%v" .Values.fastapi.service.port }}
{{- end }}
{{- end }}