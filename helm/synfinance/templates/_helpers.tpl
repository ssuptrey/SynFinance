{{/*
Expand the name of the chart.
*/}}
{{- define "synfinance.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "synfinance.fullname" -}}
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
{{- define "synfinance.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "synfinance.labels" -}}
helm.sh/chart: {{ include "synfinance.chart" . }}
{{ include "synfinance.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/part-of: synfinance
environment: {{ .Values.global.environment }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "synfinance.selectorLabels" -}}
app.kubernetes.io/name: {{ include "synfinance.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
API labels
*/}}
{{- define "synfinance.api.labels" -}}
{{ include "synfinance.labels" . }}
app.kubernetes.io/component: api
{{- end }}

{{/*
API selector labels
*/}}
{{- define "synfinance.api.selectorLabels" -}}
app.kubernetes.io/name: {{ .Values.api.name }}
app.kubernetes.io/component: api
{{- end }}

{{/*
PostgreSQL labels
*/}}
{{- define "synfinance.postgres.labels" -}}
{{ include "synfinance.labels" . }}
app.kubernetes.io/component: database
{{- end }}

{{/*
PostgreSQL selector labels
*/}}
{{- define "synfinance.postgres.selectorLabels" -}}
app.kubernetes.io/name: {{ .Values.postgres.name }}
app.kubernetes.io/component: database
{{- end }}

{{/*
Redis labels
*/}}
{{- define "synfinance.redis.labels" -}}
{{ include "synfinance.labels" . }}
app.kubernetes.io/component: cache
{{- end }}

{{/*
Redis selector labels
*/}}
{{- define "synfinance.redis.selectorLabels" -}}
app.kubernetes.io/name: {{ .Values.redis.name }}
app.kubernetes.io/component: cache
{{- end }}

{{/*
Create the name of the service account to use for API
*/}}
{{- define "synfinance.api.serviceAccountName" -}}
{{- if .Values.api.serviceAccount.create }}
{{- default .Values.api.name .Values.api.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.api.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Create the name of the service account to use for PostgreSQL
*/}}
{{- define "synfinance.postgres.serviceAccountName" -}}
{{- if .Values.postgres.serviceAccount.create }}
{{- default .Values.postgres.name .Values.postgres.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.postgres.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Create the name of the service account to use for Redis
*/}}
{{- define "synfinance.redis.serviceAccountName" -}}
{{- if .Values.redis.serviceAccount.create }}
{{- default .Values.redis.name .Values.redis.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.redis.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Image pull secrets
*/}}
{{- define "synfinance.imagePullSecrets" -}}
{{- if .Values.global.imagePullSecrets }}
imagePullSecrets:
{{- range .Values.global.imagePullSecrets }}
  - name: {{ . }}
{{- end }}
{{- end }}
{{- end }}
