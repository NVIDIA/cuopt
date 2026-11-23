{{/*
Expand the name of the chart.
*/}}
{{- define "cuopt-server.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "cuopt-server.fullname" -}}
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
{{- define "cuopt-server.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "cuopt-server.labels" -}}
helm.sh/chart: {{ include "cuopt-server.chart" . }}
{{ include "cuopt-server.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "cuopt-server.selectorLabels" -}}
app.kubernetes.io/name: {{ include "cuopt-server.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "cuopt-server.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "cuopt-server.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
rapids-pre-commit-hooks: disable-next-line[verify-hardcoded-version]
serverType requires an image >= 26.10. "" leaves the image entrypoint in
charge and uses HTTP probes. proxy and legacy also use HTTP. grpc uses the
standard gRPC health probe.
*/}}
{{- define "cuopt-server.validate" -}}
{{- $serverType := .Values.serverType | default "" -}}
{{- if and $serverType (not (has $serverType (list "proxy" "grpc" "legacy"))) -}}
{{- fail (printf "serverType must be empty, proxy, grpc, or legacy, got %q" $serverType) -}}
{{- end -}}
{{- if and (eq $serverType "grpc") .Values.ingress.enabled -}}
{{- fail "ingress is HTTP-only; set ingress.enabled to false when serverType is grpc" -}}
{{- end -}}
{{- end -}}

{{- define "cuopt-server.listenPort" -}}
{{- if eq (.Values.serverType | default "") "grpc" -}}
{{- .Values.grpc.port -}}
{{- else -}}
{{- .Values.service.targetPort -}}
{{- end -}}
{{- end -}}

{{- define "cuopt-server.servicePort" -}}
{{- if eq (.Values.serverType | default "") "grpc" -}}
{{- .Values.grpc.port -}}
{{- else -}}
{{- .Values.service.port -}}
{{- end -}}
{{- end -}}

{{- define "cuopt-server.portName" -}}
{{- if eq (.Values.serverType | default "") "grpc" -}}
grpc
{{- else -}}
http
{{- end -}}
{{- end -}}
