#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#
# CuOpt NIM Operator Deployment Script
#
# This script automates the deployment of NVIDIA cuOpt using the NIM Operator.
#
# Usage:
#   ./deploy.sh                           # Deploy with defaults
#   ./deploy.sh --namespace my-ns         # Custom namespace
#   ./deploy.sh --uninstall               # Remove deployment
#   ./deploy.sh --help                    # Show help
#

set -e

# Default values
NAMESPACE="nim-service"
CUOPT_IMAGE_TAG="25.12.0-cuda12.9-py3.13"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNINSTALL=false
SKIP_PREREQUISITES=false
WAIT_TIMEOUT=300

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

usage() {
    cat << EOF
CuOpt NIM Operator Deployment Script

Usage: $(basename "$0") [OPTIONS]

Options:
    -n, --namespace NAME      Kubernetes namespace (default: nim-service)
    -t, --tag TAG             CuOpt image tag (default: ${CUOPT_IMAGE_TAG})
    -u, --uninstall           Uninstall CuOpt deployment
    -s, --skip-prerequisites  Skip prerequisite checks
    -w, --wait SECONDS        Timeout for waiting on resources (default: 300)
    -h, --help                Show this help message

Environment Variables:
    NGC_API_KEY               Required. Your NVIDIA NGC API key

Examples:
    # Deploy with defaults
    export NGC_API_KEY=your-key
    ./deploy.sh

    # Deploy to custom namespace
    ./deploy.sh --namespace my-cuopt

    # Uninstall
    ./deploy.sh --uninstall

EOF
    exit 0
}

check_prerequisites() {
    print_info "Checking prerequisites..."

    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        print_error "kubectl is not installed"
        exit 1
    fi

    # Check cluster connectivity
    if ! kubectl cluster-info &> /dev/null; then
        print_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi

    # Check NGC_API_KEY
    if [[ -z "${NGC_API_KEY}" ]]; then
        print_error "NGC_API_KEY environment variable is not set"
        echo "Please set it with: export NGC_API_KEY=<your-ngc-api-key>"
        exit 1
    fi

    # Check NIM Operator
    if ! kubectl get crd nimservices.apps.nvidia.com &> /dev/null; then
        print_error "NIM Operator is not installed"
        echo "Please install it with:"
        echo "  helm upgrade --install nim-operator nvidia/k8s-nim-operator -n nim-operator --version=3.0.2"
        exit 1
    fi

    # Check GPU Operator
    if ! kubectl get pods -n gpu-operator 2>/dev/null | grep -q "Running"; then
        print_warn "GPU Operator may not be running. Deployment might fail."
    fi

    print_info "All prerequisites met"
}

create_namespace() {
    print_info "Creating namespace ${NAMESPACE}..."

    if kubectl get namespace "${NAMESPACE}" &> /dev/null; then
        print_info "Namespace ${NAMESPACE} already exists"
    else
        kubectl create namespace "${NAMESPACE}"
    fi
}

create_secrets() {
    print_info "Creating secrets..."

    # Create image pull secret
    if kubectl get secret ngc-secret -n "${NAMESPACE}" &> /dev/null; then
        print_info "Secret ngc-secret already exists, updating..."
        kubectl delete secret ngc-secret -n "${NAMESPACE}"
    fi

    kubectl create secret docker-registry ngc-secret \
        -n "${NAMESPACE}" \
        --docker-server=nvcr.io \
        --docker-username='$oauthtoken' \
        --docker-password="${NGC_API_KEY}"

    # Create NGC API key secret
    if kubectl get secret ngc-api-secret -n "${NAMESPACE}" &> /dev/null; then
        print_info "Secret ngc-api-secret already exists, updating..."
        kubectl delete secret ngc-api-secret -n "${NAMESPACE}"
    fi

    kubectl create secret generic ngc-api-secret \
        -n "${NAMESPACE}" \
        --from-literal=NGC_API_KEY="${NGC_API_KEY}"

    print_info "Secrets created successfully"
}

deploy_cuopt() {
    print_info "Deploying CuOpt NIMService..."

    # Update namespace in manifest if different from default
    if [[ "${NAMESPACE}" != "nim-service" ]]; then
        print_info "Using custom namespace: ${NAMESPACE}"
        sed "s/namespace: nim-service/namespace: ${NAMESPACE}/" \
            "${SCRIPT_DIR}/cuopt-nimservice.yaml" | kubectl apply -f -
    else
        kubectl apply -f "${SCRIPT_DIR}/cuopt-nimservice.yaml"
    fi

    print_info "CuOpt NIMService created"
}

wait_for_deployment() {
    print_info "Waiting for CuOpt to be ready (timeout: ${WAIT_TIMEOUT}s)..."

    local start_time
    start_time=$(date +%s)
    local ready=false

    while [[ $(($(date +%s) - start_time)) -lt ${WAIT_TIMEOUT} ]]; do
        local status
        status=$(kubectl get nimservice cuopt-service -n "${NAMESPACE}" -o jsonpath='{.status.state}' 2>/dev/null || echo "Unknown")

        if [[ "${status}" == "Ready" ]]; then
            ready=true
            break
        fi

        echo -ne "\r  Status: ${status}... "
        sleep 5
    done

    echo ""

    if [[ "${ready}" == "true" ]]; then
        print_info "CuOpt is ready!"

        # Get service details
        local cluster_ip
        cluster_ip=$(kubectl get svc cuopt-service -n "${NAMESPACE}" -o jsonpath='{.spec.clusterIP}' 2>/dev/null)
        local port
        port=$(kubectl get svc cuopt-service -n "${NAMESPACE}" -o jsonpath='{.spec.ports[0].port}' 2>/dev/null)

        echo ""
        echo "============================================"
        echo "  CuOpt Deployment Successful!"
        echo "============================================"
        echo ""
        echo "  Service: cuopt-service.${NAMESPACE}.svc.cluster.local"
        echo "  ClusterIP: ${cluster_ip}"
        echo "  Port: ${port}"
        echo ""
        echo "  To test locally:"
        echo "    kubectl port-forward svc/cuopt-service -n ${NAMESPACE} 8000:8000"
        echo ""
        echo "  Then access: http://localhost:8000"
        echo "============================================"
    else
        print_warn "Deployment timed out. CuOpt may still be starting."
        print_info "Check status with: kubectl get nimservice cuopt-service -n ${NAMESPACE}"
        print_info "Check logs with: kubectl logs -f deployment/cuopt-service -n ${NAMESPACE}"
    fi
}

uninstall() {
    print_info "Uninstalling CuOpt from namespace ${NAMESPACE}..."

    # Delete NIMService
    if kubectl get nimservice cuopt-service -n "${NAMESPACE}" &> /dev/null; then
        kubectl delete nimservice cuopt-service -n "${NAMESPACE}"
        print_info "NIMService deleted"
    fi

    # Delete secrets
    kubectl delete secret ngc-secret ngc-api-secret -n "${NAMESPACE}" --ignore-not-found
    print_info "Secrets deleted"

    # Optionally delete namespace
    read -p "Delete namespace ${NAMESPACE}? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        kubectl delete namespace "${NAMESPACE}"
        print_info "Namespace deleted"
    fi

    print_info "Uninstall complete"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        -t|--tag)
            CUOPT_IMAGE_TAG="$2"
            shift 2
            ;;
        -u|--uninstall)
            UNINSTALL=true
            shift
            ;;
        -s|--skip-prerequisites)
            SKIP_PREREQUISITES=true
            shift
            ;;
        -w|--wait)
            WAIT_TIMEOUT="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            print_error "Unknown option: $1"
            usage
            ;;
    esac
done

# Main execution
if [[ "${UNINSTALL}" == "true" ]]; then
    uninstall
    exit 0
fi

if [[ "${SKIP_PREREQUISITES}" != "true" ]]; then
    check_prerequisites
fi

create_namespace
create_secrets
deploy_cuopt
wait_for_deployment
