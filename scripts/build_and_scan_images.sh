#!/bin/bash
# Build and security scan Docker images for production

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
VERSION=${VERSION:-latest}
REGISTRY=${DOCKER_REGISTRY:-}
SCAN_SEVERITY=${SCAN_SEVERITY:-CRITICAL,HIGH}

echo -e "${GREEN}Building Zeitgeist Docker images...${NC}"

# Function to build and scan image
build_and_scan() {
    local dockerfile=$1
    local image_name=$2
    local context_dir=${3:-.}
    
    echo -e "\n${YELLOW}Building ${image_name}...${NC}"
    
    # Build image
    docker build \
        -f "${dockerfile}" \
        -t "${image_name}:${VERSION}" \
        --build-arg BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
        --build-arg VCS_REF="$(git rev-parse --short HEAD)" \
        --no-cache \
        "${context_dir}"
    
    # Get image size
    SIZE=$(docker images "${image_name}:${VERSION}" --format "{{.Size}}")
    echo -e "${GREEN}Image size: ${SIZE}${NC}"
    
    # Security scan with Trivy
    echo -e "${YELLOW}Running security scan...${NC}"
    if command -v trivy &> /dev/null; then
        trivy image \
            --severity "${SCAN_SEVERITY}" \
            --no-progress \
            --format table \
            "${image_name}:${VERSION}"
    else
        echo -e "${YELLOW}Trivy not installed, skipping security scan${NC}"
        echo "Install with: brew install aquasecurity/trivy/trivy"
    fi
    
    # Check image size
    SIZE_MB=$(docker images "${image_name}:${VERSION}" --format "{{.Size}}" | sed 's/MB//')
    if [[ "${SIZE_MB}" =~ ^[0-9]+$ ]] && [ "${SIZE_MB}" -gt 200 ]; then
        echo -e "${YELLOW}Warning: Image size (${SIZE}MB) exceeds 200MB target${NC}"
    fi
    
    # Tag for registry if specified
    if [ -n "${REGISTRY}" ]; then
        docker tag "${image_name}:${VERSION}" "${REGISTRY}/${image_name}:${VERSION}"
        echo -e "${GREEN}Tagged as ${REGISTRY}/${image_name}:${VERSION}${NC}"
    fi
}

# Build collector image
build_and_scan \
    "docker/Dockerfile.collector.production" \
    "envy/collector" \
    "."

# Build zeitgeist image
build_and_scan \
    "docker/Dockerfile.zeitgeist.production" \
    "envy/zeitgeist" \
    "."

# Run container structure tests if available
if command -v container-structure-test &> /dev/null; then
    echo -e "\n${YELLOW}Running container structure tests...${NC}"
    if [ -f "tests/container/collector-tests.yaml" ]; then
        container-structure-test test \
            --image "envy/collector:${VERSION}" \
            --config "tests/container/collector-tests.yaml"
    fi
    
    if [ -f "tests/container/zeitgeist-tests.yaml" ]; then
        container-structure-test test \
            --image "envy/zeitgeist:${VERSION}" \
            --config "tests/container/zeitgeist-tests.yaml"
    fi
else
    echo -e "${YELLOW}Container structure test not installed, skipping${NC}"
fi

# Summary
echo -e "\n${GREEN}Build complete!${NC}"
echo "Images built:"
echo "  - envy/collector:${VERSION}"
echo "  - envy/zeitgeist:${VERSION}"

# Check for vulnerabilities
echo -e "\n${YELLOW}Vulnerability Summary:${NC}"
if command -v trivy &> /dev/null; then
    for image in "envy/collector" "envy/zeitgeist"; do
        VULNS=$(trivy image --severity "${SCAN_SEVERITY}" --format json "${image}:${VERSION}" 2>/dev/null | jq '.Results[].Vulnerabilities | length' | awk '{s+=$1} END {print s}')
        if [ "${VULNS:-0}" -gt 0 ]; then
            echo -e "${RED}${image}: ${VULNS} vulnerabilities found${NC}"
        else
            echo -e "${GREEN}${image}: No critical/high vulnerabilities found${NC}"
        fi
    done
fi

# Push images if requested
if [ "${PUSH_IMAGES:-false}" == "true" ] && [ -n "${REGISTRY}" ]; then
    echo -e "\n${YELLOW}Pushing images to registry...${NC}"
    docker push "${REGISTRY}/envy/collector:${VERSION}"
    docker push "${REGISTRY}/envy/zeitgeist:${VERSION}"
    echo -e "${GREEN}Images pushed successfully${NC}"
fi