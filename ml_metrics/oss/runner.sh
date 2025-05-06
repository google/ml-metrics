#!/bin/bash
# This script copies ml_metrics from internal repo, builds a docker, and
# builds pip wheels for all Python versions.

set -e -x

export TMP_FOLDER="/tmp/ml_metrics"

# Clean previous folders/images.
[ -f $TMP_FOLDER ] && rm -rf $TMP_FOLDER

PYTHON_MAJOR_VERSION="3"
PYTHON_MINOR_VERSION="11"
PYTHON_VERSION="${PYTHON_MAJOR_VERSION}.${PYTHON_MINOR_VERSION}"
ML_METRICS_RUN_TESTS=true

AUDITWHEEL_PLATFORM="manylinux2014_x86_64"


docker rmi -f ml_metrics:${PYTHON_VERSION}
docker rm -f ml_metrics

# Synchronize Copybara in $TMP_FOLDER.
cp -r . $TMP_FOLDER

cd $TMP_FOLDER

DOCKER_BUILDKIT=1 docker build --progress=plain --no-cache \
  --build-arg PYTHON_VERSION=${PYTHON_VERSION} \
  --build-arg PYTHON_MAJOR_VERSION=${PYTHON_MAJOR_VERSION} \
  --build-arg PYTHON_MINOR_VERSION=${PYTHON_MINOR_VERSION} \
  -t ml_metrics:${PYTHON_VERSION} - < ml_metrics/oss/build.Dockerfile

docker run --rm -a stdin -a stdout -a stderr \
  --env PYTHON_VERSION=${PYTHON_VERSION} \
  --env PYTHON_MAJOR_VERSION=${PYTHON_MAJOR_VERSION} \
  --env PYTHON_MINOR_VERSION=${PYTHON_MINOR_VERSION} \
  --env ML_METRICS_RUN_TESTS=${ML_METRICS_RUN_TESTS} \
  --env AUDITWHEEL_PLATFORM=${AUDITWHEEL_PLATFORM} \
  -v /tmp/ml_metrics:/tmp/ml_metrics \
  --name ml_metrics ml_metrics:${PYTHON_VERSION} \
  bash ml_metrics/oss/build_whl.sh

ls $TMP_FOLDER/all_dist/*.whl