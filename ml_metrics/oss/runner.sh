#!/bin/bash
# This script copies ml_metrics from internal repo, builds a docker, and
# builds pip wheels for all Python versions.

set -e -x

export TMP_FOLDER="/tmp/ml_metrics/copybara"

# Clean previous folders/images.
[ -f $TMP_FOLDER ] && rm -rf $TMP_FOLDER

PYTHON_MAJOR_VERSION="3"
PYTHON_MINOR_VERSION="10"
PYTHON_VERSION="${PYTHON_MAJOR_VERSION}.${PYTHON_MINOR_VERSION}"

AUDITWHEEL_PLATFORM="manylinux2014_x86_64"


docker rmi -f ml_metrics:${PYTHON_VERSION}
docker rm -f ml_metrics

# Synchronize Copybara in $TMP_FOLDER.
copybara third_party/py/ml_metrics/oss/copy.bara.sky local .. \
  --init-history --folder-dir=$TMP_FOLDER --ignore-noop

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
  --env AUDITWHEEL_PLATFORM=${AUDITWHEEL_PLATFORM} \
  -v /tmp/ml_metrics:/tmp/ml_metrics \
  --name ml_metrics ml_metrics:${PYTHON_VERSION} \
  bash copybara/ml_metrics/oss/build_whl.sh

ls $TMP_FOLDER/all_dist/*.whl