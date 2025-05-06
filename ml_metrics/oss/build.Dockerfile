# Constructs the environment within which we will build the ml-metrics pip wheels.
#
# From /tmp/ml_metrics,
# ❯ DOCKER_BUILDKIT=1 docker build \
#     --build-arg PYTHON_VERSION=${PYTHON_VERSION} \
#     -t ml_metrics:${PYTHON_VERSION} - < ml_metrics/oss/build.Dockerfile
# ❯ docker run --rm -it -v /tmp/ml_metrics:/tmp/ml_metrics \
#      ml_metrics:${PYTHON_VERSION} bash

FROM quay.io/pypa/manylinux2014_x86_64
LABEL maintainer="ml-metrics team <ml-metrics-dev@google.com>"

ARG PYTHON_MAJOR_VERSION
ARG PYTHON_MINOR_VERSION
ARG PYTHON_VERSION

ENV DEBIAN_FRONTEND=noninteractive

RUN ulimit -n 1024 && yum install -y rsync

ENV PATH="/opt/python/cp${PYTHON_MAJOR_VERSION}${PYTHON_MINOR_VERSION}-cp${PYTHON_MAJOR_VERSION}${PYTHON_MINOR_VERSION}/bin:${PATH}"

# Install dependencies needed for ml-metrics
RUN --mount=type=cache,target=/root/.cache \
  python${PYTHON_VERSION} -m pip install -U \
    absl-py \
    build \
    cloudpickle \
    more-itertools\
    numpy;

# Install dependencies needed for ml-metrics tests
RUN --mount=type=cache,target=/root/.cache \
  python${PYTHON_VERSION} -m pip install -U \
    auditwheel;

WORKDIR "/tmp/ml_metrics"