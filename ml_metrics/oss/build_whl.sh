#!/bin/bash
# build wheel for python version specified in $PYTHON

set -e -x

CP_VERSION="cp${PYTHON_MAJOR_VERSION}${PYTHON_MINOR_VERSION}"
PYTHON_BIN_PATH="/opt/python/${CP_VERSION}-${CP_VERSION}/bin/python"

function main() {

  DEST="/tmp/ml_metrics/all_dist"
  mkdir -p "${DEST}"

  echo "=== Destination directory: ${DEST}"

  if [ "$ML_METRICS_RUN_TESTS" = true ] ; then
    python3 -m unittest discover -s ml_metrics -p '*_test.py'
  fi

  TMPDIR=$(mktemp -d -t tmp.XXXXXXXXXX)

  echo $(date) : "=== Using tmpdir: ${TMPDIR}"

  echo "=== Copy ml_metrics files"

  cp ./setup.py "${TMPDIR}"
  cp ./pyproject.toml "${TMPDIR}"
  cp ./LICENSE "${TMPDIR}"
  rsync -avm -L --exclude="__pycache__/*" ./ml_metrics "${TMPDIR}"
  # rsync -avm -L  --include="*.so" --include="*_pb2.py" \
  #   --exclude="*.runfiles" --exclude="*_obj" --include="*/" --exclude="*" \
  #   bazel-bin/ml_metrics "${TMPDIR}"

  pushd ${TMPDIR}
  echo $(date) : "=== Building wheel"

  "python${PYTHON_VERSION}" setup.py bdist_wheel --python-tag py3${PYTHON_MINOR_VERSION}
  cp dist/*.whl "${DEST}"

  echo $(date) : "=== Auditing wheel"
  auditwheel repair --plat ${AUDITWHEEL_PLATFORM} -w dist dist/*.whl

  echo $(date) : "=== Listing wheel"
  ls -lrt dist/*.whl
  cp dist/*.whl "${DEST}"
  popd

  echo $(date) : "=== Output wheel file is in: ${DEST}"
}

main "$@"