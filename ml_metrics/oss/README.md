# Steps to build a new py-ml-metrics pip package

1. Update the version number in project.toml

2. To build pypi wheel, run:

```
cd <ml_metrics_dir>
sh ml_metrics/oss/runner.sh
```

3. Wheels are in `/tmp/ml_metrics/all_dist`.

4. Upload to PyPI:

```
python3 -m pip install --upgrade twine
python3 -m twine upload /tmp/ml_metrics/all_dist/*-any.whl
```

Authenticate with Twine by following https://pypi.org/help/#apitoken and editing
your `~/.pypirc`.

5. Draft the new release in github: https://github.com/google/ml-metrics/releases.
 Tag the release commit with the version number.

# Optional

* To build for a different python version, change the PYTHON_MINOR_VERSION
  and/or PYTHON_MAJOR_VERSION in `ml_metrics/oss/runner.sh`.

* To use a different docker image, switch it out under
  `ml_metrics/oss/build.Dockerfile`.

* All the dependencies have to be installed manually in
  `ml_metrics/oss/build.Dockerfile`.

* Refer the required dependencies from the dependencies section of `pyproject.toml`.

* When you do not need to update version number, you can manually adds or
  increment build number by renaming the whl files under
  `/tmp/ml_metrics/all_dist` following the format of
  `py_ml_metrics-{version}-[{build version}]-{py version}-non-any.whl`
  e.g.,:

```
mv py_ml_metrics-0.0.1-py310-none-any.whl py_ml_metrics-0.0.1-1-py310-none-any.whl
```
