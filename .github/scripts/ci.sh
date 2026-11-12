#!/usr/bin/env bash
# Run from the GitHub Actions workspace; dependencies are checked out by ci.yml.
set -eo pipefail
source /opt/ros/lyrical/setup.bash

for package in rpp_common rpp_plugin_registrator rpp_orchestrator rpp_py rpp_cli rpp_testing; do
  python -m pip install --no-deps -e "rpp_dev/src/$package"
done
python -m pip install "casadi>=3.5.5" numpy matplotlib
python -m pip install --no-deps -e libraries/more_common
python -m pip install --no-deps -e libraries/more_dynamics
python -m pip install --no-deps -e libraries/more_simulation

colcon build --base-paths rpp_dev/src/rpp_cpp --symlink-install \
  --cmake-args -DBUILD_TESTING=OFF \
  -DPython3_EXECUTABLE=/opt/rpp-test-venv/bin/python
source install/setup.bash
rpp registry config set RPP_CPP_CORE_PATH "$PWD/build/rpp_cpp"
rpp init-home
rpp library register "$PWD/libraries/more_dynamics" --link

rpp library register "$PWD/libraries/more_simulation" --link

python -m pytest libraries/more_simulation/test
