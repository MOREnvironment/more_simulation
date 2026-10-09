# more_simulation

RPP vessel simulation package.

Build and run it from a sourced ROS 2 workspace:

```bash
colcon build --packages-select more_simulation --symlink-install
source install/setup.bash
ros2 run more_simulation simulation
```

The default RPP workspace in `.rppws` configures one
`more_dynamics::HullVessel`. Open this package with `rpp ws` to change the
vessel list or its parameters.

The configured `Simulation` stores the time, state, output, and input arrays
for each vessel in `simulation.results`. The `simulation` executable runs the
configured simulation and plots those results.

## Localization of the simulated jetski

`launch/jetski_localization.launch.py` runs the `JetLocalization`
configuration, a Jetski with IMU, DVL and GNSS sensors, together with the
`rpp_localization` filter:

```bash
ros2 launch more_simulation jetski_localization.launch.py
ros2 topic pub -r 10 /cmd_out std_msgs/msg/Float64MultiArray "{data: [0.2, 0.3]}"
ros2 run more_simulation odometry_error --warmup 5 --duration 20
```

`filter:=ekf` (the default) runs the 15-state EKF. With it,
`process_model:=jetski` (the default) predicts with the jetski dynamics and
the commands on `cmd_out`, and `process_model:=constant_acceleration` uses
the generic model of `rpp_localization`. `filter:=inekf` runs the invariant
EKF, which predicts with the IMU. GNSS fixes reach the filter through
`navsat_pose_node`, whose `datum` in `config/jetski_localization.yaml` must
match the `fix_location` of the simulated GNSS. `odometry_error` reports the
error of `odometry/filtered` against `sim/odometry`.
