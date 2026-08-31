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
