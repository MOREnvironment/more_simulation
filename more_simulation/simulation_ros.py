"""ROS 2 node for running a configured vehicle model in real time."""

from __future__ import annotations

import time
from math import ceil, cos, sin
from pathlib import Path

import casadi as ca
import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Trigger

from more_common.casadi_graph import RppCasadiGraph
from more_simulation.sensor_publishing import SensorPublisher
from more_simulation.simulation import Simulation


class SimulationRos(Node):
    """Advance one configured vehicle with a fixed-step RK4 timer."""

    _VEHICLE_STATE_SIZE = 12
    _PERFORMANCE_ESTIMATE_DURATION = 15.0
    _MAX_PERFORMANCE_ESTIMATE_STEPS = 1_500

    def __init__(self) -> None:
        super().__init__("simulation")

        self._delta_t = self._positive_float_parameter("delta_t", 0.1)
        vessel_index = int(self.declare_parameter("vessel_index", 0).value)
        rpp_workspace = str(
            self.declare_parameter("rpp_workspace", "").value
        ).strip()
        if not rpp_workspace:
            raise ValueError("rpp_workspace must be specified")
        rpp_configuration = str(
            self.declare_parameter("rpp_configuration", "").value
        ).strip()
        self._simulation = Simulation(
            delta_t=self._delta_t,
            rpp_workspace=rpp_workspace,
            rpp_configuration=rpp_configuration or None,
        )
        if not 0 <= vessel_index < len(self._simulation.vessels):
            raise IndexError(
                "vessel_index must select a configured vessel "
                f"(received {vessel_index})"
            )

        vessel = self._simulation.vessels[vessel_index]
        self._graph = RppCasadiGraph(vessel.graph())
        if self._graph.step is None:
            raise ValueError("configured vehicle graph does not define dynamics")
        if self._graph.num_outputs < self._VEHICLE_STATE_SIZE:
            raise ValueError(
                "configured vehicle output must contain at least 12 values"
            )

        self._state = Simulation._extract_initial_conditions(
            self._graph.payload
        )
        self._zero_command = ca.DM.zeros(self._graph.num_inputs, 1)
        self._command = self._zero_command
        self._last_command_received_at_nanoseconds: int | None = None
        self._rk4_step = self._create_rk4_step()
        self._last_realtime_step_at = time.monotonic()
        self._step_accumulator = 0.0
        self._last_realtime_warning_at: float | None = None
        self._ros_time_origin_nanoseconds: int | None = None
        self._command_latch_duration = self._non_negative_float_parameter(
            "command_latch_duration", 0.5
        )
        self._output_logging = bool(
            self.declare_parameter("output_logging", False).value
        )
        self._info_log_period = self._positive_float_parameter(
            "info_log_period", 1.0
        )
        self._last_info_log_time: float | None = None

        self._frame_id = self.declare_parameter("frame_id", "odom").value
        self._child_frame_id = self.declare_parameter(
            "child_frame_id", "base_link"
        ).value
        self._prefix_sensor_topics_with_sim = bool(
            self.declare_parameter(
                "prefix_sensor_topics_with_sim", False
            ).value
        )

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._command_subscriber = self.create_subscription(
            Float64MultiArray,
            "cmd_out",
            self._command_callback,
            qos,
        )
        self._state_publisher = self.create_publisher(
            Float64MultiArray,
            "sim/state",
            qos,
        )
        self._odometry_publisher = self.create_publisher(
            Odometry,
            "sim/odometry",
            qos,
        )
        self._sensor_publisher = SensorPublisher(
            self,
            vessel.sensors,
            self._child_frame_id,
            qos,
            prefix_topics_with_sim=self._prefix_sensor_topics_with_sim,
        )
        self._io_descriptions_service = self.create_service(
            Trigger,
            "sim/get_io_descriptions",
            self._io_descriptions_callback,
        )

        self._plot_positions = bool(
            self.declare_parameter("plot_positions", False).value
        )
        self._plot_velocities = bool(
            self.declare_parameter("plot_velocities", False).value
        )
        self._plot_orientation = bool(
            self.declare_parameter("plot_orientation", False).value
        )
        self._plot_xy = bool(self.declare_parameter("plot_xy", False).value)
        plot_save_directory = str(
            self.declare_parameter("plot_save_directory", "").value
        ).strip()
        self._plot_save_directory = (
            Path(plot_save_directory).expanduser()
            if plot_save_directory
            else None
        )
        self._plot_update_period = self._positive_float_parameter(
            "plot_update_period", 0.1
        )
        self._plot_window_size = self._non_negative_float_parameter(
            "window_size", 0.0
        )
        self._plot_time: list[float] = []
        self._plot_states: list[np.ndarray] = []
        self._last_plot_update_time: float | None = None
        self._plt = None
        self._plot_figures = []
        self._position_axis = None
        self._position_lines = []
        self._orientation_axis = None
        self._orientation_lines = []
        self._yaw_axis = None
        self._yaw_line = None
        self._linear_velocity_axis = None
        self._linear_velocity_lines = []
        self._angular_velocity_axis = None
        self._angular_velocity_lines = []
        self._xy_axis = None
        self._xy_line = None
        if (
            self._plot_positions
            or self._plot_velocities
            or self._plot_orientation
            or self._plot_xy
        ):
            self._initialize_realtime_plots()

        self._timer = self.create_timer(
            self._delta_t,
            self._timer_callback,
            clock=self.get_clock(),
        )
        self._publish_current_state()
        if self._output_logging:
            self._log_configuration_estimate()
        self.get_logger().info(
            "Simulating configured vessel %d at %.6f s with %d command "
            "values and a %.3f s command latch"
            % (
                vessel_index,
                self._delta_t,
                self._graph.num_inputs,
                self._command_latch_duration,
            )
        )

    def _timer_callback(self) -> None:
        """Advance fixed simulation steps until the wall-clock deadline."""
        callback_started_at = time.perf_counter()
        current_realtime = time.monotonic()
        elapsed_realtime = max(
            0.0, current_realtime - self._last_realtime_step_at
        )
        self._last_realtime_step_at = current_realtime
        self._step_accumulator += elapsed_realtime
        step_count = int(self._step_accumulator / self._delta_t)
        if step_count == 0:
            return
        self._step_accumulator -= step_count * self._delta_t

        current_time_nanoseconds = self.get_clock().now().nanoseconds
        command_age_nanoseconds = None
        if self._last_command_received_at_nanoseconds is not None:
            command_age_nanoseconds = (
                current_time_nanoseconds
                - self._last_command_received_at_nanoseconds
            )
        command_is_fresh = (
            command_age_nanoseconds is not None
            and 0.0 <= command_age_nanoseconds
            <= self._command_latch_duration * 1_000_000_000
        )
        applied_command = (
            self._command if command_is_fresh else self._zero_command
        )

        integration_started_at = time.perf_counter()
        for _ in range(step_count):
            self._state = self._rk4_step(self._state, applied_command)
        integration_duration = time.perf_counter() - integration_started_at

        publish_started_at = time.perf_counter()
        self._publish_current_state()
        publish_duration = time.perf_counter() - publish_started_at
        if self._output_logging:
            self._log_runtime_status(applied_command)
        self._log_realtime_overrun(
            current_realtime,
            elapsed_realtime,
            step_count,
            integration_duration,
            publish_duration,
            time.perf_counter() - callback_started_at,
        )

    def _log_realtime_overrun(
        self,
        current_realtime: float,
        elapsed_realtime: float,
        step_count: int,
        integration_duration: float,
        publish_duration: float,
        callback_duration: float,
    ) -> None:
        """Report deadline misses without allowing rendering to slow dynamics."""
        if (
            elapsed_realtime <= 1.5 * self._delta_t
            and callback_duration <= self._delta_t
        ):
            return
        if (
            self._last_realtime_warning_at is not None
            and current_realtime - self._last_realtime_warning_at < 1.0
        ):
            return
        self._last_realtime_warning_at = current_realtime
        self.get_logger().warning(
            "Real-time overrun: elapsed=%.3f s, catch_up_steps=%d, "
            "integration=%.3f s, publishing=%.3f s, callback=%.3f s"
            % (
                elapsed_realtime,
                step_count,
                integration_duration,
                publish_duration,
                callback_duration,
            )
        )

    def _command_callback(self, message: Float64MultiArray) -> None:
        """Save a valid command for one upcoming timer callback."""
        command = np.asarray(message.data, dtype=float)
        if command.size != self._graph.num_inputs:
            self.get_logger().warning(
                "Ignoring command with %d values; configured dynamics require %d"
                % (command.size, self._graph.num_inputs)
            )
            return
        if not np.all(np.isfinite(command)):
            self.get_logger().warning("Ignoring command with non-finite values")
            return

        command_value = ca.DM(
            command.reshape((self._graph.num_inputs, 1))
        )
        self._command = command_value
        self._last_command_received_at_nanoseconds = (
            self.get_clock().now().nanoseconds
        )

    def _io_descriptions_callback(
        self,
        _request: Trigger.Request,
        response: Trigger.Response,
    ) -> Trigger.Response:
        """Return the configured graph input and output descriptions."""
        response.success = True
        response.message = "\n\n".join(
            (
                self._format_io_descriptions(
                    "Inputs", self._graph.payload.inputDescription
                ),
                self._format_io_descriptions(
                    "Outputs", self._graph.payload.outputDescription
                ),
            )
        )
        return response

    @staticmethod
    def _format_io_descriptions(label: str, descriptions) -> str:
        lines = [f"{label}:"]
        if not descriptions:
            lines.append("  (none)")
            return "\n".join(lines)

        for index, description in enumerate(descriptions):
            name = str(description.name) or f"{label.lower()}_{index}"
            details = [f"size={description.size}"]
            minimum = list(description.min)
            maximum = list(description.max)
            if minimum:
                details.append(f"min={minimum}")
            if maximum:
                details.append(f"max={maximum}")
            text = str(description.description)
            if text:
                details.append(text)
            lines.append(f"  - {name}: {', '.join(details)}")

        return "\n".join(lines)

    def _create_rk4_step(self) -> ca.Function:
        state = ca.SX.sym("state", self._graph.num_states)
        command = ca.SX.sym("command", self._graph.num_inputs)
        dynamics = self._graph.step

        k1 = dynamics(state, command)
        k2 = dynamics(state + self._delta_t * k1 / 2.0, command)
        k3 = dynamics(state + self._delta_t * k2 / 2.0, command)
        k4 = dynamics(state + self._delta_t * k3, command)
        next_state = state + self._delta_t * (
            k1 + 2.0 * k2 + 2.0 * k3 + k4
        ) / 6.0
        return ca.Function("vehicle_rk4_step", [state, command], [next_state])

    def _full_forward_command(self) -> ca.DM:
        """Build a bounded command for estimating forward performance."""
        command = np.zeros(self._graph.num_inputs, dtype=float)
        offset = 0
        for description in self._graph.payload.inputDescription:
            size = int(description.size)
            name = str(description.name).lower()
            minimum = np.asarray(description.min, dtype=float)
            maximum = np.asarray(description.max, dtype=float)
            if "thrust" in name and maximum.shape == (size,):
                command[offset:offset + size] = maximum
            elif (
                minimum.shape == (size,)
                and maximum.shape == (size,)
                and np.all(minimum <= 0.0)
                and np.all(maximum >= 0.0)
            ):
                command[offset:offset + size] = 0.0
            offset += size
        return ca.DM(command.reshape((self._graph.num_inputs, 1)))

    def _log_configuration_estimate(self) -> None:
        """Log the graph's predicted response to sustained forward command."""
        command = self._full_forward_command()
        steps = min(
            ceil(self._PERFORMANCE_ESTIMATE_DURATION / self._delta_t),
            self._MAX_PERFORMANCE_ESTIMATE_STEPS,
        )
        state = ca.DM(self._state)
        for _ in range(steps):
            state = self._rk4_step(state, command)

        vehicle_state = np.asarray(
            self._graph.output(state, command).full(),
            dtype=float,
        ).reshape(-1)[:self._VEHICLE_STATE_SIZE]
        command_values = np.asarray(
            command.full(), dtype=float
        ).reshape(-1).tolist()
        estimate_duration = steps * self._delta_t
        self.get_logger().info(
            "Configured full-forward estimate over %.2f s: command=%s, "
            "u=%.3f m/s, v=%.3f m/s, r=%.3f rad/s"
            % (
                estimate_duration,
                command_values,
                vehicle_state[6],
                vehicle_state[7],
                vehicle_state[11],
            )
        )

    def _log_runtime_status(self, applied_command: ca.DM) -> None:
        """Log the command and acceleration that were used by this timer."""
        current_time = self._elapsed_ros_time_seconds(
            self.get_clock().now().nanoseconds
        )
        if (
            self._last_info_log_time is not None
            and current_time - self._last_info_log_time
            < self._info_log_period
        ):
            return

        state_derivative = np.asarray(
            self._graph.step(self._state, applied_command).full(),
            dtype=float,
        ).reshape(-1)
        vehicle_state = self._vehicle_state()
        command_values = np.asarray(
            applied_command.full(), dtype=float
        ).reshape(-1).tolist()
        self.get_logger().info(
            "Runtime t=%.2f s: applied_command=%s, "
            "u=%.3f m/s, du=%.3f m/s^2, r=%.3f rad/s"
            % (
                current_time,
                command_values,
                vehicle_state[6],
                state_derivative[-6],
                vehicle_state[11],
            )
        )
        self._last_info_log_time = current_time

    def _publish_current_state(self) -> None:
        vehicle_state = self._vehicle_state()
        ros_time_nanoseconds = self.get_clock().now().nanoseconds

        state_message = Float64MultiArray()
        state_message.data = vehicle_state.tolist()
        self._state_publisher.publish(state_message)
        self._odometry_publisher.publish(
            self._odometry_message(vehicle_state, ros_time_nanoseconds)
        )
        self._sensor_publisher.publish(
            vehicle_state,
            ros_time_nanoseconds,
        )

        if (
            self._plot_positions
            or self._plot_velocities
            or self._plot_orientation
            or self._plot_xy
        ):
            self._plot_time.append(
                self._elapsed_ros_time_seconds(ros_time_nanoseconds)
            )
            self._plot_states.append(vehicle_state)
            self._update_realtime_plots()

    def _vehicle_state(self) -> np.ndarray:
        output = self._graph.output(self._state, self._command)
        return np.asarray(output.full(), dtype=float).reshape(-1)[
            : self._VEHICLE_STATE_SIZE
        ]

    def _odometry_message(
        self,
        state: np.ndarray,
        ros_time_nanoseconds: int,
    ) -> Odometry:
        message = Odometry()
        message.header.stamp.sec = ros_time_nanoseconds // 1_000_000_000
        message.header.stamp.nanosec = (
            ros_time_nanoseconds % 1_000_000_000
        )
        message.header.frame_id = self._frame_id
        message.child_frame_id = self._child_frame_id

        message.pose.pose.position.x = float(state[0])
        message.pose.pose.position.y = float(state[1])
        message.pose.pose.position.z = float(state[2])
        quaternion = self._quaternion_from_euler(state[3], state[4], state[5])
        message.pose.pose.orientation.x = quaternion[0]
        message.pose.pose.orientation.y = quaternion[1]
        message.pose.pose.orientation.z = quaternion[2]
        message.pose.pose.orientation.w = quaternion[3]

        message.twist.twist.linear.x = float(state[6])
        message.twist.twist.linear.y = float(state[7])
        message.twist.twist.linear.z = float(state[8])
        message.twist.twist.angular.x = float(state[9])
        message.twist.twist.angular.y = float(state[10])
        message.twist.twist.angular.z = float(state[11])
        return message

    def _elapsed_ros_time_seconds(self, ros_time_nanoseconds: int) -> float:
        if self._ros_time_origin_nanoseconds is None:
            self._ros_time_origin_nanoseconds = ros_time_nanoseconds
        return (
            ros_time_nanoseconds - self._ros_time_origin_nanoseconds
        ) / 1_000_000_000

    def destroy_node(self) -> bool:
        if self._plot_time:
            self._update_realtime_plots(force=True)
            if self._plot_save_directory is not None:
                self._save_realtime_plots()
            if self._plt is not None:
                self._plt.ioff()
                if self._plot_save_directory is None:
                    self._plt.show()
        return super().destroy_node()

    def _initialize_realtime_plots(self) -> None:
        if self._plot_save_directory is not None:
            import matplotlib

            matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        self._plt = plt
        plt.rcParams["figure.raise_window"] = False
        if self._plot_save_directory is None:
            plt.ion()

        if self._plot_orientation:
            figure, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 8))
            self._orientation_axis = axes[0]
            self._orientation_lines = [
                axes[0].plot([], [], label=label)[0]
                for label in ("roll", "pitch")
            ]
            axes[0].set_ylabel("Roll/pitch [rad]")
            axes[0].set_title("Vehicle orientation and position")
            axes[0].grid(True)
            self._yaw_axis = axes[0].twinx()
            self._yaw_line = self._yaw_axis.plot(
                [], [], label="yaw", color="tab:green"
            )[0]
            self._yaw_axis.set_ylabel("Yaw [rad]")
            axes[0].legend(
                self._orientation_lines + [self._yaw_line],
                [line.get_label() for line in self._orientation_lines]
                + [self._yaw_line.get_label()],
            )

            self._position_axis = axes[1]
            self._position_lines = [
                axes[1].plot([], [], label=label)[0]
                for label in ("x", "y", "z")
            ]
            axes[1].set_xlabel("Time [s]")
            axes[1].set_ylabel("Position [m]")
            axes[1].grid(True)
            axes[1].legend()
            figure.tight_layout()
            self._plot_figures.append(figure)

        elif self._plot_positions:
            figure, axis = plt.subplots(figsize=(10, 5))
            self._position_axis = axis
            self._position_lines = [
                axis.plot([], [], label=label)[0]
                for label in ("x", "y", "z")
            ]
            axis.set_xlabel("Time [s]")
            axis.set_ylabel("Position [m]")
            axis.set_title("Vehicle position")
            axis.grid(True)
            axis.legend()
            figure.tight_layout()
            self._plot_figures.append(figure)

        if self._plot_velocities:
            figure, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 8))
            self._linear_velocity_axis = axes[0]
            self._linear_velocity_lines = [
                axes[0].plot([], [], label=label)[0]
                for label in ("u", "v", "w")
            ]
            axes[0].set_ylabel("Linear velocity [m/s]")
            axes[0].set_title("Vehicle velocity")
            axes[0].grid(True)
            axes[0].legend()

            self._angular_velocity_axis = axes[1]
            self._angular_velocity_lines = [
                axes[1].plot([], [], label=label)[0]
                for label in ("p", "q", "r")
            ]
            axes[1].set_xlabel("Time [s]")
            axes[1].set_ylabel("Angular velocity [rad/s]")
            axes[1].grid(True)
            axes[1].legend()
            figure.tight_layout()
            self._plot_figures.append(figure)

        if self._plot_xy:
            figure, axis = plt.subplots(figsize=(7, 7))
            self._xy_axis = axis
            self._xy_line = axis.plot([], [])[0]
            axis.set_xlabel("x [m]")
            axis.set_ylabel("y [m]")
            axis.set_title("Vehicle XY trajectory")
            axis.grid(True)
            axis.set_aspect("equal", adjustable="box")
            figure.tight_layout()
            self._plot_figures.append(figure)

        if self._plot_save_directory is None:
            plt.show(block=False)

    def _save_realtime_plots(self) -> None:
        if self._plot_save_directory is None:
            return

        self._plot_save_directory.mkdir(parents=True, exist_ok=True)
        for name, figure in zip(self._plot_file_names(), self._plot_figures):
            figure.savefig(self._plot_save_directory / f"{name}.png", dpi=150)
        self.get_logger().info(
            f"Saved simulation plots to {self._plot_save_directory}"
        )

    def _plot_file_names(self) -> list[str]:
        names = []
        if self._plot_orientation:
            names.append("orientation_position")
        elif self._plot_positions:
            names.append("positions")
        if self._plot_velocities:
            names.append("velocities")
        if self._plot_xy:
            names.append("xy")
        return names

    def _update_realtime_plots(self, force: bool = False) -> None:
        if self._plt is None:
            return

        current_time = self._plot_time[-1]
        if (
            not force
            and self._last_plot_update_time is not None
            and current_time - self._last_plot_update_time
            < self._plot_update_period
        ):
            return

        time = np.asarray(self._plot_time)
        states = np.asarray(self._plot_states)
        if self._plot_window_size > 0.0:
            first_visible_index = np.searchsorted(
                time,
                current_time - self._plot_window_size,
                side="left",
            )
            time = time[first_visible_index:]
            states = states[first_visible_index:]

        if self._position_axis is not None:
            for index, line in enumerate(self._position_lines):
                line.set_data(time, states[:, index])
            self._rescale_axis(self._position_axis)

        if self._orientation_axis is not None:
            orientation = np.arctan2(
                np.sin(states[:, 3:6]),
                np.cos(states[:, 3:6]),
            )
            for index, line in enumerate(self._orientation_lines):
                line.set_data(time, orientation[:, index])
            self._rescale_axis(self._orientation_axis)
            if self._yaw_axis is not None and self._yaw_line is not None:
                self._yaw_line.set_data(time, orientation[:, 2])
                self._rescale_axis(self._yaw_axis)

        if self._linear_velocity_axis is not None:
            for index, line in enumerate(self._linear_velocity_lines):
                line.set_data(time, states[:, 6 + index])
            self._rescale_axis(self._linear_velocity_axis)

        if self._angular_velocity_axis is not None:
            for index, line in enumerate(self._angular_velocity_lines):
                line.set_data(time, states[:, 9 + index])
            self._rescale_axis(self._angular_velocity_axis)

        if self._xy_axis is not None:
            self._xy_line.set_data(states[:, 0], states[:, 1])
            self._rescale_xy_axis(self._xy_axis)

        if self._plot_window_size > 0.0 and current_time > 0.0:
            window_start = max(0.0, current_time - self._plot_window_size)
            for axis in (
                self._position_axis,
                self._orientation_axis,
                self._linear_velocity_axis,
                self._angular_velocity_axis,
            ):
                if axis is not None:
                    axis.set_xlim(window_start, current_time)

        for figure in self._plot_figures:
            figure.canvas.draw_idle()
            figure.canvas.flush_events()
        self._last_plot_update_time = current_time

    @staticmethod
    def _rescale_axis(axis) -> None:
        axis.relim()
        axis.autoscale_view()

    @staticmethod
    def _rescale_xy_axis(axis, minimum_span: float = 20.0) -> None:
        axis.relim()
        x_minimum, x_maximum = axis.dataLim.intervalx
        y_minimum, y_maximum = axis.dataLim.intervaly
        if not np.all(np.isfinite([x_minimum, x_maximum, y_minimum, y_maximum])):
            return

        x_center = (x_minimum + x_maximum) / 2.0
        y_center = (y_minimum + y_maximum) / 2.0
        x_span = max(x_maximum - x_minimum, minimum_span)
        y_span = max(y_maximum - y_minimum, minimum_span)
        axis.set_xlim(x_center - x_span / 2.0, x_center + x_span / 2.0)
        axis.set_ylim(y_center - y_span / 2.0, y_center + y_span / 2.0)

    def _positive_float_parameter(self, name: str, default: float) -> float:
        value = float(self.declare_parameter(name, default).value)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be a positive finite value")
        return value

    def _non_negative_float_parameter(self, name: str, default: float) -> float:
        value = float(self.declare_parameter(name, default).value)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be a non-negative finite value")
        return value

    @staticmethod
    def _quaternion_from_euler(
        roll: float,
        pitch: float,
        yaw: float,
    ) -> tuple[float, float, float, float]:
        half_roll = roll / 2.0
        half_pitch = pitch / 2.0
        half_yaw = yaw / 2.0
        return (
            sin(half_roll) * cos(half_pitch) * cos(half_yaw)
            - cos(half_roll) * sin(half_pitch) * sin(half_yaw),
            cos(half_roll) * sin(half_pitch) * cos(half_yaw)
            + sin(half_roll) * cos(half_pitch) * sin(half_yaw),
            cos(half_roll) * cos(half_pitch) * sin(half_yaw)
            - sin(half_roll) * sin(half_pitch) * cos(half_yaw),
            cos(half_roll) * cos(half_pitch) * cos(half_yaw)
            + sin(half_roll) * sin(half_pitch) * sin(half_yaw),
        )


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = SimulationRos()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
