"""Interactive plotting for a running simulation, isolated from its timer."""

from __future__ import annotations

from math import asin, atan2
from pathlib import Path

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile


class SimulationPlotter(Node):
    """Render simulation telemetry in a process separate from dynamics."""

    _STATE_SIZE = 12

    def __init__(self) -> None:
        super().__init__("simulation_plotter")
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
        self._plot_update_period = self._positive_float_parameter(
            "plot_update_period", 0.1
        )
        self._plot_window_size = self._non_negative_float_parameter(
            "window_size", 60.0
        )
        plot_save_directory = str(
            self.declare_parameter("plot_save_directory", "").value
        ).strip()
        self._plot_save_directory = (
            Path(plot_save_directory).expanduser()
            if plot_save_directory
            else None
        )

        self._plot_time: list[float] = []
        self._plot_states: list[np.ndarray] = []
        self._time_origin_nanoseconds: int | None = None
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

        self._initialize_plots()
        self._odometry_subscription = self.create_subscription(
            Odometry,
            "sim/odometry",
            self._on_odometry,
            QoSProfile(depth=10),
        )
        self._render_timer = self.create_timer(
            self._plot_update_period,
            self._render,
        )
        self.get_logger().info(
            "Plotting simulation telemetry independently at %.3f s intervals."
            % self._plot_update_period
        )

    def _on_odometry(self, message: Odometry) -> None:
        stamp_nanoseconds = (
            message.header.stamp.sec * 1_000_000_000
            + message.header.stamp.nanosec
        )
        if self._time_origin_nanoseconds is None:
            self._time_origin_nanoseconds = stamp_nanoseconds
        elif stamp_nanoseconds < self._time_origin_nanoseconds:
            self._time_origin_nanoseconds = stamp_nanoseconds
            self._plot_time.clear()
            self._plot_states.clear()

        self._plot_time.append(
            (stamp_nanoseconds - self._time_origin_nanoseconds) / 1_000_000_000
        )
        self._plot_states.append(self._state_from_odometry(message))
        self._prune_history()

    @classmethod
    def _state_from_odometry(cls, message: Odometry) -> np.ndarray:
        state = np.zeros(cls._STATE_SIZE, dtype=float)
        position = message.pose.pose.position
        orientation = message.pose.pose.orientation
        twist = message.twist.twist
        state[:3] = [position.x, position.y, position.z]
        state[3:6] = cls._euler_from_quaternion(
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w,
        )
        state[6:9] = [twist.linear.x, twist.linear.y, twist.linear.z]
        state[9:12] = [twist.angular.x, twist.angular.y, twist.angular.z]
        return state

    def _prune_history(self) -> None:
        if self._plot_window_size <= 0.0 or not self._plot_time:
            return
        minimum_time = self._plot_time[-1] - self._plot_window_size
        first_visible_index = int(
            np.searchsorted(self._plot_time, minimum_time, side="left")
        )
        if first_visible_index:
            self._plot_time = self._plot_time[first_visible_index:]
            self._plot_states = self._plot_states[first_visible_index:]

    def _initialize_plots(self) -> None:
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

    def _render(self) -> None:
        if not self._plot_time:
            return
        time = np.asarray(self._plot_time)
        states = np.asarray(self._plot_states)
        current_time = time[-1]

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

    def destroy_node(self) -> bool:
        if self._plot_time and self._plot_save_directory is not None:
            self._plot_save_directory.mkdir(parents=True, exist_ok=True)
            for name, figure in zip(self._plot_file_names(), self._plot_figures):
                figure.savefig(self._plot_save_directory / f"{name}.png", dpi=150)
        if self._plt is not None:
            self._plt.ioff()
            if self._plot_save_directory is None:
                self._plt.show()
        return super().destroy_node()

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
    def _euler_from_quaternion(
        x: float,
        y: float,
        z: float,
        w: float,
    ) -> tuple[float, float, float]:
        roll = atan2(
            2.0 * (w * x + y * z),
            1.0 - 2.0 * (x * x + y * y),
        )
        pitch = asin(float(np.clip(2.0 * (w * y - z * x), -1.0, 1.0)))
        yaw = atan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        )
        return roll, pitch, yaw


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = SimulationPlotter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
