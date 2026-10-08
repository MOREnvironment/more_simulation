"""Report the error of an estimated odometry against a reference odometry."""

from __future__ import annotations

from argparse import ArgumentParser
from math import atan2

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node


def _yaw(message: Odometry) -> float:
    orientation = message.pose.pose.orientation
    return atan2(
        2.0 * (orientation.w * orientation.z + orientation.x * orientation.y),
        1.0 - 2.0 * (orientation.y**2 + orientation.z**2),
    )


def _stamp(message: Odometry) -> float:
    return message.header.stamp.sec + message.header.stamp.nanosec * 1e-9


def _values(message: Odometry) -> np.ndarray:
    """Return stamp, position, yaw, and body-frame linear velocity."""
    position = message.pose.pose.position
    velocity = message.twist.twist.linear
    return np.array(
        [
            _stamp(message),
            position.x, position.y, position.z, _yaw(message),
            velocity.x, velocity.y, velocity.z,
        ]
    )


class OdometryError(Node):
    """Record a reference and an estimated odometry for comparison."""

    def __init__(self, reference_topic: str, estimate_topic: str) -> None:
        super().__init__("odometry_error")
        self.references: list[np.ndarray] = []
        self.estimates: list[np.ndarray] = []
        self.create_subscription(
            Odometry,
            reference_topic,
            lambda message: self.references.append(_values(message)),
            100,
        )
        self.create_subscription(
            Odometry,
            estimate_topic,
            lambda message: self.estimates.append(_values(message)),
            100,
        )

    def clear(self) -> None:
        self.references.clear()
        self.estimates.clear()


def estimate_errors(
    references: list[np.ndarray],
    estimates: list[np.ndarray],
) -> np.ndarray:
    """Subtract the reference interpolated at each estimate stamp."""
    if len(references) < 2 or not estimates:
        return np.zeros((0, 7))
    reference = np.asarray(references)
    estimate = np.asarray(estimates)
    reference = reference[np.argsort(reference[:, 0])]
    reference[:, 4] = np.unwrap(reference[:, 4])
    covered = (estimate[:, 0] >= reference[0, 0]) & (
        estimate[:, 0] <= reference[-1, 0]
    )
    estimate = estimate[covered]
    interpolated = np.column_stack(
        [
            np.interp(estimate[:, 0], reference[:, 0], reference[:, column])
            for column in range(1, 8)
        ]
    )
    errors = estimate[:, 1:] - interpolated
    errors[:, 3] = np.arctan2(np.sin(errors[:, 3]), np.cos(errors[:, 3]))
    return errors


def root_mean_square_errors(errors: np.ndarray) -> dict[str, float]:
    """Summarise errors as position, yaw, and velocity RMSE."""
    return {
        "position_m": float(
            np.sqrt(np.mean(np.sum(errors[:, 0:3] ** 2, axis=1)))
        ),
        "yaw_rad": float(np.sqrt(np.mean(errors[:, 3] ** 2))),
        "velocity_mps": float(
            np.sqrt(np.mean(np.sum(errors[:, 4:7] ** 2, axis=1)))
        ),
    }


def main(args: list[str] | None = None) -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--reference", default="sim/odometry")
    parser.add_argument("--estimate", default="odometry/filtered")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--warmup", type=float, default=5.0)
    arguments, ros_arguments = parser.parse_known_args(args)

    rclpy.init(args=ros_arguments)
    node = OdometryError(arguments.reference, arguments.estimate)
    clock = node.get_clock()
    started_at = clock.now()
    warmed_up = False
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            elapsed = (clock.now() - started_at).nanoseconds * 1e-9
            if not warmed_up and elapsed >= arguments.warmup:
                node.clear()
                warmed_up = True
            if elapsed >= arguments.warmup + arguments.duration:
                break
    finally:
        errors = estimate_errors(node.references, node.estimates)
        node.destroy_node()
        rclpy.shutdown()

    if not len(errors):
        raise SystemExit("no estimate overlaps the reference in time")
    print(f"samples: {len(errors)}")
    for name, value in root_mean_square_errors(errors).items():
        print(f"{name} rmse: {value:.4f}")
    names = ["x", "y", "z", "yaw", "vx", "vy", "vz"]
    means = ", ".join(
        f"{name} {value:+.3f}" for name, value in zip(names, errors.mean(axis=0))
    )
    print(f"mean error: {means}")


if __name__ == "__main__":
    main()
