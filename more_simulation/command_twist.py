"""Republish simulator commands as the twist a localization filter reads."""

from __future__ import annotations

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


class CommandTwist(Node):
    """Map command values onto the axes of a control twist.

    axes lists, for each command value in order, the twist axis it drives:
    0 to 2 are linear x, y, z and 3 to 5 are angular x, y, z. A command
    older than timeout is replaced by zeros, like the simulator's latch.
    """

    def __init__(self) -> None:
        super().__init__("command_twist")
        self._axes = list(
            self.declare_parameter("axes", [0, 5]).value
        )
        if any(not 0 <= axis <= 5 for axis in self._axes):
            raise ValueError("axes must be twist axis indices from 0 to 5")
        self._timeout = float(self.declare_parameter("timeout", 0.5).value)
        rate = float(self.declare_parameter("rate", 10.0).value)
        if self._timeout < 0.0 or rate <= 0.0:
            raise ValueError("timeout cannot be negative and rate must be positive")
        self._values = [0.0] * 6
        self._received_at = None
        self._publisher = self.create_publisher(Twist, "cmd_vel", 1)
        self.create_subscription(
            Float64MultiArray, "cmd_out", self._command_callback, 10
        )
        self.create_timer(1.0 / rate, self._publish)

    def _command_callback(self, message: Float64MultiArray) -> None:
        values = [0.0] * 6
        for axis, value in zip(self._axes, message.data):
            values[axis] = float(value)
        self._values = values
        self._received_at = self.get_clock().now()

    def _publish(self) -> None:
        values = self._values
        if (
            self._received_at is None
            or (self.get_clock().now() - self._received_at).nanoseconds * 1e-9
            > self._timeout
        ):
            values = [0.0] * 6
        twist = Twist()
        twist.linear.x, twist.linear.y, twist.linear.z = values[0:3]
        twist.angular.x, twist.angular.y, twist.angular.z = values[3:6]
        self._publisher.publish(twist)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = CommandTwist()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
