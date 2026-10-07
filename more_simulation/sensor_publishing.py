"""Publish configured MORE sensors as ROS messages."""

from collections.abc import Sequence
from dataclasses import dataclass
from math import cos, sin
from typing import Any

import numpy as np
from geometry_msgs.msg import PointStamped, PoseStamped, TwistStamped
from rclpy.node import Node
from rclpy.publisher import Publisher
from rclpy.qos import QoSProfile
from sensor_msgs.msg import FluidPressure, Imu, MagneticField, NavSatFix

from more_common.casadi_graph import RppCasadiGraph
from more_sensors.models import SensorSampler


SensorMessage = (
    FluidPressure
    | Imu
    | MagneticField
    | NavSatFix
    | PointStamped
    | PoseStamped
    | TwistStamped
)


@dataclass(frozen=True)
class _SensorPublication:
    """A configured sensor sampler and its ROS publisher."""

    sampler: SensorSampler
    publisher: Publisher
    message_name: str


class SensorPublisher:
    """Map sensor message names to ROS messages and publish them."""

    _VESSEL_STATE_SIZE = 12
    _VESSEL_ACCELERATION_SIZE = 6
    _MESSAGE_TYPES: dict[str, tuple[type, str]] = {
        "PoseStamped": (PoseStamped, "pose"),
        "TwistStamped": (TwistStamped, "dvl"),
        "Imu": (Imu, "imu"),
        "NavSatFix": (NavSatFix, "gnss"),
        "FluidPressure": (FluidPressure, "pressure"),
        "PointStamped": (PointStamped, "sbl"),
        "MagneticField": (MagneticField, "mag"),
    }

    def __init__(
        self,
        node: Node,
        sensors: Sequence[Any],
        frame_id: str,
        qos: QoSProfile,
        prefix_topics_with_sim: bool = False,
    ) -> None:
        self._node = node
        self._frame_id = frame_id
        self._qos = qos
        self._prefix_topics_with_sim = prefix_topics_with_sim
        self._publications = self._create_publications(sensors)

    def publish(
        self,
        vessel_state: np.ndarray,
        ros_time_nanoseconds: int,
        vessel_acceleration: np.ndarray | None = None,
    ) -> None:
        """Sample and publish every sensor measurement that is due."""
        for publication in self._publications:
            values = publication.sampler.sample(
                vessel_state,
                ros_time_nanoseconds * 1e-9,
                vessel_acceleration,
            )
            if values is None:
                continue
            if not np.all(np.isfinite(values)):
                raise ValueError(
                    f"sensor {publication.message_name} returned "
                    "non-finite values"
                )
            publication.publisher.publish(
                self._message_from_values(
                    publication.message_name,
                    values,
                    ros_time_nanoseconds,
                    publication.sampler.noise.white_noise_std**2,
                )
            )

    def _create_publications(
        self,
        sensors: Sequence[Any],
    ) -> list[_SensorPublication]:
        publications = []
        topic_counts: dict[str, int] = {}
        for sensor_index, sensor in enumerate(sensors):
            sampler = SensorSampler(sensor.graph())
            graph = sampler.graph
            self._validate_sensor_graph(sensor_index, graph)

            message_name = str(
                getattr(graph.payload, "messageName", "")
            ).strip()
            message_type_and_topic = self._MESSAGE_TYPES.get(message_name)
            if message_type_and_topic is None:
                supported_types = ", ".join(self._MESSAGE_TYPES)
                raise ValueError(
                    f"sensor {sensor_index} declares unsupported message "
                    f"{message_name!r}; supported types: {supported_types}"
                )

            message_type, topic_base = message_type_and_topic
            topic_count = topic_counts.get(topic_base, 0)
            topic_counts[topic_base] = topic_count + 1
            topic = (
                f"sim/{topic_base}"
                if self._prefix_topics_with_sim
                else topic_base
            )
            if topic_count:
                topic = f"{topic}_{topic_count}"
            publications.append(
                _SensorPublication(
                    sampler=sampler,
                    publisher=self._node.create_publisher(
                        message_type,
                        topic,
                        self._qos,
                    ),
                    message_name=message_name,
                )
            )
        return publications

    @classmethod
    def _validate_sensor_graph(
        cls,
        sensor_index: int,
        graph: RppCasadiGraph,
    ) -> None:
        if graph.num_inputs not in (
            cls._VESSEL_STATE_SIZE,
            cls._VESSEL_STATE_SIZE + cls._VESSEL_ACCELERATION_SIZE,
        ):
            raise ValueError(
                f"sensor {sensor_index} must accept a "
                f"{cls._VESSEL_STATE_SIZE}-value vessel state, optionally "
                f"followed by a {cls._VESSEL_ACCELERATION_SIZE}-value "
                "body acceleration"
            )
        if graph.num_states != 0 or graph.step is not None:
            raise ValueError(f"sensor {sensor_index} must be stateless")
        if graph.num_outputs == 0:
            raise ValueError(f"sensor {sensor_index} has no output")

    def _message_from_values(
        self,
        message_name: str,
        values: np.ndarray,
        ros_time_nanoseconds: int,
        variances: np.ndarray,
    ) -> SensorMessage:
        if message_name == "PoseStamped":
            self._require_size(message_name, values, 6)
            message = PoseStamped()
            self._set_header(message, ros_time_nanoseconds)
            message.pose.position.x = float(values[0])
            message.pose.position.y = float(values[1])
            message.pose.position.z = float(values[2])
            quaternion = self._quaternion_from_euler(
                values[3], values[4], values[5]
            )
            message.pose.orientation.x = quaternion[0]
            message.pose.orientation.y = quaternion[1]
            message.pose.orientation.z = quaternion[2]
            message.pose.orientation.w = quaternion[3]
            return message

        if message_name == "TwistStamped":
            self._require_size(message_name, values, 3)
            message = TwistStamped()
            self._set_header(message, ros_time_nanoseconds)
            message.twist.linear.x = float(values[0])
            message.twist.linear.y = float(values[1])
            message.twist.linear.z = float(values[2])
            return message

        if message_name == "Imu":
            self._require_size(message_name, values, 9)
            message = Imu()
            self._set_header(message, ros_time_nanoseconds)
            quaternion = self._quaternion_from_euler(
                values[0], values[1], values[2]
            )
            message.orientation.x = quaternion[0]
            message.orientation.y = quaternion[1]
            message.orientation.z = quaternion[2]
            message.orientation.w = quaternion[3]
            message.angular_velocity.x = float(values[3])
            message.angular_velocity.y = float(values[4])
            message.angular_velocity.z = float(values[5])
            message.linear_acceleration.x = float(values[6])
            message.linear_acceleration.y = float(values[7])
            message.linear_acceleration.z = float(values[8])
            for axis in range(3):
                message.orientation_covariance[4 * axis] = float(
                    variances[axis]
                )
                message.angular_velocity_covariance[4 * axis] = float(
                    variances[3 + axis]
                )
                message.linear_acceleration_covariance[4 * axis] = float(
                    variances[6 + axis]
                )
            return message

        if message_name == "NavSatFix":
            self._require_size(message_name, values, 3)
            message = NavSatFix()
            self._set_header(message, ros_time_nanoseconds)
            message.latitude = float(values[0])
            message.longitude = float(values[1])
            message.altitude = float(values[2])
            return message

        if message_name == "FluidPressure":
            self._require_size(message_name, values, 1)
            message = FluidPressure()
            self._set_header(message, ros_time_nanoseconds)
            message.fluid_pressure = float(values[0])
            message.variance = float(variances[0])
            return message

        if message_name == "MagneticField":
            self._require_size(message_name, values, 3)
            message = MagneticField()
            self._set_header(message, ros_time_nanoseconds)
            message.magnetic_field.x = float(values[0])
            message.magnetic_field.y = float(values[1])
            message.magnetic_field.z = float(values[2])
            for axis in range(3):
                message.magnetic_field_covariance[4 * axis] = float(
                    variances[axis]
                )
            return message

        if message_name == "PointStamped":
            self._require_size(message_name, values, 3)
            message = PointStamped()
            self._set_header(message, ros_time_nanoseconds)
            message.point.x = float(values[0])
            message.point.y = float(values[1])
            message.point.z = float(values[2])
            return message

        raise ValueError(f"unsupported sensor message {message_name!r}")

    def _set_header(
        self,
        message: SensorMessage,
        ros_time_nanoseconds: int,
    ) -> None:
        message.header.stamp.sec = ros_time_nanoseconds // 1_000_000_000
        message.header.stamp.nanosec = ros_time_nanoseconds % 1_000_000_000
        message.header.frame_id = self._frame_id

    @staticmethod
    def _require_size(
        message_name: str,
        values: np.ndarray,
        expected_size: int,
    ) -> None:
        if values.size != expected_size:
            raise ValueError(
                f"{message_name} requires {expected_size} sensor values, "
                f"received {values.size}"
            )

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
