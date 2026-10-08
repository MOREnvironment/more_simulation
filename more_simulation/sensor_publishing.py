"""Publish configured MORE sensors as ROS messages."""

from collections.abc import Sequence
from dataclasses import dataclass
from math import cos, sin
from typing import Any

import numpy as np
from geometry_msgs.msg import (
    PointStamped,
    PoseStamped,
    PoseWithCovarianceStamped,
    TransformStamped,
    TwistStamped,
    TwistWithCovarianceStamped,
)
from rclpy.node import Node
from rclpy.publisher import Publisher
from rclpy.qos import QoSProfile
from sensor_msgs.msg import FluidPressure, Imu, MagneticField, NavSatFix
from tf2_ros import StaticTransformBroadcaster

from more_common.casadi_graph import RppCasadiGraph
from more_sensors.models import SensorMessage, SensorSampler


SensorMessage = (
    FluidPressure
    | Imu
    | MagneticField
    | NavSatFix
    | PointStamped
    | PoseStamped
    | PoseWithCovarianceStamped
    | TwistStamped
    | TwistWithCovarianceStamped
)


@dataclass(frozen=True)
class _MessagePublication:
    """One message of a sensor and its ROS publisher."""

    message: SensorMessage
    publisher: Publisher
    frame_id: str


@dataclass(frozen=True)
class _SensorPublication:
    """A configured sensor sampler and the messages it publishes."""

    sampler: SensorSampler
    messages: list[_MessagePublication]


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
        "PoseWithCovarianceStamped": (PoseWithCovarianceStamped, "pose"),
        "TwistWithCovarianceStamped": (TwistWithCovarianceStamped, "dvl"),
    }
    # Measurements expressed in the world frame instead of a sensor frame.
    _WORLD_FRAME_MESSAGES = frozenset({"PoseWithCovarianceStamped"})

    def __init__(
        self,
        node: Node,
        sensors: Sequence[Any],
        frame_id: str,
        qos: QoSProfile,
        prefix_topics_with_sim: bool = False,
        world_frame_id: str = "odom",
    ) -> None:
        self._node = node
        self._frame_id = frame_id
        self._world_frame_id = world_frame_id
        self._static_transform_broadcaster = None
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
            sample = publication.sampler.sample(
                vessel_state,
                ros_time_nanoseconds * 1e-9,
                vessel_acceleration,
            )
            if sample is None:
                continue
            noise_std = publication.sampler.noise.white_noise_std
            for output in publication.messages:
                values = output.message.values(sample)
                if not np.all(np.isfinite(values)):
                    raise ValueError(
                        f"sensor {output.message.name} returned "
                        "non-finite values"
                    )
                output.publisher.publish(
                    self._message_from_values(
                        output.message.name,
                        values,
                        ros_time_nanoseconds,
                        output.message.variances(noise_std),
                        output.frame_id,
                    )
                )

    def _create_publications(
        self,
        sensors: Sequence[Any],
    ) -> list[_SensorPublication]:
        publications = []
        transforms = []
        topic_counts: dict[str, int] = {}
        for sensor_index, sensor in enumerate(sensors):
            sampler = SensorSampler(sensor.graph())
            graph = sampler.graph
            self._validate_sensor_graph(sensor_index, graph)

            mounting = getattr(graph.payload, "mounting", None)
            sensor_frame_id = str(getattr(mounting, "frameId", "")).strip()
            if sensor_frame_id and mounting.publishTf:
                transforms.append(
                    self._mounting_transform(sensor_frame_id, mounting.location)
                )

            messages = []
            for message in sampler.messages:
                message_type_and_topic = self._MESSAGE_TYPES.get(message.name)
                if message_type_and_topic is None:
                    supported_types = ", ".join(self._MESSAGE_TYPES)
                    raise ValueError(
                        f"sensor {sensor_index} declares unsupported "
                        f"message {message.name!r}; supported types: "
                        f"{supported_types}"
                    )
                message_type, default_topic = message_type_and_topic
                topic_base = message.topic or default_topic
                topic_count = topic_counts.get(topic_base, 0)
                topic_counts[topic_base] = topic_count + 1
                topic = (
                    f"sim/{topic_base}"
                    if self._prefix_topics_with_sim
                    else topic_base
                )
                if topic_count:
                    topic = f"{topic}_{topic_count}"
                messages.append(
                    _MessagePublication(
                        message=message,
                        publisher=self._node.create_publisher(
                            message_type,
                            topic,
                            self._qos,
                        ),
                        frame_id=(
                            self._world_frame_id
                            if message.name in self._WORLD_FRAME_MESSAGES
                            else sensor_frame_id or self._frame_id
                        ),
                    )
                )
            publications.append(
                _SensorPublication(sampler=sampler, messages=messages)
            )
        if transforms:
            self._static_transform_broadcaster = StaticTransformBroadcaster(
                self._node
            )
            self._static_transform_broadcaster.sendTransform(transforms)
        return publications

    def _mounting_transform(
        self,
        sensor_frame_id: str,
        location: Sequence[float],
    ) -> TransformStamped:
        """Build the fixed vessel-to-sensor transform of a mounted sensor."""
        if len(location) != 3:
            raise ValueError(
                f"sensor frame {sensor_frame_id!r} needs a 3-value location"
            )
        transform = TransformStamped()
        transform.header.stamp = self._node.get_clock().now().to_msg()
        transform.header.frame_id = self._frame_id
        transform.child_frame_id = sensor_frame_id
        transform.transform.translation.x = float(location[0])
        transform.transform.translation.y = float(location[1])
        transform.transform.translation.z = float(location[2])
        transform.transform.rotation.w = 1.0
        return transform

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
        frame_id: str,
    ) -> SensorMessage:
        if message_name == "PoseStamped":
            self._require_size(message_name, values, 6)
            message = PoseStamped()
            self._set_header(message, ros_time_nanoseconds, frame_id)
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
            self._set_header(message, ros_time_nanoseconds, frame_id)
            message.twist.linear.x = float(values[0])
            message.twist.linear.y = float(values[1])
            message.twist.linear.z = float(values[2])
            return message

        if message_name == "Imu":
            self._require_size(message_name, values, 9)
            message = Imu()
            self._set_header(message, ros_time_nanoseconds, frame_id)
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
            self._set_header(message, ros_time_nanoseconds, frame_id)
            message.latitude = float(values[0])
            message.longitude = float(values[1])
            message.altitude = float(values[2])
            return message

        if message_name == "FluidPressure":
            self._require_size(message_name, values, 1)
            message = FluidPressure()
            self._set_header(message, ros_time_nanoseconds, frame_id)
            message.fluid_pressure = float(values[0])
            message.variance = float(variances[0])
            return message

        if message_name == "MagneticField":
            self._require_size(message_name, values, 3)
            message = MagneticField()
            self._set_header(message, ros_time_nanoseconds, frame_id)
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
            self._set_header(message, ros_time_nanoseconds, frame_id)
            message.point.x = float(values[0])
            message.point.y = float(values[1])
            message.point.z = float(values[2])
            return message

        if message_name == "TwistWithCovarianceStamped":
            self._require_size(message_name, values, 3)
            message = TwistWithCovarianceStamped()
            self._set_header(message, ros_time_nanoseconds, frame_id)
            message.twist.twist.linear.x = float(values[0])
            message.twist.twist.linear.y = float(values[1])
            message.twist.twist.linear.z = float(values[2])
            for axis in range(3):
                message.twist.covariance[7 * axis] = float(variances[axis])
            return message

        if message_name == "PoseWithCovarianceStamped":
            return self._pose_with_covariance_message(
                values, ros_time_nanoseconds, variances, frame_id
            )

        raise ValueError(f"unsupported sensor message {message_name!r}")

    def _pose_with_covariance_message(
        self,
        values: np.ndarray,
        ros_time_nanoseconds: int,
        variances: np.ndarray,
        frame_id: str,
    ) -> PoseWithCovarianceStamped:
        """Build a pose from a full pose, a position, or a z position."""
        elements_by_size = {6: range(6), 3: range(3), 1: [2]}
        elements = elements_by_size.get(values.size)
        if elements is None:
            raise ValueError(
                "PoseWithCovarianceStamped requires 6, 3, or 1 sensor "
                f"values, received {values.size}"
            )
        pose = np.zeros(6)
        message = PoseWithCovarianceStamped()
        self._set_header(message, ros_time_nanoseconds, frame_id)
        for value, variance, element in zip(values, variances, elements):
            pose[element] = value
            message.pose.covariance[7 * element] = float(variance)
        message.pose.pose.position.x = float(pose[0])
        message.pose.pose.position.y = float(pose[1])
        message.pose.pose.position.z = float(pose[2])
        quaternion = self._quaternion_from_euler(pose[3], pose[4], pose[5])
        message.pose.pose.orientation.x = quaternion[0]
        message.pose.pose.orientation.y = quaternion[1]
        message.pose.pose.orientation.z = quaternion[2]
        message.pose.pose.orientation.w = quaternion[3]
        return message

    def _set_header(
        self,
        message: SensorMessage,
        ros_time_nanoseconds: int,
        frame_id: str,
    ) -> None:
        message.header.stamp.sec = ros_time_nanoseconds // 1_000_000_000
        message.header.stamp.nanosec = ros_time_nanoseconds % 1_000_000_000
        message.header.frame_id = frame_id

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
