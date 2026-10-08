"""Run the jetski simulation with its sensors and rpp_localization."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share_directory = Path(get_package_share_directory("more_simulation"))
    return LaunchDescription([
        DeclareLaunchArgument(
            "rpp_workspace",
            default_value=str(share_directory),
            description="RPP workspace holding the simulation script",
        ),
        DeclareLaunchArgument(
            "rpp_configuration",
            default_value="JetLocalization",
            description="Simulation configuration with the sensor suite",
        ),
        DeclareLaunchArgument(
            "localization_parameters",
            default_value=str(
                share_directory / "config" / "jetski_localization.yaml"
            ),
            description="Parameters of the rpp_localization node",
        ),
        ExecuteProcess(
            cmd=[
                "python3", "-m", "more_simulation.simulation_ros",
                "--ros-args",
                "-p", ["rpp_workspace:=", LaunchConfiguration("rpp_workspace")],
                "-p", [
                    "rpp_configuration:=",
                    LaunchConfiguration("rpp_configuration"),
                ],
            ],
            output="screen",
        ),
        Node(
            package="rpp_localization",
            executable="localization_node",
            name="localization_node",
            output="screen",
            parameters=[LaunchConfiguration("localization_parameters")],
        ),
    ])
