"""Run the jetski simulation with its sensors and rpp_localization."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    share_directory = Path(get_package_share_directory("more_simulation"))
    uses_inekf = PythonExpression(
        ["'", LaunchConfiguration("filter"), "' == 'inekf'"]
    )
    uses_jetski_model = PythonExpression(
        [
            "'", LaunchConfiguration("filter"), "' == 'ekf' and '",
            LaunchConfiguration("process_model"), "' == 'jetski'",
        ]
    )
    uses_constant_acceleration = PythonExpression(
        [
            "'", LaunchConfiguration("filter"), "' == 'ekf' and '",
            LaunchConfiguration("process_model"), "' != 'jetski'",
        ]
    )
    return LaunchDescription([
        DeclareLaunchArgument(
            "filter",
            default_value="ekf",
            choices=["ekf", "inekf"],
            description="Filter: the 15-state EKF or the invariant EKF",
        ),
        DeclareLaunchArgument(
            "process_model",
            default_value="jetski",
            choices=["jetski", "constant_acceleration"],
            description="Model the EKF predicts with; unused by the InEKF",
        ),
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
            description="Parameters of the EKF and of navsat_pose_node",
        ),
        DeclareLaunchArgument(
            "inekf_parameters",
            default_value=str(share_directory / "config" / "jetski_inekf.yaml"),
            description="Parameters of the invariant EKF",
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
            executable="navsat_pose_node",
            name="navsat_pose_node",
            output="screen",
            parameters=[LaunchConfiguration("localization_parameters")],
        ),
        Node(
            package="rpp_localization",
            executable="localization_node",
            name="localization_node",
            output="screen",
            parameters=[LaunchConfiguration("localization_parameters")],
            condition=IfCondition(uses_constant_acceleration),
        ),
        Node(
            package="rpp_localization",
            executable="inekf_node",
            name="inekf_node",
            output="screen",
            parameters=[LaunchConfiguration("inekf_parameters")],
            condition=IfCondition(uses_inekf),
        ),
        Node(
            package="rpp_localization",
            executable="localization_node",
            name="localization_node",
            output="screen",
            parameters=[
                LaunchConfiguration("localization_parameters"),
                str(share_directory / "config" / "jetski_model.yaml"),
                {"rpp_workspace": LaunchConfiguration("rpp_workspace")},
            ],
            condition=IfCondition(uses_jetski_model),
        ),
        Node(
            package="more_simulation",
            executable="command_twist",
            name="command_twist",
            output="screen",
            condition=IfCondition(uses_jetski_model),
        ),
    ])
