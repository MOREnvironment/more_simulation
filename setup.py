from pathlib import Path

from setuptools import find_packages, setup


package_name = "more_simulation"
rppws_root = Path(".rppws")
rppws_data_files = [
    (
        str(Path("share") / package_name / path.parent),
        [str(path)],
    )
    for path in sorted(rppws_root.rglob("*"))
    if path.is_file()
]


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            [f"resource/{package_name}"],
        ),
        (f"share/{package_name}", ["package.xml"]),
        (
            f"share/{package_name}/launch",
            [str(path) for path in sorted(Path("launch").glob("*.launch.py"))],
        ),
        (
            f"share/{package_name}/config",
            [str(path) for path in sorted(Path("config").glob("*.yaml"))],
        ),
    ] + rppws_data_files,
    install_requires=[
        "casadi>=3.5.5",
        "matplotlib",
        "more_common>=0.1.0",
        "more_dynamics>=0.1.0",
        "more_sensors>=0.1.0",
        "numpy",
        "rpp-py>=0.1.0",
        "setuptools",
    ],
    zip_safe=True,
    maintainer="luka",
    maintainer_email="luka.mandic@fer.hr",
    description="RPP simulations for vessel dynamics models",
    license="Apache-2.0",
    url="https://github.com/MOREnvironment/more_simulation",
    entry_points={
        "console_scripts": [
            "simulation = more_simulation.main:main",
            "simulation_ros = more_simulation.simulation_ros:main",
            "simulation_plotter = more_simulation.simulation_plotter:main",
            "odometry_error = more_simulation.odometry_error:main",
            "command_twist = more_simulation.command_twist:main",
        ],
    },
)
