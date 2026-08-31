from setuptools import find_packages, setup


package_name = "more_simulation"
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
            f"share/{package_name}/.rppws/script_descriptions",
            [".rppws/script_descriptions/simulation.json"],
        ),
        (
            f"share/{package_name}/.rppws/parts/"
            "more_dynamics__hull_vessel/"
            "b8844e3c-0958-4ec9-b7b2-e186be6fbf56",
            [
                ".rppws/parts/more_dynamics__hull_vessel/"
                "b8844e3c-0958-4ec9-b7b2-e186be6fbf56/callbacks.py",
                ".rppws/parts/more_dynamics__hull_vessel/"
                "b8844e3c-0958-4ec9-b7b2-e186be6fbf56/description.json",
            ],
        ),
        (
            f"share/{package_name}/.rppws/parts/"
            "more_dynamics__hull_vessel/"
            "b8844e3c-0958-4ec9-b7b2-e186be6fbf56/params",
            [
                ".rppws/parts/more_dynamics__hull_vessel/"
                "b8844e3c-0958-4ec9-b7b2-e186be6fbf56/params/parameters.py",
            ],
        ),
    ],
    install_requires=[
        "casadi>=3.5.5",
        "matplotlib",
        "more_common>=0.1.0",
        "more_dynamics>=0.1.0",
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
        ],
    },
)
