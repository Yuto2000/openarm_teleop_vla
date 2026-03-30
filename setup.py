from setuptools import setup
import os
from glob import glob

package_name = "openarm_teleop"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="eig-01",
    maintainer_email="eig@example.com",
    description="Quest2ROS2 teleop IK node for OpenArm bimanual robot",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "ik_node = openarm_teleop.ik_node:main",
            "record_node = openarm_teleop.record_node:main",
        ],
    },
)
