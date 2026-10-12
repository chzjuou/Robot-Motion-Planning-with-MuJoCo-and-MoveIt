from setuptools import setup

package_name = "panda_moveit_task"

setup(
    name=package_name,
    version="0.0.1",
    packages=[package_name],
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        (
            "share/" + package_name,
            ["package.xml"],
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="user",
    maintainer_email="user@example.com",
    description="Panda MoveIt MuJoCo task scripts",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            # BEGIN AUTO TASKS
            'move = panda_moveit_task.move:main',
            'pick_place = panda_moveit_task.pick_place:main',
            'regrasp = panda_moveit_task.regrasp:main',
            'return = panda_moveit_task.return:main',
            'screw = panda_moveit_task.screw:main',
            # END AUTO TASKS
        ],
    },
)
