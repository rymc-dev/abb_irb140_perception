from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch_ros.actions import Node 

def generate_launch_description():

    perception = Node(
        package="abb_irb140_perception",
        executable="abb_irb140_perception_node",
        parameters=[{
            # CPU inference competing with Gazebo for cycles was causing a
            # multi-second processing backlog (observed ~4.7s of steady-state
            # lag between a detection's capture stamp and the TF the node
            # had caught up to) -- see README's "Known issue" section.
            "device": "0",
        }],
    )

    return LaunchDescription(
        [
            perception
        ]
    )