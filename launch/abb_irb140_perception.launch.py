from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():

    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="false",
        description="Use simulation (Gazebo) clock if true",
    )

    perception = Node(
        package="abb_irb140_perception",
        executable="abb_irb140_perception_node",
        parameters=[{
            # CPU inference competing with Gazebo for cycles was causing a
            # multi-second processing backlog (observed ~4.7s of steady-state
            # lag between a detection's capture stamp and the TF the node
            # had caught up to) -- see README's "Known issue" section.
            "device": "0",
            "use_sim_time": LaunchConfiguration("use_sim_time"),
        }],
    )

    return LaunchDescription(
        [
            use_sim_time_arg,
            perception,
        ]
    )