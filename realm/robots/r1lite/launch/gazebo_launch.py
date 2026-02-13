from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution, FindPackageShare
import os

def generate_launch_description():
    # Gazebo基础环境配置
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('gazebo_ros'),
                'launch/gzserver.launch.py'
            ])
        ]),
        launch_arguments={'world': 'empty'}.items()
    )

    # 静态坐标变换发布器(替代ROS1的tf节点)
    static_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='tf_footprint_base',
        arguments=['0', '0', '0', '0', '0', '0', 'base_link', 'base_footprint'],
        output='screen'
    )

    # 模型生成器(替代ROS1的spawn_model)
    model_path = PathJoinSubstitution([
        FindPackageShare('r1lite'),
        'urdf/r1lite.urdf'
    ])
    spawn_entity = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-file', model_path,
            '-name', 'r1lite',
            '-allow_renaming', 'true'
        ],
        output='screen'
    )

    # 关节校准状态发布器(替代ROS1的rostopic命令)
    calibration_publisher = ExecuteProcess(
        cmd=['ros2', 'topic', 'pub', '/calibrated', 'std_msgs/msg/Bool',
             '{data: true}', '--once'],
        output='screen'
    )

    return LaunchDescription([
        gazebo_launch,
        static_tf_node,
        spawn_entity,
        calibration_publisher
    ])