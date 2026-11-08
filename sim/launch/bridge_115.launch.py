"""Project 115 ROS<->gz bridge (sensor-load cut, 2026-11-04): the stock turtlebot4_gz_bringup ros_gz_bridge.launch.py
minus what nothing in the project consumes.
Dropped: rgbd_camera/points (PointCloud2; with no gz subscriber gz-sensors skips filling the cloud) and the HMI
display / buttons / LED bridges (only the stock turtlebot4_node's on-robot display uses them).
Kept unchanged: the create3 bridge (cmd_vel, odom/tf, ground-truth pose for the create3 nodes, bumper, IR hazards),
/scan, and rgbd image / depth / camera_info (photos.py + the agent's camera)."""
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

ARGUMENTS = [DeclareLaunchArgument(n, default_value=d) for n, d in
             [("use_sim_time", "true"), ("robot_name", "turtlebot4"), ("dock_name", "standard_dock"),
              ("namespace", ""), ("world", "warehouse"), ("model", "standard")]]


def generate_launch_description() -> LaunchDescription:
    world, robot = LaunchConfiguration("world"), LaunchConfiguration("robot_name")
    sim_time = [{"use_sim_time": LaunchConfiguration("use_sim_time")}]
    create3 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [get_package_share_directory("irobot_create_gz_bringup"), "/launch/create3_ros_gz_bridge.launch.py"]),
        launch_arguments=[(k, LaunchConfiguration(k)) for k in ("robot_name", "dock_name", "namespace", "world")])

    def gz(path: str) -> list:
        return ["/world/", world, "/model/", robot, path]

    lidar = Node(package="ros_gz_bridge", executable="parameter_bridge", name="lidar_bridge", output="screen",
                 parameters=sim_time,
                 arguments=[gz("/link/rplidar_link/sensor/rplidar/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan")],
                 remappings=[(gz("/link/rplidar_link/sensor/rplidar/scan"), "scan")])
    cam = "/link/oakd_rgb_camera_frame/sensor/rgbd_camera/"
    topics = [("image", "sensor_msgs/msg/Image[gz.msgs.Image", "oakd/rgb/preview/image_raw"),
              ("depth_image", "sensor_msgs/msg/Image[gz.msgs.Image", "oakd/rgb/preview/depth"),
              ("camera_info", "sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo", "oakd/rgb/preview/camera_info")]
    camera = Node(package="ros_gz_bridge", executable="parameter_bridge", name="camera_bridge", output="screen",
                  parameters=sim_time,
                  arguments=[gz(cam + t + "@" + typ) for t, typ, _ in topics],
                  remappings=[(gz(cam + t), ros) for t, _, ros in topics])
    return LaunchDescription([*ARGUMENTS, create3, lidar, camera])
