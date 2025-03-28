# 탐색 Mapping 시
ros2 launch turtlebot4_viz view_robot.launch.py 
ros2 launch turtlebot4_navigation slam.launch.py 
ros2 launch turtlebot4_navigation nav2.launch.py params_file:='$HOME/turtlebot4_ws/nav2.yaml'
ros2 run map_listener test
ros2 service call /oakd/start_camera std_srvs/srv/Trigger "{}"
# 특징점 검출 시
ros2 launch turtlebot4_viz view_robot.launch.py
ros2 launch turtlebot4_navigation slam.launch.py
ros2 run teleop_twist_keyboard teleop_twist_keyboard
ros2 run map_listener map_pic_pos

