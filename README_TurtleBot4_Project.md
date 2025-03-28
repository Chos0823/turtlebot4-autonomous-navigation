# TurtleBot4 Autonomous Navigation & Feature Detection

This project is a ROS2-based TurtleBot4 workspace that supports autonomous navigation using SLAM, as well as visual object recognition using onboard cameras and RViz2 visualization.

## 📦 Features

- **SLAM-based Mapping**
  - Uses `turtlebot4_navigation` and `nav2` to perform SLAM and navigate the environment.
- **Frontier-based Exploration**
  - Dynamically identifies unexplored regions and navigates towards them (`test.py`).
- **Visual Feature Detection**
  - Detects predefined objects (e.g., fire extinguisher, human figure) using ORB feature matching (`map_pic_pos.py`).
  - Estimates 3D pose with solvePnP and visualizes positions in RViz using markers.
- **Teleoperation & Manual Mapping**
  - Supports keyboard-based teleoperation for manual exploration and data collection.

## 🚀 Quick Start

### 1. Launch SLAM and Navigation

```bash
ros2 launch turtlebot4_viz view_robot.launch.py
ros2 launch turtlebot4_navigation slam.launch.py
ros2 launch turtlebot4_navigation nav2.launch.py params_file:='$HOME/turtlebot4_ws/nav2.yaml'
```

### 2. Start Frontier Exploration

```bash
ros2 run map_listener test
```

### 3. Start Camera and Feature Detection

```bash
ros2 service call /oakd/start_camera std_srvs/srv/Trigger "{}"
ros2 run map_listener map_pic_pos
```

### 4. Optional: Manual Control

```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

## 📁 Project Structure

```
turtlebot4_ws/
├── src/
│   ├── map_listener/
│   │   ├── map_pic_pos.py       # Feature detection and visualization
│   │   └── test.py              # Frontier-based autonomous exploration
├── nav2.yaml                    # Navigation parameter file
├── ext_orig.png                 # Fire extinguisher reference image
├── man_orig.png                 # Human reference image
├── README.md                    # This file
```

## 🧠 Technologies

- ROS2 (Humble)
- OpenCV (ORB, BFMatcher, solvePnPRansac)
- RViz2 (Marker visualization)
- tf2_ros
- Nav2 stack

## 📜 License

MIT License

---

Feel free to contribute or adapt this project for your own robotic platform!
