import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
import numpy as np

class FrontierExplorer(Node):
    def __init__(self):
        super().__init__('frontier_explorer')
        
        # 맵 데이터 구독
        self.subscription = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10
        )
        
        # 1초마다 목표 갱신
        self.create_timer(1.0, self.update_goal)

        # 내비게이션 액션 클라이언트 초기화
        self.nav_to_pose_client = ActionClient(self, NavigateToPose, '/navigate_to_pose')

        self.map_data = None
        self.width = 0
        self.height = 0
        self.resolution = 0.01
        self.origin_x = 0
        self.origin_y = 0
        self.map_received = False  # 맵 수신 여부 확인
        self.goal_count = {}  # 목표 지점 카운트 (중복 체크)
        self.blacklist = set()  # 블랙리스트 관리

    def map_callback(self, msg):
        """맵 데이터를 저장하고 상태 업데이트"""
        self.map_data = np.array(msg.data, dtype=np.int8).reshape((msg.info.height, msg.info.width))
        self.map_data = np.flipud(self.map_data)
        self.width = msg.info.width
        self.height = msg.info.height
        self.resolution = msg.info.resolution
        self.origin_x = msg.info.origin.position.x
        self.origin_y = msg.info.origin.position.y
        self.map_received = True  # 맵 수신됨

    def update_goal(self):
        """새로운 탐색 목표를 설정하고 네비게이션 요청"""
        if not self.map_received:
            self.get_logger().warn("⚠️ 맵 데이터를 아직 수신하지 못함")
            return

        new_goal = self.find_frontier()
        if new_goal:
            goal_tuple = tuple(new_goal)  # 튜플로 변환하여 딕셔너리 키로 사용
            
            # 동일한 목표 10회 이상 요청 시 블랙리스트 추가
            if goal_tuple in self.goal_count:
                self.goal_count[goal_tuple] += 1
            else:
                self.goal_count[goal_tuple] = 1
            
            if self.goal_count[goal_tuple] >= 10:
                self.blacklist.add(goal_tuple)
                self.get_logger().warn(f"🚫 목표 {goal_tuple} 블랙리스트 추가됨")
                return
            
            self.navigate_to_goal(new_goal)
        else:
            self.get_logger().warn("⚠️ 목표를 찾지 못함")

    def find_frontier(self):
        """탐색할 수 있는 경계를 찾음 (블랙리스트 제외)"""
        frontier_points = []
        for y in range(1, self.height - 1):
            for x in range(1, self.width - 1):
                if self.map_data[y, x] == 0:
                    if any(self.map_data[y+dy, x+dx] == -1 for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]):
                        world_coords = self.convert_to_world_coordinates((x, y))
                        if tuple(world_coords) not in self.blacklist:
                            frontier_points.append(world_coords)
        
        if frontier_points:
            return frontier_points[0]
        return None

    def convert_to_world_coordinates(self, grid_point):
        """맵 좌표를 실제 좌표계로 변환"""
        x = grid_point[0] * self.resolution + self.origin_x
        y = (self.height - grid_point[1]) * self.resolution + self.origin_y
        return x, y

    def navigate_to_goal(self, goal):
        """탐색한 지점으로 이동 요청"""
        if not self.nav_to_pose_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().warn("❌ 내비게이션 서버가 실행되지 않음")
            return

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = goal[0]
        goal_msg.pose.pose.position.y = goal[1]
        goal_msg.pose.pose.orientation.w = 1.0

        self.get_logger().info(f"🚀 새로운 목표 설정: {goal}")

        self.nav_to_pose_client.send_goal_async(goal_msg)

def main():
    rclpy.init()
    explorer = FrontierExplorer()
    rclpy.spin(explorer)
    explorer.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
