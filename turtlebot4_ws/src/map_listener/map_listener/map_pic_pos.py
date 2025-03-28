import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
import tf2_ros
import geometry_msgs.msg
from visualization_msgs.msg import Marker  # Marker 메시지 임포트

class FeatureExtractor(Node):
    def __init__(self):
        super().__init__('feature_extractor_node')
        self.subscription = self.create_subscription(
            CompressedImage,
            '/oakd/rgb/preview/image_raw/compressed',  # 이미지 토픽
            self.image_callback,
            10)
        self.bridge = CvBridge()

        # ORB 특징점 검출기 초기화 (더 많은 특징점 추출)
        self.orb = cv2.ORB_create(nfeatures=500)

        # 기존 이미지 경로 설정 (소화기, 사람)
        self.ext_orig_path = '/home/johyunsuk/turtlebot4_ws/ext_orig.png'
        self.man_orig_path = '/home/johyunsuk/turtlebot4_ws/man_orig.png'

        # 기존 이미지 읽기 (그레이스케일)
        self.ext_orig_img = cv2.imread(self.ext_orig_path, cv2.IMREAD_GRAYSCALE)
        self.man_orig_img = cv2.imread(self.man_orig_path, cv2.IMREAD_GRAYSCALE)

        # 기존 이미지에서 특징점 및 기술자 추출
        self.kp_ext_orig, self.des_ext_orig = self.orb.detectAndCompute(self.ext_orig_img, None)
        self.kp_man_orig, self.des_man_orig = self.orb.detectAndCompute(self.man_orig_img, None)

        # Brute Force Matcher 초기화 (Hamming 거리 사용)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # 인식 로직 실행 간격 (초 단위)
        self.recognition_interval = 1.5  # 1.5초 간격
        self.last_recognition_time = time.time()

        # 카메라 intrinsic 파라미터와 distortion 계수 (202>>390카메라 스펙에 따라)
        self.camera_matrix = np.array([[390.6661376953125, 0, 123.86566162109375],
                                       [0, 390.6661376953125, 124.75257873535156],
                                       [0, 0, 1]])
        self.dist_coeffs = np.array([-2.6200926303863525, -38.589866638183594,
                                      -0.0010925641981884837, 0.00021615292644128203,
                                      262.2897644042969, -2.7986717224121094,
                                      -36.708839416503906, 255.18719482421875])

        # TF2 리스너 초기화
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # RViz2에서 Marker 퍼블리시를 위한 퍼블리셔 생성
        self.marker_pub = self.create_publisher(Marker, 'visualization_marker', 10)

    def perform_pnp(self, object_keypoints, image_keypoints, good_matches, object_type):
        """
        good_matches로부터 object_points (참조 이미지의 keypoint, z=0 가정)와
        image_points (현재 프레임 keypoint)를 추출한 후, solvePnPRansac을 통해
        변환 행렬(회전, 평행 이동)을 구합니다.
        object_type 파라미터로 소화기('extinguisher')와 사람('person')을 구분합니다.
        """
        object_points = []
        image_points = []
        
        # object_type에 따라 pixel_to_meter 비율 다르게 설정
        if object_type == 'extinguisher':
            pixel_to_meter = 0.18 / 680  # 소화기 비율
        elif object_type == 'person':
            pixel_to_meter1 = 0.23 / 860  # 사람 x축 비율
            pixel_to_meter2 = 0.18 / 680  # 사람 y축 비율
        
        for match in good_matches:
            # queryIdx: 참조 이미지, trainIdx: 현재 이미지
            pt_obj = object_keypoints[match.queryIdx].pt  # (u,v)
            pt_img = image_keypoints[match.trainIdx].pt     # (u,v)
            
            if object_type == 'extinguisher':
                # 객체 포인트를 픽셀에서 미터 단위로 변환
                object_points.append([pt_obj[0] * pixel_to_meter, pt_obj[1] * pixel_to_meter, 0])  
            elif object_type == 'person':
                # 사람의 경우 x와 y의 비율이 다르므로 각각 다른 비율을 적용
                object_points.append([pt_obj[0] * pixel_to_meter1, pt_obj[1] * pixel_to_meter2, 0]) 
            
            # 이미지 포인트를 픽셀에서 미터로 변환
            image_points.append([pt_img[0], pt_img[1]]) 
        
        object_points = np.array(object_points, dtype=np.float32)
        image_points = np.array(image_points, dtype=np.float32)

        # PnP 파라미터 튜닝: 반복 횟수, reprojection error threshold 등 조정
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points,
            image_points,
            self.camera_matrix,
            self.dist_coeffs
        )
        if success and inliers is not None and len(inliers) > 5:
            # 추정된 회전 벡터를 회전 행렬로 변환
            R, _ = cv2.Rodrigues(rvec)
            # [R | t] 형태의 3x4 변환 행렬 생성
            transformation_matrix_3x4 = np.hstack((R, tvec))  # 3x4 변환 행렬
            # 4x4 변환 행렬로 확장
            transformation_matrix_4x4 = np.vstack([transformation_matrix_3x4, [0, 0, 0, 1]])
            
            self.get_logger().info("4x4 Transformation Matrix:\n{}".format(transformation_matrix_4x4))

            # TF2 변환을 사용하여 base_link -> 카메라 좌표계 변환 계산 (object_type 전달)
            self.compute_base_link_to_camera_transformation(transformation_matrix_4x4, object_type)
        else:
            self.get_logger().error("solvePnPRansac 실패: 유효한 포즈를 찾지 못했습니다.")


    def compute_base_link_to_camera_transformation(self, transformation_matrix_4x4, object_type):
        """
        TF2를 사용하여 base_link -> oakd_rgb_camera_optical_frame 변환을 계산합니다.
        object_type을 전달하여 이후 마커 표시 시 구분합니다.
        """
        try:
            # oakd_rgb_camera_optical_frame -> base_link 변환 추출
            transform = self.tf_buffer.lookup_transform(
                "base_link",  # 기준 좌표계 (로봇)
                "oakd_rgb_camera_optical_frame",  # 대상 좌표계 (카메라))
                rclpy.time.Time())  # 현재 시간
            translation = transform.transform.translation
            rotation = transform.transform.rotation

            # 변환 행렬 구성
            translation_vector = np.array([translation.x, translation.y, translation.z])
            rotation_matrix = np.array([
                [1-2*(rotation.y**2 + rotation.z**2), 2*(rotation.x*rotation.y - rotation.z*rotation.w), 2*(rotation.x*rotation.z + rotation.y*rotation.w)],
                [2*(rotation.x*rotation.y + rotation.z*rotation.w), 1-2*(rotation.x**2 + rotation.z**2), 2*(rotation.y*rotation.z - rotation.x*rotation.w)],
                [2*(rotation.x*rotation.z - rotation.y*rotation.w), 2*(rotation.y*rotation.z + rotation.x*rotation.w), 1-2*(rotation.x**2 + rotation.y**2)]
            ])

            # 4x4 변환 행렬 구성
            transformation_matrix = np.vstack((
                np.hstack((rotation_matrix, translation_vector.reshape(3, 1))),
                np.array([0, 0, 0, 1])
            ))

            self.get_logger().info("Base Link to Camera Transformation Matrix:\n{}".format(transformation_matrix))

            # /map -> /base_link 변환 계산 (object_type 전달)
            self.compute_map_to_base_link_transformation(transformation_matrix, transformation_matrix_4x4, object_type)

        except (tf2_ros.TransformException, Exception) as e:
            self.get_logger().error(f"Transform 오류 발생: {e}")

    def compute_map_to_base_link_transformation(self, transformation_matrix, transformation_matrix_4x4, object_type):
        """
        /map/ ->  base_link 변환을 계산한 후, object_type에 따라 다른 마커로 RViz2에 표시합니다.
        """
        try:
            # map -> base_link 변환 추출
            transform = self.tf_buffer.lookup_transform(
                "map",  # 기준 좌표계 (로봇)
                "base_link",        # 대상 좌표계 (맵)
                rclpy.time.Time())  # 현재 시간
            translation = transform.transform.translation
            rotation = transform.transform.rotation

            # 변환 행렬 구성
            translation_vector = np.array([translation.x, translation.y, translation.z])
            rotation_matrix = np.array([
                [1-2*(rotation.y**2 + rotation.z**2), 2*(rotation.x*rotation.y - rotation.z*rotation.w), 2*(rotation.x*rotation.z + rotation.y*rotation.w)],
                [2*(rotation.x*rotation.y + rotation.z*rotation.w), 1-2*(rotation.x**2 + rotation.z**2), 2*(rotation.y*rotation.z - rotation.x*rotation.w)],
                [2*(rotation.x*rotation.z - rotation.y*rotation.w), 2*(rotation.y*rotation.z + rotation.x*rotation.w), 1-2*(rotation.x**2 + rotation.y**2)]
            ])

            # 4x4 변환 행렬 구성
            transformation_matrixs = np.vstack((
                np.hstack((rotation_matrix, translation_vector.reshape(3, 1))),
                np.array([0, 0, 0, 1])
            ))

            self.get_logger().info("Map to Base Link Transformation Matrix:\n{}".format(transformation_matrixs))

            # 최종 변환 행렬 계산: transformation_matrix_4x4 * transformation_matrix * transformation_matrixs
            #final_matrix = np.dot(np.dot(transformation_matrix_4x4, transformation_matrix), transformation_matrixs)transformation_matrix_4x4
            final_matrix = np.dot(np.dot(transformation_matrixs,transformation_matrix),transformation_matrix_4x4 )
            self.get_logger().info("Final Transformation Matrix:\n{}".format(final_matrix))

            # 최종 행렬의 translation 값 추출 (x, y, z)
            translation_x = final_matrix[0, 3]
            translation_y = final_matrix[1, 3]
            translation_z = final_matrix[2, 3]

            # Marker 메시지 생성 및 퍼블리시 (RViz2에서 지점 표시)
            marker = Marker()
            marker.header.frame_id = "map"  # RViz에서 사용하는 좌표계 (필요시 수정)
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "final_point"
            # object_type에 따라 서로 다른 marker id 사용 (겹침 방지)
            marker.id = 0 if object_type == 'extinguisher' else 1
            marker.action = Marker.ADD

            # object_type에 따른 마커 형태와 색상 설정
            if object_type == 'extinguisher':
                marker.type = Marker.SPHERE
                marker.color.a = 1.0  # 투명도
                marker.color.r = 1.0  # 빨강색
                marker.color.g = 0.0
                marker.color.b = 0.0
            elif object_type == 'person':
                marker.type = Marker.CUBE
                marker.color.a = 1.0  # 투명도
                marker.color.r = 0.0
                marker.color.g = 0.0
                marker.color.b = 1.0  # 파랑색

            marker.pose.position.x = translation_x
            marker.pose.position.y = translation_y
            marker.pose.position.z = 0.0  # 필요시 z값 조정
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.2  # 마커 크기 (필요에 따라 조정)
            marker.scale.y = 0.2
            marker.scale.z = 0.2

            self.marker_pub.publish(marker)
            self.get_logger().info("RViz2에 Marker 퍼블리시 완료: x={}, y={}, type={}".format(translation_x, translation_y, object_type))

        except (tf2_ros.TransformException, Exception) as e:
            self.get_logger().error(f"Transform 오류 발생: {e}")

    def image_callback(self, msg):
        try:
            # CompressedImage 메시지를 OpenCV 이미지로 변환
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='passthrough')

            # 이미지를 그레이스케일로 변환
            gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

            # 현재 이미지에서 특징점 및 기술자 추출
            kp, des = self.orb.detectAndCompute(gray_image, None)
            if des is None:
                self.get_logger().info("현재 프레임에서 특징점을 찾지 못했습니다.")
                return

            # 기존 이미지와 현재 이미지의 특징점 매칭
            matches_ext = self.bf.match(self.des_ext_orig, des)
            matches_man = self.bf.match(self.des_man_orig, des)

            # 매칭 결과를 거리 순으로 정렬
            matches_ext = sorted(matches_ext, key=lambda x: x.distance)
            matches_man = sorted(matches_man, key=lambda x: x.distance)

            # 임계값 적용 (거리 < 60 또는 62) 후 좋은 매칭만 선택
            good_matches_ext = [m for m in matches_ext if m.distance < 59]
            good_matches_man = [m for m in matches_man if m.distance < 59]

            # 인식 로직은 지정된 간격마다 실행
            current_time = time.time()
            if current_time - self.last_recognition_time >= self.recognition_interval:
                if len(good_matches_ext) > 38:
                    self.get_logger().info("소화기 이미지 인식됨!")
                    # 소화기 reference keypoint와 현재 프레임 keypoint로 PnP 수행 (object_type: 'extinguisher')
                    self.perform_pnp(self.kp_ext_orig, kp, good_matches_ext, 'extinguisher')
                elif len(good_matches_man) > 42:
                    self.get_logger().info("사람 이미지 인식됨!")
                    # 사람 reference keypoint와 현재 프레임 keypoint로 PnP 수행 (object_type: 'person')
                    self.perform_pnp(self.kp_man_orig, kp, good_matches_man, 'person')
                else:
                    self.get_logger().info("사진이 인식되지 않았습니다.")                    

                self.last_recognition_time = current_time

            # 카메라 영상과 ORB 특징점이 그려진 영상을 윈도우에 출력
            keypoints_img = cv2.drawKeypoints(cv_image, kp, None, color=(0, 255, 0), flags=0)
            cv2.imshow("ORB Keypoints", keypoints_img)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"이미지 처리 중 오류 발생: {e}")

def main(args=None):
    rclpy.init(args=args)
    feature_extractor = FeatureExtractor()
    rclpy.spin(feature_extractor)
    feature_extractor.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
