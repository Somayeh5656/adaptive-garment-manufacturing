#!/usr/bin/env python3
"""
perception_node.py
------------------
ROS 2 node that subscribes to the Gazebo camera feed (/camera/image_raw),
runs YOLOS to detect fabric pieces, publishes a fixed fabric pose
(placeholder) and a defect visualisation topic.

Part of the Adaptive Garment Manufacturing thesis.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose
import cv2
from cv_bridge import CvBridge

from .yolos_detector import YOLOSDetector


class FabricPerceptionNode(Node):
    """Detects fabric in images and publishes its pose (and defects)."""

    def __init__(self):
        super().__init__('fabric_perception')

        self.detector = YOLOSDetector()
        self.bridge = CvBridge()

        # Subscribe to the Gazebo camera topic
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        # Publishers
        self.fabric_pose_pub = self.create_publisher(Pose, '/fabric_pose', 10)
        self.defect_pub = self.create_publisher(Image, '/defect_map', 10)

        self.get_logger().info('Fabric Perception Node started.')

    def image_callback(self, msg):
        """
        Process each camera frame:
        - Run YOLOS detection
        - Publish the largest detection as fixed fabric pose (simplified)
        - Simulate defect detection and publish a visualisation
        """
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            detections = self.detector.detect_fabric(cv_image)

            if detections:
                # Pick the largest fabric detection by bounding‑box area
                main_fabric = max(detections, key=lambda d: d['area'])

                # Publish a fixed pose (placeholder – real 3D localisation is future work)
                pose_msg = Pose()
                pose_msg.position.x = 0.5
                pose_msg.position.y = 0.0
                pose_msg.position.z = 0.05
                pose_msg.orientation.w = 1.0
                self.fabric_pose_pub.publish(pose_msg)

                # Defect detection (currently simulated)
                defects = self.detector.detect_defects(cv_image)
                if defects:
                    self.get_logger().info(f'Detected {len(defects)} simulated defect(s).')
                    defect_img = self.create_defect_map(cv_image, defects)
                    defect_msg = self.bridge.cv2_to_imgmsg(defect_img, encoding='bgr8')
                    self.defect_pub.publish(defect_msg)

        except Exception as e:
            self.get_logger().error(f'Error in image callback: {e}')

    def create_defect_map(self, image, defects):
        """Draw bounding boxes and labels for simulated defects."""
        img_copy = image.copy()
        for defect in defects:
            x, y, w, h = defect['bbox']
            cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(img_copy, defect['type'], (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        return img_copy


def main(args=None):
    rclpy.init(args=args)
    node = FabricPerceptionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()