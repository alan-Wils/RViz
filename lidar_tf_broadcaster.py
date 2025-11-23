#!/usr/bin/env python3
import math

import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster


class LidarTFBroadcaster(Node):
    """Publishes a static transform from base_link to laser."""

    def __init__(self):
        super().__init__("lidar_tf_broadcaster")
        self.declare_parameter("use_sim_time", False)

        self.broadcaster = StaticTransformBroadcaster(self)
        self.publish_transform()

    def publish_transform(self):
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = "base_link"
        transform.child_frame_id = "laser"

        # TODO: Replace with your actual laser mounting transform
        transform.transform.translation.x = 0.15
        transform.transform.translation.y = 0.0
        transform.transform.translation.z = 0.10

        roll = 0.0
        pitch = 0.0
        yaw = 0.0
        qx, qy, qz, qw = self.euler_to_quaternion(roll, pitch, yaw)
        transform.transform.rotation.x = qx
        transform.transform.rotation.y = qy
        transform.transform.rotation.z = qz
        transform.transform.rotation.w = qw

        self.broadcaster.sendTransform(transform)
        self.get_logger().info("Published static transform base_link -> laser")

    @staticmethod
    def euler_to_quaternion(roll: float, pitch: float, yaw: float):
        """Convert Euler angles to quaternion."""
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)

        qw = cy * cr * cp + sy * sr * sp
        qx = cy * sr * cp - sy * cr * sp
        qy = cy * cr * sp + sy * sr * cp
        qz = sy * cr * cp - cy * sr * sp
        return qx, qy, qz, qw


def main(args=None):
    rclpy.init(args=args)
    node = LidarTFBroadcaster()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
