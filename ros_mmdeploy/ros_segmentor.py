from typing import Union
import cv2
import numpy as np
import time

import rclpy
from std_msgs.msg import String
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge

from mmdeploy_runtime import Segmentor

class ROSSegmentor:
    def __init__(self, 
                 model_path,
                 use_compressed_rgb=True,
                 enable_visualization=False):
        rclpy.init()
        self.segmentor = Segmentor(model_path, device_name='cuda')
        self.bridge = CvBridge()
        self.node = rclpy.create_node('ros_segmentor_node')
        
        
        # 订阅图像话题
        topic = '/camera/color/image_raw/compressed' if use_compressed_rgb else '/camera/color/image_raw'
        msg_type = CompressedImage if use_compressed_rgb else Image
        self.image_subscribe = self.node.create_subscription(
            msg_type, topic, self.image_callback, 10
        )
        
        self.semantic_pub = self.node.create_publisher(Image, 'semantic_lable', 10)
        self.health_pub = self.node.create_publisher(String, 'health', 10)

        self.node.get_logger().info('ros_segmentor_node started')
        
        # 可视化相关
        self.enable_visualization = enable_visualization
        if enable_visualization:
            self.vis_pub = self.node.create_publisher(Image, 'semantic_colormap', 10)
            self.node.get_logger().set_level(rclpy.logging.LoggingSeverity.DEBUG)
    def image_callback(self, image: Union[Image, CompressedImage]):
        try:
            # 图像预处理
            if isinstance(image, CompressedImage):
                im_cv = self.bridge.compressed_imgmsg_to_cv2(image)
            else:
                im_cv = self.bridge.imgmsg_to_cv2(image, desired_encoding="passthrough")

            # 语义分割（已预初始化）
            t0 = time.perf_counter()
            semantic_cv = self.segmentor(im_cv)  # segmentor 应在 __init__ 中初始化
            infer_time = (time.perf_counter() - t0) * 1000
            self.node.get_logger().debug(f"Inference time: {infer_time:.1f}ms")
            
            # 确保内存连续性并转换数据类型
            semantic_np = np.ascontiguousarray(semantic_cv, dtype=np.uint8)
            
            # 创建并发布语义消息（使用更通用的编码）
            semantic_msg = self.bridge.cv2_to_imgmsg(
                semantic_np, 
                encoding="mono8"  # 或 "8UC1"（需确保标签值 <= 255）
            )
            semantic_msg.header = image.header  # 保持原始消息头
            self.semantic_pub.publish(semantic_msg)
            
            # 如果启用可视化，生成并发布可视化结果
            if self.enable_visualization:
                vis_img = cv2.applyColorMap(semantic_np * 10, cv2.COLORMAP_JET)
                vis_msg = self.bridge.cv2_to_imgmsg(vis_img, encoding="bgr8")
                vis_msg.header = image.header
                self.vis_pub.publish(vis_msg)
                
        except Exception as e:
            self.node.get_logger().error(f"Processing failed: {str(e)}", exc_info=True)
        
def main(args=None):
    ros_node = ROSSegmentor(
        model_path='/home/nypyp/code/nerf_bridge/mmdeploy_model/deeplabv3plus-r50-d8_sunrgb',
        use_compressed_rgb=True,
        enable_visualization=False)  # 默认关闭可视化
    rclpy.spin(ros_node.node)
    ros_node.node.destroy_node()
    rclpy.shutdown()
    
if __name__ == '__main__':
    main()