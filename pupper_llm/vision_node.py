import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
# from vision_msgs.mg import Detection2DArray
import depthai as dai
import numpy as np
# import cv2
import blobconverter
from priorbox import PriorBox

# resize input to smaller size for faster inference
NN_WIDTH, NN_HEIGHT = 160, 120
VIDEO_WIDTH, VIDEO_HEIGHT = 640, 480
class DepthAICameraNode(Node):
    def __init__(self):
        super().__init__('face_detection_node')
        # self.publisher_detections = self.create_publisher(Detection2DArray, 'face_detections', 10)
        self.publisher_image = self.create_publisher(Image, '/image', 10)
        self.publisher_depth = self.create_publisher(Image, '/depth', 10)
        self.publisher_point = self.create_publisher(Point, '/point_topic', 10)
        self.pipeline = dai.Pipeline()
        
        detection_nn = self.pipeline.create(dai.node.NeuralNetwork)

        detection_nn.setBlobPath(blobconverter.from_zoo(name="face_detection_yunet_160x120", zoo_type="depthai", shaves=6))
        detection_nn.input.setBlocking(False)

        # Define camera
        left = self.pipeline.create(dai.node.ColorCamera)
        left.setPreviewSize(VIDEO_WIDTH, VIDEO_HEIGHT)
        left.setInterleaved(False)
        left.setFps(30)
        left.setResolution(dai.ColorCameraProperties.SensorResolution.THE_800_P)
        left.setCamera("left")

        right = self.pipeline.create(dai.node.ColorCamera)
        right.setPreviewSize(VIDEO_WIDTH, VIDEO_HEIGHT)
        right.setInterleaved(False)
        right.setFps(30)
        right.setResolution(dai.ColorCameraProperties.SensorResolution.THE_800_P)
        right.setCamera("right")

        # Define depth camera
        depth = self.pipeline.create(dai.node.StereoDepth)
        depth.setLeftRightCheck(True)
        depth.setExtendedDisparity(True)

        left.preview.link(depth.left)
        right.preview.link(depth.right)

        # Define manip
        manip = self.pipeline.create(dai.node.ImageManip)
        manip.initialConfig.setResize(NN_WIDTH, NN_HEIGHT)
        manip.initialConfig.setFrameType(dai.RawImgFrame.Type.BGR888p)
        manip.inputConfig.setWaitForMessage(False)

        # Create outputs
        xout_cam = self.pipeline.create(dai.node.XLinkOut)
        xout_cam.setStreamName("cam")

        xout_depth = self.pipeline.create(dai.node.XLinkOut)
        xout_depth.setStreamName("depth")

        xout_nn = self.pipeline.create(dai.node.XLinkOut)
        xout_nn.setStreamName("nn")

        left.preview.link(manip.inputImage)
        left.preview.link(xout_cam.input)
        depth.disparity.link(xout_depth.input)
        manip.out.link(detection_nn.input)
        detection_nn.out.link(xout_nn.input)
        self.device = dai.Device(self.pipeline)
        self.q_cam = self.device.getOutputQueue("cam", maxSize=4, blocking=False)
        self.q_depth = self.device.getOutputQueue("depth", maxSize=4, blocking=False)
        self.q_nn = self.device.getOutputQueue(name="nn", maxSize=4, blocking=False)
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.px, self.py = None, None
    def timer_callback(self):
        in_frame = self.q_cam.tryGet()
        in_depth = self.q_depth.tryGet()
        in_nn = self.q_nn.tryGet()
        if in_nn is not None:
            frame = in_frame.getCvFrame()
            msg = Image()
            msg.header.stamp = Node.get_clock(self).now().to_msg()
            msg.encoding = "bgr8"
            msg.height = np.shape(frame)[0]
            msg.width = np.shape(frame)[1]
            msg.step = np.shape(frame)[1] * 3
            msg.data = np.array(frame).tobytes()
            self.publisher_image.publish(msg)
            
            depth_frame = in_depth.getFrame()
            msg = Image()
            msg.header.stamp = Node.get_clock(self).now().to_msg()
            msg.encoding = "mono8"
            msg.height = np.shape(depth_frame)[0]
            msg.width = np.shape(depth_frame)[1]
            msg.step = np.shape(depth_frame)[1] * 2
            msg.data = np.array(depth_frame).tobytes()
            self.publisher_depth.publish(msg)

            conf = np.array(in_nn.getLayerFp16("conf")).reshape((1076, 2))
            iou = np.array(in_nn.getLayerFp16("iou")).reshape((1076, 1))
            loc = np.array(in_nn.getLayerFp16("loc")).reshape((1076, 14))

            # decode
            pb = PriorBox(input_shape=(NN_WIDTH, NN_HEIGHT), output_shape=(640, 480))
            dets = pb.decode(loc, conf, iou, 0.6)
            # print("dets", dets.shape[0])
            if dets.shape[0] > 0:
                bboxes = dets[:, 0:4]
                msg = Point()
                closest_dist = 100000000
                for i in range(min(dets.shape[0], 10)):
                    x = bboxes[i, 0] + bboxes[i, 2] / 2
                    y = bboxes[i, 1] + bboxes[i, 3] / 2 + 150 # /2
                    norm_x_pot = ((x / 640) - 0.5) * 2 
                    norm_y_pot = ((y / 480) - 0.5) * 2
                    if self.px is None:
                        norm_x = ((x / 640) - 0.5) * 2 
                        norm_y = ((y / 480) - 0.5) * 2
                        break
                    dist = (abs(norm_x_pot - self.px)**2 + abs(norm_y_pot - self.py)**2) ** 0.5
                    if dist < closest_dist:
                        norm_x = ((x / 640) - 0.5) * 2 
                        norm_y = ((y / 480) - 0.5) * 2
                        closest_dist = dist

                
                self.px = norm_x
                self.py = norm_y
                
                msg.x = norm_x
                msg.y = norm_y
                
                self.publisher_point.publish(msg)
            if dets.shape[0] == 0:
                self.px = None
                self.py = None
                # det_array = Detection2DArray()

                # # Bounding box
                # for i in range(dets.shape[0]):
                #     detections_msg = Detection2D()
                #     detection_msg.header = self.get_clock().now().to_msg()  # Use node's clock
                #     detection_msg.header.frame_id = "Detections"  # Or whatever your frame ID is
                #     bbox = detection_msg.bbox
                #     bbox.size_x = det[2]
                #     bbox.size_y = det[3]
                #     bbox.center.pose.position.x = det[0] + bbox.size_x / 2
                #     bbox.center.pose.position.y = det[1] + bbox.size_y / 2
                #     bbox.center.pose.orientation.w = 1  # Assuming no rotation
                #     det_array.detections.append(detection_msg)

                # self.publisher_detections.publish(det_array)

def main(args=None):
    rclpy.init(args=args)
    node = DepthAICameraNode()
    rclpy.spin(node)

    rclpy.shutdown()

if __name__=='__main__':
    main()            
