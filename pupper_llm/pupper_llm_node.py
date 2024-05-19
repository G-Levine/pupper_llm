import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from geometry_msgs.msg import Twist

import cv2
from cv_bridge import CvBridge
import base64

from openai import OpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt

import time

import prompt_utils
import json

GPT_MODEL = "gpt-4o"
client = OpenAI()

class LLMNode(Node):

    def __init__(self):
        super().__init__('llm_node')
        self.image_subscriber = self.create_subscription(Image, '/image', self.image_callback, 10)
        self.depth_subscriber = self.create_subscription(Image, '/depth', self.depth_callback, 10)
        self.transcription_subscriber = self.create_subscription(String, '/transcription', self.transcription_callback, 10)
        self.timer = self.create_timer(prompt_utils.IMAGE_PERIOD, self.timer_callback)
        self.image = None
        self.depth = None
        self.bridge = CvBridge()
        self.messages = [prompt_utils.system_message]
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)

    def image_callback(self, msg):
        self.image = msg
    
    def depth_callback(self, msg):
        self.depth = msg

    def transcription_callback(self, msg):
        self.call_llm(self.image, msg.data)
    
    def timer_callback(self):
        self.call_llm(self.image)

    def call_llm(self, image=None, text=None):
        content = []
        if image:
            cv_image = self.bridge.imgmsg_to_cv2(image, "bgr8")
            _, buffer = cv2.imencode('.jpg', cv_image)
            jpg_as_text = base64.b64encode(buffer).decode("utf-8")
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{jpg_as_text}", "detail": "low"}})
        if text:
            content.append({"type": "text", "text": text})
        self.messages.append({"role": "user", "content": content})
        response = chat_completion_request(self.messages)
        self.messages.append(response.dict())
        breakpoint()
        if response.tool_calls:
            for tool_call in response.tool_calls:
                if tool_call.function.name == "walk":
                    tool_fn = self.walk
                elif tool_call.function.name == "say":
                    tool_fn = self.say
                elif tool_call.function.name == "bark":
                    tool_fn = self.bark
                elif tool_call.function.name == "wag":
                    tool_fn = self.wag
                elif tool_call.function.name == "spin_around":
                    tool_fn = self.spin_around
                elif tool_call.function.name == "wait":
                    tool_fn = self.wait
                result = tool_fn(json.loads(tool_call.function.arguments))
                self.messages.append({"role": "tool", "tool_call_id": tool_call.id, "name": tool_call.function.name, "content": result})

    def walk(self, arguments):
        print(f"Walk: {arguments}")
        depth_np = np.frombuffer(self.depth.data, dtype=np.uint8)
        if arguments["linear_x"] > -0.01 and np.histogram(depth_np)[0][-1] > 100:
            return "Obstacle detected, can't move forward"
        cmd_vel_msg = Twist()
        cmd_vel_msg.linear.x = float(arguments["linear_x"])
        cmd_vel_msg.linear.y = float(arguments["linear_y"])
        cmd_vel_msg.linear.z = 0.0
        cmd_vel_msg.angular.x = 0.0
        cmd_vel_msg.angular.y = 0.0
        cmd_vel_msg.angular.z = float(arguments["angular_z"])
        self.cmd_vel_publisher.publish(cmd_vel_msg)
        time.sleep(arguments["duration"])
        cmd_vel_msg.linear.x = 0.0
        cmd_vel_msg.linear.y = 0.0
        cmd_vel_msg.angular.z = 0.0
        self.cmd_vel_publisher.publish(cmd_vel_msg)
        return "Success"

    def wag(self, arguments):
        print("Wag")
        cmd_vel_msg = Twist()
        cmd_vel_msg.linear.x = 0.0
        cmd_vel_msg.linear.y = 0.0
        cmd_vel_msg.linear.z = 0.0
        cmd_vel_msg.angular.x = 0.0
        cmd_vel_msg.angular.y = 0.0
        cmd_vel_msg.angular.z = 1.0
        for _ in range(10):
            self.cmd_vel_publisher.publish(cmd_vel_msg)
            time.sleep(0.1)
            cmd_vel_msg.angular.z *= -1
        cmd_vel_msg.angular.z = 0.0
        self.cmd_vel_publisher.publish(cmd_vel_msg)
        return "Success"

    def spin_around(self, arguments):
        print("Spin around")
        cmd_vel_msg = Twist()
        cmd_vel_msg.linear.x = 0.0
        cmd_vel_msg.linear.y = 0.0
        cmd_vel_msg.linear.z = 0.0
        cmd_vel_msg.angular.x = 0.0
        cmd_vel_msg.angular.y = 0.0
        cmd_vel_msg.angular.z = 2.0
        self.cmd_vel_publisher.publish(cmd_vel_msg)
        time.sleep(3.0)
        cmd_vel_msg.angular.z = 0.0
        self.cmd_vel_publisher.publish(cmd_vel_msg)
        return "Success"

    def say(self, arguments):
        print(f"Say: {arguments}")
        return "Success"

    def bark(self, arguments):
        print("Bark")
        return "Success"

    def wait(self, arguments):
        print("Wait")
        return "Success"

@retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
def chat_completion_request(messages, tools=prompt_utils.tools, tool_choice=None, model=GPT_MODEL):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="required",
        ).choices[0].message
        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e

def main(args=None):
    rclpy.init(args=args)
    node = LLMNode()
    rclpy.spin(node)

    rclpy.shutdown()

if __name__ == '__main__':
    main()
