import numpy as np

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from geometry_msgs.msg import Twist

import cv2
from cv_bridge import CvBridge
import base64

import openai
from openai import OpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt

import pyaudio
import samplerate as sr
import numpy as np

import time

import prompt_utils
import json

GPT_MODEL = "gpt-4o"
client = OpenAI()
resampler = sr.Resampler()

player_stream = pyaudio.PyAudio().open(format=pyaudio.paInt16, channels=1, rate=48000, output=True, output_device_index=0)

class LLMNode(Node):

    def __init__(self):
        super().__init__('llm_node')
        self.image_subscriber = self.create_subscription(Image, '/image', self.image_callback, 1)
        self.depth_subscriber = self.create_subscription(Image, '/depth', self.depth_callback, 1)
        self.transcription_subscriber = self.create_subscription(String, '/transcription', self.transcription_callback, 1)
        self.timer = self.create_timer(prompt_utils.IMAGE_PERIOD, self.timer_callback)
        self.image = None
        self.depth = None
        self.bridge = CvBridge()
        self.messages = []
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)

    def image_callback(self, msg):
        self.image = msg
    
    def depth_callback(self, msg):
        self.depth = msg

    def transcription_callback(self, msg):
        print(f"Transcription: {msg.data}")
        self.call_llm(self.image, msg.data)
        # self.messages.append({"role": "user", "content": [{"type": "text", "text": msg.data}]})
        self.most_recent_message = msg.data
    
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
        # self.messages.pop(-1)
        if self.messages[-1]["role"] == "user":
            self.messages[-1]["content"] = [content_item for content_item in self.messages[-1]["content"] if content_item["type"] != "image"]# else {"type": "text", "text": "(old image)"}]
        # remove all images from messages
        # self.messages = [message for message in self.messages if not any(content["type"] == "image_url" for content in message["content"])]
        self.messages.append(response.dict())
        if response.tool_calls:
            tool_fn = None
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
                result = ""
                if tool_fn:
                    result = tool_fn(json.loads(tool_call.function.arguments))
                self.messages.append({"role": "tool", "tool_call_id": tool_call.id, "name": tool_call.function.name, "content": result})

    def walk(self, arguments):
        print(f"Walk: {arguments}")
        depth_np = np.frombuffer(self.depth.data, dtype=np.uint8)
        if arguments["linear_x"] > 0.0 and np.histogram(depth_np)[0][-1] > 100:
            return "Obstacle detected, can't move forward"
        cmd_vel_msg = Twist()
        cmd_vel_msg.linear.x = float(arguments["linear_x"])
        cmd_vel_msg.linear.y = float(arguments["linear_y"])
        cmd_vel_msg.linear.z = 0.0
        cmd_vel_msg.angular.x = 0.0
        cmd_vel_msg.angular.y = 0.0
        cmd_vel_msg.angular.z = float(arguments["angular_z"])
        self.cmd_vel_publisher.publish(cmd_vel_msg)
        # time.sleep(arguments["duration"])
        self.get_clock().sleep_for(Duration(seconds=float(arguments["duration"])))
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
            # time.sleep(0.1)
            self.get_clock().sleep_for(Duration(seconds=0.1))
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
        # time.sleep(3.0)
        self.get_clock().sleep_for(Duration(seconds=3.0))
        cmd_vel_msg.angular.z = 0.0
        self.cmd_vel_publisher.publish(cmd_vel_msg)
        return "Success"

    def say(self, arguments):
        print(f"Say: {arguments["message"]}")
        speak(arguments["message"])
        return "Success"

    def bark(self, arguments):
        print("Bark")
        player_stream.write(open("/home/pi/bark.wav", "rb").read())
        return "Success"

    def wait(self, arguments):
        print("Wait")
        return "Success"

@retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
def chat_completion_request(messages, tools=prompt_utils.tools, tool_choice=None, model=GPT_MODEL):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[prompt_utils.system_message] + messages,
            tools=tools,
            tool_choice="required",
        ).choices[0].message
        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e

def speak(text):
    with openai.audio.speech.with_streaming_response.create(
            model="tts-1",
            voice="alloy",
            response_format="pcm",  # similar to WAV, but without a header chunk at the start.
            input=text,
        ) as response:
            for chunk in response.iter_bytes(chunk_size=1024):
                resampled_chunk = resampler.process(np.frombuffer(chunk, dtype=np.int16), 2.0)
                resampled_chunk = resampled_chunk.astype(np.int16).tobytes()
                player_stream.write(resampled_chunk)

def main(args=None):
    rclpy.init(args=args)
    node = LLMNode()
    rclpy.spin(node)

    rclpy.shutdown()

if __name__ == '__main__':
    main()
