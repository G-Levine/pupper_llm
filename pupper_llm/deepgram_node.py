import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from deepgram import DeepgramClient, LiveTranscriptionEvents, LiveOptions, Microphone

class DeepgramNode(Node):
    def __init__(self):
        super().__init__('deepgram_node')
        self.transcription_publisher = self.create_publisher(String, '/transcription', 10)
        self.latest_transcription = ""

        self.deepgram = DeepgramClient()
        self.dg_connection = self.deepgram.listen.live.v("1")
        
        # Set up event listeners
        self.dg_connection.on(LiveTranscriptionEvents.Transcript, self.create_callback(self.on_message))
        self.dg_connection.on(LiveTranscriptionEvents.UtteranceEnd, self.create_callback(self.on_utterance_end))

        # Start the connection
        options = LiveOptions(
            model="nova-2-conversationalai",
            punctuate=True,
            language="en-US",
            encoding="linear16",
            channels=1,
            sample_rate=48000,
            interim_results=True,
            utterance_end_ms="1000",
            vad_events=True,
            keywords=["Pupper"],
            diarize=True,
            endpointing="1000",
        )
        self.dg_connection.start(options)

        # Create and start the microphone
        # self.microphone = Microphone(self.dg_connection.send)
        self.microphone = Microphone(self.dg_connection.send, rate=48000, input_device_index=1, channels=1)
        self.microphone.start()

    def create_callback(self, func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper

    def on_message(self, *args, **kwargs):
        result = kwargs.get('result')
        sentence = result.channel.alternatives[0].transcript
        if len(sentence) == 0:
            return
        self.latest_transcription = sentence
        self.get_logger().info(f"Partial Transcription: {sentence}")
        if result.speech_final:
            msg = String()
            self.get_logger().info(f"Transcription: {sentence}")
            msg.data = self.latest_transcription
            self.transcription_publisher.publish(msg)
            self.latest_transcription = ""

    def on_utterance_end(self, *args, **kwargs):
        utterance_end = kwargs.get('utterance_end')
        self.get_logger().info(f"Utterance ended: {utterance_end}")
        if not self.latest_transcription:
            return
        msg = String()
        msg.data = self.latest_transcription
        self.transcription_publisher.publish(msg)
        self.latest_transcription = ""

def main(args=None):
    rclpy.init(args=args)
    deepgram_node = DeepgramNode()

    try:
        rclpy.spin(deepgram_node)
    except KeyboardInterrupt:
        pass

    deepgram_node.microphone.finish()
    deepgram_node.dg_connection.finish()
    deepgram_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
