CMD_X_VEL_LIMIT = 0.75
CMD_Y_VEL_LIMIT = 0.5
CMD_YAW_VEL_LIMIT = 2.0
IMAGE_PERIOD = 15

system_message = {
    "role": "system",
    "content": [
    {
        "type": "text",
        "text": f"You are Pupper, a small, cute, and friendly robot puppy. You can receive verbal commands from humans, and you also receive images from your camera every {IMAGE_PERIOD} seconds. If you do not receive a user command, you should act autonomously. Do your best to behave like a real puppy! Important note: when you use the say tool, you will hear the message spoken out loud. Don't get mixed up between the human voice and your own voice! If the last user command is similar to your last say command, you're probably just hearing your own voice and you should disregard it.",
    }
    ]
}

tools = [
    {
        "type": "function",
        "function": {
            "name": "walk",
            "description": "Moves the robot by an xy linear and yaw angular velocity for a specified duration.",
            "parameters": {
                "type": "object",
                "properties": {
                    "linear_x": {
                        "type": "number",
                        "description": f"The linear velocity along the x-axis in m/s (positive is forwards, negative is backwards). Must be between -{CMD_X_VEL_LIMIT} and {CMD_X_VEL_LIMIT}.",
                    },
                    "linear_y": {
                        "type": "number",
                        "description": f"The linear velocity along the y-axis in m/s (positive is left, negative is right). Must be between -{CMD_Y_VEL_LIMIT} and {CMD_Y_VEL_LIMIT}.",
                    },
                    "angular_z": {
                        "type": "number",
                        "description": f"The angular velocity around the z-axis in rad/s (positive is turning left, negative is turning right). Must be between -{CMD_YAW_VEL_LIMIT} and {CMD_YAW_VEL_LIMIT}.",
                    },
                    "duration": {
                        "type": "number",
                        "description": "The duration of the movement in seconds.",
                    },
                },
                "required": ["linear_x", "linear_y", "angular_z", "duration"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "say",
            "description": "Speaks a message out loud. Without calling this function, the robot will not speak.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The message to speak.",
                    },
                },
                "required": ["message"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bark",
            "description": "Makes a barking sound.",
            "parameters": {},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wag",
            "description": "Wags the robot's tail.",
            "parameters": {},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spin_around",
            "description": "Spins the robot around in place.",
            "parameters": {},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wait",
            "description": f"Waits for the next {IMAGE_PERIOD} seconds.",
            "parameters": {},
        },
    }
]