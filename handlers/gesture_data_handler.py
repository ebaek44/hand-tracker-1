# gesture_data_handler
import json
import requests
import time
import threading

class GestureHandler:
    """
    Created a thread-safe gesture state & emitter
    """
    def __init__(self) -> None:
        self._gesture_id = None
        self._gesture_duration = 0
        self._pointer_id = None

    @property
    def gesture_id(self):
        return self._gesture_id
    
    @property
    def gesture_duration(self):
        return self._gesture_duration
    
    @property
    def pointer_id(self):
        return self._pointer_id
    
    @staticmethod
    def _send_gesture_data(data):
        url = "https://localhost:3000/api/gesture"
        headers = {'Content-Type': 'application/json'}
        response = requests.post(url, headers=headers, data=data)
        return response
    
    def update_gesture(self, hand_id, duration, pointer_id):
        self._gesture_id = hand_id
        self._gesture_duration = duration
        self._pointer_id = pointer_id

    def process_gesture(self):
        if self._gesture_duration >= 3:
            send_data = {
                'hand_id': self._gesture_id,
                'pointer_id': self._pointer_id
            }
            send_data_json = json.dumps(send_data)
            response = self._send_gesture_data(send_data_json)
            if response.status_code == 200:
                return response
            else:
                return {"status": "failure", "message": response.text}
        return None

        