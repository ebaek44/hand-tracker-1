# shared_data.py


class GestureData:
    def __init__(self) -> None:
        self.gesture_id = None
        self.gesture_duration = 0.0
        self.finger_id = None

    def update_gesture(self, gesture_id, duration, finger_id):
        if gesture_id == 2:
            self.finger_id = finger_id
        else:
            self.finger_id = None
        self.gesture_id = gesture_id
        self.gesture_duration = duration

    def get_gesture_info(self):
        if self.finger_id is None:
            return self.gesture_id, self.gesture_duration
        else:
            return self.gesture_id, self.gesture_duration, self.finger_id
