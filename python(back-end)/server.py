from flask import Flask
from flask_socketio import SocketIO, emit
from shared_data import GestureData
from hand_tracker import app
import threading
import time

gesture_data = GestureData

def run_hand_tracker():
    app.hand_tracker(gesture_data)

if __name__ == "__main__":
    run_hand_tracker()