# server.py 
# Uses a hybrid of socket.io (to work with backend and threading) and flask to communicate with front end
import eventlet


import signal
import sys
import time
from threading import Lock
import threading
import socketio
from flask import Flask, jsonify, request
from flask_cors import CORS
from handlers.gesture_data_handler import GestureHandler
from hand_tracker.app import hand_tracker

HOST = "0.0.0.0"
PORT = 3333
NAMESPACE = "/gestures"

# Create Flask app for HTTP API
flask_app = Flask(__name__)
CORS(flask_app, origins="*") 

# Create Socket.IO server
sio = socketio.Server(
    async_mode="eventlet", 
    cors_allowed_origins="*",
    cors_credentials=False
)

# combine Flask + Socket.IO
app = socketio.WSGIApp(sio, flask_app)

# Shared state for the latest gesture
latest_gesture = {"payload": None, "timestamp": 0}
gesture_lock = Lock()

@sio.on("connect", namespace=NAMESPACE)
def on_connect(sid, environ):
    print(f"[sio] connected: {sid}")
    sio.emit("gesture", {"type": "control", "value": "connection_test"},
             namespace=NAMESPACE, to=sid)

@sio.on("disconnect", namespace=NAMESPACE)
def on_disconnect(sid):
    print(f"[sio] disconnected: {sid}")

# HTTP API endpoint for Chrome extension polling
@flask_app.route('/poll-gestures', methods=['GET'])
def poll_gestures():
    """Simple HTTP endpoint that Chrome extension can poll"""
    with gesture_lock:
        current_time = time.time()
        
        # only return gesture if it's recent (within last 2 seconds)
        if (latest_gesture["payload"] and 
            current_time - latest_gesture["timestamp"] < 2.0):
            
            gesture_data = latest_gesture["payload"]
            # Clear it so we don't send the same gesture multiple times
            latest_gesture["payload"] = None
            
            print(f"[http] Serving gesture to extension: {gesture_data}")
            return jsonify(gesture_data)
        
        # No recent gesture
        return jsonify({"type": "none"})
# Tells us if the health is ok (not needed)
@flask_app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "ok", "server": "gesture-server"})



def main():
    # This will be used to emit from gesture handler
    def emit_gesture(payload: dict):
        print(f"[server] Broadcasting gesture: {payload}")
        
        # Store for HTTP polling
        with gesture_lock:
            latest_gesture["payload"] = payload
            latest_gesture["timestamp"] = time.time()
        
        # Also broadcast via Socket.IO for any web clients
        sio.emit("gesture", payload, namespace=NAMESPACE)
  
    handler = GestureHandler(
        emit_fn=emit_gesture,
        burst_window=0.25,
        swipe_cooldown=1.5,
        lockout_secs=1.0,
        sign_dwell=0.35,
        control_cooldown=1.0,
        loop_hz=90.0,
        emit_event="gesture",
    )
    
    handler.start()
    
    # Shutdown handler (safely shuts down the handler)
    def _shutdown(_sig, _frm):
        print("[server.py] shutting down...")
        try:
            handler.close()
        finally:
            time.sleep(0.1)
            sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)
    
    # Run the hand tracker loop with the app.py in as the subthread
    threading.Thread(target=hand_tracker, args=(handler,), daemon=True).start()
    
    print(f"[server.py] Hybrid server running on http://{HOST}:{PORT}")
    print(f"[server.py] Socket.IO: http://{HOST}:{PORT}{NAMESPACE}")
    print(f"[server.py] HTTP API: http://{HOST}:{PORT}/poll-gestures")

    # Run server
    eventlet.wsgi.server(eventlet.listen((HOST, PORT)), app)

if __name__ == "__main__":
  main()