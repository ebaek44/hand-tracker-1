# server.py — HTTP-only (no eventlet, no socketio)
import signal, sys, time, os
from threading import Lock, Thread
from flask import Flask, jsonify
from flask_cors import CORS

from handlers.gesture_data_handler import GestureHandler
from hand_tracker.app import hand_tracker

HOST = "0.0.0.0"
PORT = 3333

flask_app = Flask(__name__)
CORS(flask_app, origins="*")

_latest = {"payload": None, "ts": 0.0}
_LOCK = Lock()

@flask_app.get("/poll-gestures")
def poll_gestures():
    now = time.time()
    with _LOCK:
        p, ts = _latest["payload"], _latest["ts"]
        if p and (now - ts) < 2.0:
            data = p
            _latest["payload"] = None  # one-shot
            print(f"[http] Serving gesture to extension: {data}")
            return jsonify(data)
    return jsonify({"type": "none"})

@flask_app.get("/health")
def health():
    return jsonify({"status": "ok", "server": "gesture-server"})

def main():
    def emit_gesture(payload: dict):
        print(f"[server] Broadcasting gesture: {payload}")
        with _LOCK:
            _latest["payload"] = payload
            _latest["ts"] = time.time()

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

    # Run tracker headless in a thread (for training, run app.py directly)
    Thread(target=hand_tracker, args=(handler,), daemon=True).start()

    def _shutdown(_sig, _frm):
        print("[server.py] shutting down...")
        try:
            handler.close()
        finally:
            time.sleep(0.1)
            sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    print(f"[server.py] HTTP server running on http://{HOST}:{PORT}")
    print(f"[server.py] HTTP API: http://{HOST}:{PORT}/poll-gestures")
    flask_app.run(host=HOST, port=PORT, threaded=True)

if __name__ == "__main__":
    main()
