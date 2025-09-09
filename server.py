# server.py
import signal
import sys
import time

from handlers.gesture_data_handler import GestureHandler
from hand_tracker.app import hand_tracker  # wherever your function lives

RUN_SOCKET_URL = "http://localhost:3000"  # must match your Next.js server
RUN_NAMESPACE = "/gestures"

def main():
    handler = GestureHandler(
        sio_url=RUN_SOCKET_URL,
        sio_namespace=RUN_NAMESPACE,
        burst_window=0.25,
        swipe_cooldown=1.5,
        lockout_secs=1.0,
        sign_dwell=0.35,
        control_cooldown=1.0,
        loop_hz=90.0,
        emit_event="gesture",
    )

    handler.start()
    
    # Graceful shutdown on SIGTERM from Node
    def _shutdown(_sig, _frm):
        print("[server.py] shutting down...")
        handler.close()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)
    
    # Run the hand tracker loop (blocks until user quits)
    try:
        hand_tracker(handler)
    finally:
        handler.close()
        # small delay to flush any final emits
        time.sleep(0.1)


if __name__ == "__main__":
    main()
