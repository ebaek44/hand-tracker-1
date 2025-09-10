# gesture_data_handler
import time
import threading
from collections import deque, Counter
from typing import Iterable, Optional, Callable

class GestureHandler:
    """
    Created a thread-safe gesture state & emitter
    - Call update_gesture() every frame from your tracking thread
    - It will call start() once to create an internal worker that decides when to emit
    - Call close() when the app shuts down
    """
    # the hand_id that toggles play and pause
    TOGGLE_PLAY_PAUSE = {0}
    def __init__(
            self, 
            emit_fn: Callable[[dict], None],
            burst_window: float = 0.25,   # seconds for swipe majority
            swipe_cooldown: float = 0.35, # min gap between swipe emits
            lockout_secs: float = 0.30,   # block immediate opposite after a swipe
            sign_dwell: float = 0.35,     # how long hand sign must be held
            control_cooldown: float = 0.50, # min gap between control emits
            emit_event:str = "gesture", # socket.io event name
            loop_hz: float = 90.0 # the loop rate of the worker
        ) -> None:

         # live state (written by tracking thread)
        self._gesture_id = 0       # hand_sign_id
        self._gesture_duration = 0.0
        self._pointer_id = 0       # 0 stop, 1 right, 2 left

        # swipe buffers
        self._recent_ptr: deque[tuple[float,int]] = deque(maxlen=60)
        self._last_swipe_emit = 0.0
        self._lock_dir = 0         # +1 after right, -1 after left, 0 none
        self._lock_until = 0.0

        # control / sign
        self._last_control_emit = 0.0
        self._control_latched = False  # internal memory for toggle mode

        # params
        # NOTE: If sluggish lower swipe_cooldown or shorten burst_window
        # NOTE: Increase lockout_secs or swipe_cooldown if double swips
        # NOTE: If play/pause fires to early increase sign_dwell
        self.burst_window = burst_window
        self.swipe_cooldown = swipe_cooldown
        self.lockout_secs = lockout_secs
        self.sign_dwell = sign_dwell
        self.control_cooldown = control_cooldown
        self.loop_dt = 1.0 / float(loop_hz)
        self.emit_event = emit_event

        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._worker: Optional[threading.Thread] = None

        if emit_fn is None:
            raise ValueError("emit_fn is required in server mode")
        self.emit_fn = emit_fn

        
    # ----- Public API -----

    def start(self) -> None:
        """ This will start the worker if it isn't already running """
        if self._worker and self._worker.is_alive():
            return
        self._stop.clear()
        self._worker = threading.Thread(target=self._run, name="GestureHandlerWorker", daemon=True)
        self._worker.start()

    def close(self, timeout:float = 1.0) -> None:
        """ This will close the GestureHandler thread"""
        self._stop.set()
        if self._worker:
            self._worker.join(timeout=timeout)
        
        
    

    def update_gesture(self, hand_id: int, duration: float, pointer_id: int):
        """ This will update our current gesture and pointer ids while pushing onto the deque"""
        now = time.time()
        with self._lock:
            self._gesture_id = hand_id
            self._gesture_duration = duration
            self._pointer_id = pointer_id
            self._recent_ptr.append((now, self._pointer_id))
    
    # ----- Thread worker logic -----
    def _run(self) -> None:
        """
        This will try to send a "tick" to emit a sign
        Wakes up around 90 times a sec to read those values
        """
        while not self._stop.is_set():
            try:
                self._tick()
            except Exception as e:
                print("[GestureHandler] tick error:", e)
            time.sleep(self.loop_dt)

    def _tick(self) -> None:
        now = time.time()
        with self._lock:
            hand_id = self._gesture_id
            dur = self._gesture_duration

            self._prune_recent(now)

            # If its to toggle play and pause then we do that with priority (returns the payload to emniut)
            if hand_id in self.TOGGLE_PLAY_PAUSE:
                payload = self._maybe_emit_control(now, dur, hand_id)
                
            else:
                payload = self._maybe_emit_swipe(now)
        
        if payload:
            self._emit(payload)

    # ----- Control Helpers -----
    def _maybe_emit_control(self, now: float, dur: float, hand_id: int):
        """ This will help emit the payload to the front end for play pause"""
        if dur >= self.sign_dwell and not self._control_latched and (now - self._last_control_emit) >= self.control_cooldown and hand_id in self.TOGGLE_PLAY_PAUSE:
            payload = {"type": "control", "value": "toggle_play_pause"}
            self._last_control_emit = now
            self._control_latched = True
            return payload

        if dur < 0.05 or hand_id not in self.TOGGLE_PLAY_PAUSE:
            self._control_latched = False
        return None

    
    def _maybe_emit_swipe(self, now: float):
        if not self._recent_ptr:
            return None
        
        locked_dir = self._lock_dir if now < self._lock_until else 0

        ptr_vals = [p for (_, p) in self._recent_ptr]
        winner = self._majority_nonzero(ptr_vals)

        if winner == 0:
            return None
        dir_sign = +1 if winner == 1 else -1
        if locked_dir != 0 and dir_sign == -locked_dir:
            return None
        if (now - self._last_swipe_emit) < self.swipe_cooldown:
            return None

        payload = {"type": "swipe", "value": "right" if dir_sign == +1 else "left"}
        self._last_swipe_emit = now
        self._lock_dir = dir_sign
        self._lock_until = now + self.lockout_secs
        return payload

    # ------ Swipe Helpers ------

    def _prune_recent(self, now: float) -> None:
        # It will make sure all the recent gestures are inside the burst window
        cutoff = now - self.burst_window
        while self._recent_ptr and self._recent_ptr[0][0] < cutoff:
            self._recent_ptr.popleft()

    @staticmethod
    def _majority_nonzero(values: Iterable[int]) -> int:
        # this will pick the "winner" for the swipe
        vals = [v for v in values if v in (1, 2)]
        if not vals:
            return 0
        if len(vals) < 3:
            return 0
        c = Counter(vals)
        best_n = max(c.values())
        candidates = [v for v, n in c.items() if n == best_n]  # [1], [2], or [1,2]
        if len(candidates) == 1:
            return candidates[0]
        # tie → prefer the most recent nonzero in the window
        for v in reversed(vals):
            if v in candidates:
                return v
        return 0

    # ---- Emission to Front End -----

    def _emit(self, data: dict) -> None:
        # Will try to emit to front end
        try:
            self.emit_fn(data) 
        except Exception as e:
            print("[GestureHandler] emit failed:", e)

    def process_gesture(self):
        """ Unneeded now because the worker will handle emitting"""
        return None