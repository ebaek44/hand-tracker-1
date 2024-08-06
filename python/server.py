from flask import Flask, request, jsonify
from flask_socketio import SocketIO
from handlers.gesture_data_handler import GestureHandler

app = Flask(__name__)
socketio = SocketIO(app)
gesture_handler = GestureHandler()


@app.route('api/gesture', methods=["POST"])
def handle_gesture():
    data = request.json
    hand_id = data.get('hand_id')
    duration = data.get('duration')
    pointer_id = data.get('pointer_id')

    gesture_handler.update_gesture(hand_id, duration, pointer_id)
    response = gesture_handler.process_gesture()

    if response:
        socketio.emit('gesture_data', response)
        return jsonify({"status": "success", "data": response})
    else:
        return jsonify({"status": "failure", "message": "Gesture duration less than 3"}), 400
    

if __name__ == "__main__":
    socketio.run(app, debug=True)

"""
Add to frontend
<!-- Add this to your HTML file -->
<script src="/socket.io/socket.io.js"></script>
<script>
  const socket = io('http://localhost:5000');

  socket.on('gesture_data', (data) => {
    console.log('Received gesture data:', data);
    // Handle the received data and update the UI
  });
</script>

### Make sure to look at how to change it to be in react js format (it is in html format rn)
"""

"""
NEXT STEPS
See if I need to make the change from using https requests between the flask server and the hand_tracker 
function to using socketio and then find out how to simoutanoesly run the hand_tracker and the flask server
"""