
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load ASL model
model = load_model('catdogmodel.h5')

# Load class labels
with open('labels.names', 'r') as f:
    labels = f.read().split('\n')

# Open the camera
cam = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_data=cv2.resize(frame_rgb,(150,150))
    input_data = np.expand_dims(input_data, axis=0)

    prediction = model.predict(input_data)
    print(prediction)
    class_id = np.argmax(prediction)
    class_name = labels[class_id]

    # Display the result on the frame
    cv2.putText(frame, class_name, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the frame
    cv2.imshow("Dog Cat Recognition", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release resources
cam.release()
cv2.destroyAllWindows()
