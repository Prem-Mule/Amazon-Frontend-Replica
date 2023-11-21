import cv2
import numpy as np

# Load the virtual hat, shirt, and pant images
hat = cv2.imread('hat.jpg', -1)
shirt = cv2.imread('shirt.png', -1)
pant = cv2.imread('pant.png', -1)
# Load the face cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
upper_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')
lower_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_lowerbody.xml')

def overlay_image(background, overlay, x, y, h_target):
    h, w = overlay.shape[0:2]

    # Resize the overlay image to match the target height
    scale_factor = h_target / h
    overlay = cv2.resize(overlay, (int(w * scale_factor), int(h_target)))

    # If the overlay image has an alpha channel, use it
    if overlay.shape[2] == 4:
        alpha = overlay[:, :, 3] / 255.0
        for c in range(0, 3):
            background[y:y+h_target, x:x+overlay.shape[1], c] = (
                (1 - alpha) * background[y:y+h_target, x:x+overlay.shape[1], c] +
                alpha * overlay[:, :, c]
            )
    else:
        # If no alpha channel, simply overlay the RGB channels onto the background
        background[y:y+h_target, x:x+overlay.shape[1], 0:3] = overlay[:, :, 0:3]

# Open the camera
cap = cv2.VideoCapture(0)

while True:
    # Capture the frame
    ret, frame = cap.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    upper_bodies = upper_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    lower_bodies = lower_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    total_detections = len(faces) + len(upper_bodies) + len(lower_bodies)
    
    # Calculate confidence level only if there are detections
    face_confidence = len(faces) / total_detections if total_detections > 0 else 0

    print(f"Face Confidence: {face_confidence:.2f}")

    for (x, y, w, h) in faces:
        # Overlay the hat on the face
        overlay_image(frame, hat, x, y - int(h / 4), int(h / 2))

    if face_confidence > 0.5:  # Show virtual clothing if face is confidently detected
        for (x, y, w, h) in upper_bodies:
            # Overlay the shirt on the upper body
            overlay_image(frame, shirt, x, y, int(h * 0.8))  # Adjust the scaling factor as needed

        for (x, y, w, h) in lower_bodies:
            # Overlay the pant on the lower body
            overlay_image(frame, pant, x, y, int(h * 1.2))  # Adjust the scaling factor as needed

    # Display the frame
    cv2.imshow('Virtual Dressing Room', frame)

    # Exit the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
