import cv2
import matplotlib.pyplot as plt

# Initialize the camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
else:
    print("Webcam opened successfully. Press 'q' to quit.")

plt.ion()  # Turn on interactive mode for Matplotlib
fig, ax = plt.subplots()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Convert the frame from BGR to RGB for Matplotlib
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    ax.imshow(frame_rgb)
    plt.draw()
    plt.pause(0.001)
    ax.clear()  # Clear the frame after each capture to update

    # Exit if 'q' is pressed
    if plt.waitforbuttonpress(0.001) and plt.get_current_fig_manager().canvas.key_press_handler_id == ord('q'):
        break

cap.release()
plt.close()

