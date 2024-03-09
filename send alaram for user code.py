from ultralytics import YOLO
import cv2
import cvzone
import math
import time
import telebot

# Create a Telegram bot
bot = telebot.TeleBot("6432095951:AAF-PF2M8xUAJ8fUGiabNszCWG3kbUJjmaA")
# Set the chat ID to send messages to
chat_id = "345855374"

# Load the YOLO model
model = YOLO("../Yolo-Weights/kaggle_n++.pt")
# Open the video stream
cap = cv2.VideoCapture("images/c3.mp4")

# Initialize the variables
classNames = ['UAV']
prev_frame_time = 0
new_frame_time = 0

# Loop through the video frames
while True:
    # Get the current frame
    success, img = cap.read()

    # Run YOLO on the frame
    results = model(img, stream=True, conf=0.5)

    # Iterate over the results
    for r in results:
        # Get the bounding boxes
        boxes = r.boxes

        # Iterate over the bounding boxes
        for box in boxes:
            # Get the coordinates of the bounding box
            x1, y1, x2, y2 = box.xyxy[0]

            # Convert the coordinates to integers
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Draw the bounding box on the image
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)

            # Calculate the confidence
            conf = math.ceil((box.conf[0] * 100)) / 100

            # Get the class name
            cls = int(box.cls[0])
            label = f'{classNames[cls]} {conf}%'

            # Put the label on the image
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.1, (255, 255, 0), 1)

    # Check if a drone was detected
    if len(boxes) > 0:
        # Send a message to Telegram
        bot.send_message(chat_id, f"تحذير: تم الكشف عن وجود طائرة مسيرة بنسبة  = {conf}%!")
    # Display the image
    cv2.imshow("Image", img)

    # Wait for a key press
    key = cv2.waitKey(1)

    # Quit if the user presses ESC
    if key == 27:
        break

# Close the video stream
cap.release()
# Close all windows
cv2.destroyAllWindows()
