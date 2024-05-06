import os  # Importing the os module for operating system functionalities
import cv2  # Importing OpenCV library for image processing
import numpy as np  # Importing numpy library for numerical computations
import tkinter as tk  # Importing tkinter library for GUI
from tkinter import ttk, colorchooser, filedialog, messagebox  # Importing specific modules/classes from tkinter

# Function to process the video file
def process_video(video_file):
    global cap, video_output, pause, show_confidences, bounding_box_color, object_count, font_scale
    pause = False
    object_count = 0

    # Open the video file
    cap = cv2.VideoCapture(video_file)
    # Create a VideoWriter object to save the processed video
    video_output = cv2.VideoWriter('output.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    # Loop through the video frames
    while cap.isOpened():
        if not pause:
            ret, frame = cap.read()  # Read a frame from the video
            if not ret:
                break

            height, width, _ = frame.shape  # Get the dimensions of the frame
            # Preprocess the frame for object detection
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (224, 224), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outs = net.forward(output_layers)

            class_ids = []
            confidences = []
            boxes = []

            # Loop through the detected objects
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > confidence_threshold:
                        # Calculate the bounding box coordinates
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            # Apply non-maximum suppression to remove overlapping boxes
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
            object_count = len(indexes)
            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    if class_ids[i] < len(classes):
                        label = str(classes[class_ids[i]])
                        color = bounding_box_color
                        # Draw bounding box and label on the frame
                        cv2.rectangle(frame, (x, y), (x + w, y + h), tuple(color), 2)
                        if show_confidences:
                            cv2.putText(frame, label, (x, y + 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)
                            cv2.putText(frame, f'{confidences[i] * 100:.2f}%', (x, y + 60), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)

            # Display object count on the frame
            cv2.putText(frame, f'Objects: {object_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)
            video_output.write(frame)  # Write the frame to the output video
            cv2.imshow('frame', frame)  # Display the frame

        key = cv2.waitKey(1)
        if key == 27:
            break
        elif key == 32:
            pause = not pause

    cap.release()  # Release the video capture object
    video_output.release()  # Release the video writer object
    cv2.destroyAllWindows()  # Close all OpenCV windows

# Function to update parameters based on user input
def update_parameters():
    global confidence_threshold, nms_threshold, show_confidences, bounding_box_color, font_scale
    if not video_file_path:
        messagebox.showwarning("Warning", "Please select a video file.")
        return
    confidence_threshold = float(confidence_entry.get())
    nms_threshold = float(nms_entry.get())
    show_confidences = show_confidences_var.get()
    bounding_box_color = eval(bounding_box_color_var.get())
    font_scale = float(font_scale_entry.get())
    print("Updated parameters:")
    print("Confidence Threshold:", confidence_threshold)
    print("NMS Threshold:", nms_threshold)
    print("Show Confidences:", show_confidences)
    print("Bounding Box Color:", bounding_box_color)
    print("Font Scale:", font_scale)
    root.destroy()  # Close the tkinter window
    process_video(video_file_path)  # Process the video with updated parameters

# Function to choose a color using tkinter colorchooser
def choose_color():
    color_hex = colorchooser.askcolor()[1]  # Get the selected color in hexadecimal format
    if color_hex:
        # Convert hexadecimal color to RGB tuple
        color_rgb = tuple(int(color_hex[i:i+2], 16) for i in (1, 3, 5))
        # Ensure values are within 0-255 range
        color_rgb = tuple(min(max(0, c), 255) for c in color_rgb)
        bounding_box_color_var.set(str(color_rgb))

# Function to choose a video file using tkinter filedialog
def choose_video_file():
    global video_file_path
    initial_dir = os.path.dirname(__file__)
    video_file_path = filedialog.askopenfilename(initialdir=initial_dir)
    if video_file_path:
        video_file_label.config(text=f"Selected Video: {video_file_path}")

# Function to confirm closing the application
def confirm_close():
    if messagebox.askokcancel("Quit", "Do you want to quit the application?"):
        root.destroy()  # Close the tkinter window

# Load YOLO model and classes
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
output_layers = net.getUnconnectedOutLayersNames()

# Define the codec for saving the video
fourcc = cv2.VideoWriter_fourcc(*'XVID')

# Default parameter values
confidence_threshold = 0.5
nms_threshold = 0.4
show_confidences = True
bounding_box_color = (0, 255, 0)
font_scale = 1.0
video_file_path = None

# Create tkinter window
root = tk.Tk()
root.title("Parameters Adjustment")
root.geometry("400x400")

# Widgets for selecting video file
video_file_label = tk.Label(root, text="Selected Video: None")
video_file_label.grid(row=0, column=0, columnspan=2, padx=5, pady=5)
video_file_button = ttk.Button(root, text="Choose Video File", command=choose_video_file)
video_file_button.grid(row=1, column=0, columnspan=2, padx=5, pady=5)

# Widgets for adjusting parameters
confidence_label = tk.Label(root, text="Confidence Threshold:")
confidence_label.grid(row=2, column=0, padx=5, pady=5)
confidence_entry = ttk.Entry(root)
confidence_entry.insert(0, str(confidence_threshold))
confidence_entry.grid(row=2, column=1, padx=5, pady=5)

nms_label = tk.Label(root, text="NMS Threshold:")
nms_label.grid(row=3, column=0, padx=5, pady=5)
nms_entry = ttk.Entry(root)
nms_entry.insert(0, str(nms_threshold))
nms_entry.grid(row=3, column=1, padx=5, pady=5)

show_confidences_var = tk.BooleanVar()
show_confidences_var.set(True)
show_confidences_check = ttk.Checkbutton(root, text="Show Confidences", variable=show_confidences_var)
show_confidences_check.grid(row=4, columnspan=2, padx=5, pady=5)

bounding_box_color_var = tk.StringVar()
bounding_box_color_var.set('(0, 255, 0)')
bounding_box_color_button = ttk.Button(root, text="Choose Bounding Box Color", command=choose_color)
bounding_box_color_button.grid(row=5, columnspan=2, padx=5, pady=5)
color_label = tk.Label(root, text="Current Color:")
color_label.grid(row=6, column=0, padx=5, pady=5)
color_display = tk.Label(root, textvariable=bounding_box_color_var)
color_display.grid(row=6, column=1, padx=5, pady=5)

font_scale_label = tk.Label(root, text="Font Scale:")
font_scale_label.grid(row=7, column=0, padx=5, pady=5)
font_scale_entry = ttk.Entry(root)
font_scale_entry.insert(0, str(font_scale))
font_scale_entry.grid(row=7, column=1, padx=5, pady=5)

# Button to update parameters
update_button = ttk.Button(root, text="Update Parameters", command=update_parameters)
update_button.grid(row=8, columnspan=2, padx=5, pady=5)

# Bind closing event to confirm_close function
root.protocol("WM_DELETE_WINDOW", confirm_close)

# Run the tkinter event loop
root.mainloop()
