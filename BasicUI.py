import os
import sys
import cv2
from ultralytics import YOLO
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import threading
import torch  # Add this import

# Define path to model and other user variables
model_path = 'my_model.pt'  # Path to model
min_thresh = 0.50  # Minimum detection threshold
cam_index = 0  # Index of USB camera
imgW, imgH = 1280, 720  # Resolution to run USB camera at

# Create dictionary to hold info about recycling suggestions
recycling_suggestions = {
    'plastic': "Recycle this plastic item in your local recycling bin. Consider reusing it if possible.",
    'cardboard': "Flatten this cardboard and place it in your recycling bin. Ensure it's clean and dry."
}

# Set bounding box colors (using the Tableu 10 color scheme)
bbox_colors = {
    'plastic': (0, 255, 0),  # Green for plastic
    'cardboard': (0, 0, 255)  # Red for cardboard
}

# Check if model file exists and is valid
if not os.path.exists(model_path):
    print('WARNING: Model path is invalid or model was not found.')
    sys.exit()

# Load the model into memory and get label map
model = YOLO(model_path, task='detect')

# Fix: Load the state dictionary with strict=False
try:
    checkpoint = torch.load(model_path)
    model.model.load_state_dict(checkpoint['model'].float().state_dict(), strict=False)
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit()

labels = model.names

# Initialize camera
cap = None
is_running = False

# Function to initialize the camera
def initialize_camera():
    global cap
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        messagebox.showerror("Error", "Unable to access the camera. Please check the camera index.")
        return False
    cap.set(3, imgW)
    cap.set(4, imgH)
    return True

# Function to process an image and display results
def process_image(image_path):
    global current_frame
    frame = cv2.imread(image_path)
    if frame is None:
        messagebox.showerror("Error", "Unable to read the image file.")
        return

    # Run inference on the image
    results = model.track(frame, verbose=False)

    # Extract results
    detections = results[0].boxes

    # Initialize variable to hold every material detected in this frame
    materials_detected = []

    # Go through each detection and get bbox coords, confidence, and class
    for i in range(len(detections)):
        # Get bounding box coordinates
        xyxy_tensor = detections[i].xyxy.cpu()
        xyxy = xyxy_tensor.numpy().squeeze()
        xmin, ymin, xmax, ymax = xyxy.astype(int)

        # Get bounding box class ID and name
        classidx = int(detections[i].cls.item())
        classname = labels[classidx]

        # Get bounding box confidence
        conf = detections[i].conf.item()

        # Draw box if confidence threshold is high enough
        if conf > min_thresh:
            # Draw box around object
            color = bbox_colors.get(classname, (255, 255, 255))
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

            # Draw label for object
            label = f'{classname}: {int(conf * 100)}%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_ymin = max(ymin, labelSize[1] + 10)
            cv2.rectangle(frame, (xmin, label_ymin - labelSize[1] - 10), (xmin + labelSize[0], label_ymin + baseLine - 10), color, cv2.FILLED)
            cv2.putText(frame, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            # Add object to list of detected materials
            materials_detected.append(classname)

    # Process list of materials that have been detected to provide recycling suggestions
    suggestions_text = ""
    for material in materials_detected:
        suggestion = recycling_suggestions.get(material, "No suggestion available for this material.")
        suggestions_text += f"Detected {material}: {suggestion}\n"

    # Update the suggestions text box
    suggestions_box.config(state=tk.NORMAL)
    suggestions_box.delete(1.0, tk.END)
    suggestions_box.insert(tk.END, suggestions_text)
    suggestions_box.config(state=tk.DISABLED)

    # Convert frame to RGB and display in the GUI
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    img.thumbnail((imgW, imgH))  # Resize image to fit within the display area
    img_tk = ImageTk.PhotoImage(image=img)
    video_label.img_tk = img_tk  # Keep a reference to avoid garbage collection
    video_label.config(image=img_tk)

# Function to start the camera feed
def start_camera():
    global is_running, cap
    if cap is None or not cap.isOpened():
        if not initialize_camera():
            return  # Exit if camera initialization fails
    is_running = True
    while is_running:
        ret, frame = cap.read()
        if not ret:
            messagebox.showerror("Error", "Unable to read frames from the camera.")
            break

        # Run inference on frame with tracking enabled
        results = model.track(frame, verbose=False)

        # Extract results
        detections = results[0].boxes

        # Initialize variable to hold every material detected in this frame
        materials_detected = []

        # Go through each detection and get bbox coords, confidence, and class
        for i in range(len(detections)):
            # Get bounding box coordinates
            xyxy_tensor = detections[i].xyxy.cpu()
            xyxy = xyxy_tensor.numpy().squeeze()
            xmin, ymin, xmax, ymax = xyxy.astype(int)

            # Get bounding box class ID and name
            classidx = int(detections[i].cls.item())
            classname = labels[classidx]

            # Get bounding box confidence
            conf = detections[i].conf.item()

            # Draw box if confidence threshold is high enough
            if conf > min_thresh:
                # Draw box around object
                color = bbox_colors.get(classname, (255, 255, 255))
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

                # Draw label for object
                label = f'{classname}: {int(conf * 100)}%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                label_ymin = max(ymin, labelSize[1] + 10)
                cv2.rectangle(frame, (xmin, label_ymin - labelSize[1] - 10), (xmin + labelSize[0], label_ymin + baseLine - 10), color, cv2.FILLED)
                cv2.putText(frame, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                # Add object to list of detected materials
                materials_detected.append(classname)

        # Process list of materials that have been detected to provide recycling suggestions
        suggestions_text = ""
        for material in materials_detected:
            suggestion = recycling_suggestions.get(material, "No suggestion available for this material.")
            suggestions_text += f"Detected {material}: {suggestion}\n"

        # Update the suggestions text box
        suggestions_box.config(state=tk.NORMAL)
        suggestions_box.delete(1.0, tk.END)
        suggestions_box.insert(tk.END, suggestions_text)
        suggestions_box.config(state=tk.DISABLED)

        # Convert frame to RGB and display in the GUI
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        img.thumbnail((imgW, imgH))  # Resize image to fit within the display area
        img_tk = ImageTk.PhotoImage(image=img)
        video_label.img_tk = img_tk  # Keep a reference to avoid garbage collection
        video_label.config(image=img_tk)

    # Release camera when stopped
    if not is_running:
        cap.release()
        cap = None

# Function to stop the camera feed
def stop_camera():
    global is_running
    is_running = False

# Function to save the current frame as an image
def save_image():
    if current_frame is not None:
        cv2.imwrite("recycle_capture.png", current_frame)
        messagebox.showinfo("Success", "Image saved as recycle_capture.png")

# Function to upload an image
def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        process_image(file_path)

# Function to exit the application
def exit_app():
    stop_camera()
    if cap is not None:
        cap.release()
    root.destroy()

# Create the main GUI window
root = tk.Tk()
root.title("My Recycling Identifier")
root.geometry("1280x800")  # Set window size

# Modern color scheme
bg_color = "#2E3440"  # Dark background
fg_color = "#D8DEE9"  # Light text
button_color = "#5E81AC"  # Blue buttons
text_color = "#333333"  # White text

# Apply custom styling
style = ttk.Style()
style.theme_use("clam")  # Use a modern theme
style.configure("TFrame", background=bg_color)
style.configure("TLabel", background=bg_color, foreground=fg_color, font=("Helvetica", 12))
style.configure("TButton", background=button_color, foreground=text_color, font=("Helvetica", 12), padding=10)
style.configure("TText", background=bg_color, foreground=fg_color, font=("Helvetica", 12))

# Create a frame for the video feed
video_frame = ttk.Frame(root)
video_frame.pack(pady=10, fill=tk.BOTH, expand=True)

# Label to display the video feed
video_label = ttk.Label(video_frame)
video_label.pack(fill=tk.BOTH, expand=True)

# Create a frame for buttons
button_frame = ttk.Frame(root)
button_frame.pack(pady=10, fill=tk.X)

# Start Camera Button
start_button = ttk.Button(button_frame, text="Start Camera", command=lambda: threading.Thread(target=start_camera).start())
start_button.grid(row=0, column=0, padx=5)

# Stop Camera Button
stop_button = ttk.Button(button_frame, text="Stop Camera", command=stop_camera)
stop_button.grid(row=0, column=1, padx=5)

# Upload Image Button
upload_button = ttk.Button(button_frame, text="Upload Image", command=upload_image)
upload_button.grid(row=0, column=2, padx=5)

# Save Image Button
save_button = ttk.Button(button_frame, text="Save Image", command=save_image)
save_button.grid(row=0, column=3, padx=5)

# Exit Button
exit_button = ttk.Button(button_frame, text="Exit", command=exit_app)
exit_button.grid(row=0, column=4, padx=5)

# Create a frame for recycling suggestions
suggestions_frame = ttk.Frame(root)
suggestions_frame.pack(pady=10, fill=tk.BOTH, expand=True)

# Label for suggestions
suggestions_label = ttk.Label(suggestions_frame, text="Recycling Suggestions:")
suggestions_label.pack()

# Text box to display recycling suggestions
suggestions_box = tk.Text(suggestions_frame, height=5, width=80, state=tk.DISABLED, bg=bg_color, fg=fg_color, font=("Helvetica", 12))
suggestions_box.pack(fill=tk.BOTH, expand=True)

# Run the GUI
root.mainloop()

# Clean up
if cap is not None:
    cap.release()
cv2.destroyAllWindows()