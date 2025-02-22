import os
import sys
import cv2
from ultralytics import YOLO
import torch
from pathlib import Path
from tkinter import Tk, Canvas, Text, Button, PhotoImage, messagebox, filedialog
from PIL import Image, ImageTk
import threading

# Path configurations
OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r"../my_modelV5n/build/assets/frame0")

def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

# AI Model configurations
model_path = 'my_model.pt'
min_thresh = 0.50
imgW, imgH = 725, 456

# Recycling information
recycling_suggestions = {
    'plastic': "Recycle this plastic item in your local recycling bin. Consider reusing it if possible.",
    'cardboard': "Flatten this cardboard and place it in your recycling bin. Ensure it's clean and dry."
}

bbox_colors = {
    'plastic': (0, 255, 0),
    'cardboard': (0, 0, 255)
}

# Initialize AI model
if not os.path.exists(model_path):
    print('WARNING: Model path is invalid or model was not found.')
    sys.exit()

model = YOLO(model_path, task='detect')

try:
    checkpoint = torch.load(model_path)
    model.model.load_state_dict(checkpoint['model'].float().state_dict(), strict=False)
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit()

labels = model.names

# Global variables
cap = None
is_running = False
current_frame = None

class RecycleAI:
    def __init__(self):
        self.window = Tk()
        self.window.title("Recycle AI")
        self.window.geometry("1280x720")
        self.window.configure(bg="#FFFFFF")
        self.window.resizable(False, False)
        
        self.setup_canvas()
        self.load_images()
        self.create_buttons()
        self.setup_text_area()
        
        self.is_running = False
        self.cap = None

    def setup_canvas(self):
        self.canvas = Canvas(
            self.window,
            bg="#FFFFFF",
            height=720,
            width=1280,
            bd=0,
            highlightthickness=0,
            relief="ridge"
        )
        self.canvas.place(x=0, y=0)

    def load_images(self):
        # Load all your original images
        self.image_1 = PhotoImage(file=relative_to_assets("image_1.png"))
        self.image_2 = PhotoImage(file=relative_to_assets("image_2.png"))
        self.image_3 = PhotoImage(file=relative_to_assets("image_3.png"))
        
        # Create image elements
        self.canvas.create_image(640.0, 48.0, image=self.image_1)
        self.canvas.create_image(35.0, 49.0, image=self.image_2)
        self.canvas.create_image(400.0, 356.0, image=self.image_3)
        
        # Create title text
        self.canvas.create_text(
            69.0, 25.0,
            anchor="nw",
            text="RECYCLE AI",
            fill="#FFFFFF",
            font=("Montserrat Bold", 32 * -1)
        )

    def create_buttons(self):
        # Load button images
        button_images = []
        for i in range(1, 6):
            img = PhotoImage(file=relative_to_assets(f"button_{i}.png"))
            button_images.append(img)
            setattr(self, f'button_image_{i}', img)

        # Create buttons with their respective commands
        self.buttons = []
        button_commands = [
            self.start_camera_thread,  # Button 1: Start Camera
            self.stop_camera,          # Button 2: Stop Camera
            self.upload_image,         # Button 3: Upload Image
            self.save_image,         # Button 4: Save Image
            self.exit_app        # Button 5: Exit
        ]
        button_positions = [
            (94.0, 617.0),    # Button 1 position
            (311.0, 617.0),   # Button 2 position
            (528.0, 617.0),   # Button 3 position
            (745.0, 618.0),   # Button 4 position
            (962.0, 617.0)    # Button 5 position
        ]

        for i, (img, cmd, pos) in enumerate(zip(button_images, button_commands, button_positions)):
            btn = Button(
                image=img,
                borderwidth=0,
                highlightthickness=0,
                command=cmd,
                relief="flat"
            )
            btn.place(x=pos[0], y=pos[1], width=183.0, height=50.0)
            self.buttons.append(btn)

    def setup_text_area(self):
        # Create the suggestions text area
        entry_image_1 = PhotoImage(file=relative_to_assets("entry_1.png"))
        self.canvas.create_image(1029.5, 356.5, image=entry_image_1)
        self.entry_image_1 = entry_image_1  # Prevent garbage collection
        
        self.suggestions_text = Text(
            self.window,
            bd=0,
            bg="#ffffff",
            fg="#000716",
            highlightthickness=0
        )
        self.suggestions_text.place(
            x=816.0,
            y=126.0,
            width=427.0,
            height=459.0
        )

    def start_camera_thread(self):
        if not self.is_running:
            threading.Thread(target=self.start_camera, daemon=True).start()

    def start_camera(self):
        global cap
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Unable to access the camera.")
                return
            self.cap.set(3, imgW)
            self.cap.set(4, imgH)
        
        self.is_running = True
        self.process_camera_feed()

    def process_camera_feed(self):
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Process frame with AI model
            results = model.track(frame, verbose=False)
            processed_frame = self.process_detections(frame, results[0].boxes)
            
            # Update GUI
            self.update_display(processed_frame)

        if self.cap:
            self.cap.release()
            self.cap = None

    def process_detections(self, frame, detections):
        materials_detected = []
        
        for detection in detections:
            # Get detection details
            xyxy = detection.xyxy.cpu().numpy().squeeze()
            xmin, ymin, xmax, ymax = xyxy.astype(int)
            classname = labels[int(detection.cls.item())]
            conf = detection.conf.item()

            if conf > min_thresh:
                # Draw bounding box
                color = bbox_colors.get(classname, (255, 255, 255))
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                
                # Add label
                label = f'{classname}: {int(conf * 100)}%'
                self.draw_label(frame, label, xmin, ymin, color)
                
                materials_detected.append(classname)

        # Update suggestions
        self.update_suggestions(materials_detected)
        return frame

    def draw_label(self, frame, label, x, y, color):
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y = max(y, labelSize[1] + 10)
        cv2.rectangle(frame, (x, y - labelSize[1] - 10),
                     (x + labelSize[0], y + baseLine - 10), color, cv2.FILLED)
        cv2.putText(frame, label, (x, y - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    def update_display(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img.thumbnail((imgW, imgH))
        img_tk = ImageTk.PhotoImage(image=img)
        
        # Update the image in canvas
        self.canvas.create_image(400.0, 356.0, image=img_tk)
        self.canvas.image = img_tk  # Keep reference

    def update_suggestions(self, materials):
        suggestions_text = ""
        for material in materials:
            suggestion = recycling_suggestions.get(material, "No suggestion available for this material.")
            suggestions_text += f"Detected {material}: {suggestion}\n"
        
        self.suggestions_text.delete(1.0, "end")
        self.suggestions_text.insert("1.0", suggestions_text)

    def stop_camera(self):
        self.is_running = False

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
        if file_path:
            frame = cv2.imread(file_path)
            if frame is not None:
                results = model.track(frame, verbose=False)
                processed_frame = self.process_detections(frame, results[0].boxes)
                self.update_display(processed_frame)

    def save_image(self):
        if self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                cv2.imwrite("recycle_capture.png", frame)
                messagebox.showinfo("Success", "Image saved as recycle_capture.png")

    def exit_app(self):
        self.stop_camera()
        self.window.destroy()

    def run(self):
        self.window.mainloop()

if __name__ == "__main__":
    app = RecycleAI()
    app.run()