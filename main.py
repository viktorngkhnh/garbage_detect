from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

import cv2
import torch
from PIL import Image, ImageTk
from torchvision import transforms

from pytorch_resnet import ResNet


CHECKPOINT_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "pytorch_resnet"
    / "waste_classifier_resnet50.pth"
)


def load_classifier(checkpoint_path, device):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Model checkpoint was not found:\n{checkpoint_path}\n\n"
            "Run `python pytorch_resnet.py` first."
        )

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    required_keys = {
        "architecture",
        "state_dict",
        "class_names",
        "input_size",
        "normalization_mean",
        "normalization_std",
    }
    missing_keys = required_keys.difference(checkpoint)
    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise ValueError(f"Checkpoint metadata is missing: {missing}")
    if checkpoint["architecture"] != "resnet50":
        raise ValueError(f"Unsupported architecture: {checkpoint['architecture']}")

    class_names = checkpoint["class_names"]
    model = ResNet(
        num_classes=len(class_names),
        pretrained=False,
        freeze_backbone=False,
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    image_transform = transforms.Compose([
        transforms.Resize((checkpoint["input_size"], checkpoint["input_size"])),
        transforms.ToTensor(),
        transforms.Normalize(
            checkpoint["normalization_mean"],
            checkpoint["normalization_std"],
        ),
    ])
    return model, class_names, image_transform


class WasteClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Waste Classifier")
        self.root.geometry("860x720")
        self.root.minsize(680, 600)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.class_names, self.image_transform = load_classifier(
            CHECKPOINT_PATH,
            self.device,
        )
        self.latest_frame = None
        self.preview_image = None

        self.camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.camera.isOpened():
            self.camera.release()
            self.camera = cv2.VideoCapture(0)
        self.camera_available = self.camera.isOpened()

        if self.camera_available:
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.build_interface()
        self.update_preview()

    def build_interface(self):
        container = ttk.Frame(self.root, padding=18)
        container.pack(fill="both", expand=True)

        title = ttk.Label(container, text="Waste Classifier", font=("Segoe UI", 22, "bold"))
        title.pack(pady=(0, 12))

        self.preview_label = ttk.Label(container, anchor="center")
        self.preview_label.pack(fill="both", expand=True)

        if not self.camera_available:
            black_preview = Image.new("RGB", (760, 480), "black")
            self.preview_image = ImageTk.PhotoImage(image=black_preview)
            self.preview_label.configure(image=self.preview_image)

        self.capture_button = ttk.Button(
            container,
            text="Capture & Classify",
            command=self.classify_current_frame,
        )
        self.capture_button.pack(pady=14, ipadx=18, ipady=8)

        if not self.camera_available:
            self.capture_button.configure(state="disabled")

        self.result_label = ttk.Label(
            container,
            text=(
                "Point the camera at one waste item, then capture."
                if self.camera_available
                else "Camera unavailable - interface preview mode"
            ),
            anchor="center",
            font=("Segoe UI", 16, "bold"),
        )
        self.result_label.pack(fill="x", pady=(0, 8))

        self.probability_label = ttk.Label(
            container,
            text="",
            anchor="center",
            justify="center",
            font=("Segoe UI", 11),
        )
        self.probability_label.pack(fill="x")

        device_label = ttk.Label(container, text=f"Inference device: {self.device}")
        device_label.pack(pady=(10, 0))

    def update_preview(self):
        if not self.camera_available:
            return

        success, frame = self.camera.read()
        if success:
            self.latest_frame = frame
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb_frame)
            image.thumbnail((760, 480), Image.Resampling.LANCZOS)
            self.preview_image = ImageTk.PhotoImage(image=image)
            self.preview_label.configure(image=self.preview_image)

        self.root.after(30, self.update_preview)

    def classify_current_frame(self):
        if self.latest_frame is None:
            messagebox.showwarning("Camera", "No camera frame is available yet.")
            return

        self.capture_button.configure(state="disabled")
        self.root.update_idletasks()

        rgb_frame = cv2.cvtColor(self.latest_frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_frame)
        image_tensor = self.image_transform(image).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            logits = self.model(image_tensor)
            probabilities = torch.softmax(logits[0], dim=0).cpu()

        predicted_index = int(torch.argmax(probabilities).item())
        predicted_class = self.class_names[predicted_index]
        confidence = probabilities[predicted_index].item() * 100
        self.result_label.configure(
            text=f"Prediction: {predicted_class} ({confidence:.1f}%)"
        )

        ranked_indices = torch.argsort(probabilities, descending=True)
        details = [
            f"{self.class_names[index]}: {probabilities[index].item() * 100:.1f}%"
            for index in ranked_indices.tolist()
        ]
        self.probability_label.configure(text="   |   ".join(details))
        self.capture_button.configure(state="normal")

    def close(self):
        if self.camera_available:
            self.camera.release()
        self.root.destroy()


def main():
    root = tk.Tk()
    try:
        WasteClassifierApp(root)
    except Exception as error:
        root.withdraw()
        messagebox.showerror("Waste Classifier", str(error))
        root.destroy()
        return
    root.mainloop()


if __name__ == "__main__":
    main()