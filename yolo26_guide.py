```python
"""
YOLO26 Training and Deployment Guide
====================================

A simple practical CLI for:

1. Selecting a YOLO model
2. Testing a pretrained model
3. Training a custom model
4. Evaluating the trained model
5. Running inference on images
6. Running inference on videos
7. Running real-time detection with a webcam
8. Running inference on an IP camera
9. Exporting the trained model

Requirements:
    pip install ultralytics opencv-python

Important:
    Update the paths and model names based on your project.
"""

from pathlib import Path

import cv2
import torch
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

# Start with a small model for your first experiment.
DEFAULT_MODEL = "yolo26n.pt"

# Dataset configuration.
DATASET_CONFIG = "dataset/data.yaml"

# Default trained model location.
TRAINED_MODEL = "runs/train/yolo26_experiment/weights/best.pt"

# Default image size.
IMAGE_SIZE = 640

# Default confidence threshold.
CONFIDENCE_THRESHOLD = 0.25

# Default GPU device.
DEVICE = 0


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def print_title(title: str) -> None:
    """Print a formatted title."""

    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def check_environment() -> None:
    """Check Python, PyTorch, and CUDA availability."""

    print_title("ENVIRONMENT INFORMATION")

    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Count: {torch.cuda.device_count()}")
    else:
        print(
            "\nWarning: CUDA GPU was not detected."
        )
        print(
            "The model may run on the CPU, "
            "but training and inference can be slower."
        )


def get_device():
    """
    Return the best available device.

    GPU is preferred when CUDA is available.
    """

    if torch.cuda.is_available():
        return DEVICE

    return "cpu"


def load_model(model_path: str) -> YOLO:
    """Load a YOLO model."""

    print(f"\nLoading model: {model_path}")

    model = YOLO(model_path)

    print("Model loaded successfully.")

    return model


def check_file(file_path: str) -> bool:
    """Check whether a file exists."""

    if not Path(file_path).exists():
        print(f"\nError: File not found: {file_path}")
        return False

    return True


# ============================================================
# 1. SELECT AND INSPECT MODEL
# ============================================================

def select_model() -> None:
    """
    Load and display model information.

    Example models:

    yolo26n.pt
    yolo26s.pt
    yolo26m.pt
    yolo26l.pt
    """

    print_title("1. SELECT AND INSPECT MODEL")

    model_path = input(
        f"Model path [{DEFAULT_MODEL}]: "
    ).strip()

    if not model_path:
        model_path = DEFAULT_MODEL

    try:
        model = load_model(model_path)

        print("\nModel Information:")
        model.info()

    except Exception as error:
        print(f"\nError while loading model:\n{error}")


# ============================================================
# 2. TEST PRETRAINED MODEL ON IMAGE
# ============================================================

def test_pretrained_model() -> None:
    """Run a pretrained model on an image."""

    print_title("2. TEST PRETRAINED MODEL")

    model_path = input(
        f"Model path [{DEFAULT_MODEL}]: "
    ).strip() or DEFAULT_MODEL

    image_path = input(
        "Image path: "
    ).strip()

    if not check_file(image_path):
        return

    try:
        model = load_model(model_path)

        results = model.predict(
            source=image_path,
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE_THRESHOLD,
            device=get_device(),
            save=True
        )

        print("\nInference completed successfully.")

        for result in results:
            print(
                f"Detected objects: "
                f"{len(result.boxes)}"
            )

    except Exception as error:
        print(f"\nInference error:\n{error}")


# ============================================================
# 3. TRAIN CUSTOM MODEL
# ============================================================

def train_model() -> None:
    """
    Train a YOLO model on a custom dataset.

    The dataset must contain a valid data.yaml file.
    """

    print_title("3. TRAIN CUSTOM MODEL")

    model_path = input(
        f"Base model [{DEFAULT_MODEL}]: "
    ).strip() or DEFAULT_MODEL

    dataset_config = input(
        f"Dataset config [{DATASET_CONFIG}]: "
    ).strip() or DATASET_CONFIG

    if not check_file(dataset_config):
        return

    try:
        epochs = int(
            input("Epochs [100]: ").strip() or 100
        )

        batch = int(
            input("Batch size [8]: ").strip() or 8
        )

        imgsz = int(
            input(
                f"Image size [{IMAGE_SIZE}]: "
            ).strip() or IMAGE_SIZE
        )

        model = load_model(model_path)

        print("\nTraining started...")

        model.train(
            data=dataset_config,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=get_device(),
            workers=4,
            pretrained=True,
            project="runs/train",
            name="yolo26_experiment",
            exist_ok=True
        )

        print("\nTraining completed.")

        print(
            "Check the following directory "
            "for training results:"
        )

        print(
            "runs/train/yolo26_experiment/"
        )

    except ValueError:
        print("\nError: Invalid numeric value.")

    except Exception as error:
        print(f"\nTraining error:\n{error}")


# ============================================================
# 4. EVALUATE TRAINED MODEL
# ============================================================

def evaluate_model() -> None:
    """Evaluate a trained YOLO model."""

    print_title("4. EVALUATE MODEL")

    model_path = input(
        f"Model path [{TRAINED_MODEL}]: "
    ).strip() or TRAINED_MODEL

    dataset_config = input(
        f"Dataset config [{DATASET_CONFIG}]: "
    ).strip() or DATASET_CONFIG

    if not check_file(model_path):
        return

    if not check_file(dataset_config):
        return

    try:
        model = load_model(model_path)

        print("\nEvaluation started...")

        metrics = model.val(
            data=dataset_config,
            imgsz=IMAGE_SIZE,
            device=get_device()
        )

        print("\nEvaluation completed.")

        print("\nModel Metrics:")

        print(
            f"Precision: "
            f"{metrics.box.mp:.4f}"
        )

        print(
            f"Recall: "
            f"{metrics.box.mr:.4f}"
        )

        print(
            f"mAP50: "
            f"{metrics.box.map50:.4f}"
        )

        print(
            f"mAP50-95: "
            f"{metrics.box.map:.4f}"
        )

    except Exception as error:
        print(f"\nEvaluation error:\n{error}")


# ============================================================
# 5. PREDICT ON IMAGE
# ============================================================

def predict_image() -> None:
    """Run a trained model on an image."""

    print_title("5. IMAGE INFERENCE")

    model_path = input(
        f"Model path [{TRAINED_MODEL}]: "
    ).strip() or TRAINED_MODEL

    image_path = input(
        "Image path: "
    ).strip()

    if not check_file(model_path):
        return

    if not check_file(image_path):
        return

    try:
        model = load_model(model_path)

        results = model.predict(
            source=image_path,
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE_THRESHOLD,
            device=get_device(),
            save=True
        )

        print("\nDetection Results:")

        for result in results:

            for box in result.boxes:

                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                class_name = (
                    result.names[class_id]
                )

                print(
                    f"Class: {class_name} | "
                    f"Confidence: {confidence:.2%}"
                )

    except Exception as error:
        print(f"\nInference error:\n{error}")


# ============================================================
# 6. PREDICT ON VIDEO
# ============================================================

def predict_video() -> None:
    """Run a trained model on a video."""

    print_title("6. VIDEO INFERENCE")

    model_path = input(
        f"Model path [{TRAINED_MODEL}]: "
    ).strip() or TRAINED_MODEL

    video_path = input(
        "Video path: "
    ).strip()

    if not check_file(model_path):
        return

    if not check_file(video_path):
        return

    try:
        model = load_model(model_path)

        results = model.predict(
            source=video_path,
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE_THRESHOLD,
            device=get_device(),
            stream=True,
            save=True
        )

        frame_count = 0

        for _ in results:
            frame_count += 1

            if frame_count % 100 == 0:
                print(
                    f"Processed frames: "
                    f"{frame_count}"
                )

        print(
            f"\nVideo processing completed."
        )

        print(
            f"Total processed frames: "
            f"{frame_count}"
        )

    except Exception as error:
        print(f"\nVideo inference error:\n{error}")


# ============================================================
# 7. REAL-TIME WEBCAM DETECTION
# ============================================================

def run_webcam() -> None:
    """Run real-time detection using a webcam."""

    print_title("7. WEBCAM DETECTION")

    model_path = input(
        f"Model path [{TRAINED_MODEL}]: "
    ).strip() or TRAINED_MODEL

    if not check_file(model_path):
        return

    try:
        camera_id = int(
            input("Camera ID [0]: ").strip() or 0
        )

        model = load_model(model_path)

        cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            print(
                "\nError: Could not open webcam."
            )
            return

        print(
            "\nWebcam started."
        )

        print(
            "Press 'q' to exit."
        )

        while True:

            success, frame = cap.read()

            if not success:
                print(
                    "Error: Could not read frame."
                )
                break

            results = model(
                frame,
                imgsz=IMAGE_SIZE,
                conf=CONFIDENCE_THRESHOLD,
                device=get_device(),
                verbose=False
            )

            annotated_frame = (
                results[0].plot()
            )

            cv2.imshow(
                "YOLO26 Real-Time Detection",
                annotated_frame
            )

            if (
                cv2.waitKey(1) & 0xFF
                == ord("q")
            ):
                break

        cap.release()
        cv2.destroyAllWindows()

    except Exception as error:
        print(
            f"\nWebcam error:\n{error}"
        )


# ============================================================
# 8. IP CAMERA DETECTION
# ============================================================

def run_ip_camera() -> None:
    """
    Run real-time detection using an IP camera.

    Example RTSP URL:

    rtsp://username:password@192.168.1.100:554/stream
    """

    print_title("8. IP CAMERA DETECTION")

    model_path = input(
        f"Model path [{TRAINED_MODEL}]: "
    ).strip() or TRAINED_MODEL

    if not check_file(model_path):
        return

    stream_url = input(
        "RTSP/Stream URL: "
    ).strip()

    if not stream_url:
        print("Error: Stream URL is required.")
        return

    try:
        model = load_model(model_path)

        cap = cv2.VideoCapture(stream_url)

        if not cap.isOpened():
            print(
                "\nError: Could not connect "
                "to IP camera."
            )
            return

        print(
            "\nIP camera connected."
        )

        print(
            "Press 'q' to exit."
        )

        while True:

            success, frame = cap.read()

            if not success:
                print(
                    "Warning: Could not read frame."
                )
                break

            results = model(
                frame,
                imgsz=IMAGE_SIZE,
                conf=CONFIDENCE_THRESHOLD,
                device=get_device(),
                verbose=False
            )

            annotated_frame = (
                results[0].plot()
            )

            cv2.imshow(
                "YOLO26 IP Camera",
                annotated_frame
            )

            if (
                cv2.waitKey(1) & 0xFF
                == ord("q")
            ):
                break

        cap.release()
        cv2.destroyAllWindows()

    except Exception as error:
        print(
            f"\nIP camera error:\n{error}"
        )


# ============================================================
# 9. EXPORT MODEL
# ============================================================

def export_model() -> None:
    """Export a trained model."""

    print_title("9. EXPORT MODEL")

    model_path = input(
        f"Model path [{TRAINED_MODEL}]: "
    ).strip() or TRAINED_MODEL

    if not check_file(model_path):
        return

    export_format = input(
        "Export format [onnx]: "
    ).strip() or "onnx"

    try:
        model = load_model(model_path)

        print(
            f"\nExporting model to "
            f"{export_format}..."
        )

        export_path = model.export(
            format=export_format,
            imgsz=IMAGE_SIZE
        )

        print("\nModel exported successfully.")

        print(
            f"Exported file: {export_path}"
        )

    except Exception as error:
        print(
            f"\nExport error:\n{error}"
        )


# ============================================================
# MAIN MENU
# ============================================================

def show_menu() -> None:
    """Display the application menu."""

    print_title(
        "YOLO26 TRAINING AND DEPLOYMENT GUIDE"
    )

    print(
        """
1. Check Environment
2. Select and Inspect Model
3. Test Pretrained Model
4. Train Custom Model
5. Evaluate Trained Model
6. Predict on Image
7. Predict on Video
8. Run Webcam Detection
9. Run IP Camera Detection
10. Export Model

0. Exit
"""
    )


def main() -> None:
    """Run the main application."""

    while True:

        show_menu()

        choice = input(
            "Select an option: "
        ).strip()

        if choice == "1":
            check_environment()

        elif choice == "2":
            select_model()

        elif choice == "3":
            test_pretrained_model()

        elif choice == "4":
            train_model()

        elif choice == "5":
            evaluate_model()

        elif choice == "6":
            predict_image()

        elif choice == "7":
            predict_video()

        elif choice == "8":
            run_webcam()

        elif choice == "9":
            run_ip_camera()

        elif choice == "10":
            export_model()

        elif choice == "0":

            print(
                "\nExiting application..."
            )

            break

        else:

            print(
                "\nInvalid option. "
                "Please try again."
            )

        input(
            "\nPress Enter to continue..."
        )


if __name__ == "__main__":
    main()
```
