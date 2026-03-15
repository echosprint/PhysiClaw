import cv2
import sys
from datetime import datetime


def take_picture(output_path: str | None = None) -> str:
    """Open the camera, take a picture, and save it."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        sys.exit(1)

    # Allow camera to warm up by grabbing a few frames
    for _ in range(30):
        cap.read()

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Error: Could not capture frame.")
        sys.exit(1)

    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"photo_{timestamp}.jpg"

    cv2.imwrite(output_path, frame)
    print(f"Picture saved to {output_path}")
    return output_path


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else None
    take_picture(path)
