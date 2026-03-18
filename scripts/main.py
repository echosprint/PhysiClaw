import subprocess
import sys
from datetime import datetime


def take_picture(output_path: str | None = None) -> str:
    """Take a picture using imagesnap (macOS)."""
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"photo_{timestamp}.jpg"

    result = subprocess.run(
        ["imagesnap", "-d", "UGREEN Camera 1080P", "-w", "1", output_path],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"Error: {result.stderr.strip()}")
        sys.exit(1)

    print(f"Picture saved to {output_path}")
    return output_path


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else None
    take_picture(path)
