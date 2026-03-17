"""
Demo: simulate how an AI agent uses the stylus arm after calibration.

Run after calibrate.py has completed successfully:
    uv run python demo_agent.py
"""

import time

from stylus_arm import StylusArm


def main():
    arm = StylusArm()
    arm.setup()
    arm.load_calibration()  # reads calibration.json

    try:
        # --- Example: unlock phone and open an app ---

        # 1. Tap center of screen to wake
        print("\n--- Tap to wake ---")
        arm.tap()
        time.sleep(2)

        # 2. Swipe up to unlock
        print("\n--- Swipe up to unlock ---")
        arm.swipe('up')
        time.sleep(2)

        # 3. Move to where an app icon might be (top-left area)
        print("\n--- Move to app icon ---")
        arm.move('up-left', 'large')
        time.sleep(1)

        # 4. Tap to open the app
        print("\n--- Tap app ---")
        arm.tap()
        time.sleep(2)

        # 5. Scroll down to read content
        print("\n--- Scroll down ---")
        arm.swipe('down', 'slow')
        time.sleep(1)
        arm.swipe('down', 'slow')
        time.sleep(2)

        # 6. Long press on an item
        print("\n--- Long press ---")
        arm.move('down', 'medium')
        arm.long_press()
        time.sleep(2)

        # 7. Double tap to zoom
        print("\n--- Double tap to zoom ---")
        arm.move('up', 'small')
        arm.double_tap()
        time.sleep(2)

        # 8. Swipe right to go back
        print("\n--- Swipe right to go back ---")
        arm.swipe('right')
        time.sleep(1)

        print("\n--- Demo complete ---")

    finally:
        arm._pen_up()
        arm._fast_move(0, 0)
        arm.close()


if __name__ == "__main__":
    main()
