"""
Demo: simulate how an AI agent uses the stylus arm after calibration.

Run after calibrate.py has completed successfully:
    uv run python scripts/demo_agent.py
"""

import time

from physiclaw import PhysiClaw


def main():
    hw = PhysiClaw()

    try:
        # hw.arm triggers lazy init (connects arm, identifies cameras, loads calibration)
        arm = hw.arm

        # --- Example: unlock phone and open an app ---

        # 1. Tap center of screen to wake
        print("\n--- Tap to wake ---")
        arm.tap()
        time.sleep(2)

        # 2. Swipe up to unlock
        print("\n--- Swipe up to unlock ---")
        arm.swipe('top')
        time.sleep(2)

        # 3. Move to where an app icon might be (top-left area)
        print("\n--- Move to app icon ---")
        arm.move('top-left', 'large')
        time.sleep(1)

        # 4. Tap to open the app
        print("\n--- Tap app ---")
        arm.tap()
        time.sleep(2)

        # 5. Scroll down to read content
        print("\n--- Scroll down ---")
        arm.swipe('bottom', 'slow')
        time.sleep(1)
        arm.swipe('bottom', 'slow')
        time.sleep(2)

        # 6. Long press on an item
        print("\n--- Long press ---")
        arm.move('bottom', 'medium')
        arm.long_press()
        time.sleep(2)

        # 7. Double tap to zoom
        print("\n--- Double tap to zoom ---")
        arm.move('top', 'small')
        arm.double_tap()
        time.sleep(2)

        # 8. Swipe right to go back
        print("\n--- Swipe right to go back ---")
        arm.swipe('right')
        time.sleep(1)

        print("\n--- Demo complete ---")

    finally:
        hw.shutdown()


if __name__ == "__main__":
    main()
