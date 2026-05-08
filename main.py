import sys
import time
import cv2
import config
from detector import PostureDetector
from alerts import draw_overlay, draw_hud, play_alert

_WIN_CAL = "Slouch Shamer - Calibrating"
_WIN_RUN = "Slouch Shamer"


def calibrate(detector: PostureDetector, cap: cv2.VideoCapture) -> float | None:
    """
    Show webcam feed for CALIBRATION_DURATION seconds and collect posture angles.
    Returns the computed threshold, or None if the user quit early.
    """
    angles: list[float] = []
    deadline = time.time() + config.CALIBRATION_DURATION

    while time.time() < deadline:
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)

        results = detector.process(frame)
        landmarks = detector.get_landmarks(results, frame.shape)
        if landmarks:
            angles.append(detector.calculate_angle(**landmarks))
            draw_overlay(frame, landmarks, is_slouching=False)

        remaining = max(0.0, deadline - time.time())
        cv2.putText(
            frame,
            f"Sit up straight!  Calibrating... {remaining:.1f}s",
            (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 230, 200), 2,
        )
        cv2.imshow(_WIN_CAL, frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            return None

    cv2.destroyWindow(_WIN_CAL)

    if not angles:
        print("[calibration] No pose detected; using default threshold.")
        return float(config.ANGLE_THRESHOLD)

    baseline = sum(angles) / len(angles)
    threshold = max(baseline - config.CALIBRATION_MARGIN, 100.0)
    print(f"[calibration] baseline {baseline:.1f}°  ->  threshold {threshold:.1f}°")
    return threshold


def main() -> None:
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        sys.exit(
            f"Could not open webcam (index {config.CAMERA_INDEX}). "
            "Check CAMERA_INDEX in config.py."
        )

    detector = PostureDetector()

    print("=== Slouch Shamer ===")
    print(
        f"Calibration: sit in good posture for "
        f"{config.CALIBRATION_DURATION} seconds, then monitoring begins."
    )

    threshold = calibrate(detector, cap)
    if threshold is None:
        print("Quitting.")
        cap.release()
        detector.close()
        cv2.destroyAllWindows()
        return

    print(f"Monitoring with threshold {threshold:.1f}°. Press Q to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)

        results = detector.process(frame)
        landmarks = detector.get_landmarks(results, frame.shape)

        angle = 180.0
        is_slouching = False
        has_pose = landmarks is not None

        if has_pose:
            angle = detector.calculate_angle(**landmarks)
            is_slouching = angle < threshold
            draw_overlay(frame, landmarks, is_slouching)
            if is_slouching:
                play_alert()

        draw_hud(frame, angle, threshold, is_slouching, has_pose)
        cv2.imshow(_WIN_RUN, frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    detector.close()
    cap.release()
    cv2.destroyAllWindows()
    print("Bye.")


if __name__ == "__main__":
    main()
