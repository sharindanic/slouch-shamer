import os
import time
import subprocess
import cv2
import config

# ---------------------------------------------------------------------------
# Sound
# ---------------------------------------------------------------------------
_PYGAME_OK = False
_beep = None

def _init_pygame():
    global _PYGAME_OK, _beep
    try:
        import pygame
        import numpy as np
        pygame.mixer.pre_init(44100, -16, 2, 512)
        pygame.mixer.init()
        t = np.linspace(0, 0.35, int(44100 * 0.35), False)
        wave = (np.sin(2 * np.pi * 660 * t) * 28000).astype(np.int16)
        stereo = np.ascontiguousarray(np.column_stack([wave, wave]))
        _beep = pygame.sndarray.make_sound(stereo)
        _PYGAME_OK = True
    except Exception:
        pass

_init_pygame()

# macOS fallback sound (guaranteed to exist)
_MACOS_SOUND = "/System/Library/Sounds/Funk.aiff"
_HAS_MACOS_SOUND = os.path.exists(_MACOS_SOUND)

_last_alert: float = 0.0


def play_alert() -> None:
    global _last_alert
    if not config.SOUND_ENABLED:
        return
    now = time.time()
    if now - _last_alert < config.ALERT_COOLDOWN:
        return
    _last_alert = now

    if _PYGAME_OK and _beep is not None:
        _beep.play()
    elif _HAS_MACOS_SOUND:
        subprocess.Popen(
            ["afplay", _MACOS_SOUND],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


# ---------------------------------------------------------------------------
# Visuals
# ---------------------------------------------------------------------------

def draw_overlay(frame, landmarks, is_slouching: bool) -> None:
    if landmarks is None:
        return
    color = config.COLOR_SLOUCH if is_slouching else config.COLOR_GOOD
    ear, shoulder, hip = landmarks["ear"], landmarks["shoulder"], landmarks["hip"]
    cv2.line(frame, ear, shoulder, color, config.LINE_THICKNESS)
    cv2.line(frame, shoulder, hip, color, config.LINE_THICKNESS)
    for pt in (ear, shoulder, hip):
        cv2.circle(frame, pt, config.LANDMARK_RADIUS, color, -1)


def draw_hud(frame, angle: float, threshold: float, is_slouching: bool,
             has_pose: bool) -> None:
    h, w = frame.shape[:2]

    if has_pose:
        cv2.putText(frame,
                    f"Angle: {angle:5.1f}   Threshold: {threshold:.0f}",
                    (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (230, 230, 230), 2)
    else:
        cv2.putText(frame, "No pose detected – move into frame",
                    (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (160, 160, 160), 2)

    cv2.putText(frame, "Q  quit", (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 140, 140), 1)

    if is_slouching:
        label = "SIT UP STRAIGHT!"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 1.3, 3)
        x = (w - tw) // 2
        y = h // 2 + th // 2
        # shadow for readability
        cv2.putText(frame, label, (x + 2, y + 2),
                    cv2.FONT_HERSHEY_DUPLEX, 1.3, (0, 0, 0), 4)
        cv2.putText(frame, label, (x, y),
                    cv2.FONT_HERSHEY_DUPLEX, 1.3, config.COLOR_SLOUCH, 3)
