# Angle at the shoulder vertex between ear->shoulder and hip->shoulder vectors.
# Upright posture ≈ 180°; drop below threshold = slouch.
ANGLE_THRESHOLD = 150       # degrees – fallback if calibration finds no pose
CALIBRATION_DURATION = 3    # seconds to hold good posture during calibration
CALIBRATION_MARGIN = 15     # degrees subtracted from calibrated baseline

# BGR colors (OpenCV convention)
COLOR_GOOD = (220, 80, 200)   # purple-pink
COLOR_SLOUCH = (0, 0, 255)    # red

LANDMARK_RADIUS = 8
LINE_THICKNESS = 3

ALERT_COOLDOWN = 4.0    # minimum seconds between successive audio alerts
SOUND_ENABLED = True
CAMERA_INDEX = 0
