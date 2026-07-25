"""
pose_detector.py

Wraps MediaPipe's Pose Landmarker Task so the rest of the app never has
to touch the MediaPipe API directly.

IMPORTANT: This uses MediaPipe's current "Tasks" API (mediapipe.tasks),
not the old mp.solutions.pose API. Google removed mp.solutions entirely
starting in mediapipe 0.10.30 — it no longer exists in current installs,
regardless of Python version. The Tasks API is the supported replacement
and needs a separate downloaded model file (see the .task file setup
below) instead of a model bundled invisibly inside the package.

Responsibility boundary is unchanged from before:
    camera.py        -> gets raw frames (no idea what a "person" is)
    pose_detector.py -> takes a frame, finds a body, returns landmarks
    (future) angle logic / exercise counters -> use those landmarks
"""

import os
import time

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision as mp_vision

# The model file MediaPipe needs is not bundled in the pip package
# anymore — you download it once and point the detector at it.
# See the setup instructions that came with this file.
_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
DEFAULT_MODEL_PATH = os.path.join(_MODEL_DIR, "pose_landmarker_lite.task")

# MediaPipe's official 33-landmark BlazePose topology: which landmark
# indices are connected by a "bone". This used to be exposed as
# mp.solutions.pose.POSE_CONNECTIONS, but that lived inside the removed
# solutions API, so it's defined directly here instead.
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
]


class PoseDetector:
    """
    Object-oriented wrapper around mediapipe.tasks.python.vision.PoseLandmarker.

    Same public interface as before (find_pose, find_landmarks), so
    nothing about how main.py will use this class needs to change —
    only the internals had to change when Google removed the old API.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        num_poses: int = 1,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        """
        Args:
            model_path: path to the downloaded .task model file
            num_poses: max number of people to detect at once (1 is
                right for a fitness coach app — you're the only one
                the camera needs to track)
            min_detection_confidence / min_presence_confidence /
            min_tracking_confidence: same idea as the old API's
                thresholds — how confident MediaPipe must be before
                it trusts a detection.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Pose model not found at: {model_path}\n"
                "Download it first — see the setup instructions."
            )

        options = mp_vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            # VIDEO mode (rather than LIVE_STREAM) keeps this synchronous:
            # detect_for_video() blocks and returns the result directly,
            # which fits a simple frame-by-frame loop far better than
            # LIVE_STREAM's async callback pattern.
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=num_poses,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.landmarker = mp_vision.PoseLandmarker.create_from_options(options)

        # VIDEO mode requires a timestamp per frame that strictly
        # increases. We track our own clock rather than relying on a
        # frame counter, since real elapsed time is what MediaPipe
        # actually needs for its internal tracking.
        self._start_time_ms = time.time() * 1000

        # Cached after each find_pose() call so find_landmarks() can
        # reuse it without re-running detection.
        self.results = None

    def find_pose(self, frame, draw: bool = True):
        """
        Run pose detection on a single frame.

        Args:
            frame: a BGR image from OpenCV (e.g. from Camera.read_frame())
            draw: if True, draws the skeleton directly onto the frame

        Returns:
            frame: the same frame, with skeleton drawn on it if draw=True
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        timestamp_ms = int(time.time() * 1000 - self._start_time_ms)
        self.results = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        if self.results.pose_landmarks and draw:
            # pose_landmarks is a list (one entry per detected person);
            # we only asked for num_poses=1, so we use the first one.
            self._draw_landmarks(frame, self.results.pose_landmarks[0])

        return frame

    def _draw_landmarks(self, frame, landmarks):
        """
        Draw the skeleton manually, since drawing_utils lived inside
        the now-removed solutions API and has no Tasks-API equivalent
        as simple as before.
        """
        height, width = frame.shape[:2]
        points = [(int(lm.x * width), int(lm.y * height)) for lm in landmarks]

        for start_idx, end_idx in POSE_CONNECTIONS:
            cv2.line(frame, points[start_idx], points[end_idx], (0, 255, 0), 2)

        for point in points:
            cv2.circle(frame, point, 4, (0, 0, 255), -1)

    def find_landmarks(self, frame):
        """
        Extract landmark coordinates from the last find_pose() call.

        Returns:
            A list of dicts, one per landmark:
                {"id": int, "x": int, "y": int, "z": float, "visibility": float}
            Empty list if no person was detected.
        """
        landmark_list = []

        if not self.results or not self.results.pose_landmarks:
            return landmark_list

        height, width = frame.shape[:2]

        for idx, lm in enumerate(self.results.pose_landmarks[0]):
            landmark_list.append({
                "id": idx,
                "x": int(lm.x * width),
                "y": int(lm.y * height),
                "z": lm.z,
                "visibility": lm.visibility,
            })

        return landmark_list

    def close(self):
        """Release the landmarker's resources."""
        self.landmarker.close()


def run_pose_preview():
    """
    Milestone 2 verification: combines Camera + PoseDetector to show
    a live skeleton overlay, and prints how many landmarks were found.
    Press 'q' to quit.
    """
    from camera import Camera  # local import keeps this file testable alone

    camera = Camera(source=0)
    camera.start()
    detector = PoseDetector()

    print("Pose preview started. Press 'q' to quit.")

    try:
        while True:
            frame = camera.read_frame()
            if frame is None:
                print("Failed to read frame from camera.")
                break

            frame = detector.find_pose(frame, draw=True)
            landmarks = detector.find_landmarks(frame)

            if landmarks:
                cv2.putText(
                    frame,
                    f"Landmarks detected: {len(landmarks)}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

            cv2.imshow("AI Fitness Coach - Pose Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        camera.release()
        detector.close()


if __name__ == "__main__":
    run_pose_preview()