"""
camera.py

Handles webcam access and frame capture for the AI Fitness Coach project.

This module is the "eyes" layer: it only deals with opening the camera,
reading frames, displaying them, and cleanup. It has no knowledge of
poses, landmarks, or exercises — that logic lives in later modules
like pose_detector.py. Keeping concerns separated like this is what
makes it easy to plug a PoseDetector in later without touching this file.
"""

import cv2



class Camera:
    """
    A thin, object-oriented wrapper around OpenCV's VideoCapture.

    Wrapping VideoCapture in a class (instead of using it directly in
    main.py) means:
      - main.py doesn't need to know OpenCV's raw API
      - pose_detector.py can later receive frames from this class
        without caring where they came from
      - swapping camera sources later (webcam -> video file -> IP
        camera) only requires changing this one class
    """

    def __init__(self, source: int = 0, width: int = 1920, height: int = 1080):
        """
        Args:
            source: camera index (0 = default webcam) or a video file path
            width, height: requested capture resolution
        """
        self.source = source
        self.width = width
        self.height = height
        self.cap = None  # created in start(), not here — see note below

    def start(self):
        """
        Open the video capture device.

        This is separate from __init__ on purpose: creating an object
        shouldn't automatically grab hardware resources. You want to
        control exactly when the camera turns on.
        """
        self.cap = cv2.VideoCapture(self.source)

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        print(f"Camera FPS: {fps}")

        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera source: {self.source}")

        # Request a resolution. The camera may ignore this if it
        # doesn't support the exact size, but most webcams honor it.
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def read_frame(self):
        """
        Grab a single frame from the camera.

        Returns:
            frame (numpy.ndarray) if successful, otherwise None.
            Returning None instead of raising lets the caller decide
            whether a dropped frame is fatal or can just be skipped.
        """
        if self.cap is None:
            raise RuntimeError("Camera not started. Call start() first.")

        success, frame = self.cap.read()
        if not success:
            return None

        # Mirror the frame horizontally so movement feels natural —
        # like looking in a mirror rather than at a security camera.
        # This matters a lot for a fitness app where you're watching
        # yourself move in real time.
        frame = cv2.flip(frame, 1)
        return frame

    def release(self):
        """Release the camera and close any OpenCV windows."""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()


def run_live_preview():
    """
    Milestone 1 verification: open the webcam and show a live video
    window, with no pose detection involved yet.

    This exists purely to prove Camera works on its own before
    pose_detector.py and main.py get wired in. Press 'q' to quit.
    """
    camera = Camera(source=0)
    camera.start()

    print("Live preview started. Press 'q' to quit.")

    try:
        while True:
            frame = camera.read_frame()
            if frame is None:
                print("Failed to read frame from camera.")
                break

            cv2.imshow("AI Fitness Coach - Camera Feed", frame)

            # waitKey(1) checks for a keypress every 1ms without
            # blocking frame capture. The 0xFF mask makes the
            # comparison work consistently across platforms.
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # Always release the camera, even if something goes wrong —
        # otherwise the webcam can stay "locked" by your process.
        camera.release()


if __name__ == "__main__":
    run_live_preview()