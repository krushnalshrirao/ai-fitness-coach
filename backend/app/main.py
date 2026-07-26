"""
main.py

Entry point for Phase 1 of the AI Fitness Coach.

Wires the two pieces built so far into the pipeline described in the
roadmap:

    Camera -> Pose Detector -> Skeleton -> Display

No rep counting, no form analysis yet — this file exists purely to
prove the foundation works end-to-end before Phase 2 (geometry: turning
landmarks into angles and distances) and Phase 3 (exercise logic) build
on top of it.
"""

import cv2

from camera import Camera
from pose_detector import PoseDetector
from geometry import calculate_angle, LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST


def main():
    camera = Camera(source=0)
    camera.start()
    detector = PoseDetector()

    print("AI Fitness Coach starting. Press 'q' to quit.")

    try:
        while True:
            frame = camera.read_frame()
            if frame is None:
                print("Failed to read frame from camera. Stopping.")
                break

            frame = detector.find_pose(frame, draw=True)
            landmarks = detector.find_landmarks(frame)

            # Simple on-screen status — useful right now for confirming
            # detection is working; Phase 4 will replace this with real
            # form-analysis feedback.
            if landmarks:
                status_text = f"Landmarks detected: {len(landmarks)}"
                status_color = (0, 255, 0)  # green
            else:
                status_text = "No person detected"
                status_color = (0, 0, 255)  # red

            cv2.putText(
                frame,
                status_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                status_color,
                2,
            )

            # Phase 2 verification: prove geometry.py works by showing
            # a live left elbow angle. Full list always has 33 entries
            # when a person is detected, so indexing by landmark id
            # here is safe.
            if landmarks:
                elbow_angle = calculate_angle(
                    landmarks[LEFT_SHOULDER],
                    landmarks[LEFT_ELBOW],
                    landmarks[LEFT_WRIST],
                )
                cv2.putText(
                    frame,
                    f"Left elbow angle: {int(elbow_angle)} deg",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2,
                )

            cv2.imshow("AI Fitness Coach", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # Always clean up both resources, even if something throws —
        # an unreleased camera or landmarker can cause weird behavior
        # the next time you run the app.
        camera.release()
        detector.close()
        print("Shut down cleanly.")


if __name__ == "__main__":
    main()