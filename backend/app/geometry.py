"""
geometry.py

Turns raw landmark coordinates into the measurements a fitness app
actually cares about: angles and distances.

This sits between "where is the body" (pose_detector.py) and "what
exercise is happening" (Phase 3). It has zero knowledge of curls,
squats, or reps — it only knows how to do the math. Exercise logic in
Phase 3 will call these functions but live in its own file.
"""

import math

# Named indices for MediaPipe's 33-point pose model, so exercise code
# can write LEFT_ELBOW instead of the magic number 13. Only the ones
# we need for now are listed — add more here as later phases need them
# (full list: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker)
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28


def calculate_angle(point_a, point_b, point_c):
    """
    Calculate the angle at point_b, formed by the lines b->a and b->c.

    This is the core building block for rep counting. For an elbow
    angle: point_a = shoulder, point_b = elbow, point_c = wrist. That
    angle shrinks as you curl a weight up and opens back toward 180
    degrees as you extend — a curl counter in Phase 3 is really just
    this function plus a threshold check.

    Args:
        point_a, point_b, point_c: dicts with "x" and "y" keys (pixel
            coordinates), e.g. items from PoseDetector.find_landmarks()

    Returns:
        angle in degrees, between 0 and 180
    """
    a = (point_a["x"], point_a["y"])
    b = (point_b["x"], point_b["y"])
    c = (point_c["x"], point_c["y"])

    # atan2 gives the angle of a vector relative to the x-axis;
    # subtracting the two vector angles gives the angle between them.
    # This is more robust than a dot-product formula — it won't
    # divide by zero or lose direction information at extreme poses.
    radians = (
        math.atan2(c[1] - b[1], c[0] - b[0])
        - math.atan2(a[1] - b[1], a[0] - b[0])
    )
    angle = abs(radians * 180.0 / math.pi)

    # A joint angle is always described as the smaller angle between
    # the two segments (0-180), but atan2 subtraction can return up
    # to 360 depending on direction — this folds it back down.
    if angle > 180:
        angle = 360 - angle

    return angle


def calculate_distance(point_a, point_b):
    """
    Straight-line (Euclidean) distance between two landmarks, in pixels.

    Useful for things later phases will need: checking whether wrists
    are above shoulders, or normalizing measurements relative to body
    size (e.g. shoulder width) so thresholds work regardless of how
    close someone is standing to the camera.

    Args:
        point_a, point_b: dicts with "x" and "y" keys

    Returns:
        distance in pixels
    """
    return math.hypot(point_b["x"] - point_a["x"], point_b["y"] - point_a["y"])