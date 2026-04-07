"""
Screen matching — ORB feature matching against reference screenshots.

During skill building, the agent saves reference screenshots for each screen.
During skill execution, this module compares the current camera frame against
reference screenshots to identify which screen the phone is showing.

Also provides frame differencing for "changed or not" detection, and
brightness checking for popup/overlay detection.

Architecture plan Phase 5: "Screen Matching + Skill Execution"
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

# ─── ORB feature matching ────────────────────────────────────

# ORB config: balance between speed and accuracy
_ORB_FEATURES = 1000      # max keypoints to detect
_MATCH_RATIO = 0.75       # Lowe's ratio test threshold
_MIN_MATCHES = 15         # minimum good matches for a valid identification
_HOMOGRAPHY_THRESH = 5.0  # RANSAC reprojection threshold in pixels


@dataclass
class MatchResult:
    """Result of comparing a camera frame to a reference screenshot."""
    matched: bool           # True if enough good matches found
    confidence: float       # 0-1 score (good_matches / total_keypoints)
    good_matches: int       # number of matches passing ratio test
    total_keypoints: int    # keypoints in reference image
    homography: np.ndarray | None  # 3x3 perspective transform (if matched)
    inliers: int            # RANSAC inlier count


def match_screen(camera_frame: np.ndarray,
                 reference: np.ndarray,
                 min_matches: int = _MIN_MATCHES) -> MatchResult:
    """Compare a camera frame against a reference screenshot using ORB.

    Both images are converted to grayscale internally. The reference can be
    a clean phone screenshot or a previous camera capture.

    Args:
        camera_frame: current BGR image from camera
        reference: BGR reference screenshot to match against
        min_matches: minimum good matches for a positive identification

    Returns:
        MatchResult with match status, confidence, and homography.
    """
    gray_cam = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2GRAY)
    gray_ref = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=_ORB_FEATURES)
    kp_ref, desc_ref = orb.detectAndCompute(gray_ref, None)
    kp_cam, desc_cam = orb.detectAndCompute(gray_cam, None)

    if desc_ref is None or desc_cam is None or len(kp_ref) < 10:
        return MatchResult(
            matched=False, confidence=0.0,
            good_matches=0, total_keypoints=len(kp_ref) if kp_ref else 0,
            homography=None, inliers=0)

    # BFMatcher with Hamming distance (ORB uses binary descriptors)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw_matches = bf.knnMatch(desc_ref, desc_cam, k=2)

    # Lowe's ratio test
    good = []
    for pair in raw_matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < _MATCH_RATIO * n.distance:
                good.append(m)

    total_kp = len(kp_ref)
    confidence = len(good) / total_kp if total_kp > 0 else 0.0

    if len(good) < min_matches:
        return MatchResult(
            matched=False, confidence=round(confidence, 3),
            good_matches=len(good), total_keypoints=total_kp,
            homography=None, inliers=0)

    # Compute homography with RANSAC
    src_pts = np.float32([kp_ref[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_cam[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,
                                  _HOMOGRAPHY_THRESH)
    inliers = int(mask.sum()) if mask is not None else 0

    matched = H is not None and inliers >= min_matches // 2
    return MatchResult(
        matched=matched,
        confidence=round(confidence, 3),
        good_matches=len(good),
        total_keypoints=total_kp,
        homography=H,
        inliers=inliers,
    )


def match_best(camera_frame: np.ndarray,
               references: dict[str, np.ndarray],
               min_matches: int = _MIN_MATCHES) -> tuple[str | None, MatchResult]:
    """Match a camera frame against multiple reference screenshots.

    Args:
        references: dict of {screen_name: reference_image}

    Returns:
        (best_name, best_result) — the screen with highest confidence,
        or (None, empty_result) if nothing matched.
    """
    best_name = None
    best_result = MatchResult(matched=False, confidence=0.0,
                              good_matches=0, total_keypoints=0,
                              homography=None, inliers=0)

    for name, ref in references.items():
        result = match_screen(camera_frame, ref, min_matches)
        if result.confidence > best_result.confidence:
            best_name = name
            best_result = result

    if not best_result.matched:
        return None, best_result
    return best_name, best_result


# ─── Frame differencing (state change detection) ─────────────

def frames_differ(frame_a: np.ndarray, frame_b: np.ndarray,
                  threshold: float = 0.05) -> bool:
    """Check if two frames are visually different.

    Uses normalized histogram comparison. Returns True if the frames
    differ significantly (e.g., screen transition happened).

    Args:
        threshold: difference threshold (0-1). Higher = more different needed.
    """
    gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)

    hist_a = cv2.calcHist([gray_a], [0], None, [64], [0, 256])
    hist_b = cv2.calcHist([gray_b], [0], None, [64], [0, 256])

    cv2.normalize(hist_a, hist_a)
    cv2.normalize(hist_b, hist_b)

    # Correlation: 1.0 = identical, lower = different
    corr = cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL)
    return corr < (1.0 - threshold)


# ─── Popup / overlay detection ───────────────────────────────

def detect_dark_overlay(frame: np.ndarray,
                        dark_ratio_threshold: float = 0.3,
                        brightness_threshold: int = 60) -> bool:
    """Detect a dark overlay (modal popup, alert) on the screen.

    Many apps dim the background when showing a popup. This checks if
    a significant portion of the frame is unusually dark.

    Args:
        dark_ratio_threshold: fraction of pixels that must be dark (0-1)
        brightness_threshold: V-channel value below which a pixel is "dark"
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]
    dark_pixels = np.count_nonzero(v_channel < brightness_threshold)
    ratio = dark_pixels / v_channel.size
    return ratio > dark_ratio_threshold
