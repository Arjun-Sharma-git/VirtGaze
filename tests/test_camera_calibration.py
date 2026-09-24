"""Tests for camera intrinsics: estimation, undistortion, serialisation, calibration.

The chessboard views are synthesised by warping a rendered board, so
``calibrate_from_images`` runs for real without a physical target.  If the
installed OpenCV cannot detect the synthetic boards the calibration test skips
rather than failing on an environment limitation.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from gaze_estimation.utils.camera_calibration import (
    calibrate_from_images,
    estimate_camera_matrix,
    load_intrinsics,
    precompute_undistort_maps,
    save_intrinsics,
    undistort_frame,
    undistort_points,
    zero_dist_coeffs,
)

W, H = 640, 480


@pytest.fixture
def intrinsics():
    return estimate_camera_matrix(W, H)


# ── Estimation ────────────────────────────────────────────────────────────────


def test_estimated_focal_length_is_the_long_edge():
    matrix = estimate_camera_matrix(W, H)
    assert matrix[0, 0] == pytest.approx(float(max(W, H)))
    assert matrix[1, 1] == pytest.approx(float(max(W, H)))


def test_principal_point_is_the_image_centre():
    matrix = estimate_camera_matrix(W, H)
    assert matrix[0, 2] == pytest.approx(W / 2)
    assert matrix[1, 2] == pytest.approx(H / 2)


@pytest.mark.parametrize("size", [(1280, 720), (1920, 1080), (320, 240)])
def test_estimate_handles_any_resolution(size):
    matrix = estimate_camera_matrix(*size)
    assert matrix.shape == (3, 3)
    assert matrix.dtype == np.float64
    assert matrix[2, 2] == pytest.approx(1.0)


def test_zero_distortion_coefficients():
    coeffs = zero_dist_coeffs()
    assert coeffs.shape == (5,)
    assert coeffs.dtype == np.float64
    assert not coeffs.any()


# ── Undistortion ──────────────────────────────────────────────────────────────


def test_undistort_maps_match_the_frame_size(intrinsics):
    map1, map2 = precompute_undistort_maps(intrinsics, zero_dist_coeffs(), W, H)

    assert map1.shape == (H, W)
    assert map2.shape == (H, W)
    assert map1.dtype == np.float32
    assert map2.dtype == np.float32


def test_zero_distortion_maps_are_near_identity(intrinsics):
    """With no lens distortion the maps must not move pixels around."""
    map1, map2 = precompute_undistort_maps(intrinsics, zero_dist_coeffs(), W, H)

    xs = np.tile(np.arange(W, dtype=np.float32), (H, 1))
    ys = np.tile(np.arange(H, dtype=np.float32).reshape(-1, 1), (1, W))
    assert np.allclose(map1, xs, atol=1.0)
    assert np.allclose(map2, ys, atol=1.0)


def test_undistort_frame_preserves_shape_and_content_without_distortion(intrinsics):
    maps = precompute_undistort_maps(intrinsics, zero_dist_coeffs(), W, H)
    frame = np.tile(np.arange(W, dtype=np.uint8), (H, 1))  # horizontal ramp

    out = undistort_frame(frame, *maps)

    assert out.shape == frame.shape
    assert out.dtype == frame.dtype
    assert np.abs(out.astype(int) - frame.astype(int)).mean() < 1.0


def test_undistort_frame_moves_pixels_when_distortion_is_present(intrinsics):
    barrel = np.array([0.35, -0.2, 0.0, 0.0, 0.0])
    maps = precompute_undistort_maps(intrinsics, barrel, W, H)
    frame = np.tile(np.arange(W, dtype=np.uint8), (H, 1))

    out = undistort_frame(frame, *maps)

    assert np.abs(out.astype(int) - frame.astype(int)).mean() > 1.0


def test_undistort_points_is_identity_without_distortion(intrinsics):
    points = np.array([[10.0, 20.0], [300.0, 240.0], [639.0, 479.0]], dtype=np.float32)

    out = undistort_points(points, intrinsics, zero_dist_coeffs())

    assert out.shape == (3, 2)
    np.testing.assert_allclose(out, points, atol=1e-6)


def test_undistort_points_moves_points_with_distortion(intrinsics):
    barrel = np.array([0.35, -0.2, 0.0, 0.0, 0.0])
    points = np.array([[10.0, 20.0]], dtype=np.float64)

    out = undistort_points(points, intrinsics, barrel)

    assert out.shape == (1, 2)
    assert not np.allclose(out, points, atol=1.0)


def test_undistort_points_accepts_float32_and_returns_float64(intrinsics):
    points = np.array([[100.0, 100.0]], dtype=np.float32)

    out = undistort_points(points, intrinsics, zero_dist_coeffs())

    assert out.dtype == np.float64


# ── Serialisation ─────────────────────────────────────────────────────────────


def test_intrinsics_round_trip_through_disk(tmp_path, intrinsics):
    path = str(tmp_path / "camera_intrinsics.npz")
    coeffs = np.array([0.1, -0.05, 0.001, 0.002, 0.0])

    save_intrinsics(path, intrinsics, coeffs)
    loaded_matrix, loaded_coeffs = load_intrinsics(path)

    np.testing.assert_allclose(loaded_matrix, intrinsics)
    np.testing.assert_allclose(loaded_coeffs, coeffs)
    assert loaded_matrix.dtype == np.float64
    assert loaded_coeffs.dtype == np.float64


def test_load_rejects_a_file_without_the_expected_keys(tmp_path):
    path = str(tmp_path / "wrong.npz")
    np.savez(path, something_else=np.zeros(3))

    with pytest.raises(KeyError):
        load_intrinsics(path)


# ── Chessboard calibration ────────────────────────────────────────────────────

BOARD_COLS, BOARD_ROWS = 9, 6  # inner corners


def _render_board(square_px: int = 60) -> np.ndarray:
    """Render a flat chessboard on a white margin."""
    squares_x, squares_y = BOARD_COLS + 1, BOARD_ROWS + 1
    board_w, board_h = squares_x * square_px, squares_y * square_px
    image = np.full((board_h + 2 * square_px, board_w + 2 * square_px), 255, dtype=np.uint8)
    for row in range(squares_y):
        for col in range(squares_x):
            if (row + col) % 2 == 0:
                continue
            y0 = square_px + row * square_px
            x0 = square_px + col * square_px
            image[y0 : y0 + square_px, x0 : x0 + square_px] = 0
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def _warp_into_frame(board: np.ndarray, seed: int, frame_shape=(720, 1280)) -> np.ndarray:
    """Place the board in a frame under a mild random perspective."""
    frame_h, frame_w = frame_shape
    board_h, board_w = board.shape[:2]
    src = np.float32([[0, 0], [board_w, 0], [board_w, board_h], [0, board_h]])
    jitter = np.random.default_rng(seed).uniform(-0.18, 0.18, (4, 2)) * np.array(
        [board_w, board_h], dtype=np.float64
    )
    dst = src + jitter.astype(np.float32)
    homography = cv2.getPerspectiveTransform(src, dst)

    canvas = np.full((frame_h, frame_w, 3), 200, dtype=np.uint8)
    warped = cv2.warpPerspective(board, homography, (frame_w, frame_h))
    mask = cv2.warpPerspective(
        np.full((board_h, board_w), 255, dtype=np.uint8),
        homography,
        (frame_w, frame_h),
        flags=cv2.INTER_NEAREST,
    )
    canvas[mask > 0] = warped[mask > 0]
    return canvas


def _is_detectable(image: np.ndarray) -> bool:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    found, _ = cv2.findChessboardCorners(gray, (BOARD_COLS, BOARD_ROWS))
    return bool(found)


@pytest.fixture
def chessboard_views():
    board = _render_board()
    views = [_warp_into_frame(board, seed) for seed in range(8)]
    detectable = sum(1 for view in views if _is_detectable(view))
    if detectable < 5:
        pytest.skip(f"this OpenCV build detected only {detectable}/8 synthetic chessboards")
    return views


def test_calibration_needs_at_least_five_valid_frames():
    blank = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(10)]

    with pytest.raises(ValueError, match="at least 5"):
        calibrate_from_images(blank)


def test_calibration_reports_how_many_frames_were_valid():
    noise = [
        np.random.default_rng(i).integers(0, 256, (480, 640, 3), dtype=np.uint8) for i in range(3)
    ]

    with pytest.raises(ValueError, match=r"Only \d+ valid chessboard frames"):
        calibrate_from_images(noise)


def test_calibration_recovers_a_camera_matrix(chessboard_views):
    matrix, coeffs, rms = calibrate_from_images(
        chessboard_views, board_cols=BOARD_COLS, board_rows=BOARD_ROWS
    )

    assert matrix.shape == (3, 3)
    assert matrix.dtype == np.float64
    assert matrix[2, 2] == pytest.approx(1.0)
    assert coeffs.ndim == 1 and coeffs.dtype == np.float64
    assert np.isfinite(rms) and rms >= 0.0


def test_calibration_result_is_usable_for_undistortion(chessboard_views):
    matrix, coeffs, _ = calibrate_from_images(
        chessboard_views, board_cols=BOARD_COLS, board_rows=BOARD_ROWS
    )

    map1, map2 = precompute_undistort_maps(matrix, coeffs, 1280, 720)
    frame = np.full((720, 1280, 3), 128, dtype=np.uint8)
    out = undistort_frame(frame, map1, map2)

    assert out.shape == frame.shape


def test_square_size_scales_the_recovered_translation(chessboard_views):
    """A larger square is the same geometry, so the intrinsics must not change."""
    small = calibrate_from_images(chessboard_views, BOARD_COLS, BOARD_ROWS, square_size_mm=10.0)
    large = calibrate_from_images(chessboard_views, BOARD_COLS, BOARD_ROWS, square_size_mm=50.0)

    np.testing.assert_allclose(small[0], large[0], rtol=1e-6)
    np.testing.assert_allclose(small[1], large[1], atol=1e-6)
