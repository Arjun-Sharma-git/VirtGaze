"""Tests for ProfileManager: CRUD + recalibration logic."""
from __future__ import annotations

import os

import numpy as np
import pytest

from gaze_estimation.profile.profile_manager import ProfileManager


@pytest.fixture
def tmp_profile_dir(tmp_path):
    return str(tmp_path / "profiles")


def test_create_profile(tmp_profile_dir):
    pm = ProfileManager(tmp_profile_dir)
    path = pm.create_profile("alice", (1920, 1080), "test_cam")
    assert os.path.isdir(path)


def test_save_and_load_profile(tmp_profile_dir):
    pm = ProfileManager(tmp_profile_dir)
    cm = np.eye(3, dtype=np.float64)
    dc = np.zeros(5, dtype=np.float64)
    pm.save_calibration(
        user_id="bob",
        kappa_yaw=2.5, kappa_pitch=-1.0,
        eyeball_radius=12.0,
        mlp_path="models/bob.pt",
        screen_resolution=(1920, 1080),
        camera_name="cam0",
        camera_matrix=cm,
        dist_coeffs=dc,
        calibration_samples=2500,
    )
    profile = pm.load_profile("bob")
    assert profile is not None
    assert profile.user_id == "bob"
    assert abs(profile.kappa_yaw - 2.5) < 1e-6
    assert profile.calibration_samples == 2500


def test_list_profiles(tmp_profile_dir):
    pm = ProfileManager(tmp_profile_dir)
    cm = np.eye(3)
    dc = np.zeros(5)
    for uid in ["user1", "user2", "user3"]:
        pm.save_calibration(uid, 0.0, 0.0, 12.0, "x.pt", (1920, 1080),
                            "cam", cm, dc, 100)
    profiles = pm.list_profiles()
    assert "user1" in profiles
    assert "user2" in profiles
    assert len(profiles) == 3


def test_load_missing_profile(tmp_profile_dir):
    pm = ProfileManager(tmp_profile_dir)
    assert pm.load_profile("nobody") is None


def test_needs_recalibration_camera_change(tmp_profile_dir):
    pm = ProfileManager(tmp_profile_dir)
    cm = np.eye(3)
    dc = np.zeros(5)
    pm.save_calibration("alice", 0.0, 0.0, 12.0, "x.pt", (1920, 1080),
                        "old_cam", cm, dc, 100)
    assert pm.needs_recalibration("alice", "new_cam", (1920, 1080))


def test_needs_recalibration_screen_change(tmp_profile_dir):
    pm = ProfileManager(tmp_profile_dir)
    cm = np.eye(3)
    dc = np.zeros(5)
    pm.save_calibration("carol", 0.0, 0.0, 12.0, "x.pt", (1920, 1080),
                        "cam", cm, dc, 100)
    assert pm.needs_recalibration("carol", "cam", (2560, 1440))


def test_delete_profile(tmp_profile_dir):
    pm = ProfileManager(tmp_profile_dir)
    cm = np.eye(3)
    dc = np.zeros(5)
    pm.save_calibration("dave", 0.0, 0.0, 12.0, "x.pt", (1920, 1080),
                        "cam", cm, dc, 100)
    pm.delete_profile("dave")
    assert pm.load_profile("dave") is None
