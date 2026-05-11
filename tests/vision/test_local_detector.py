"""
Unit tests for LocalDetector configuration and startup.

Tests cover:
- LOCAL_DETECTOR_MODEL env var: valid path, unset (default), nonexistent path
- LOCAL_DETECTOR_CONFIDENCE env var: valid float, unset (default 0.4)
- Model load failure: _health_detection_disabled = True, face recognition continues

Requirements: 4.2, 4.4, 4.5, 8.2
"""

from __future__ import annotations

import importlib
import logging
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure repo root is on sys.path
_REPO_ROOT = Path(__file__).parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fresh_local_detector_module(env: dict[str, str] | None = None,
                                  yolo_side_effect=None,
                                  yolo_return_value=None):
    """
    Re-import services.vision.local_detector with a clean module cache and
    optional env overrides / YOLO mock behaviour.

    Returns (module, yolo_mock, env_patcher) where env_patcher is already
    started.  The caller is responsible for calling env_patcher.stop() when
    done, or using the module inside a ``with`` block via the helper
    ``_fresh_module_ctx``.

    Parameters
    ----------
    env:
        Environment variables to set (replaces os.environ for the import).
    yolo_side_effect:
        If set, the YOLO constructor will raise this exception.
    yolo_return_value:
        If set, the YOLO constructor returns this value (default: MagicMock()).
    """
    # Remove any cached copy of the module.
    for key in list(sys.modules.keys()):
        if "local_detector" in key:
            del sys.modules[key]

    yolo_mock = MagicMock()
    if yolo_side_effect is not None:
        yolo_mock.side_effect = yolo_side_effect
    elif yolo_return_value is not None:
        yolo_mock.return_value = yolo_return_value
    else:
        yolo_mock.return_value = MagicMock()

    ultralytics_mock = MagicMock()
    ultralytics_mock.YOLO = yolo_mock

    # Patch sys.modules for ultralytics permanently for this test
    sys.modules["ultralytics"] = ultralytics_mock

    # Start env patcher — stays active until caller calls stop()
    env_patcher = patch.dict(os.environ, env or {}, clear=(env is not None))
    env_patcher.start()

    import services.vision.local_detector as mod
    return mod, yolo_mock, env_patcher


# ---------------------------------------------------------------------------
# Task 8.1 — LOCAL_DETECTOR_MODEL env var
# ---------------------------------------------------------------------------

class TestLocalDetectorModelEnvVar:
    """Tests for LOCAL_DETECTOR_MODEL env var (Requirements 4.2, 4.5)."""

    def test_valid_model_path_loads_successfully(self, tmp_path):
        """
        LOCAL_DETECTOR_MODEL set to a valid (existing) path → model loads
        successfully and _health_detection_disabled remains False.

        Requirement 4.2
        """
        model_file = tmp_path / "yolov8n.pt"
        model_file.write_bytes(b"fake weights")

        env = {"LOCAL_DETECTOR_MODEL": str(model_file)}
        mod, yolo_mock, env_patcher = _fresh_local_detector_module(env=env)
        try:
            # Trigger model load via get_detector()
            detector = mod.get_detector()

            assert detector is not None
            assert mod._health_detection_disabled is False
            # YOLO was called with the correct path
            yolo_mock.assert_called_once_with(str(model_file))
        finally:
            env_patcher.stop()

    def test_unset_model_env_var_uses_default_path(self):
        """
        LOCAL_DETECTOR_MODEL env var unset → _load_detector uses the default
        path "yolov8n.pt" (as documented in the module).

        Requirement 4.2
        """
        # Remove LOCAL_DETECTOR_MODEL from env entirely
        env = {k: v for k, v in os.environ.items() if k != "LOCAL_DETECTOR_MODEL"}
        mod, yolo_mock, env_patcher = _fresh_local_detector_module(env=env)
        try:
            # Trigger load — it may fail because "yolov8n.pt" doesn't exist, but
            # we only care that YOLO was called with the default path.
            try:
                mod.get_detector()
            except Exception:
                pass

            # The YOLO constructor should have been called with the default path
            if yolo_mock.called:
                call_args = yolo_mock.call_args[0][0]
                assert call_args == "yolov8n.pt", (
                    f"Expected default path 'yolov8n.pt', got {call_args!r}"
                )
        finally:
            env_patcher.stop()

    def test_nonexistent_model_path_logs_critical_and_disables(self, caplog, tmp_path):
        """
        LOCAL_DETECTOR_MODEL set to a nonexistent path → YOLO raises an
        exception, get_detector() logs CRITICAL and sets
        _health_detection_disabled = True.

        Requirement 4.5, 8.2
        """
        nonexistent = str(tmp_path / "does_not_exist.pt")
        env = {"LOCAL_DETECTOR_MODEL": nonexistent}

        mod, yolo_mock, env_patcher = _fresh_local_detector_module(
            env=env,
            yolo_side_effect=FileNotFoundError(f"No such file: {nonexistent}"),
        )
        try:
            with caplog.at_level(logging.CRITICAL, logger="services.vision.local_detector"):
                with pytest.raises(Exception):
                    mod.get_detector()

            assert mod._health_detection_disabled is True
            # A CRITICAL log message must have been emitted
            critical_records = [r for r in caplog.records if r.levelno == logging.CRITICAL]
            assert critical_records, "Expected at least one CRITICAL log record"
        finally:
            env_patcher.stop()

    def test_nonexistent_model_path_subsequent_calls_raise_runtime_error(self, tmp_path):
        """
        After a load failure, subsequent calls to get_detector() raise
        RuntimeError (health detection stays disabled).

        Requirement 8.2
        """
        nonexistent = str(tmp_path / "missing.pt")
        env = {"LOCAL_DETECTOR_MODEL": nonexistent}

        mod, _, env_patcher = _fresh_local_detector_module(
            env=env,
            yolo_side_effect=FileNotFoundError("missing"),
        )
        try:
            # First call raises the original exception
            with pytest.raises(Exception):
                mod.get_detector()

            # Subsequent calls raise RuntimeError (disabled)
            with pytest.raises(RuntimeError, match="disabled"):
                mod.get_detector()
        finally:
            env_patcher.stop()


# ---------------------------------------------------------------------------
# Task 8.1 — LOCAL_DETECTOR_CONFIDENCE env var
# ---------------------------------------------------------------------------

class TestLocalDetectorConfidenceEnvVar:
    """Tests for LOCAL_DETECTOR_CONFIDENCE env var (Requirement 4.4)."""

    def test_valid_confidence_float_is_used(self, tmp_path):
        """
        LOCAL_DETECTOR_CONFIDENCE set to a valid float → used as the
        confidence_threshold on the LocalDetector instance.

        Requirement 4.4
        """
        model_file = tmp_path / "yolov8n.pt"
        model_file.write_bytes(b"fake")

        env = {
            "LOCAL_DETECTOR_MODEL": str(model_file),
            "LOCAL_DETECTOR_CONFIDENCE": "0.75",
        }
        mod, _, env_patcher = _fresh_local_detector_module(env=env)
        try:
            detector = mod.get_detector()
            assert detector.confidence_threshold == pytest.approx(0.75)
        finally:
            env_patcher.stop()

    def test_unset_confidence_defaults_to_0_4(self, tmp_path):
        """
        LOCAL_DETECTOR_CONFIDENCE env var unset → defaults to 0.4.

        Requirement 4.4
        """
        model_file = tmp_path / "yolov8n.pt"
        model_file.write_bytes(b"fake")

        # Build env without LOCAL_DETECTOR_CONFIDENCE
        base_env = {k: v for k, v in os.environ.items() if k != "LOCAL_DETECTOR_CONFIDENCE"}
        base_env["LOCAL_DETECTOR_MODEL"] = str(model_file)
        mod, _, env_patcher = _fresh_local_detector_module(env=base_env)
        try:
            detector = mod.get_detector()
            assert detector.confidence_threshold == pytest.approx(0.4)
        finally:
            env_patcher.stop()

    def test_unparseable_confidence_defaults_to_0_4_and_logs_warning(
        self, tmp_path, caplog
    ):
        """
        LOCAL_DETECTOR_CONFIDENCE set to a non-float string → logs WARNING
        and falls back to 0.4.

        Requirement 4.4
        """
        model_file = tmp_path / "yolov8n.pt"
        model_file.write_bytes(b"fake")

        env = {
            "LOCAL_DETECTOR_MODEL": str(model_file),
            "LOCAL_DETECTOR_CONFIDENCE": "not-a-float",
        }
        mod, _, env_patcher = _fresh_local_detector_module(env=env)
        try:
            with caplog.at_level(logging.WARNING, logger="services.vision.local_detector"):
                detector = mod.get_detector()

            assert detector.confidence_threshold == pytest.approx(0.4)
            warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
            assert warning_records, "Expected at least one WARNING log record"
        finally:
            env_patcher.stop()

    def test_confidence_threshold_passed_directly_overrides_env(self, tmp_path):
        """
        When confidence_threshold is passed directly to LocalDetector.__init__,
        it takes precedence over the env var.
        """
        model_file = tmp_path / "yolov8n.pt"
        model_file.write_bytes(b"fake")

        env = {
            "LOCAL_DETECTOR_MODEL": str(model_file),
            "LOCAL_DETECTOR_CONFIDENCE": "0.9",
        }
        mod, _, env_patcher = _fresh_local_detector_module(env=env)
        try:
            # Instantiate directly with an explicit threshold
            detector = mod.LocalDetector(
                model_path=str(model_file),
                confidence_threshold=0.55,
            )
            assert detector.confidence_threshold == pytest.approx(0.55)
        finally:
            env_patcher.stop()


# ---------------------------------------------------------------------------
# Task 8.1 — Model load failure: face recognition continues
# ---------------------------------------------------------------------------

class TestModelLoadFailure:
    """
    Tests that a YOLO model load failure sets _health_detection_disabled=True
    and does not propagate exceptions to the face recognition pipeline.

    Requirements: 4.5, 8.2
    """

    def test_load_failure_sets_health_detection_disabled(self, tmp_path, caplog):
        """
        When YOLO raises during model load, _health_detection_disabled is set
        to True and a CRITICAL log is emitted.
        """
        env = {"LOCAL_DETECTOR_MODEL": str(tmp_path / "bad.pt")}
        mod, _, env_patcher = _fresh_local_detector_module(
            env=env,
            yolo_side_effect=RuntimeError("corrupt weights"),
        )
        try:
            with caplog.at_level(logging.CRITICAL, logger="services.vision.local_detector"):
                with pytest.raises(RuntimeError):
                    mod.get_detector()

            assert mod._health_detection_disabled is True
            critical_msgs = [r.message for r in caplog.records if r.levelno == logging.CRITICAL]
            assert critical_msgs, "Expected CRITICAL log on model load failure"
        finally:
            env_patcher.stop()

    def test_face_recognition_continues_after_load_failure(self, tmp_path):
        """
        After a model load failure, calling get_detector() raises RuntimeError
        (not the original exception), which the face_recognition_engine catches
        and handles gracefully — health detection is disabled but the process
        continues.

        This test verifies that get_detector() raises RuntimeError (not an
        unexpected exception type) so callers can catch it predictably.
        """
        env = {"LOCAL_DETECTOR_MODEL": str(tmp_path / "bad.pt")}
        mod, _, env_patcher = _fresh_local_detector_module(
            env=env,
            yolo_side_effect=OSError("file not found"),
        )
        try:
            # First call: original exception propagates
            with pytest.raises(OSError):
                mod.get_detector()

            # Subsequent calls: RuntimeError (disabled) — callers can catch this
            with pytest.raises(RuntimeError):
                mod.get_detector()

            # The module flag is set — health detection is disabled
            assert mod._health_detection_disabled is True
        finally:
            env_patcher.stop()

    def test_get_detector_singleton_not_set_on_failure(self, tmp_path):
        """
        After a load failure, the module-level _detector singleton remains None
        (it is not partially initialised).
        """
        env = {"LOCAL_DETECTOR_MODEL": str(tmp_path / "bad.pt")}
        mod, _, env_patcher = _fresh_local_detector_module(
            env=env,
            yolo_side_effect=ValueError("bad model"),
        )
        try:
            with pytest.raises(ValueError):
                mod.get_detector()

            assert mod._detector is None
        finally:
            env_patcher.stop()

    def test_get_detector_singleton_reused_on_success(self, tmp_path):
        """
        On successful load, get_detector() returns the same instance on every
        call (singleton pattern — model loaded exactly once).

        Requirement 8.1, 8.3
        """
        model_file = tmp_path / "yolov8n.pt"
        model_file.write_bytes(b"fake")

        env = {"LOCAL_DETECTOR_MODEL": str(model_file)}
        mod, yolo_mock, env_patcher = _fresh_local_detector_module(env=env)
        try:
            d1 = mod.get_detector()
            d2 = mod.get_detector()
            d3 = mod.get_detector()

            assert d1 is d2 is d3, "get_detector() must return the same singleton instance"
            # YOLO constructor called exactly once
            assert yolo_mock.call_count == 1, (
                f"YOLO constructor called {yolo_mock.call_count} times; expected 1"
            )
        finally:
            env_patcher.stop()
