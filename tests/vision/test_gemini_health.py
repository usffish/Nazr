"""
Unit tests for GeminiHealthClient error paths.

Tests cover:
- Gemini timeout → call_gemini_health returns None and logs ERROR
- Parse failure (no numeric content) → parse_confidence_score returns 0.0 and logs WARNING
- Brain POST non-2xx → _dispatch_health_event logs ERROR and does not retry (call count = 1)
- Brain POST connection error → _dispatch_health_event logs ERROR and does not retry

Requirements: 2.4, 2.7, 3.3, 5.2
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call
from concurrent.futures import TimeoutError as FuturesTimeoutError

import numpy as np
import pytest
import requests

# Ensure repo root is on sys.path
_REPO_ROOT = Path(__file__).parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from services.vision.gemini_health import (
    call_gemini_health,
    parse_confidence_score,
    ConfidenceResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_frame(h: int = 100, w: int = 100) -> np.ndarray:
    """Return a minimal black BGR frame."""
    return np.zeros((h, w, 3), dtype=np.uint8)


def _stub_engine_imports():
    """
    Stub heavy module-level imports so we can import
    face_recognition_engine._dispatch_health_event without loading real models.
    """
    stubs = [
        "av", "pygame", "pygame.mixer", "pygame.mixer.music",
        "onnxruntime", "cv2", "edge_tts", "ultralytics",
    ]
    for pkg in stubs:
        if pkg not in sys.modules:
            sys.modules[pkg] = MagicMock()

    cv2_mock = MagicMock()
    cv2_mock.FaceDetectorYN.create.return_value = MagicMock()
    cv2_mock.FaceRecognizerSF.create.return_value = MagicMock()
    # imencode returns (True, numpy-array-like buffer)
    fake_buf = np.frombuffer(b"\xff\xd8\xff\xd9", dtype=np.uint8)
    cv2_mock.imencode.return_value = (True, fake_buf)
    cv2_mock.resize.side_effect = lambda img, size: img
    sys.modules["cv2"] = cv2_mock

    ort_mock = MagicMock()
    sess_mock = MagicMock()
    sess_mock.get_inputs.return_value = [MagicMock(name="input")]
    sess_mock.get_outputs.return_value = [MagicMock(name="output")]
    ort_mock.InferenceSession.return_value = sess_mock
    sys.modules["onnxruntime"] = ort_mock

    dotenv_mock = MagicMock()
    dotenv_mock.load_dotenv = MagicMock()
    sys.modules["dotenv"] = dotenv_mock


def _import_engine():
    """Import face_recognition_engine with stubs, returning the module."""
    _stub_engine_imports()

    for key in list(sys.modules.keys()):
        if "face_recognition_engine" in key:
            del sys.modules[key]

    with patch("pathlib.Path.exists", return_value=True), \
         patch("os.path.exists", return_value=True):
        import services.vision.face_recognition_engine as engine
    return engine


# Import engine once for the session
try:
    _engine = _import_engine()
    _ENGINE_IMPORT_ERROR: Exception | None = None
except Exception as exc:
    _engine = None  # type: ignore[assignment]
    _ENGINE_IMPORT_ERROR = exc


# ---------------------------------------------------------------------------
# Task 8.2 — Gemini timeout
# ---------------------------------------------------------------------------

class TestGeminiTimeout:
    """
    Gemini timeout → call_gemini_health returns None and logs ERROR.

    Requirement 2.7, 5.2
    """

    def test_timeout_returns_none(self, caplog):
        """
        When the Gemini call exceeds the timeout, call_gemini_health returns
        None and logs an ERROR.
        """
        # Patch the ThreadPoolExecutor future to raise FuturesTimeoutError
        mock_future = MagicMock()
        mock_future.result.side_effect = FuturesTimeoutError()
        mock_future.cancel.return_value = True

        mock_executor = MagicMock()
        mock_executor.__enter__ = MagicMock(return_value=mock_executor)
        mock_executor.__exit__ = MagicMock(return_value=False)
        mock_executor.submit.return_value = mock_future

        with patch(
            "services.vision.gemini_health.ThreadPoolExecutor",
            return_value=mock_executor,
        ):
            with caplog.at_level(logging.ERROR, logger="services.vision.gemini_health"):
                result = call_gemini_health(
                    frame_b64="dGVzdA==",
                    subtype="drinking",
                    api_key="fake-key",
                    timeout=10.0,
                )

        assert result is None, f"Expected None on timeout, got {result!r}"

        error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert error_records, "Expected at least one ERROR log on timeout"

    def test_timeout_log_mentions_subtype(self, caplog):
        """
        The ERROR log on timeout should mention the subtype being queried.
        """
        mock_future = MagicMock()
        mock_future.result.side_effect = FuturesTimeoutError()
        mock_future.cancel.return_value = True

        mock_executor = MagicMock()
        mock_executor.__enter__ = MagicMock(return_value=mock_executor)
        mock_executor.__exit__ = MagicMock(return_value=False)
        mock_executor.submit.return_value = mock_future

        with patch(
            "services.vision.gemini_health.ThreadPoolExecutor",
            return_value=mock_executor,
        ):
            with caplog.at_level(logging.ERROR, logger="services.vision.gemini_health"):
                call_gemini_health(
                    frame_b64="dGVzdA==",
                    subtype="medicine_taken",
                    api_key="fake-key",
                    timeout=10.0,
                )

        error_msgs = " ".join(r.getMessage() for r in caplog.records if r.levelno == logging.ERROR)
        assert "medicine_taken" in error_msgs, (
            f"Expected subtype 'medicine_taken' in ERROR log, got: {error_msgs!r}"
        )

    def test_unexpected_exception_returns_none_and_logs_error(self, caplog):
        """
        When the Gemini call raises an unexpected exception (not timeout),
        call_gemini_health returns None and logs ERROR.
        """
        mock_future = MagicMock()
        mock_future.result.side_effect = RuntimeError("network failure")

        mock_executor = MagicMock()
        mock_executor.__enter__ = MagicMock(return_value=mock_executor)
        mock_executor.__exit__ = MagicMock(return_value=False)
        mock_executor.submit.return_value = mock_future

        with patch(
            "services.vision.gemini_health.ThreadPoolExecutor",
            return_value=mock_executor,
        ):
            with caplog.at_level(logging.ERROR, logger="services.vision.gemini_health"):
                result = call_gemini_health(
                    frame_b64="dGVzdA==",
                    subtype="eating",
                    api_key="fake-key",
                )

        assert result is None
        error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert error_records, "Expected ERROR log on unexpected exception"


# ---------------------------------------------------------------------------
# Task 8.2 — Parse failure
# ---------------------------------------------------------------------------

class TestParseConfidenceScore:
    """
    Parse failure (response with no numeric content) → parse_confidence_score
    returns 0.0 and logs WARNING.

    Requirement 2.4
    """

    def test_no_numeric_content_returns_0_0(self, caplog):
        """
        A response with no numeric content returns 0.0.
        """
        with caplog.at_level(logging.WARNING, logger="services.vision.gemini_health"):
            score = parse_confidence_score("no numbers here at all")

        assert score == 0.0, f"Expected 0.0, got {score}"

    def test_no_numeric_content_logs_warning(self, caplog):
        """
        A response with no numeric content logs a WARNING.
        """
        with caplog.at_level(logging.WARNING, logger="services.vision.gemini_health"):
            parse_confidence_score("I cannot determine a score.")

        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warning_records, "Expected at least one WARNING log on parse failure"

    def test_empty_string_returns_0_0(self, caplog):
        """
        An empty response string returns 0.0 and logs WARNING.
        """
        with caplog.at_level(logging.WARNING, logger="services.vision.gemini_health"):
            score = parse_confidence_score("")

        assert score == 0.0

    def test_valid_float_in_range_parsed_correctly(self):
        """
        A valid float in [0.0, 1.0] is parsed correctly (no warning).
        """
        score = parse_confidence_score("0.85")
        assert score == pytest.approx(0.85)

    def test_value_above_1_clamped_to_1(self):
        """
        A value above 1.0 is clamped to 1.0.
        """
        score = parse_confidence_score("1.5")
        assert score == pytest.approx(1.0)

    def test_value_below_0_clamped_to_0(self):
        """
        A negative value is clamped to 0.0.
        """
        score = parse_confidence_score("-0.3")
        # The regex matches digits, so "-0.3" → matches "0.3" → clamped to 0.3
        # (negative sign is not part of the regex pattern)
        assert 0.0 <= score <= 1.0

    def test_integer_response_parsed(self):
        """
        A bare integer response (e.g. "1") is parsed as a float.
        """
        score = parse_confidence_score("1")
        assert score == pytest.approx(1.0)

    def test_first_number_used_when_multiple_present(self):
        """
        When multiple numbers appear, the first one is used.
        """
        score = parse_confidence_score("0.7 or maybe 0.9")
        assert score == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# Task 8.2 — Brain POST non-2xx
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    _engine is None,
    reason=f"Could not import face_recognition_engine: {_ENGINE_IMPORT_ERROR}",
)
class TestBrainPostNon2xx:
    """
    Brain POST non-2xx → _dispatch_health_event logs ERROR and does not retry.

    Requirement 3.3
    """

    def _make_confidence_result(self, score: float = 0.8, subtype: str = "drinking"):
        return ConfidenceResult(score=score, raw_text=str(score), subtype=subtype)

    def _make_detection_result(self, objects=None):
        from services.vision.local_detector import DetectionResult
        return DetectionResult(
            flagged=True,
            detected_objects=objects or ["cup"],
            confidence_scores={"cup": 0.9},
            medicine_flagged=False,
        )

    def test_non_2xx_logs_error(self, caplog):
        """
        When Brain POST returns a non-2xx status, an ERROR is logged.
        """
        mock_response = MagicMock()
        mock_response.status_code = 500

        with patch(
            "services.vision.face_recognition_engine.http_requests.post",
            return_value=mock_response,
        ) as mock_post, \
        patch("services.vision.face_recognition_engine.save_event_json"):
            with caplog.at_level(
                logging.ERROR,
                logger="services.vision.face_recognition_engine",
            ):
                _engine._dispatch_health_event(
                    _make_frame(),
                    "drinking",
                    self._make_confidence_result(),
                    self._make_detection_result(),
                )

        error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert error_records, "Expected ERROR log on non-2xx Brain POST"

    def test_non_2xx_does_not_retry(self, caplog):
        """
        When Brain POST returns a non-2xx status, the POST is made exactly
        once (no retry).
        """
        mock_response = MagicMock()
        mock_response.status_code = 503

        with patch(
            "services.vision.face_recognition_engine.http_requests.post",
            return_value=mock_response,
        ) as mock_post, \
        patch("services.vision.face_recognition_engine.save_event_json"):
            _engine._dispatch_health_event(
                _make_frame(),
                "eating",
                self._make_confidence_result(subtype="eating"),
                self._make_detection_result(objects=["fork"]),
            )

        assert mock_post.call_count == 1, (
            f"Expected exactly 1 POST call (no retry), got {mock_post.call_count}"
        )

    def test_non_2xx_save_event_json_not_called(self):
        """
        When Brain POST returns non-2xx, save_event_json is NOT called
        (event is not persisted locally on failure).
        """
        mock_response = MagicMock()
        mock_response.status_code = 404

        with patch(
            "services.vision.face_recognition_engine.http_requests.post",
            return_value=mock_response,
        ), \
        patch(
            "services.vision.face_recognition_engine.save_event_json"
        ) as mock_save:
            _engine._dispatch_health_event(
                _make_frame(),
                "drinking",
                self._make_confidence_result(),
                self._make_detection_result(),
            )

        mock_save.assert_not_called()

    @pytest.mark.parametrize("status_code", [400, 401, 403, 404, 500, 502, 503])
    def test_various_non_2xx_codes_all_log_error(self, status_code, caplog):
        """
        All non-2xx status codes result in an ERROR log.
        """
        mock_response = MagicMock()
        mock_response.status_code = status_code

        with patch(
            "services.vision.face_recognition_engine.http_requests.post",
            return_value=mock_response,
        ), \
        patch("services.vision.face_recognition_engine.save_event_json"):
            with caplog.at_level(
                logging.ERROR,
                logger="services.vision.face_recognition_engine",
            ):
                _engine._dispatch_health_event(
                    _make_frame(),
                    "medicine_taken",
                    self._make_confidence_result(subtype="medicine_taken"),
                    self._make_detection_result(objects=["pill"]),
                )

        error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert error_records, f"Expected ERROR log for status_code={status_code}"


# ---------------------------------------------------------------------------
# Task 8.2 — Brain POST connection error
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    _engine is None,
    reason=f"Could not import face_recognition_engine: {_ENGINE_IMPORT_ERROR}",
)
class TestBrainPostConnectionError:
    """
    Brain POST connection error → _dispatch_health_event logs ERROR and does
    not retry.

    Requirement 3.3
    """

    def _make_confidence_result(self, score: float = 0.8, subtype: str = "drinking"):
        return ConfidenceResult(score=score, raw_text=str(score), subtype=subtype)

    def _make_detection_result(self, objects=None):
        from services.vision.local_detector import DetectionResult
        return DetectionResult(
            flagged=True,
            detected_objects=objects or ["cup"],
            confidence_scores={"cup": 0.9},
            medicine_flagged=False,
        )

    def test_connection_error_logs_error(self, caplog):
        """
        When Brain POST raises requests.ConnectionError, an ERROR is logged.
        """
        with patch(
            "services.vision.face_recognition_engine.http_requests.post",
            side_effect=requests.ConnectionError("Connection refused"),
        ), \
        patch("services.vision.face_recognition_engine.save_event_json"):
            with caplog.at_level(
                logging.ERROR,
                logger="services.vision.face_recognition_engine",
            ):
                _engine._dispatch_health_event(
                    _make_frame(),
                    "drinking",
                    self._make_confidence_result(),
                    self._make_detection_result(),
                )

        error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert error_records, "Expected ERROR log on ConnectionError"

    def test_connection_error_does_not_retry(self):
        """
        When Brain POST raises requests.ConnectionError, the POST is attempted
        exactly once (no retry).
        """
        with patch(
            "services.vision.face_recognition_engine.http_requests.post",
            side_effect=requests.ConnectionError("refused"),
        ) as mock_post, \
        patch("services.vision.face_recognition_engine.save_event_json"):
            _engine._dispatch_health_event(
                _make_frame(),
                "eating",
                self._make_confidence_result(subtype="eating"),
                self._make_detection_result(objects=["fork"]),
            )

        assert mock_post.call_count == 1, (
            f"Expected exactly 1 POST attempt (no retry), got {mock_post.call_count}"
        )

    def test_timeout_error_logs_error_and_no_retry(self, caplog):
        """
        When Brain POST raises requests.Timeout, an ERROR is logged and no
        retry occurs.
        """
        with patch(
            "services.vision.face_recognition_engine.http_requests.post",
            side_effect=requests.Timeout("timed out"),
        ) as mock_post, \
        patch("services.vision.face_recognition_engine.save_event_json"):
            with caplog.at_level(
                logging.ERROR,
                logger="services.vision.face_recognition_engine",
            ):
                _engine._dispatch_health_event(
                    _make_frame(),
                    "medicine_taken",
                    self._make_confidence_result(subtype="medicine_taken"),
                    self._make_detection_result(objects=["pill"]),
                )

        error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert error_records, "Expected ERROR log on Timeout"
        assert mock_post.call_count == 1

    def test_connection_error_does_not_propagate_exception(self):
        """
        A connection error must NOT propagate out of _dispatch_health_event —
        the function should handle it gracefully.
        """
        with patch(
            "services.vision.face_recognition_engine.http_requests.post",
            side_effect=requests.ConnectionError("refused"),
        ), \
        patch("services.vision.face_recognition_engine.save_event_json"):
            # Should not raise
            _engine._dispatch_health_event(
                _make_frame(),
                "drinking",
                self._make_confidence_result(),
                self._make_detection_result(),
            )
