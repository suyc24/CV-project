from __future__ import annotations

import time
from dataclasses import dataclass
from threading import Event
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class RGBDFrame:
    color_bgr: np.ndarray
    depth: Optional[np.ndarray]
    confidence: Optional[np.ndarray]
    timestamp: float
    intrinsics: Optional[Any] = None
    source: str = "record3d"


class Record3DCamera:
    """Small adapter around the optional Record3D Python SDK.

    Record3D streams RGB-D data from an iPhone/iPad. The SDK is callback based;
    this class exposes a blocking `read()` method so the main loop can treat it
    like a normal camera source.
    """

    def __init__(
        self,
        device_index: int = 0,
        timeout_seconds: float = 2.0,
        rotate_degrees: int = 0,
        mirror: bool = False,
        depth_unit: str = "auto",
    ) -> None:
        try:
            from record3d import Record3DStream
        except Exception as exc:  # pragma: no cover - depends on optional SDK
            raise RuntimeError(
                "Record3D SDK is not installed. Install the optional dependency with "
                "`pip install record3d`, then enable USB streaming in the Record3D iOS app."
            ) from exc

        self._stream_cls = Record3DStream
        self._session = Record3DStream()
        self._event = Event()
        self._timeout_seconds = timeout_seconds
        self._rotate_degrees = rotate_degrees
        self._mirror = mirror
        self._depth_unit = depth_unit
        self._closed = False
        self._last_event_time: Optional[float] = None
        self._last_event_interval_ms: Optional[float] = None

        self._session.on_new_frame = self._on_new_frame
        self._session.on_stream_stopped = self._on_stream_stopped

        devices = list_connected_record3d_devices(Record3DStream)
        if not devices:
            raise RuntimeError(
                "No Record3D device found. Connect an iPhone/iPad by USB, trust this computer, "
                "open Record3D, and enable Live RGBD Video Streaming over USB."
            )
        if not 0 <= device_index < len(devices):
            raise RuntimeError(f"Record3D device index {device_index} is out of range; found {len(devices)} device(s).")

        self.device = devices[device_index]
        self._session.connect(self.device)
        print(f"Record3D: connected to device index {device_index}: {self.device}")

    @property
    def rotate_degrees(self) -> int:
        return self._rotate_degrees

    def set_rotation(self, rotate_degrees: int) -> None:
        self._rotate_degrees = rotate_degrees % 360

    def rotate_clockwise(self) -> int:
        self._rotate_degrees = (self._rotate_degrees + 90) % 360
        return self._rotate_degrees

    def read(self) -> tuple[bool, Optional[RGBDFrame]]:
        ok, frame, _ = self.read_profiled()
        return ok, frame

    def read_profiled(self) -> Tuple[bool, Optional[RGBDFrame], Dict[str, float]]:
        timings: Dict[str, float] = {}
        read_start = time.perf_counter()
        if self._closed:
            return False, None, timings
        if self._last_event_interval_ms is not None:
            timings["record3d_callback_interval_ms"] = self._last_event_interval_ms
        wait_start = time.perf_counter()
        if not self._event.wait(self._timeout_seconds):
            timings["record3d_wait_ms"] = (time.perf_counter() - wait_start) * 1000.0
            timings["record3d_read_total_ms"] = (time.perf_counter() - read_start) * 1000.0
            return False, None, timings
        timings["record3d_wait_ms"] = (time.perf_counter() - wait_start) * 1000.0
        self._event.clear()

        try:
            step_start = time.perf_counter()
            rgb = self._session.get_rgb_frame()
            timings["record3d_get_rgb_ms"] = (time.perf_counter() - step_start) * 1000.0
            step_start = time.perf_counter()
            depth = self._session.get_depth_frame()
            timings["record3d_get_depth_ms"] = (time.perf_counter() - step_start) * 1000.0
            step_start = time.perf_counter()
            confidence = self._get_optional_confidence()
            timings["record3d_get_confidence_ms"] = (time.perf_counter() - step_start) * 1000.0
            step_start = time.perf_counter()
            intrinsics = self._get_optional_intrinsics()
            timings["record3d_get_intrinsics_ms"] = (time.perf_counter() - step_start) * 1000.0
        except Exception:
            timings["record3d_read_total_ms"] = (time.perf_counter() - read_start) * 1000.0
            return False, None, timings

        if rgb is None:
            timings["record3d_read_total_ms"] = (time.perf_counter() - read_start) * 1000.0
            return False, None, timings

        step_start = time.perf_counter()
        color_bgr = self._rgb_to_bgr(np.asarray(rgb))
        timings["record3d_rgb_convert_ms"] = (time.perf_counter() - step_start) * 1000.0
        step_start = time.perf_counter()
        depth_m = normalize_depth_units(depth, self._depth_unit)
        timings["record3d_depth_normalize_ms"] = (time.perf_counter() - step_start) * 1000.0
        step_start = time.perf_counter()
        confidence_arr = np.asarray(confidence) if confidence is not None else None
        timings["record3d_confidence_convert_ms"] = (time.perf_counter() - step_start) * 1000.0

        step_start = time.perf_counter()
        color_bgr = self._orient_image(color_bgr)
        if depth_m is not None:
            depth_m = self._orient_image(depth_m)
        if confidence_arr is not None:
            confidence_arr = self._orient_image(confidence_arr)
        timings["record3d_orient_ms"] = (time.perf_counter() - step_start) * 1000.0

        frame = RGBDFrame(
            color_bgr=color_bgr,
            depth=depth_m,
            confidence=confidence_arr,
            timestamp=time.perf_counter(),
            intrinsics=intrinsics,
        )
        timings["record3d_read_total_ms"] = (time.perf_counter() - read_start) * 1000.0
        return True, frame, timings

    def release(self) -> None:
        self._closed = True
        try:
            self._session.disconnect()
        except Exception:
            pass

    def _on_new_frame(self) -> None:
        now = time.perf_counter()
        if self._last_event_time is not None:
            self._last_event_interval_ms = (now - self._last_event_time) * 1000.0
        self._last_event_time = now
        self._event.set()

    def _on_stream_stopped(self) -> None:
        self._closed = True
        self._event.set()

    def _get_optional_confidence(self):
        getter = getattr(self._session, "get_confidence_frame", None)
        if getter is None:
            return None
        try:
            return getter()
        except Exception:
            return None

    def _get_optional_intrinsics(self):
        getter = getattr(self._session, "get_camera_intrinsics", None)
        if getter is None:
            return None
        try:
            return getter()
        except Exception:
            return None

    def _rgb_to_bgr(self, rgb: np.ndarray) -> np.ndarray:
        if rgb.ndim == 2:
            return cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)
        if rgb.shape[-1] == 4:
            return cv2.cvtColor(rgb, cv2.COLOR_RGBA2BGR)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def _orient_image(self, image: np.ndarray) -> np.ndarray:
        if self._rotate_degrees == 90:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif self._rotate_degrees == 180:
            image = cv2.rotate(image, cv2.ROTATE_180)
        elif self._rotate_degrees == 270:
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if self._mirror:
            image = cv2.flip(image, 1)
        return np.ascontiguousarray(image)


def list_record3d_devices() -> List[Any]:
    try:
        from record3d import Record3DStream
    except Exception:
        return []
    return list_connected_record3d_devices(Record3DStream)


def list_connected_record3d_devices(stream_cls) -> List[Any]:
    try:
        devices = stream_cls.get_connected_devices()
    except Exception:
        return []
    return list(devices or [])


def normalize_depth_units(depth, depth_unit: str = "auto") -> Optional[np.ndarray]:
    if depth is None:
        return None
    depth_arr = np.asarray(depth, dtype=np.float32)
    if depth_arr.size == 0:
        return None

    finite = depth_arr[np.isfinite(depth_arr) & (depth_arr > 0)]
    if finite.size == 0:
        return depth_arr
    median = float(np.median(finite))

    if depth_unit == "mm" or (depth_unit == "auto" and median > 20.0):
        depth_arr = depth_arr / 1000.0
    elif depth_unit == "cm" or (depth_unit == "auto" and 5.0 < median <= 20.0):
        depth_arr = depth_arr / 100.0
    return depth_arr
