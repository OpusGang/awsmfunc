from typing import TypedDict


class DetectProgressState(TypedDict):
    start_time: float
    frames_done: int
    fps: float
    last_fps_report_time: float
