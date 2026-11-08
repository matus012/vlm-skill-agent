"""The robot as the skills see it (INTERFACES section 1). No ROS here; `adapters/` implements `Adapter`.

All poses are map frame, metres / radians. The VLM never sees any of them: it gets only `Photo.image`."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Frame:
    """One camera capture, already cropped by the robot's self_crop."""
    rgb: Image.Image          # cropped RGB, what the VLM sees
    seg: np.ndarray | None    # cropped label image (uint8/uint16 label id per pixel), same crop; scorer only
    meta: dict[str, Any]      # sim_time_s, heading_target_rad|None, loc_pose [x,y,yaw], crop {top_px,bottom_px}


@dataclass(frozen=True)
class Photo:
    """A photo as shown in one packet: its number there (1-based; report(photo=n) refers to it) and the image.
    `frame` is where it came from, for the run log only; the context builder uses `image`."""
    number: int
    image: Image.Image
    frame: Frame | None = None


@dataclass(frozen=True)
class GoResult:
    status: str       # "reached" | "blocked"  (Nav2 ABORTED/CANCELED/REJECTED or timeout -> "blocked")
    nav_status: str   # raw Nav2 status name or "TIMEOUT"
    sim_s: float
    wall_s: float


class Adapter(ABC):
    @abstractmethod
    def go_to(self, x: float, y: float, yaw: float | None, timeout_s: float) -> GoResult:
        """Nav2 goal; waits. timeout_s is sim seconds."""

    @abstractmethod
    def turn_to(self, yaw: float) -> float:
        """Absolute heading (localization); returns the final error in rad."""

    @abstractmethod
    def photo(self) -> Frame:
        """A frame stamped after the robot settled."""

    @abstractmethod
    def pose(self) -> tuple[float, float, float]:
        """Localization (AMCL) x, y, yaw."""

    @abstractmethod
    def where(self) -> str:
        """Room name of pose()."""

    @abstractmethod
    def stop(self) -> None:
        """Zero velocity, cancel any goal."""

    @abstractmethod
    def sim_time(self) -> float: ...
