from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import cos, sin, radians, sqrt
from typing import Any, Dict, Optional, Tuple, Sequence


class IndicatorShape(Enum):
    ARROW = "arrow"
    DOUBLE_ARROW = "double_arrow"
    BRACKET = "bracket"
    DOUBLE_BRACKET = "double_bracket"
    LINE = "line"
    TICKED = "ticked"
    CUSTOM = "custom"


class DimensionMode(Enum):
    AUTO = "auto"
    D2 = "2d"
    D3 = "3d"


Color = Tuple[float, float, float, float]
Point3D = Tuple[float, float, float]  # (z, y, x) ordering now


def _ensure_point3(p: Sequence[float]) -> Point3D:
    # Updated for (z, y, x). For 2D inputs assume provided (x, y) -> (z=0, y, x)
    if len(p) == 2:
        x, y = float(p[0]), float(p[1])
        return (0.0, y, x)
    if len(p) == 3:
        # Treat incoming triple as already (z, y, x). External code must adapt.
        return (float(p[0]), float(p[1]), float(p[2]))
    raise ValueError(f"Point must have length 2 or 3, got {p}")


def _validate_color(c: Sequence[float]) -> Color:
    if len(c) != 4:
        raise ValueError("Color must be RGBA length 4.")
    rgba = tuple(float(x) for x in c)
    if not all(0.0 <= x <= 1.0 for x in rgba):
        raise ValueError(f"Color components must be in [0,1], got {rgba}")
    return rgba  # type: ignore[return-value]


@dataclass
class ThicknessIndicatorData:
    """
    multibacjend thickness indicator representation.

    Two distinct 'z' concepts:
      - z_index: axial slice index (int) for 2-D overlays.
      - geometric z: first element of start/end tuples.

    If z_index is not provided and geometry is planar (same z in start/end) we
    auto-derive: z_index = round(start[0]).
    For 3D (DimensionMode.D3) z_index remains optional.
    """

    indicator_id: str
    group_id: Optional[str] = None

    # Core geometry
    start: Point3D = (0.0, 0.0, 0.0)
    end: Optional[Point3D] = None
    angle_deg: float | None = None
    distance: float | None = None
    dimension_mode: DimensionMode = DimensionMode.AUTO

    # Measurement & display
    thickness_value: float | None = None
    units: str = "mm"
    value_format: str = "{value:.2f} {units}"
    show_value: bool = True
    rounding: int = 2

    # Shape & style
    indicator_shape: IndicatorShape = IndicatorShape.DOUBLE_ARROW
    color: Color = (1.0, 0.2, 0.2, 1.0)
    label_color: Optional[Color] = None
    line_width: float = 2.0
    arrow_size: float = 8.0
    bracket_size: float = 10.0
    tick_size: float = 6.0
    dash_pattern: Optional[Tuple[int, int]] = None
    opacity: float = 1.0
    visible: bool = True
    z_order: int = 10  # render layering
    z_index: int | None = None  # volume slice index

    # Positional offsets
    label_offset: Point3D = (0.0, 0.0, 0.0)
    global_offset: Point3D = (0.0, 0.0, 0.0)

    # Text
    label: str = "Thickness"
    font_family: str = "default"
    font_size: int = 14
    bold: bool = False
    italic: bool = False

    # Feature toggles
    show_arrows: bool = True
    show_brackets: bool = False
    show_ticks: bool = False

    # Extra data
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Derived internals
    _computed_end: Optional[Point3D] = field(default=None, init=False, repr=False)
    _computed_distance: float | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.start = _ensure_point3(self.start)
        if self.end is not None:
            self.end = _ensure_point3(self.end)

        self.color = _validate_color(self.color)
        self.label_color = (
            _validate_color(self.label_color) if self.label_color else self.color
        )
        self.opacity = max(0.0, min(1.0, float(self.opacity)))

        # Derive end from angle if needed
        if (
            self.end is None
            and self.angle_deg is not None
            and self.distance is not None
        ):
            self._computed_end = self._derive_end_from_angle()
        else:
            self._computed_end = self.end

        # Compute distance if we have both points
        if self._computed_end is not None:
            self._computed_distance = self._calculate_distance(
                self.start, self._computed_end
            )
        else:
            self._computed_distance = self.distance

        if self.distance is None and self._computed_distance is not None:
            self.distance = self._computed_distance
        if self.thickness_value is None:
            self.thickness_value = self.distance

        # Determine dimension mode
        if self.dimension_mode == DimensionMode.AUTO:
            planar = self._computed_end is None or (
                self._computed_end[0] == self.start[0]
            )
            self.dimension_mode = DimensionMode.D2 if planar else DimensionMode.D3

        # Auto z_index if planar and not provided
        if self.z_index is None and self.dimension_mode == DimensionMode.D2:
            z_val = self.start[0]
            # Only set if z is finite and non-negative
            if z_val >= 0:
                self.z_index = int(round(z_val))

        # Validate z_index
        if self.z_index is not None and self.z_index < 0:
            raise ValueError("z_index must be >= 0")

        # Styke adjustments based on shape
        if self.indicator_shape == IndicatorShape.LINE:
            self.show_arrows = False
            self.show_brackets = False
        if self.indicator_shape in (
            IndicatorShape.BRACKET,
            IndicatorShape.DOUBLE_BRACKET,
        ):
            self.show_brackets = True
        if self.indicator_shape == IndicatorShape.TICKED:
            self.show_ticks = True

    # Geometry helpers
    def _derive_end_from_angle(self) -> Point3D:
        # Angle applied in (y, x) plane; z (index 0) constant.
        assert self.angle_deg is not None and self.distance is not None
        theta = radians(self.angle_deg)
        dx = cos(theta) * self.distance
        dy = sin(theta) * self.distance
        sz, sy, sx = self.start
        return (sz, sy + dy, sx + dx)

    @staticmethod
    def _calculate_distance(a: Point3D, b: Point3D) -> float:
        az, ay, ax = a
        bz, by, bx = b
        return sqrt((bz - az) ** 2 + (by - ay) ** 2 + (bx - ax) ** 2)

    @property
    def effective_end(self) -> Optional[Point3D]:
        return self._computed_end

    @property
    def measurement_vector(self) -> Optional[Point3D]:
        if self.effective_end is None:
            return None
        sz, sy, sx = self.start
        ez, ey, ex = self.effective_end
        return (ez - sz, ey - sy, ex - sx)

    # Formatting & serialization
    def formatted_value(self) -> str:
        if not self.show_value or self.thickness_value is None:
            return self.label
        value = round(self.thickness_value, self.rounding)
        dist = round(self.distance or value, self.rounding)
        text = self.value_format.format(value=value, distance=dist, units=self.units)
        return f"{self.label}: {text}" if self.label else text

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.indicator_id,
            "group_id": self.group_id,
            "start": self.start,
            "end": self.effective_end,
            "angle_deg": self.angle_deg,
            "distance": self.distance,
            "thickness_value": self.thickness_value,
            "indicator_shape": self.indicator_shape.value,
            "dimension_mode": self.dimension_mode.value,
            "color": self.color,
            "label_color": self.label_color,
            "line_width": self.line_width,
            "arrow_size": self.arrow_size,
            "bracket_size": self.bracket_size,
            "tick_size": self.tick_size,
            "dash_pattern": self.dash_pattern,
            "opacity": self.opacity,
            "visible": self.visible,
            "z_order": self.z_order,
            "z_index": self.z_index,
            "label_offset": self.label_offset,
            "global_offset": self.global_offset,
            "label": self.label,
            "font_family": self.font_family,
            "font_size": self.font_size,
            "bold": self.bold,
            "italic": self.italic,
            "show_arrows": self.show_arrows,
            "show_brackets": self.show_brackets,
            "show_ticks": self.show_ticks,
            "units": self.units,
            "value_format": self.value_format,
            "show_value": self.show_value,
            "rounding": self.rounding,
            "formatted_value": self.formatted_value(),
            "measurement_vector": self.measurement_vector,
            "metadata": self.metadata,
        }

    def update(self, **kwargs: Any) -> ThicknessIndicatorData:
        data = self.to_dict()
        # Remove derived fields
        data.pop("formatted_value", None)
        data.pop("measurement_vector", None)
        # Rehydrate enums
        data["indicator_shape"] = IndicatorShape(data["indicator_shape"])
        data["dimension_mode"] = DimensionMode(data["dimension_mode"])
        data.update(kwargs)
        return ThicknessIndicatorData(
            indicator_id=data["id"],
            group_id=data.get("group_id"),
            start=data["start"],
            end=data.get("end"),
            angle_deg=data.get("angle_deg"),
            distance=data.get("distance"),
            thickness_value=data.get("thickness_value"),
            indicator_shape=data["indicator_shape"],
            dimension_mode=data["dimension_mode"],
            color=data["color"],
            label_color=data.get("label_color"),
            line_width=data["line_width"],
            arrow_size=data["arrow_size"],
            bracket_size=data["bracket_size"],
            tick_size=data["tick_size"],
            dash_pattern=data.get("dash_pattern"),
            opacity=data["opacity"],
            visible=data["visible"],
            z_order=data["z_order"],
            z_index=data.get("z_index"),
            label_offset=data["label_offset"],
            global_offset=data["global_offset"],
            label=data["label"],
            font_family=data["font_family"],
            font_size=data["font_size"],
            bold=data["bold"],
            italic=data["italic"],
            show_arrows=data["show_arrows"],
            show_brackets=data["show_brackets"],
            show_ticks=data["show_ticks"],
            units=data["units"],
            value_format=data["value_format"],
            show_value=data["show_value"],
            rounding=data["rounding"],
            metadata=data.get("metadata", {}),
        )


def test_auto_z_index_from_planar():
    # Updated expectation: z_index derives from first element (z)
    t = ThicknessIndicatorData(
        indicator_id="remaaining_thickness_1",
        start=(10.0, 5.0, 3.0),  # (z=10 (slice index), y=5, x=3)
        end=(10.0, 15.0, 13.0),  # same z=10 plane, (sicne its mostly 2d), y=15, x=13
        angle_deg=0.0,
    )
    assert t.z_index == 10  # derived from start[0] (z)
    assert (
        t.dimension_mode == DimensionMode.D2
    )  # usually what we want for thickness indicators fr nnunet
