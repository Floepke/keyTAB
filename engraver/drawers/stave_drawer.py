from __future__ import annotations

from ui.widgets.draw_util import DrawUtil


def stave_drawer(du: DrawUtil, pre_calc: dict) -> None:
    """Draw stave black-key lines from pre-calculated geometry."""
    system = dict(pre_calc.get('system', {}) or {})
    y0 = float(pre_calc.get('y0', 0.0) or 0.0)
    y1 = float(pre_calc.get('y1', 0.0) or 0.0)
    notation_color = tuple(pre_calc.get('notation_color', (0.0, 0.0, 0.0, 1.0)) or (0.0, 0.0, 0.0, 1.0))

    for stv in list(system.get('staves', []) or []):
        stv: dict
        for ln in list(stv.get('black_lines', []) or []):
            x_mm = float(ln.get('x_mm', 0.0) or 0.0)
            du.add_line(
                x_mm,
                y0,
                x_mm,
                y1,
                color=notation_color,
                width_mm=float(ln.get('width_mm', 0.5) or 0.5),
                dash_pattern=ln.get('dash', None),
                id=0,
                tags=['stave'],
            )
