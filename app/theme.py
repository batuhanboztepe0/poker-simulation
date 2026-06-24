"""
theme.py
--------
Single source of truth for all Plotly figure styling in this project.

Every chart factory in charts.py starts with:
    fig.update_layout(**THEME, title=..., xaxis_title=..., ...)

This keeps all margin, font, grid, and background decisions in one place
so regenerating figures after a style change requires touching only this file.
"""

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------

COLORS = {
    # Semantic: directional signal
    "pos":     "#2ecc71",   # green  — positive edge, win, above zero
    "neg":     "#e74c3c",   # red    — negative edge, loss, below zero
    "neutral": "#95a5a6",   # gray   — CI straddles zero, placebo, break-even

    # Curve series (exploitability / learning curves)
    "curve_a": "#2ecc71",   # CFR time-average / trained agent
    "curve_b": "#e67e22",   # CFR last-iterate
    "curve_c": "#e74c3c",   # Q-learning last-iterate
    "curve_d": "#3498db",   # NFSP average-policy

    # Annotation / reference lines
    "zero_line": "#7f8c8d",
    "ribbon":    "rgba(46,204,113,0.15)",  # SEM ribbon fill

    # Heatmap: RdBu diverging stays, but colorscale override available
    "heatmap_scale": "RdBu",
}

# ---------------------------------------------------------------------------
# Base layout — applied to every figure via fig.update_layout(**THEME, ...)
# ---------------------------------------------------------------------------

_FONT = dict(family="Inter, -apple-system, sans-serif", size=13, color="#2c3e50")

_AXIS_COMMON = dict(
    gridcolor="#ecf0f1",
    gridwidth=1,
    linecolor="#bdc3c7",
    linewidth=1,
    tickfont=dict(size=11),
    title_font=dict(size=12),
    zeroline=False,
)

THEME = dict(
    paper_bgcolor="#ffffff",
    plot_bgcolor="#fafbfc",
    font=_FONT,
    margin=dict(t=72, b=80, l=90, r=60),
    xaxis=_AXIS_COMMON,
    yaxis=_AXIS_COMMON,
    legend=dict(
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="#dde1e4",
        borderwidth=1,
        font=dict(size=11),
    ),
    hoverlabel=dict(
        bgcolor="white",
        font_size=12,
        bordercolor="#bdc3c7",
    ),
    title_font=dict(size=15, color="#1a252f"),
    title_x=0.0,
)

# Plain-audience variant — larger title + wider bottom margin for annotation
THEME_PLAIN = {
    **THEME,
    "margin": dict(t=80, b=110, l=90, r=60),
    "title_font": dict(size=21, color="#1a252f"),
    "title_x": 0.5,
    "title_xanchor": "center",
}

# ---------------------------------------------------------------------------
# Helper: wrap long strings for y-axis labels
# ---------------------------------------------------------------------------

def apply_theme(fig, **overrides):
    """
    Apply THEME to a figure, then apply any per-figure overrides.

    Callers pass overrides as keyword arguments exactly as they would to
    fig.update_layout(). Keys in THEME that conflict with overrides are
    replaced by the override value, preventing 'multiple values for keyword
    argument' errors when a factory needs to customise margin, yaxis, etc.

    Usage:
        apply_theme(fig, title="My title", yaxis=dict(type="log"), margin=dict(l=300))

    Args:
        fig: a plotly.graph_objects.Figure.
        **overrides: any update_layout keyword arguments.

    Returns:
        The same figure (mutated in place) for chaining.
    """
    merged = {**THEME}
    merged.update(overrides)
    fig.update_layout(**merged)
    return fig


def wrap_label(text: str, max_chars: int = 52) -> str:
    """
    Insert <br> breaks so Plotly renders long y-axis tick labels across
    multiple lines instead of clipping them.

    Args:
        text: label string, may contain spaces.
        max_chars: target maximum line length (characters).

    Returns:
        String with <br> inserted at word boundaries.
    """
    if len(text) <= max_chars:
        return text
    words = text.split()
    lines, current = [], []
    length = 0
    for word in words:
        if length + len(word) + 1 > max_chars and current:
            lines.append(" ".join(current))
            current, length = [word], len(word)
        else:
            current.append(word)
            length += len(word) + 1
    if current:
        lines.append(" ".join(current))
    return "<br>".join(lines)
