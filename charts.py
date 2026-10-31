"""
Chart helpers for the Employee Attrition Risk Predictor.
- make_risk_gauge(): semicircular risk gauge with colored zones + needle
- make_shap_chart(): styled horizontal bar chart of top SHAP drivers
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Wedge
import numpy as np


# Semantic palette — must match app.py constants
COLOR_LOW   = "#0EA5A4"
COLOR_MED   = "#F59E0B"
COLOR_HIGH  = "#E11D48"
COLOR_TEXT  = "#1A2330"
COLOR_MUTED = "#64748B"
COLOR_PUSH_UP = "#E11D48"
COLOR_PUSH_DOWN = "#0EA5A4"


def make_risk_gauge(prob, threshold, high_cutoff, figsize=(5, 3.4)):
    """
    Semicircular gauge from 0% to 100% with three colored zones
    (low / medium / high) and a needle indicating the current probability.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=120)

    # Standard gauge convention: 0% on the LEFT (180°), 100% on the RIGHT (0°).
    # Probability p maps to needle angle  = pi * (1 - p).
    # Therefore zone boundaries (in degrees, measured CCW from east):
    #   threshold    -> angle = (1 - threshold) * 180      (LOW/MED boundary, left side)
    #   high_cutoff  -> angle = (1 - high_cutoff) * 180    (MED/HIGH boundary, right side)
    low_med_angle  = (1 - threshold) * 180      # ~117° for threshold=0.35
    med_high_angle = (1 - high_cutoff) * 180    # ~81°  for high_cutoff=0.55

    cx, cy = 0.5, 0.1
    r_outer, r_inner = 0.42, 0.30

    # 1) Gray track (full semicircle), then three colored zones on top.
    #    LOW (teal) on the LEFT, MED (amber) in the MIDDLE, HIGH (red) on the RIGHT.
    ax.add_patch(Wedge((cx, cy), r_outer, 0, 180,
                       width=r_outer - r_inner,
                       facecolor="#E5E7EB", edgecolor="none"))
    ax.add_patch(Wedge((cx, cy), r_outer, 0, med_high_angle,
                       width=r_outer - r_inner,
                       facecolor=COLOR_HIGH, alpha=0.55, edgecolor="none"))
    ax.add_patch(Wedge((cx, cy), r_outer, med_high_angle, low_med_angle,
                       width=r_outer - r_inner,
                       facecolor=COLOR_MED, alpha=0.55, edgecolor="none"))
    ax.add_patch(Wedge((cx, cy), r_outer, low_med_angle, 180,
                       width=r_outer - r_inner,
                       facecolor=COLOR_LOW, alpha=0.55, edgecolor="none"))

    # Threshold tick marks
    for frac, label in [(threshold, None), (high_cutoff, None)]:
        angle = np.pi * (1 - frac)  # 0..1 maps to pi..0
        x1 = cx + r_inner * np.cos(angle)
        y1 = cy + r_inner * np.sin(angle)
        x2 = cx + r_outer * np.cos(angle)
        y2 = cy + r_outer * np.sin(angle)
        ax.plot([x1, x2], [y1, y2], color="white", linewidth=2.5, solid_capstyle="round")

    # Needle
    angle = np.pi * (1 - prob)
    nx = cx + (r_outer + 0.02) * np.cos(angle)
    ny = cy + (r_outer + 0.02) * np.sin(angle)
    ax.plot([cx, nx], [cy, ny], color=COLOR_TEXT, linewidth=3, solid_capstyle="round")
    ax.scatter([cx], [cy], s=90, color=COLOR_TEXT, zorder=5)

    # Big probability label in center
    ax.text(cx, cy + 0.02, f"{prob:.1%}",
            ha="center", va="center", fontsize=22, fontweight="bold",
            color=COLOR_TEXT)
    ax.text(cx, cy - 0.06, "Attrition Risk",
            ha="center", va="center", fontsize=9, color=COLOR_MUTED,
            fontweight="500")

    # Zone labels at the bottom edge
    for frac, txt, col in [(threshold/2, "LOW", COLOR_LOW),
                           ((threshold + high_cutoff)/2, "MED", COLOR_MED),
                           ((high_cutoff + 1)/2, "HIGH", COLOR_HIGH)]:
        a = np.pi * (1 - frac)
        lx = cx + (r_outer + 0.08) * np.cos(a)
        ly = cy + (r_outer + 0.08) * np.sin(a)
        ax.text(lx, ly, txt, ha="center", va="center",
                fontsize=8, fontweight="700", color=col)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.65)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout(pad=0.2)
    return fig


def make_shap_chart(top_factors, figsize=(7.5, 3.6)):
    """
    Horizontal bar chart of top SHAP drivers.
    Red = pushes toward leave (positive), teal = pushes toward stay (negative).
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=120)

    values = top_factors.values[::-1]
    labels = [_shorten(f) for f in top_factors.index[::-1]]
    colors = [COLOR_PUSH_UP if v > 0 else COLOR_PUSH_DOWN for v in values]

    bars = ax.barh(range(len(values)), values, color=colors, edgecolor="white", height=0.62)

    # Reference line at zero
    ax.axvline(0, color=COLOR_MUTED, linewidth=0.8, alpha=0.6)

    # Value labels at bar tips
    for bar, v in zip(bars, values):
        x = v + (0.003 if v >= 0 else -0.003)
        ha = "left" if v >= 0 else "right"
        ax.text(x, bar.get_y() + bar.get_height()/2,
                f"{v:+.2f}", va="center", ha=ha,
                fontsize=9, fontweight="600",
                color=COLOR_PUSH_UP if v > 0 else COLOR_PUSH_DOWN)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("SHAP value  (← toward stay     toward leave →)",
                  fontsize=10, color=COLOR_MUTED, labelpad=8)
    ax.set_xlim(min(values.min() * 1.35, -0.1), max(values.max() * 1.35, 0.1))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", linestyle="--", alpha=0.3, color=COLOR_MUTED)
    ax.set_axisbelow(True)

    # Legend
    up_patch = mpatches.Patch(color=COLOR_PUSH_UP, label="Pushes toward Leave")
    down_patch = mpatches.Patch(color=COLOR_PUSH_DOWN, label="Pushes toward Stay")
    ax.legend(handles=[up_patch, down_patch], loc="lower right",
              frameon=False, fontsize=9)

    plt.tight_layout(pad=0.4)
    return fig


def _shorten(name, max_len=28):
    """Shorten feature names for chart readability."""
    replacements = {
        "BusinessTravel_": "Travel: ",
        "Department_": "Dept: ",
        "EducationField_": "EduField: ",
        "JobRole_": "Role: ",
        "MaritalStatus_": "Marital: ",
        "OverTime_": "OT: ",
        "Gender_": "Gender: ",
    }
    out = name
    for k, v in replacements.items():
        if out.startswith(k):
            out = v + out[len(k):]
    if len(out) > max_len:
        out = out[: max_len - 1] + "…"
    return out
