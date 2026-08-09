"""
Chart Components
================

Plotly-based gauge charts and visualizations for the results dashboard.
"""

from __future__ import annotations

import plotly.graph_objects as go


def score_gauge(score: float, title: str, max_val: float = 100) -> go.Figure:
    """
    Create a semi-circular gauge chart for a score.

    Parameters
    ----------
    score : float
        Score value (0–100).
    title : str
        Chart title.
    max_val : float
        Maximum gauge value.

    Returns
    -------
    go.Figure
    """
    # Color based on score
    if score >= 75:
        bar_color = "#64ffda"
    elif score >= 50:
        bar_color = "#ffd166"
    elif score >= 25:
        bar_color = "#ff9f43"
    else:
        bar_color = "#ff6b6b"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": "%", "font": {"size": 36, "color": "#ccd6f6"}},
            title={"text": title, "font": {"size": 14, "color": "#8892b0"}},
            gauge={
                "axis": {
                    "range": [0, max_val],
                    "tickwidth": 1,
                    "tickcolor": "#233554",
                    "dtick": 25,
                    "tickfont": {"color": "#8892b0", "size": 10},
                },
                "bar": {"color": bar_color, "thickness": 0.75},
                "bgcolor": "#112240",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 25], "color": "rgba(255, 107, 107, 0.08)"},
                    {"range": [25, 50], "color": "rgba(255, 159, 67, 0.08)"},
                    {"range": [50, 75], "color": "rgba(255, 209, 102, 0.08)"},
                    {"range": [75, 100], "color": "rgba(100, 255, 218, 0.08)"},
                ],
                "threshold": {
                    "line": {"color": bar_color, "width": 3},
                    "thickness": 0.85,
                    "value": score,
                },
            },
        )
    )

    fig.update_layout(
        height=200,
        margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif"),
    )

    return fig


def score_breakdown_bar(
    labels: list[str], scores: list[float], title: str = "Score Breakdown"
) -> go.Figure:
    """
    Create a horizontal bar chart for score component breakdown.

    Parameters
    ----------
    labels : list[str]
        Component names.
    scores : list[float]
        Score values (0–100).
    title : str
        Chart title.

    Returns
    -------
    go.Figure
    """
    colors = []
    for s in scores:
        if s >= 75:
            colors.append("#64ffda")
        elif s >= 50:
            colors.append("#ffd166")
        elif s >= 25:
            colors.append("#ff9f43")
        else:
            colors.append("#ff6b6b")

    fig = go.Figure(
        go.Bar(
            x=scores,
            y=labels,
            orientation="h",
            marker=dict(
                color=colors,
                line=dict(width=0),
                cornerradius=6,
            ),
            text=[f"{s:.0f}%" for s in scores],
            textposition="auto",
            textfont=dict(color="#0a192f", size=11, family="Inter"),
        )
    )

    fig.update_layout(
        title=dict(text=title, font=dict(color="#ccd6f6", size=14)),
        height=max(200, len(labels) * 40 + 60),
        margin=dict(l=10, r=20, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            range=[0, 105],
            showgrid=True,
            gridcolor="rgba(255,255,255,0.05)",
            tickfont=dict(color="#8892b0"),
        ),
        yaxis=dict(
            tickfont=dict(color="#ccd6f6", size=11),
            autorange="reversed",
        ),
        font=dict(family="Inter, sans-serif"),
    )

    return fig


def skill_comparison_chart(
    matched: list[str], missing: list[str], extra: list[str]
) -> go.Figure:
    """
    Create a stacked bar showing skill comparison counts.

    Parameters
    ----------
    matched : list[str]
        Matched skills.
    missing : list[str]
        Missing skills from JD.
    extra : list[str]
        Extra skills in resume.

    Returns
    -------
    go.Figure
    """
    categories = ["Matched", "Missing from Resume", "Extra in Resume"]
    counts = [len(matched), len(missing), len(extra)]
    colors = ["#64ffda", "#ff6b6b", "#bd93f9"]

    fig = go.Figure(
        go.Bar(
            x=categories,
            y=counts,
            marker=dict(
                color=colors,
                line=dict(width=0),
                cornerradius=8,
            ),
            text=counts,
            textposition="auto",
            textfont=dict(color="#0a192f", size=14, family="Inter"),
        )
    )

    fig.update_layout(
        height=250,
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(tickfont=dict(color="#ccd6f6", size=11)),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.05)",
            tickfont=dict(color="#8892b0"),
        ),
        font=dict(family="Inter, sans-serif"),
    )

    return fig
