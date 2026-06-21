#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""DreamerV3 inference (acting) loop — only Encoder/Posterior + GRU + Actor."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.path import Path

FP = "/home/jovyan/.local/lib/python3.11/site-packages/matplotlib/mpl-data/fonts/ttf/NanumGothic.ttf"
font_manager.fontManager.addfont(FP)
NANUM = font_manager.FontProperties(fname=FP).get_name()
plt.rcParams["font.family"] = [NANUM, "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

FWD  = "#37474f"
ACT  = "#7c4dbf"   # action / actor flow
REC  = "#37a037"   # recurrent h flow
ARCH = "#9a5fb0"   # architecture-type label color

C = {
    "obs":   ("#dbeafe", "#3b82f6"),
    "enc":   ("#fde9d0", "#e8883a"),
    "post":  ("#cdeae6", "#1f9e8e"),
    "gru":   ("#d6efd6", "#37a037"),
    "state": ("#fff3c4", "#d4a017"),
    "actor": ("#e7ddf5", "#8a5cc8"),
}

fig, ax = plt.subplots(figsize=(15.5, 9.2))
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
DASH = (0, (5, 3))


def box(x, y, w, h, title, sub="", arch="", key="enc", tsize=14, ssize=9.6, lw=2.2):
    face, edge = C[key]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.9",
                 linewidth=lw, edgecolor=edge, facecolor=face, zorder=3))
    cxx, cyy = x + w / 2, y + h / 2
    if arch:
        ax.text(cxx, cyy + h * 0.30, title, ha="center", va="center",
                fontsize=tsize, fontweight="bold", zorder=4)
        ax.text(cxx, cyy + h * 0.08, arch, ha="center", va="center",
                fontsize=ssize + 0.4, fontstyle="italic", color=ARCH, fontweight="bold", zorder=4)
        if sub:
            ax.text(cxx, cyy - h * 0.25, sub, ha="center", va="center",
                    fontsize=ssize, color="#333333", zorder=4, linespacing=1.4)
    elif sub:
        ax.text(cxx, cyy + h * 0.24, title, ha="center", va="center",
                fontsize=tsize, fontweight="bold", zorder=4)
        ax.text(cxx, cyy - h * 0.22, sub, ha="center", va="center",
                fontsize=ssize, color="#333333", zorder=4, linespacing=1.4)
    else:
        ax.text(cxx, cyy, title, ha="center", va="center",
                fontsize=tsize, fontweight="bold", zorder=4, linespacing=1.3)
    return (x, y, w, h)


def Rt(b): return (b[0] + b[2], b[1] + b[3] / 2)
def Lf(b): return (b[0], b[1] + b[3] / 2)
def Bt(b): return (b[0] + b[2] / 2, b[1])
def Tp(b): return (b[0] + b[2] / 2, b[1] + b[3])


def line(pts, color=FWD, lw=2.8, ls="-", head=True, mut=18, z=5):
    codes = [Path.MOVETO] + [Path.LINETO] * (len(pts) - 1)
    ax.add_patch(mpatches.PathPatch(Path(pts, codes), fill=False, edgecolor=color,
                 lw=lw, linestyle=ls, zorder=z, joinstyle="round", capstyle="round"))
    if head:
        x1, y1 = pts[-2]; x2, y2 = pts[-1]
        dx, dy = x2 - x1, y2 - y1
        n = (dx * dx + dy * dy) ** 0.5 or 1
        s = (x2 - dx / n * 0.6, y2 - dy / n * 0.6)
        ax.add_patch(FancyArrowPatch(s, (x2, y2), arrowstyle="-|>", mutation_scale=mut,
                     color=color, lw=lw, zorder=z + 1, shrinkA=0, shrinkB=0))


def lab(x, y, t, size=9.6, color="#3a3a3a", weight="normal", style="normal", ha="center", bg=False):
    bbox = dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.92) if bg else None
    ax.text(x, y, t, ha=ha, va="center", fontsize=size, color=color, fontweight=weight,
            fontstyle=style, zorder=8, linespacing=1.35, bbox=bbox)


# ---- title ----
ax.text(50, 96, "DreamerV3 Inference (Acting) Loop", ha="center", va="center",
        fontsize=21, fontweight="bold")
ax.text(50, 91, "학습 완료 후 실제 환경에서 행동 — Encoder(Posterior) + Sequence Model + Actor 만 사용",
        ha="center", va="center", fontsize=11.5, color="#555555")

# ---- main perception → action pipeline (one centerline at y=68.5) ----
env  = box(3,  61.5, 15, 14, "Environment", "관측 x_t 제공", arch="(외부 환경)", key="obs")
enc  = box(22, 61.5, 13, 14, "Encoder", "enc(x_t)", arch="(CNN / MLP)", key="enc")
post = box(38, 61.5, 22, 14, "Posterior  q_φ",
           "z_t ~ q_φ(z_t | h_t, x_t)\n실제 관측 x_t 사용\n(Prior 안 씀)", arch="(MLP)", key="post")
st   = box(63, 61.5, 15, 14, "model state", "s_t = {h_t, z_t}", arch="(h_t, z_t 결합)", key="state")
actor= box(82, 61.5, 15, 14, "Actor  π_θ",
           "a_t ~ π_θ(a_t | s_t)\n이산:Softmax / 연속:Gaussian", arch="(MLP)", key="actor")

line([Rt(env), Lf(enc)]);            lab(20, 77.2, "x_t", size=9.6, color="#3b82f6", weight="bold")
line([Rt(enc), Lf(post)]);           lab(36.5, 77.2, "enc(x_t)", size=9.2, color="#e8883a", weight="bold")
line([Rt(post), Lf(st)], color="#1f9e8e"); lab(61.5, 77.2, "z_t", size=9.6, color="#1f9e8e", weight="bold")
line([Rt(st), Lf(actor)]);           lab(80, 77.2, "s_t", size=9.6, color="#d4a017", weight="bold")

# ---- Sequence Model (GRU): produces recurrent state h_t; consumes z_t, a_t ----
gru = box(48, 33, 28, 13, "Sequence Model",
          "h_{t+1} = f_φ(h_t, z_t, a_t)\n순환 상태 갱신", arch="(GRU)", key="gru")

# GRU's recurrent state h_t -> Posterior  AND  -> model state (s_t = {h_t, z_t})
line([(52, 46), (52, 61.5)], color=REC); lab(52, 54.6, "h_t", size=9.4, color="#2c7a2c", weight="bold", bg=True)
line([(73, 46), (73, 61.5)], color=REC); lab(73, 54.6, "h_t", size=9.4, color="#2c7a2c", weight="bold", bg=True)
# model state's z_t feeds back into GRU (input for next h)
line([(66, 61.5), (66, 46)], color="#1f9e8e"); lab(62.6, 54.6, "z_t", size=9.0, color="#1f9e8e", weight="bold", bg=True)
# GRU self-recurrence  h_t -> h_{t+1}  (arc bulging left, away from the box)
ax.add_patch(FancyArrowPatch((48, 43.5), (48, 35.5), arrowstyle="-|>", mutation_scale=13,
             color=REC, lw=2.0, zorder=5, connectionstyle="arc3,rad=0.6", shrinkA=2, shrinkB=2))
lab(40.8, 39.5, "h_t 순환\n(→ h_{t+1})", size=8.2, color="#2c7a2c", style="italic")

# ---- action feedback loop (bottom, orthogonal) ----
line([Bt(actor), (89.5, 20), (10.5, 20), (10.5, 61.5)], color=ACT, lw=2.8)
lab(50, 17, "a_t  환경에서 실행  →  다음 관측 x_{t+1} 받음  (한 스텝 진행)", size=10.2,
    color=ACT, weight="bold")
line([(60, 20), (60, 33)], color=ACT, lw=2.4); lab(62.8, 26.2, "a_t", size=9.2, color=ACT, weight="bold", bg=True)

# ---- "not used at inference" note ----
ax.add_patch(FancyBboxPatch((3, 2.5), 94, 6.6, boxstyle="round,pad=0.12,rounding_size=0.5",
             linewidth=1.3, edgecolor="#cccccc", facecolor="#f7f7f7", zorder=2))
ax.text(50, 6.9, "추론 시 사용하지 않음 (학습 전용)", ha="center", va="center",
        fontsize=10.5, fontweight="bold", color="#888888", zorder=4)
ax.text(50, 4.2,
        "Prior p_φ(ẑ|h)  ·  Decoder  ·  Reward Head  ·  Continue Head  ·  Critic v_ψ  ·  상상(imagination) 롤아웃",
        ha="center", va="center", fontsize=9.8, color="#999999", zorder=4)

# legend
ax.add_patch(FancyBboxPatch((64, 80.8), 33, 6.4, boxstyle="round,pad=0.1,rounding_size=0.5",
             linewidth=1.1, edgecolor="#cccccc", facecolor="#fcfcfc", zorder=3))
def leg(x, y, t, color):
    ax.plot([x, x + 3], [y, y], color=color, lw=3.0, zorder=6, solid_capstyle="round")
    ax.text(x + 3.6, y, t, ha="left", va="center", fontsize=9.0, color="#444444")
leg(66, 85.4, "지각·상태 흐름", FWD)
leg(82, 85.4, "순환 상태 h", REC)
leg(66, 82.6, "행동 · 환경 루프", ACT)

plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
out = "/home/jovyan/.claude/jobs/b6058713/tmp/inference_loop.png"
fig.savefig(out, dpi=185, facecolor="white"); print("saved", out)
