#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""DreamerV3 model architecture overview — clean orthogonal routing."""
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

# ---- flow colors --------------------------------------------------------
FWD   = "#37474f"   # forward data flow (solid dark slate)
LOSS  = "#c0392b"   # losses (dashed red)
IMAG  = "#2563eb"   # imagination / prior / λ-return (dashed blue)
BUF   = "#b8860b"   # replay-buffer feed (dashed gold)
LOOP  = "#7c4dbf"   # environment loop (dashed purple)

C = {
    "obs":   ("#dbeafe", "#3b82f6"),
    "enc":   ("#fde9d0", "#e8883a"),
    "gru":   ("#d6efd6", "#37a037"),
    "post":  ("#cdeae6", "#1f9e8e"),
    "prior": ("#dbe6f8", "#4a78c8"),
    "state": ("#fff3c4", "#d4a017"),
    "head":  ("#fde2e2", "#e05c5c"),
    "actor": ("#e7ddf5", "#8a5cc8"),
    "critic":("#dbe6f8", "#4a78c8"),
    "buf":   ("#eef1f4", "#8a97a8"),
}

fig, ax = plt.subplots(figsize=(16.0, 12.6))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

DASH = (0, (5, 3))


def box(x, y, w, h, title, sub="", key="enc", tsize=12, ssize=8.2, lw=1.8):
    face, edge = C[key]
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.9",
                       linewidth=lw, edgecolor=edge, facecolor=face, zorder=3)
    ax.add_patch(p)
    cxx, cyy = x + w / 2, y + h / 2
    if sub:
        ax.text(cxx, cyy + h * 0.22, title, ha="center", va="center",
                fontsize=tsize, fontweight="bold", zorder=4)
        ax.text(cxx, cyy - h * 0.24, sub, ha="center", va="center",
                fontsize=ssize, color="#333333", zorder=4, linespacing=1.4)
    else:
        ax.text(cxx, cyy, title, ha="center", va="center",
                fontsize=tsize, fontweight="bold", zorder=4, linespacing=1.3)
    return (x, y, w, h)


def T(b): return (b[0] + b[2] / 2, b[1] + b[3])          # top-center
def B(b): return (b[0] + b[2] / 2, b[1])                 # bottom-center
def L(b): return (b[0], b[1] + b[3] / 2)                 # left-center
def R(b): return (b[0] + b[2], b[1] + b[3] / 2)          # right-center


def line(pts, color=FWD, lw=2.3, ls="-", head=True, mut=16, z=5):
    """Rounded polyline (orthogonal waypoints) with an arrowhead at the end."""
    codes = [Path.MOVETO] + [Path.LINETO] * (len(pts) - 1)
    ax.add_patch(mpatches.PathPatch(Path(pts, codes), fill=False, edgecolor=color,
                 lw=lw, linestyle=ls, zorder=z, joinstyle="round", capstyle="round"))
    if head:
        x1, y1 = pts[-2]
        x2, y2 = pts[-1]
        dx, dy = x2 - x1, y2 - y1
        n = (dx * dx + dy * dy) ** 0.5 or 1
        s = (x2 - dx / n * 0.6, y2 - dy / n * 0.6)
        ax.add_patch(FancyArrowPatch(s, (x2, y2), arrowstyle="-|>", mutation_scale=mut,
                     color=color, lw=lw, zorder=z + 1, shrinkA=0, shrinkB=0))


def lab(x, y, t, size=8.4, color="#3a3a3a", weight="normal", style="normal",
        ha="center", bg=False):
    bbox = dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9) if bg else None
    ax.text(x, y, t, ha=ha, va="center", fontsize=size, color=color,
            fontweight=weight, fontstyle=style, zorder=8, linespacing=1.35, bbox=bbox)


# ================= TITLE =================================================
ax.text(50, 98.4, "DreamerV3 Model Architecture", ha="center", va="center",
        fontsize=21, fontweight="bold")
ax.text(50, 95.4,
        "World Model (RSSM) + Actor–Critic — 단일 고정 설정으로 모든 도메인 학습 · 상상(imagination) 궤적 위에서 정책 개선",
        ha="center", va="center", fontsize=10.5, color="#555555")

# ================= BAND A : Environment / Replay ========================
env = box(5, 87.8, 16, 5.6, "Environment", "x_t · r_t · c_t", "obs", tsize=12.5, ssize=8.7)
buf = box(30, 87.8, 17, 5.6, "Replay Buffer", "배치 (B=16, T=64)", "buf", tsize=12.5, ssize=8.7)
line([R(env), L(buf)], lw=2.3)
lab(25.5, 91.9, "x_t, a_t,\nr_t, c_t", size=8.0, color="#3b82f6")
lab(13, 85.7, "↑ 학습된 Actor π_θ 의 행동으로 환경 전이 (STEP 1)", size=8.2,
    color=LOOP, style="italic")

# ================= legend (top-right) ===================================
ax.add_patch(FancyBboxPatch((52.5, 86.0), 45.0, 8.4,
             boxstyle="round,pad=0.1,rounding_size=0.6",
             linewidth=1.2, edgecolor="#bbbbbb", facecolor="#fcfcfc", zorder=3))
ax.text(54.3, 93.0, "범례", ha="left", va="center", fontsize=9.6, fontweight="bold", color="#555555")
def leg(x, y, t, color, ls="-"):
    ax.plot([x, x + 3.0], [y, y], color=color, lw=2.6, linestyle=ls, zorder=6,
            solid_capstyle="round")
    ax.text(x + 3.6, y, t, ha="left", va="center", fontsize=8.7, color="#444444")
leg(54.3, 90.6, "데이터 흐름 (forward)", FWD)
leg(76.0, 90.6, "손실 (KL · L_pred)", LOSS, ls=DASH)
leg(54.3, 88.0, "상상 롤아웃 / λ-return", IMAG, ls=DASH)
leg(76.0, 88.0, "버퍼 시작상태 · 환경 루프", BUF, ls=DASH)

# ================= BAND B : World Model container =======================
ax.add_patch(FancyBboxPatch((3, 45.5), 94, 38.5,
             boxstyle="round,pad=0.2,rounding_size=1.2",
             linewidth=1.6, edgecolor="#37a037", facecolor="#f4fbf4",
             linestyle="--", zorder=1))
ax.text(4.6, 82.9, "[월드 모델 학습 루프]  RSSM — Posterior 사용 (실제 관측 x_t 있음)",
        ha="left", va="center", fontsize=11.5, fontweight="bold", color="#2c7a2c", zorder=8)

xin   = box(5,  73.5, 13, 5.6, "관측  x_t", "이미지 / 벡터", "obs", tsize=12, ssize=8.7)
enc   = box(5,  62.0, 13, 8.2, "Encoder",
            "이미지: CNN\n벡터: symlog→MLP", "enc", tsize=12.5, ssize=8.6)
gru   = box(23, 62.5, 22, 12.4, "Sequence Model",
            "h_t = f_φ(h_{t-1},z_{t-1},a_{t-1})\n블록 대각 GRU\n(8 블록 × 1024 = 8192)\nz, a, h 각각 embed → concat", "gru",
            tsize=12.5, ssize=8.4)
post  = box(51, 70.5, 20, 8.6, "Posterior  q_φ",
            "q_φ(z_t | h_t, x_t)\n1-layer MLP · 학습 시\nstraight-through ∇", "post", tsize=12, ssize=8.3)
prior = box(76, 70.5, 20, 8.6, "Prior (Dynamics)  p_φ",
            "p_φ(ẑ_t | h_t)\n1-layer MLP · h_t 만\n상상 롤아웃 시 사용", "prior", tsize=12, ssize=8.3)
st    = box(23, 55.5, 22, 5.8, "model state  s_t = {h_t, z_t}", "", "state", tsize=11.5)
zt    = box(51, 55.5, 20, 5.8, "z_t  ~  Categorical", "이산 표현 (소프트맥스 샘플)", "post", tsize=11.5, ssize=8.3)
dec   = box(5,  46.5, 26, 7.0, "Decoder",
            "p_φ(x̂_t | s_t)\nCNN/MLP · symlog 목표", "head", tsize=12, ssize=8.4)
rew   = box(37, 46.5, 26, 7.0, "Reward Head",
            "p_φ(r̂_t | s_t)\nsymexp twohot bin", "head", tsize=12, ssize=8.4)
con   = box(69, 46.5, 26, 7.0, "Continue Head",
            "p_φ(ĉ_t | s_t)\nsigmoid · ĉ ∈ {0,1}", "head", tsize=12, ssize=8.4)

# --- world-model connections (orthogonal) ---
# x_t -> encoder
line([B(xin), (11.5, 70.2)])
# encoder -> posterior (via clear channel between Encoder & GRU, then top channel)
line([R(enc), (20.5, 66.1), (20.5, 81.0), (61, 81.0), (61, 79.1)])
lab(22.3, 77.5, "enc(x_t)", size=8.5, color="#e8883a", ha="left", weight="bold")
# GRU recurrence (self loop arching upward above the box)
ax.add_patch(FancyArrowPatch((41, 74.9), (27, 74.9), arrowstyle="-|>", mutation_scale=14,
             color="#2c7a2c", lw=1.8, zorder=5,
             connectionstyle="arc3,rad=0.45", shrinkA=2, shrinkB=2))
lab(34, 80.2, "h_{t-1}, z_{t-1}, a_{t-1}   (직전 스텝 t−1)", size=8.3, color="#2c7a2c")
lab(34, 73.8, "출처:  h_{t-1}=순환 출력 · z_{t-1}=직전 z_t · a_{t-1}=Actor/버퍼",
    size=6.6, color="#4a6a4a")
# GRU -> posterior (h_t, horizontal into left)
line([(45, 72.0), (51, 72.0)])
lab(48, 73.3, "h_t", size=8.4, color="#2c7a2c", weight="bold")
# GRU -> prior (h_t, under posterior, in channel above the KL box)
line([(45, 67.7), (86, 67.7), (86, 70.5)])
lab(57, 68.7, "h_t", size=8.2, color="#2c7a2c", weight="bold")
# GRU -> s_t (h_t, straight down)
line([B(gru), (34, 61.3)])
lab(36.6, 62.6, "h_t", size=8.2, color="#2c7a2c", weight="bold")
# posterior -> z_t (down)
line([B(post), (61, 61.3)], color="#1f9e8e")
# z_t -> s_t (left)
line([L(zt), R(st)], color="#1f9e8e")
lab(48, 56.7, "z_t", size=8.4, color="#1f9e8e", weight="bold")
# s_t -> heads (distribution bus at y=54.6)
line([B(st), (34, 54.6)], head=False, color="#d4a017")
ax.plot([18, 82], [54.6, 54.6], color="#d4a017", lw=2.3, zorder=5, solid_capstyle="round")
for hx, hb in [(18, dec), (50, rew), (82, con)]:
    line([(hx, 54.6), (hx, hb[1] + hb[3])], color="#d4a017")
lab(28, 55.7, "s_t", size=8.4, color="#d4a017", weight="bold")

# KL losses (between posterior & prior, compact)
line([(71, 76.2), (76, 76.2)], color=LOSS, ls=DASH, lw=1.8, mut=12)
line([(76, 74.6), (71, 74.6)], color=LOSS, ls=DASH, lw=1.8, mut=12)
ax.add_patch(FancyBboxPatch((50.5, 62.3), 45.5, 4.2,
             boxstyle="round,pad=0.08,rounding_size=0.4",
             linewidth=1.0, edgecolor="#e6b3b3", facecolor="#fdf3f3", zorder=2))
lab(73.2, 65.3, "L_dyn = KL(sg[q] ‖ p), β=1     L_rep = KL(q ‖ sg[p]), β=0.1", size=8.3, color=LOSS)
lab(73.2, 63.4, "free bits = 1 nat · unimix 1%   (KL 균형으로 발산 방지)", size=7.7, color="#9a6b6b")

# L_pred under heads
lab(50, 44.4, "L_pred (복원 + 보상 + 종료),  β_pred = 1", size=9.0, color=LOSS, weight="bold")

# ================= BAND C : Actor-Critic container ======================
ax.add_patch(FancyBboxPatch((3, 3.0), 94, 38.0,
             boxstyle="round,pad=0.2,rounding_size=1.2",
             linewidth=1.6, edgecolor="#8a5cc8", facecolor="#f8f5fd",
             linestyle="--", zorder=1))
ax.text(54, 39.8, "[Actor–Critic 학습 루프]  상상 궤적 위에서만 — Prior 사용 (실제 x_t 없음)",
        ha="center", va="center", fontsize=11.3, fontweight="bold", color="#6a3fb0", zorder=8)

imag  = box(5, 19.5, 40, 14.5, "상상 궤적 생성 (Imagination, H=15)",
            "s_τ ─[Actor]→ a ─[Prior p_φ]→ h, ẑ\n→ [Reward] r̂   [Continue] ĉ\n→ s_{τ+1} = {h, ẑ}   (H=15 반복)\n전부 미분 가능 (월드 모델로 ∇ 흐름)", "prior",
            tsize=12.5, ssize=9.2)
actor = box(50, 19.5, 22, 14.5, "Actor  π_θ(a | s_t)",
            "3-layer MLP · RMSNorm + SiLU\n이산: Softmax / 연속: Gaussian\nREINFORCE + 엔트로피 (η=3×10⁻⁴)\n수익 정규화 S = EMA(Per95−Per5)", "actor",
            tsize=13, ssize=9)
critic= box(76, 19.5, 21, 14.5, "Critic  v_ψ(s_t)",
            "3-layer MLP · RMSNorm + SiLU\nsymexp twohot 분포\nbins = symexp(-20..20)\nEMA 자기정규화 · 출력가중치 0", "critic",
            tsize=13, ssize=9)

# s_t (band B) -> imagination start (buffer state, dashed gold)
line([(20, 45.5), (20, 34.0)], color=BUF, ls=DASH, lw=2.2)
lab(20, 43.6, "버퍼 시작상태 s_τ (이후 실제 환경 접촉 없음)", size=8.4, color="#9a7b1a", style="italic", bg=True)
# imagination -> actor
line([(45, 26.75), (50, 26.75)], color=IMAG, lw=2.6)
lab(47.5, 28.6, "상상 궤적", size=8.6, color=IMAG)
# actor <-> critic
line([(72, 28.4), (76, 28.4)], color=FWD, lw=2.6)
line([(76, 25.1), (72, 25.1)], color=IMAG, lw=2.6)
lab(74, 30.1, "s, a", size=8.4, color=FWD)
lab(74, 23.4, "v_t", size=8.6, color=IMAG, weight="bold")
# lambda-return formula band (moved up; gap to the 3 blocks narrowed)
ax.add_patch(FancyBboxPatch((14, 7.5), 74, 9.0,
             boxstyle="round,pad=0.1,rounding_size=0.6",
             linewidth=1.5, edgecolor="#4a78c8", facecolor="#eef3fb", zorder=2))
# critic -> lambda band (down): supplies v_t
line([(85, 19.5), (85, 16.5)], color=IMAG, ls=DASH, lw=2.2)
lab(87.4, 18.0, "v_t", size=8.2, color=IMAG, weight="bold", bg=True)
# lambda band -> imagination (R^λ closes the Actor-Critic loop)
line([(16, 16.5), (16, 19.5)], color=IMAG, ls=DASH, lw=2.2)
lab(21.5, 18.0, "R^λ", size=8.4, color=IMAG, weight="bold", bg=True)
ax.text(51, 14.2, "λ-수익 R^λ_t  ·  상상의 r̂, ĉ + Critic v_t 로 계산  →  Actor advantage & Critic 목표",
        ha="center", va="center", fontsize=9.8, fontweight="bold", color="#33558c", zorder=4)
ax.text(51, 10.4,
        r"$R^{\lambda}_t = \hat{r}_t + \gamma\,\hat{c}_t\,[\,(1-\lambda)\,v_t + \lambda\,R^{\lambda}_{t+1}\,]$"
        "      (γ=0.997,  λ=0.95)",
        ha="center", va="center", fontsize=14, color="#1f3a66", zorder=4)

plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
out = "/home/jovyan/.claude/jobs/b6058713/tmp/arch_overview.png"
fig.savefig(out, dpi=185, facecolor="white")
print("saved", out)
