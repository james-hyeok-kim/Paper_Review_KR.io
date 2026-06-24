import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False
SYM = 'DejaVu Sans'

fig, ax = plt.subplots(figsize=(15.5, 9.5))
ax.set_xlim(0, 15.5)
ax.set_ylim(0, 9.5)
ax.axis('off')
fig.patch.set_facecolor('#ffffff')


def rbox(x, y, w, h, fc, ec='#444', lw=1.4, radius=0.12, ls='-'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
                 linewidth=lw, edgecolor=ec, facecolor=fc, zorder=3, linestyle=ls))


def rect(x, y, w, h, fc, ec='#444', lw=1.4):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, lw=lw, zorder=4))


def arr(x1, y1, x2, y2, color='#555', lw=2.0, ls='-', rad=0.0, zorder=4):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                connectionstyle=f'arc3,rad={rad}', linestyle=ls), zorder=zorder)


def txt(x, y, s, sz=11, color='#222', ha='center', va='center', bold=False, italic=False, zorder=6):
    ax.text(x, y, s, fontsize=sz, color=color, ha=ha, va=va,
            fontweight='bold' if bold else 'normal',
            fontstyle='italic' if italic else 'normal', zorder=zorder)


def sym(x, y, s, sz=13, color='#222', ha='center', va='center', bold=False, zorder=6):
    ax.text(x, y, s, fontsize=sz, color=color, ha=ha, va=va, fontfamily=SYM,
            fontweight='bold' if bold else 'normal', zorder=zorder)


H_M, H_8D = 0.62, 1.25
W_VEC = 0.24


def vec(cx, cy, dim_h, fc, ec, label, dimlabel, lc='#222', w=W_VEC):
    top_y = cy + dim_h / 2
    rect(cx - w / 2, top_y - dim_h, w, dim_h, fc=fc, ec=ec, lw=1.5)
    txt(cx, top_y - dim_h - 0.22, label, sz=9.3, bold=True, color=lc)
    txt(cx, top_y - dim_h - 0.46, dimlabel, sz=8.2, color='#777')


def matrix(x_left, cy, cols_h, rows_h, blockdiag=False, fc='#eaf0ff', ec='#3a66b0'):
    y = cy - rows_h / 2
    if blockdiag:
        N = 8
        cw, ch = cols_h / N, rows_h / N
        for i in range(N):
            for j in range(N):
                rect(x_left + j * cw, y + (N - 1 - i) * ch, cw, ch,
                     fc=('#f5a623' if i == j else '#ffffff'), ec='#d6c08a', lw=0.5)
        ax.add_patch(Rectangle((x_left, y), cols_h, rows_h, facecolor='none', edgecolor=ec, lw=1.6, zorder=5))
    else:
        rect(x_left, y, cols_h, rows_h, fc=fc, ec=ec, lw=1.6)


def gate_scene(cy, Wlabel, Ulabel, gate_name, gate_fc, gate_ec, gate_lc):
    matrix(0.75, cy, H_M, H_8D, fc='#eaf0ff', ec='#3a66b0')
    txt(0.75 + H_M / 2, cy - H_8D / 2 - 0.22, Wlabel, sz=9.3, bold=True, color='#27457f')
    txt(0.75 + H_M / 2, cy - H_8D / 2 - 0.46, '(8d × m)', sz=8.2, color='#777')
    sym(1.72, cy, '·', 18, '#444')
    vec(2.10, cy, H_M, '#dddddd', '#888', 'x_t', '(m)', '#333')
    sym(2.62, cy, '=', 15, '#444')
    vec(3.02, cy, H_8D, '#cfe0ff', '#3a66b0', 'W·x', '(8d)', '#27457f')
    sym(3.62, cy, '+', 17, '#444', bold=True)
    matrix(4.05, cy, H_8D, H_8D, blockdiag=True)
    txt(4.05 + H_8D / 2, cy - H_8D / 2 - 0.22, Ulabel, sz=9.3, bold=True, color='#8a5500')
    txt(4.05 + H_8D / 2, cy - H_8D / 2 - 0.46, '(8d × 8d, block-diag)', sz=8.2, color='#8a5500')
    sym(5.55, cy, '·', 18, '#444')
    vec(5.93, cy, H_8D, '#ffe0b0', '#cc6600', 'h_(t-1)', '(8d)', '#8b3a00')
    sym(6.50, cy, '=', 15, '#444')
    vec(6.90, cy, H_8D, '#ffe7c2', '#cc8800', 'U·h', '(8d)', '#8a5500')
    arr(7.25, cy, 8.15, cy, color=gate_ec, lw=2.4)
    sym(7.70, cy + 0.30, 'σ', 14, gate_ec, bold=True)
    vec(8.62, cy, H_8D, gate_fc, gate_ec, gate_name, '(8d)', gate_lc, w=0.32)


# ═══════════════ TITLE ═══════════════
txt(7.75, 9.10, 'DreamerV3 GRU — 게이트  r_t · u_t  만들기  (둘 다 같은 구조)', sz=18, bold=True, color='#15233f')
txt(7.75, 8.66, '두 게이트는 식·입력·구조가 똑같다.  학습된 가중치(W, U)만 다르다.', sz=10.5, color='#555')

# ═══════════════ ① 리셋 게이트 r_t ═══════════════
txt(0.5, 8.05, '①  리셋 게이트  r_t', sz=12, bold=True, color='#1a4a88', ha='left')
sym(2.7, 8.05, 'rₜ = σ( W_r · xₜ + U_r · h₍ₜ₋₁₎ )', sz=12, color='#aa2222', ha='left')
gate_scene(6.80, 'W_r', 'U_r', 'r_t', '#ffd0d0', '#cc4444', '#aa2222')
txt(9.45, 7.25, 'r_t 의 쓰임:', sz=10, bold=True, color='#15233f', ha='left')
txt(9.45, 6.92, '→ 후보 c_t 계산에 사용', sz=9.4, color='#aa2222', ha='left')
txt(9.45, 6.62, '   c_t = tanh(W_c·x + U_c·(r_t⊙h))', sz=9.0, color='#555', ha='left')

# 구분선
ax.plot([0.5, 15.0], [5.55, 5.55], color='#dddddd', lw=1.0, ls='--', zorder=2)

# ═══════════════ ② 업데이트 게이트 u_t ═══════════════
txt(0.5, 5.10, '②  업데이트 게이트  u_t', sz=12, bold=True, color='#1a4a88', ha='left')
sym(3.1, 5.10, 'uₜ = σ( W_u · xₜ + U_u · h₍ₜ₋₁₎ )', sz=12, color='#15487f', ha='left')
gate_scene(3.85, 'W_u', 'U_u', 'u_t', '#d6e6ff', '#2266bb', '#15487f')
txt(9.45, 4.30, 'u_t 의 쓰임:', sz=10, bold=True, color='#15233f', ha='left')
txt(9.45, 3.97, '→ 최종 혼합(blend)에 사용', sz=9.4, color='#15487f', ha='left')
txt(9.45, 3.67, '   h_t = (1-u_t)⊙h + u_t⊙c_t', sz=9.0, color='#555', ha='left')

# ═══════════════ 하단 요약 ═══════════════
rbox(0.5, 0.45, 14.5, 1.95, fc='#f7f9ff', ec='#88a0c8', lw=1.3, radius=0.18)
txt(0.8, 2.10, '정리 — r_t 와 u_t 의 같은 점 / 다른 점', sz=11, bold=True, color='#15233f', ha='left')
txt(0.8, 1.72, '· 같은 점: 식·입력(x_t, h_(t-1))·구조가 동일. σ 로 0~1 게이트를 만든다. 학습된 가중치만 다르다.', sz=9.6, color='#2a3a55', ha='left')
txt(0.8, 1.36, '· 게이트는 h_(t-1) 를 "리셋 없이 그대로" 사용 (후보 c_t 와 다른 점). → 게이트 계산엔 r_t 가 안 들어간다.', sz=9.6, color='#aa2222', ha='left')
txt(0.8, 1.00, '· 쓰임만 다르다: r_t → 후보 c_t 만들 때,  u_t → 최종 혼합(blend) 에서.', sz=9.6, color='#15487f', ha='left')
txt(0.8, 0.64, '· 차원: x_t(m) 은 W 로, h(8d) 는 U(block-diag) 로 곱해 모두 8d → σ → 8d 게이트.', sz=9.6, color='#8a5500', ha='left')

out = '/home/jovyan/workspace/Paper_Review_KR/World_Model/figs/DreamerV3/gru_gates.png'
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
print('Saved:', out)
