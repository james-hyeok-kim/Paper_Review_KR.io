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
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle=f"round,pad=0,rounding_size={radius}",
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
    txt(cx, top_y - dim_h - 0.46, dimlabel, sz=8.3, color='#777')


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


# ═══════════════ TITLE + FORMULA ═══════════════
txt(7.75, 9.05, 'DreamerV3 GRU — 후보 상태  c_t  만들기', sz=18, bold=True, color='#15233f')
sym(7.75, 8.55, 'cₜ = tanh( W_c · xₜ  +  U_c · ( rₜ ⊙ h₍ₜ₋₁₎ ) )', sz=14, color='#8a5500', bold=True)

# ═══════════════ ① 리셋 적용 ═══════════════
txt(0.5, 7.85, '①  리셋 게이트로 "과거"를 거른다 (원소별):  r_t ⊙ h_(t-1)  →  "리셋된 과거" g', sz=12, bold=True, color='#1a4a88', ha='left')

cyA = 6.85
vec(1.05, cyA, H_8D, '#ffd0d0', '#cc4444', 'r_t', '8d (0~1)', lc='#aa2222')
sym(1.70, cyA, '⊙', sz=15, color='#444')
vec(2.35, cyA, H_8D, '#ffe0b0', '#cc6600', 'h_(t-1)', '8d', lc='#8b3a00')
sym(3.00, cyA, '=', sz=15, color='#444')
vec(3.70, cyA, H_8D, '#e7d8f5', '#8833aa', 'g = r_t⊙h', '8d', lc='#6b1f8a', w=0.30)

# 예시 박스 (원소별 계산)
rbox(5.0, 5.95, 9.6, 1.85, fc='#fafafa', ec='#aaaaaa', lw=1.2, radius=0.15)
txt(5.3, 7.55, '예시 — 칸마다(원소별) 곱한다:', sz=9.8, bold=True, color='#333', ha='left')
txt(5.3, 7.18, 'r_t        =  [  0.9    0.1    0.8    0.3  ]   ← 게이트 (0~1)', sz=9.3, color='#aa2222', ha='left')
txt(5.3, 6.85, 'h_(t-1)   =  [  1.0    2.0   -1.0    3.0  ]   ← 과거 상태', sz=9.3, color='#8b3a00', ha='left')
txt(5.3, 6.52, '곱 (r_t·h) =  [  0.9    0.2   -0.8    0.9  ]   ← 리셋된 과거 g', sz=9.3, color='#6b1f8a', ha='left', bold=True)
txt(5.3, 6.19, 'r≈0 인 칸 → 과거를 거의 0 으로 "잊음" / r≈1 인 칸 → 과거 "유지"', sz=8.8, color='#555', ha='left', italic=True)

# g → 다음 단계로
arr(3.70, cyA - H_8D / 2 - 0.50, 3.70, 5.05, color='#8833aa', lw=1.8)
txt(3.70, 4.88, '리셋된 과거 g 를 ② 로', sz=8.6, color='#6b1f8a', italic=True)

# 구분선
ax.plot([0.5, 15.0], [5.55, 5.55], color='#dddddd', lw=1.0, ls='--', zorder=2)

# ═══════════════ ② 선형변환 + 합 + tanh ═══════════════
txt(0.5, 5.20, '②  입력 x_t 와 "리셋된 과거 g" 를 각각 선형변환해 더하고  →  tanh', sz=12, bold=True, color='#1a4a88', ha='left')

cyB = 3.85
# W_c (8d×m) · x_t (m) = (8d)
matrix(0.75, cyB, H_M, H_8D, fc='#eaf0ff', ec='#3a66b0')
txt(0.75 + H_M / 2, cyB - H_8D / 2 - 0.22, 'W_c', sz=9.3, bold=True, color='#27457f')
txt(0.75 + H_M / 2, cyB - H_8D / 2 - 0.46, '(8d × m)', sz=8.2, color='#777')
sym(1.70, cyB, '·', sz=18, color='#444')
vec(2.07, cyB, H_M, '#dddddd', '#888', 'x_t', '(m)', lc='#333')
sym(2.58, cyB, '=', sz=15, color='#444')
vec(2.98, cyB, H_8D, '#cfe0ff', '#3a66b0', 'W·x', '(8d)', lc='#27457f')

sym(3.58, cyB, '+', sz=17, color='#444', bold=True)

# U_c (8d×8d, block-diag) · g (8d) = (8d)
matrix(4.0, cyB, H_8D, H_8D, blockdiag=True)
txt(4.0 + H_8D / 2, cyB - H_8D / 2 - 0.22, 'U_c', sz=9.3, bold=True, color='#8a5500')
txt(4.0 + H_8D / 2, cyB - H_8D / 2 - 0.46, '(8d × 8d, block-diag)', sz=8.2, color='#8a5500')
sym(5.48, cyB, '·', sz=18, color='#444')
vec(5.86, cyB, H_8D, '#e7d8f5', '#8833aa', 'g', '(8d)', lc='#6b1f8a')
sym(6.42, cyB, '=', sz=15, color='#444')
vec(6.82, cyB, H_8D, '#efe2fb', '#8833aa', 'U·g', '(8d)', lc='#6b1f8a')

# → tanh → c_t
arr(7.18, cyB, 8.25, cyB, color='#cc8800', lw=2.4)
sym(7.71, cyB + 0.30, 'tanh', sz=11, color='#cc8800', bold=True)
vec(8.78, cyB, H_8D, '#fff0c2', '#cc8800', 'c_t', '(8d)', lc='#8a5500', w=0.34)

# 우측 메모
txt(9.7, 4.40, '핵심:', sz=10, bold=True, color='#15233f', ha='left')
txt(9.7, 4.05, '· r_t 가 바로 여기서 쓰인다', sz=9.4, color='#aa2222', ha='left')
txt(9.7, 3.73, '  → c_t 안에 r_t 영향이 담김', sz=9.2, color='#555', ha='left')
txt(9.7, 3.30, '· W·x 와 U·g 둘 다 8d → 더할 수 있음', sz=9.2, color='#555', ha='left')
txt(9.7, 2.98, '· tanh → 각 원소 -1~1, 차원은 8d 그대로', sz=9.2, color='#555', ha='left')

# ═══════════════ 하단 요약 ═══════════════
rbox(0.5, 0.45, 14.5, 1.55, fc='#f7f9ff', ec='#88a0c8', lw=1.3, radius=0.18)
txt(0.8, 1.70, '정리', sz=11, bold=True, color='#15233f', ha='left')
txt(0.8, 1.32, '· c_t 는 "이번 스텝의 새 상태 후보".  r_t 로 과거를 거른 g 와 입력 x_t 를 합쳐 tanh 로 만든다.', sz=9.6, color='#2a3a55', ha='left')
txt(0.8, 0.96, '· 그래서 r_t 는 h_t 에 "직접"이 아니라 c_t 를 거쳐 "간접" 으로 기여한다:   r_t → c_t → h_t.', sz=9.6, color='#aa2222', ha='left')
txt(0.8, 0.62, '· 차원: 모든 항이 8d (입력 x_t 만 m).  바뀌는 건 W·x (m→8d) 한 곳뿐, 나머지는 8d 유지.', sz=9.6, color='#8a5500', ha='left')

out = '/home/jovyan/workspace/Paper_Review_KR/World_Model/figs/DreamerV3/gru_candidate.png'
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
print('Saved:', out)
