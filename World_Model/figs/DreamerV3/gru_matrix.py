import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False
SYM = 'DejaVu Sans'

fig, ax = plt.subplots(figsize=(15.5, 10.5))
ax.set_xlim(0, 15.5)
ax.set_ylim(0, 10.5)
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


H_M, H_8D = 0.62, 1.28          # m / 8d 벡터 높이 (= 차원 비례)
W_VEC = 0.24


def vec(cx, cy, dim_h, fc, ec, label, dimlabel, lc='#222', w=W_VEC):
    top_y = cy + dim_h / 2
    rect(cx - w / 2, top_y - dim_h, w, dim_h, fc=fc, ec=ec, lw=1.5)
    txt(cx, top_y - dim_h - 0.22, label, sz=9.5, bold=True, color=lc)
    txt(cx, top_y - dim_h - 0.47, dimlabel, sz=8.5, color='#777')


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


# ═══════════════ TITLE ═══════════════
txt(7.75, 10.05, 'DreamerV3 GRU — 행렬 연산과 차원 변화 (shape flow)', sz=18, bold=True, color='#15233f')
txt(7.75, 9.62, '벡터·행렬의 크기 = 차원.  핵심: "차원이 바뀌는 곳은 입력 선형변환(W·x) 뿐, 그 뒤는 모두 8d 고정"', sz=10.5, color='#555')

# ═══════════════ ① 선형 변환: 모두 8d 로 ═══════════════
txt(0.5, 8.95, '①  선형 변환 — 입력을 모두 8d 로  (여기서만 차원이 바뀜)   ·예시: 리셋 게이트 r_t', sz=12, bold=True, color='#1a4a88', ha='left')

cy = 7.70

# W_r (8d×m) · x_t (m) = (8d)
matrix(0.75, cy, H_M, H_8D, fc='#eaf0ff', ec='#3a66b0')           # 8d행 × m열 → 높이 8d, 폭 m
txt(0.75 + H_M / 2, cy - H_8D / 2 - 0.22, 'W_r', sz=9.5, bold=True, color='#27457f')
txt(0.75 + H_M / 2, cy - H_8D / 2 - 0.47, '(8d × m)', sz=8.3, color='#777')
sym(1.72, cy, '·', sz=18, color='#444')
vec(2.10, cy, H_M, '#dddddd', '#888', 'x_t', '(m)', lc='#333')
sym(2.62, cy, '=', sz=15, color='#444')
vec(3.02, cy, H_8D, '#cfe0ff', '#3a66b0', 'W·x', '(8d)', lc='#27457f')

sym(3.62, cy, '+', sz=17, color='#444', bold=True)

# U_r (8d×8d, block-diag) · h (8d) = (8d)
matrix(4.05, cy, H_8D, H_8D, blockdiag=True)
txt(4.05 + H_8D / 2, cy - H_8D / 2 - 0.22, 'U_r', sz=9.5, bold=True, color='#8a5500')
txt(4.05 + H_8D / 2, cy - H_8D / 2 - 0.47, '(8d × 8d, block-diag)', sz=8.3, color='#8a5500')
sym(5.55, cy, '·', sz=18, color='#444')
vec(5.93, cy, H_8D, '#ffe0b0', '#cc6600', 'h_(t-1)', '(8d)', lc='#8b3a00')
sym(6.50, cy, '=', sz=15, color='#444')
vec(6.90, cy, H_8D, '#ffe7c2', '#cc8800', 'U·h', '(8d)', lc='#8a5500')

arr(7.25, cy, 8.15, cy, color='#cc4444', lw=2.4)
sym(7.70, cy + 0.30, 'σ', sz=14, color='#cc4444', bold=True)
vec(8.62, cy, H_8D, '#ffd0d0', '#cc4444', 'r_t', '(8d)', lc='#aa2222', w=0.32)

# 우측 메모
txt(9.45, 8.30, '핵심:  행렬의 "행 수 = 8d" 라서', sz=10, bold=True, color='#15233f', ha='left')
txt(9.45, 7.97, '입력이 x_t(m) 이든 h(8d) 이든', sz=9.6, color='#333', ha='left')
txt(9.45, 7.67, '곱한 결과는 항상 8d 가 된다.', sz=9.6, color='#333', ha='left')
txt(9.45, 7.20, '· 짧은 x_t(m) → 긴 8d 로 늘어남', sz=9.2, color='#cc4444', ha='left')
txt(9.45, 6.90, '· u_t = σ(W_u·x + U_u·h) → 8d', sz=9.2, color='#555', ha='left')
txt(9.45, 6.60, '· c_t = tanh(W_c·x + U_c(r_t⊙h)) → 8d', sz=9.2, color='#555', ha='left')

# 구분선
ax.plot([0.5, 15.0], [5.95, 5.95], color='#dddddd', lw=1.0, ls='--', zorder=2)

# ═══════════════ ② 원소별 연산: 8d 고정 ═══════════════
txt(0.5, 5.60, '②  그 다음은 전부 "원소별(element-wise) 연산"  →  차원 8d 그대로 유지', sz=12, bold=True, color='#1a4a88', ha='left')

cy3 = 4.35
spc = 1.30
x0 = 1.05
vec(x0, cy3, H_8D, '#e6f2ff', '#2266bb', '(1-u_t)', '8d', lc='#15487f')
sym(x0 + 0.65, cy3, '⊙', sz=15, color='#444')
vec(x0 + spc, cy3, H_8D, '#ffe0b0', '#cc6600', 'h_(t-1)', '8d', lc='#8b3a00')
sym(x0 + spc + 0.65, cy3, '+', sz=17, color='#444', bold=True)
vec(x0 + 2 * spc, cy3, H_8D, '#e6f2ff', '#2266bb', 'u_t', '8d', lc='#15487f')
sym(x0 + 2 * spc + 0.65, cy3, '⊙', sz=15, color='#444')
vec(x0 + 3 * spc, cy3, H_8D, '#ffe7c2', '#cc8800', 'c_t', '8d', lc='#8a5500')
sym(x0 + 3 * spc + 0.65, cy3, '=', sz=15, color='#444')
vec(x0 + 4 * spc, cy3, H_8D, '#d0f0d0', '#228833', 'h_t', '8d', lc='#155220', w=0.34)

txt(9.45, 4.85, '모든 벡터가 같은 길이(8d)', sz=10, bold=True, color='#15233f', ha='left')
txt(9.45, 4.52, '→ ⊙ · + 가능, 결과도 8d.', sz=9.6, color='#333', ha='left')
txt(9.45, 4.08, 'σ, tanh, ⊙, blend 은 모양을', sz=9.2, color='#555', ha='left')
txt(9.45, 3.80, '바꾸지 않는 원소별 연산이다.', sz=9.2, color='#555', ha='left')

# h_t → 피드백
hx = x0 + 4 * spc
arr(hx, cy3 - H_8D / 2 - 0.55, hx, 3.02, color='#228833', lw=1.8, ls='dashed')
txt(hx, 2.84, '다음 스텝 h_(t-1) 로 (feedback)', sz=8.8, color='#228833', italic=True)

# ═══════════════ 하단 요약 ═══════════════
rbox(0.5, 0.45, 14.5, 1.85, fc='#f7f9ff', ec='#88a0c8', lw=1.3, radius=0.18)
txt(0.8, 2.02, '정리 — 차원은 어디서 변하나?', sz=11, bold=True, color='#15233f', ha='left')
txt(0.8, 1.62, '· 변하는 곳은 입력 선형변환 하나뿐:  W·x_t = (8d × m)·(m) → 8d.   (m = z·a·h 임베딩 concat 크기)', sz=9.6, color='#2a3a55', ha='left')
txt(0.8, 1.26, '· U·h, σ, tanh, ⊙, blend 은 전부 8d → 8d (원소별이라 차원 보존). 출력 h_t(8d) 가 다음 스텝 입력으로.', sz=9.6, color='#2a3a55', ha='left')
txt(0.8, 0.90, '· block-diagonal U: 8d×8d 지만 대각 8블록(d×d)만 nonzero → 큰 8d 메모리를 파라미터 8배 절감으로 구현.', sz=9.6, color='#8a5500', ha='left')

out = '/home/jovyan/workspace/Paper_Review_KR/World_Model/figs/DreamerV3/gru_matrix.png'
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
print('Saved:', out)
