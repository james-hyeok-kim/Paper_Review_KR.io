import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False
SYM = 'DejaVu Sans'   # 수학 기호(σ, ⊙, −) 전용 폰트

FS = 1.0

fig, ax = plt.subplots(figsize=(17, 12))
ax.set_xlim(0, 17)
ax.set_ylim(0, 12)
ax.axis('off')
fig.patch.set_facecolor('#ffffff')


def box(x, y, w, h, fc, ec='#444', lw=1.5, radius=0.18, ls='-'):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle=f"round,pad=0,rounding_size={radius}",
                 linewidth=lw, edgecolor=ec, facecolor=fc, zorder=3, linestyle=ls))


def arr(x1, y1, x2, y2, color='#555', lw=2.0, ls='-', rad=0.0, zorder=4):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                connectionstyle=f'arc3,rad={rad}', linestyle=ls),
                zorder=zorder)


def txt(x, y, s, sz=11, color='#222', ha='center', va='center',
        bold=False, italic=False, zorder=5):
    ax.text(x, y, s, fontsize=sz * FS, color=color, ha=ha, va=va,
            fontweight='bold' if bold else 'normal',
            fontstyle='italic' if italic else 'normal', zorder=zorder)


def sym(x, y, s, sz=12, color='#222', ha='center', va='center', bold=False, zorder=5):
    ax.text(x, y, s, fontsize=sz * FS, color=color, ha=ha, va=va,
            fontfamily=SYM, fontweight='bold' if bold else 'normal', zorder=zorder)


# ═══════════════ TITLE ═══════════════
txt(8.5, 11.55, 'DreamerV3 시퀀스 모델 (RSSM)  —  Block-Diagonal GRU', sz=20, bold=True, color='#15233f')
sym(8.5, 11.08, 'hₜ = f_φ( h₍ₜ₋₁₎ , z₍ₜ₋₁₎ , a₍ₜ₋₁₎ )', sz=14, color='#3a3a55')
txt(8.5, 10.70, '직전 순환상태·직전 latent·직전 행동 → 다음 결정론적 순환상태 h_t 를 갱신', sz=11, color='#555')

# ═══════════════ [A] 입력 구성 (좌상단) ═══════════════
txt(0.5, 9.95, '[A] GRU 입력 x_t 구성', sz=12.5, bold=True, color='#1a4a88', ha='left')

inp = [('z₍ₜ₋₁₎', '직전 latent', 0.6, '#d0f0d0', '#228833'),
       ('a₍ₜ₋₁₎', '직전 행동',   3.0, '#dde4ff', '#3355aa'),
       ('h₍ₜ₋₁₎', '직전 순환상태', 5.4, '#ffe0b0', '#cc6600')]
for label, sub, x, fc, ec in inp:
    box(x, 8.95, 2.0, 0.78, fc=fc, ec=ec, lw=1.6)
    sym(x + 1.0, 9.45, label, sz=12.5, bold=True, color=ec)
    txt(x + 1.0, 9.12, sub, sz=9, color='#555')

# 각 입력 → Linear embed → concat
for x in (1.6, 4.0, 6.4):
    arr(x, 8.95, x, 8.5, color='#888', lw=1.6)
box(1.2, 7.85, 5.6, 0.62, fc='#f3e8ff', ec='#8833aa', lw=1.6)
txt(2.05, 8.16, 'Linear embed ×3', sz=10, color='#6b1f8a')
sym(4.9, 8.16, '→  concat  →  xₜ', sz=11.5, bold=True, color='#551177')

# h 도 embed 하는 이유
txt(7.75, 9.45, '※ h_(t-1) 도 (dense) linear 로 embed →\n   8블록을 섞어줌 (cross-block mixing)',
    sz=8.8, color='#aa5500', ha='left', italic=True)

# xₜ → GRU cell
arr(4.0, 7.85, 4.0, 7.45, color='#2a2a2a', lw=2.2)
sym(4.35, 7.63, 'xₜ', sz=11, color='#2a2a2a')

# ═══════════════ [B] GRU 셀 (좌중앙) ═══════════════
box(0.4, 2.95, 9.0, 4.5, fc='#fafafa', ec='#999', lw=1.4, radius=0.25)
txt(0.7, 7.18, '[B] GRU 셀 (한 타임스텝 내부 동작)', sz=12.5, bold=True, color='#1a4a88', ha='left')

# 내부 흐름 3박스
box(0.85, 4.55, 2.55, 1.95, fc='#ffe9e9', ec='#cc4444', lw=1.6)
txt(2.12, 6.22, '① 게이트', sz=11.5, bold=True, color='#882222')
sym(2.12, 5.84, 'rₜ = σ(·)', sz=11, color='#aa2222')
txt(2.12, 5.52, '리셋 게이트', sz=8.8, color='#774444')
sym(2.12, 5.16, 'uₜ = σ(·)', sz=11, color='#aa2222')
txt(2.12, 4.84, '업데이트 게이트', sz=8.8, color='#774444')

box(3.75, 4.55, 2.55, 1.95, fc='#fff3da', ec='#cc8800', lw=1.6)
txt(5.02, 6.22, '② 후보 상태', sz=11.5, bold=True, color='#8a5500')
sym(5.02, 5.80, 'cₜ = tanh(·)', sz=11, color='#a06000')
txt(5.02, 5.42, '리셋된 과거', sz=9, color='#7a5520')
sym(5.02, 5.10, 'rₜ ⊙ h₍ₜ₋₁₎', sz=10, color='#a06000')
txt(5.02, 4.80, '를 써서 새 상태 제안', sz=8.6, color='#7a5520')

box(6.65, 4.55, 2.55, 1.95, fc='#e6f2ff', ec='#2266bb', lw=1.6)
txt(7.92, 6.22, '③ 혼합 (blend)', sz=11.5, bold=True, color='#15487f')
sym(7.92, 5.80, 'hₜ =', sz=11, color='#1a5aa0')
sym(7.92, 5.46, '(1−uₜ)⊙h₍ₜ₋₁₎', sz=9.5, color='#1a5aa0')
sym(7.92, 5.14, '+ uₜ⊙cₜ', sz=9.5, color='#1a5aa0')
txt(7.92, 4.80, '과거 ↔ 후보 비율 혼합', sz=8.4, color='#3a5a80')

# ── 박스 간 신호 배선 (정확한 데이터 흐름) ──
# x_t : concat 에서 내려와 ① 게이트 & ② 후보 '둘 다' 로 분기 (③ 혼합엔 x_t 없음)
arr(4.0, 7.45, 4.0, 6.92, color='#2a2a2a', lw=1.8)
arr(4.0, 6.92, 2.3, 6.50, color='#2a2a2a', lw=1.5)     # → ① 게이트
arr(4.0, 6.92, 4.7, 6.50, color='#2a2a2a', lw=1.5)     # → ② 후보
sym(4.32, 7.16, 'xₜ', sz=10.5, color='#2a2a2a')
txt(6.55, 6.80, 'x_t 는 ①·②에만 입력 (③ 제외)', sz=8, color='#777', ha='left')

# ① → ② : r_t (리셋 게이트가 후보 계산에 들어감)
arr(3.40, 5.45, 3.75, 5.45, color='#cc4444', lw=2.2)
sym(3.57, 5.72, 'rₜ', sz=10, bold=True, color='#cc4444')
# ② → ③ : c_t (후보가 혼합으로)
arr(6.30, 5.45, 6.65, 5.45, color='#cc8800', lw=2.2)
sym(6.47, 5.72, 'cₜ', sz=10, bold=True, color='#cc8800')
# ① → ③ : u_t (업데이트 게이트가 ②를 건너뛰고 혼합으로) — ② 위로 arc
ax.annotate('', xy=(6.9, 6.50), xytext=(3.0, 6.50),
            arrowprops=dict(arrowstyle='->', color='#cc4444', lw=1.8,
                            connectionstyle='arc3,rad=-0.32'), zorder=6)
sym(4.95, 7.18, 'uₜ', sz=10, bold=True, color='#cc4444')

# h_(t-1) 순환 입력 → ①,②,③ 모두 (block-diagonal 경로)
arr(0.95, 3.7, 2.12, 4.55, color='#cc6600', lw=2.0)
arr(0.95, 3.7, 5.02, 4.55, color='#cc6600', lw=1.6, rad=0.08)
arr(0.95, 3.7, 7.55, 4.55, color='#cc6600', lw=1.6, rad=0.12)
sym(1.2, 3.5, 'h₍ₜ₋₁₎', sz=10.5, bold=True, color='#cc6600')
txt(2.7, 3.42, '순환 입력 → ①·②·③ 모두 — U는 block-diagonal (블록 내부만 연결)', sz=9, color='#aa5500', ha='left')

# h_t 출력
arr(9.2, 5.52, 9.95, 5.52, color='#228833', lw=2.4)
box(9.55, 5.05, 1.4, 0.95, fc='#d0f0d0', ec='#228833', lw=1.8)
sym(10.25, 5.52, 'hₜ', sz=14, bold=True, color='#155220')

# 재귀 루프 (h_t → 다음 스텝 h_(t-1))
arr(10.25, 5.05, 10.25, 2.35, color='#228833', lw=1.8, ls='dashed')
arr(10.25, 2.35, 0.85, 2.35, color='#228833', lw=1.8, ls='dashed', rad=0)
arr(0.85, 2.35, 0.85, 3.55, color='#228833', lw=1.8, ls='dashed')
txt(5.4, 2.16, '다음 타임스텝의 h_(t-1) 로 재귀 (recurrence)', sz=9, color='#228833', italic=True)

# ═══════════════ [C] 수식 + 차원 (우상단) ═══════════════
box(11.1, 5.20, 5.6, 5.35, fc='#f7f9ff', ec='#3355aa', lw=1.5, radius=0.22)
txt(11.4, 10.22, '[C] GRU 수식 & 차원(dimension)', sz=12.0, bold=True, color='#1a3a78', ha='left')

eqs = ['rₜ = σ(W_r xₜ + U_r h₍ₜ₋₁₎)',
       'uₜ = σ(W_u xₜ + U_u h₍ₜ₋₁₎)',
       'cₜ = tanh(W_c xₜ + U_c (rₜ ⊙ h₍ₜ₋₁₎))',
       'hₜ = (1 − uₜ) ⊙ h₍ₜ₋₁₎ + uₜ ⊙ cₜ']
ys = [9.62, 9.06, 8.50, 7.94]
for eq, y in zip(eqs, ys):
    sym(11.35, y, eq, sz=10.5, color='#142a55', ha='left')

ax.plot([11.35, 16.5], [7.58, 7.58], color='#ccccdd', lw=1.0)
sym(11.35, 7.32, 'σ : sigmoid     ⊙ : element-wise     tanh', sz=9.0, color='#555', ha='left')

# ── 차원(dimension) ──
txt(11.35, 7.00, '[차원]  d = 모델 차원,  순환상태 크기 = 8d', sz=9.4, bold=True, color='#1a3a78', ha='left')
txt(11.40, 6.70, '· h_(t-1), h_t, r_t, u_t, c_t   →   8d-벡터', sz=9.0, color='#333', ha='left')
txt(11.40, 6.42, '· x_t   →   m-벡터  (z·a·h 각각 embed 후 concat)', sz=9.0, color='#333', ha='left')
txt(11.40, 6.14, '· W_r, W_u, W_c   →   (8d × m) 행렬 · dense', sz=9.0, color='#333', ha='left')
txt(11.40, 5.86, '· U_r, U_u, U_c   →   (8d × 8d) 행렬 · block-diagonal', sz=9.0, color='#aa5500', ha='left')
txt(11.40, 5.58, '· 게이트 곱·활성화 → 원소별, 차원 8d 그대로', sz=9.0, color='#333', ha='left')

# ═══════════════ [D] block-diagonal 구조 (우하단) ═══════════════
box(11.1, 0.5, 5.6, 4.50, fc='#fffdf5', ec='#cc8800', lw=1.5, radius=0.22)
txt(11.4, 4.74, '[D] 순환 가중치 U 의 block-diagonal 구조', sz=12.0, bold=True, color='#8a5500', ha='left')

# 8×8 행렬: 대각 블록만 채움
N = 8
gx, gy, G = 11.55, 1.95, 1.95   # 행렬 좌하단, 한 변 길이
cell = G / N
for i in range(N):
    for j in range(N):
        filled = (i == j)
        ax.add_patch(Rectangle((gx + j * cell, gy + (N - 1 - i) * cell), cell, cell,
                     facecolor='#f5a623' if filled else '#ffffff',
                     edgecolor='#d0b070', lw=0.6, zorder=4))
ax.add_patch(Rectangle((gx, gy), G, G, facecolor='none', edgecolor='#8a5500', lw=1.6, zorder=5))
sym(gx + G / 2, gy - 0.24, '8d', sz=10.5, bold=True, color='#8a5500')
sym(gx - 0.28, gy + G / 2, '8d', sz=10.5, bold=True, color='#8a5500')

# 우측 설명 (행렬 오른쪽)
txt(13.85, 3.78, '· 각 블록 = d × d\n· 대각만 학습\n  (블록 내부만 연결)\n· 블록 간 = 0\n  → 8배 절감', sz=8.8,
    color='#5a4010', ha='left', va='top')

# 하단 수식
txt(11.45, 1.40, '파라미터/FLOPs:  (8d)² = 64d² → 8·d²  (8배 절감)', sz=8.8, color='#5a4010', ha='left')
txt(11.45, 1.10, '예) 200M: d=1024 → 8d = 8192 (8블록 × 1024)', sz=8.8, color='#5a4010', ha='left')
txt(11.45, 0.80, '블록 못 섞임 → [A]에서 h_(t-1) 도 dense embed 보완', sz=8.8, color='#5a4010', ha='left')

# ═══════════════ 하단(좌) 직관 요약 ═══════════════
box(0.4, 0.5, 9.0, 1.35, fc='#eef6ff', ec='#88a0c8', lw=1.3, radius=0.2)
txt(0.7, 1.55, '직관', sz=11, bold=True, color='#33507f', ha='left')
txt(0.7, 1.18,
    '· 업데이트 게이트 u_t : 과거 h_(t-1) 를 그대로 유지할지(u_t≈0) 새 후보 c_t 로 갈아탈지(u_t≈1) 비율',
    sz=9.6, color='#2a3a55', ha='left')
txt(0.7, 0.82,
    '· 리셋 게이트 r_t : 후보 c_t 를 만들 때 과거 h_(t-1) 를 얼마나 무시할지 (r_t≈0 이면 과거 무시)',
    sz=9.6, color='#2a3a55', ha='left')

out = '/home/jovyan/workspace/Paper_Review_KR/World_Model/figs/DreamerV3/gru.png'
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
print('Saved:', out)
