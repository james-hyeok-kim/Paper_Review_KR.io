# Paper Review Agents — 사용 가이드

이 문서는 `/paper-review` slash command와 5개 paper review agent를 어떻게 호출하고, 어떤 형태로 논문을 입력해야 하는지 설명합니다.

---

## Slash command란?

Claude Code에서 `/명령어` 형태로 호출하는 **단축 명령**입니다. 매번 길게 설명하지 않고 `/명령어 인자` 한 줄로 미리 정의된 작업을 trigger합니다.

예시 (이미 존재하는 것들):
- `/help` — 도움말
- `/config` — 설정 변경
- `/review` — PR 리뷰

우리가 만들 `/paper-review`도 같은 방식 — 사용자가 한 줄 입력하면 orchestrator가 5개 agent를 자동으로 순서대로 호출합니다.

> 참고: `.claude/commands/paper-review.md` 파일이 곧 그 정의. 사용자는 파일을 직접 만질 필요 없고, 그냥 `/paper-review`라고 prompt에 치면 됩니다.

---

## 논문 주는 4가지 방법 (예시)

### 1. Arxiv URL (가장 일반적)
```
/paper-review https://arxiv.org/abs/2410.10594
```
→ VISRAG 논문을 자동으로 가져와 요약.

`arxiv.org/pdf/2410.10594` 형태(PDF 직링크)도 동일하게 처리.

### 2. 로컬 PDF 파일 경로
```
/paper-review /home/jovyan/Downloads/visrag.pdf
```
→ 인터넷 없이도 동작. 메타데이터(제목/저자/학회)는 PDF 첫 페이지에서 파싱.

### 3. Conference / OpenReview URL
```
/paper-review https://openreview.net/forum?id=abcd1234
```
→ OpenReview, NeurIPS proceedings, ACL anthology 등에서 직접 가져옴.

### 4. 논문 제목만 (가장 헐렁한 방식)
```
/paper-review VISRAG: Vision-based Retrieval-Augmented Generation
```
→ WebSearch로 자동 검색. 같은 제목 여러 개 나오면 후보 보여주고 사용자에게 질문.

---

## 실행 흐름 (입력 후 자동으로 일어나는 일)

```
사용자: /paper-review https://arxiv.org/abs/2410.10594
       │
       ▼
[1] paper-fetcher     PDF 다운로드 + 메타 추출 + 그림 뽑기 (~30초)
[2] paper-summarizer  한국어 summary 초안 작성 (~1-2분)
[3] auditor + prereq  병렬로 검수 / 사전지식 추천 (~30초)
[5] format-checker    파일명·폴더·markdown·그림 위치 검수 (~10초)
       │
       ▼
결과: General_AI/[논문][2025][Summary][VISRAG] ... .md 파일이
      해당 폴더에 자동 저장됨. Claude가 최종 위치와 audit 결과를 출력.
```

---

## 5개 Agent 역할 요약

| # | Agent | 역할 |
|---|---|---|
| 1 | `paper-fetcher` | URL/PDF/제목으로부터 논문 본문 + 메타데이터 + figure 수집 |
| 2 | `paper-summarizer` | 한국어 summary 초안 작성 (VISRAG 스타일 템플릿) |
| 3 | `summary-auditor` | 핵심 contribution / 수식 / 메타데이터 누락 검수 |
| 4 | `prereq-recommender` | 사전 지식 / 선행 논문 추천 + 이 repo 내 관련 문서 cross-link |
| 5 | `format-checker` | Markdown 문법 + figure 위치 + 파일명/폴더 convention 최종 검증 |

자세한 설계는 [`plan_001.md`](./plan_001.md) 참고.

---

## 참고: Figure 처리 방식

기존 repo의 summary (예: VISRAG)는 GitHub user-attachments URL (`<img src="https://github.com/user-attachments/..."/>`)을 사용합니다. 이는 사람이 GitHub 웹에서 이미지를 드래그-드롭으로 paste한 결과입니다.

본 agent system이 생성하는 새 summary는 **local 파일 + 상대 경로**를 사용합니다:
- 저장 위치: `<chosen_folder>/figs/<TAG>/fig_NN.png`
- 참조 방식: `<img src="figs/<TAG>/fig_NN.png" .../>`

여기서 **`<TAG>`는 논문 약칭** (예: `StreamDiT`, `VISRAG`, `GPTQ`). 파일명의 `[Tag]` 부분에서 추출합니다.
arxiv ID는 사람이 읽기 어려워 폴더명으로 사용하지 않습니다.

예시:
```
Video_DiT/
├── [논문][2025][Summary][StreamDiT] StreamDiT ... .md
└── figs/
    └── StreamDiT/
        ├── fig_01.png
        ├── ...
        └── captions.json
```

이유: 자동화 가능, 인터넷 없이도 작동, GitHub Pages에서도 정상 렌더링.
기존 user-attachments 방식도 자유롭게 함께 사용 가능합니다.
