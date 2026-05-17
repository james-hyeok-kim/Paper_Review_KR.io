# plan_001 — Paper Review Agent System

작성: 2026-05-17
상태: **승인 대기 (Awaiting user approval)**

---

## 1. 목표 (Goal)

URL / PDF / 논문 제목을 입력하면 한국어로 잘 정리된 markdown summary를 만들어 적절한 폴더에 적절한 이름으로 저장해주는 multi-agent system 구축. 기존 `General_AI/VISRAG` summary와 같은 스타일을 골든 스탠다드로 한다.

## 2. 구성 (Architecture)

**Orchestrator 1개 + Subagent 5개** (사용자 결정).

```
/paper-review <url|pdf-path|title>
        │
        ▼
  paper-review-orchestrator  (slash command)
        │
        ├─[1]─► paper-fetcher          ──► raw_text + metadata + figures/
        │
        ├─[2]─► paper-summarizer       ──► draft.md (VISRAG 스타일)
        │
        ├─[3]─► summary-auditor        ─┐
        ├─[4]─► prereq-recommender     ─┴► (병렬, draft.md 참조해서 의견)
        │
        └─[5]─► format-checker         ──► 최종 .md (naming/folder/figure 검수)
```

순서:
- (1) → (2) 직렬
- (3), (4) 병렬 (둘 다 (2)의 출력만 보면 됨)
- (5)는 마지막에 단독, 위 모든 결과 통합 + 파일명/경로/markdown 검수

## 3. Agent 설계 (각 .md 파일은 `.claude/agents/`에 저장)

### Agent 1 — `paper-fetcher`
- **Input**: arxiv URL / PDF 경로 / OpenReview URL / 논문 제목 (text)
- **Tools**: WebFetch, WebSearch, Read (PDF), Bash (pdftotext, pdfimages)
- **Output**:
  - `temp/<paper-slug>/text.md` — 본문 텍스트
  - `temp/<paper-slug>/meta.json` — 제목/저자/연도/conference/abstract/arxiv id
  - `temp/<paper-slug>/figs/fig_NN.png` — 추출된 figures (있다면)
- **Edge cases**: arxiv 접근 실패 시 mirror 시도, PDF만 있고 메타데이터 없을 때는 첫 페이지에서 파싱.

### Agent 2 — `paper-summarizer`
- **Input**: `temp/<paper-slug>/` (text + meta + figs)
- **Tools**: Read, Write
- **필수 헤더 항목 (반드시 포함, user 요청)**:
  - **저자 (Authors)** — 소속 포함
  - **학회/발표 (Venue)** — 예: `ICLR 2025`, `NeurIPS 2024`, `arXiv 2026` (학회 미정인 경우)
  - **논문 위치 (Paper link)** — `[PDF](arxiv URL)` + 가능하면 OpenReview/conference page도 추가
  - **출처 (Source)** — abstract URL (arxiv abs/, openreview forum, 또는 PDF 원본 경로)
  - **생성일** (이 summary가 만들어진 날짜)
- **Output**: `temp/<paper-slug>/draft.md` — 한국어 summary, 아래 VISRAG 스타일 템플릿:
  ```
  # {TITLE}

  저자 : {Authors with affiliations}

  발표 : {Venue YEAR}

  논문 : [PDF]({pdf_url})

  출처 : {abs/forum URL}

  ---
  ## 0. Summary
   <figure marker>
   ### 0.1. 문제 (Problem)
   ### 0.2. 핵심 아이디어 (Core Idea)
   ### 0.3. 효과 (Effects)
   ### 0.4. 결과 (Results)
   ### 0.5+. (논문 기여별 detail 섹션)
  ---
  ## 1. Introduction
  ## 2. Method
  ## 3. Experiments
  ## 4. Conclusion
  ```
- **언어**: 한국어 (technical term은 영어 병기).

### Agent 3 — `summary-auditor`
- **Input**: `temp/<paper-slug>/draft.md` + 원본 text
- **Tools**: Read
- **검사 항목**:
  - **헤더 메타데이터** (저자/학회/PDF link/출처) 모두 채워졌는가 — 비어있으면 fail
  - 핵심 contribution이 빠지지 않았는가
  - 주요 수식/알고리즘 누락 여부
  - Results 섹션에 핵심 metric 포함 여부
  - 한계점/limitation 언급 여부
  - 사용자가 헷갈릴만한 용어 (예: rotation matrix, perplexity) 설명 충분 여부
- **Output**: `temp/<paper-slug>/audit.md` — 누락 항목 리스트 + 보강 제안

### Agent 4 — `prereq-recommender`
- **Input**: `temp/<paper-slug>/draft.md` + meta
- **Tools**: Read, WebSearch
- **Output**: `temp/<paper-slug>/prereq.md` — 다음 섹션:
  - 이 논문을 이해하기 위해 알아야 할 **개념** (예: RoPE, GPTQ, ...)
  - 이전에 읽으면 좋은 **선행 논문** (이름 + 한 줄 설명 + arxiv link)
  - 이 폴더 안에 이미 있는 관련 summary 파일 cross-reference
- 결과는 최종 .md의 맨 아래에 "## 부록: 사전 지식" 으로 통합.

### Agent 5 — `format-checker`
- **Input**: `temp/<paper-slug>/draft.md` + audit + prereq + figs
- **Tools**: Read, Write, Edit, Bash (ls)
- **수행 작업**:
  1. **Naming convention 검사**:
     - 패턴: `[논문][YEAR][TAG][Summary] TITLE.md` (또는 conference 포함 시 `[논문][CONF][YEAR][Summary][TAG] TITLE.md`)
     - meta에서 추출한 year, tag, conference로 파일명 조립
  2. **Folder 결정**:
     - 기존 폴더(`Diffusion`, `Diffusion_Quantization`, `Diffusion_VLA`, `Diffusion_VLA_Quant`, `Diffusion+HW`, `General_AI`, `Ilya_Sutskever_Top30`, `LLM_Quant`) 중 가장 적합한 곳 선택
     - 애매하면 후보 2~3개 제시하고 사용자에게 질문
  3. **Figure 위치 검수**:
     - draft.md의 figure marker(`<!-- FIG: fig_01 -->` 등)와 추출된 fig 파일 매칭
     - 상대 경로로 변환 (`figs/<paper-slug>/fig_01.png`)
     - 적절한 섹션에 들어갔는지 확인 (Figure 1은 0. Summary 또는 Method 초반 등)
  4. **Markdown 검수**: heading level, list indent, 수식 ($, $$) 균형, link 깨짐 여부
  5. **출력**: 최종 파일을 결정된 폴더로 이동/저장 + 파일 위치 사용자에게 보고

## 4. Slash command 정의

`.claude/commands/paper-review.md`:
- 입력 파싱 → orchestrator 실행
- 사용 예: `/paper-review https://arxiv.org/abs/2410.10594`

## 5. 파일 위치 (Files to be created)

```
.claude/
  agents/
    paper-fetcher.md
    paper-summarizer.md
    summary-auditor.md
    prereq-recommender.md
    format-checker.md
  commands/
    paper-review.md
```

추가로 `temp/`는 `.gitignore`에 추가 (중간 파일).

## 6. Test 전략 (필수)

### Success criteria
- 임의의 arxiv URL (예: `https://arxiv.org/abs/2410.10594` VISRAG)을 입력했을 때, 5분 안에 적절한 폴더(General_AI)에 `[논문][2025][Summary][VISRAG] VISION-BASED ... .md` 형태의 파일 생성
- 출력이 기존 VISRAG.md와 다음 항목에서 일치:
  - heading 구조 (0. Summary 4-subsection)
  - 한국어 작성
  - figure embed 존재
  - 저자/발표/PDF link 헤더

### Edge cases (반드시 확인)
- (a) arxiv URL이 아닌 OpenReview URL
- (b) PDF만 있고 metadata 추출이 까다로운 경우
- (c) Figure가 없는 theory paper
- (d) 이미 같은 이름 파일이 존재하는 경우 (덮어쓰기 prompt)
- (e) Conference 정보를 추출 못한 경우 (CONF 태그 비움)

### Metric
- **정성적 비교**: 골든 VISRAG.md와 새로 생성된 VISRAG 재요약본을 side-by-side로 user가 확인
- **정량적**: 5개 audit 항목 중 통과 개수 (>=4/5 → pass)

### Fallback
- Figure 추출 실패 → marker만 남기고 사용자에게 manual upload 안내
- Folder 결정 실패 → `_Unsorted/` 임시 폴더 사용 + user 질문
- WebFetch 실패 → 최대 2회 재시도 후 user에게 PDF 직접 제공 요청

## 7. 위험 요소 (Risks)

| 위험 | 영향 | 완화 |
|---|---|---|
| Subagent context bloat (전체 논문 본문 전달) | 5개 agent 호출 시 토큰 폭증 | (a) text.md를 디스크에 저장하고 경로만 넘김, (b) summarizer만 본문 전부 봄, 나머지는 draft.md만 봄 |
| Figure 추출 품질 | pdfimages가 vector figure를 못 뽑음 | 1차로 pdfimages, 실패 시 마커만 남김 |
| Folder 잘못 선택 | 다른 사람이 못 찾음 | 애매할 때 user에게 ask, 자의적 결정 금지 |
| Naming 충돌 | 기존 파일 덮어쓰기 | 항상 dry-run 결과 보여주고 user 승인 |

## 8. 예상 작업 시간

- Agent 5개 작성: 60-90분
- Slash command + glue: 15분
- Smoke test 1회: 5-10분
- 총: ~2시간

## 9. 변경 이력

- 2026-05-17 초안 작성. Agent #5에 "naming/folder convention 검수" 추가 (user 요청).
