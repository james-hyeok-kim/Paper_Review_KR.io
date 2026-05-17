---
description: 논문 URL/PDF/제목을 받아 5개 agent를 순차/병렬로 호출해 한국어 summary를 만들고 적절한 폴더에 저장합니다.
argument-hint: <arxiv-url | openreview-url | pdf-path | paper-title>
---

# /paper-review — Paper review orchestrator

사용자 입력: `$ARGUMENTS`

당신은 paper review pipeline의 **orchestrator**입니다. 5개 subagent를 다음 순서로 호출하세요. **모든 agent 호출은 Agent tool로 명시적으로**.

## Pipeline

### Stage 1 — Fetch (직렬)
`paper-fetcher` subagent를 호출.
- prompt: 사용자 입력 `$ARGUMENTS` 그대로 전달 + "Read your full instructions and produce the temp/<slug>/ workspace. Reply with the slug."
- 결과에서 **slug**를 추출. 이후 모든 단계에서 사용.

### Stage 2 — Summarize (직렬)
`paper-summarizer` subagent를 호출.
- prompt: `"slug = <slug>. Read temp/<slug>/{text.md,meta.json,figs/captions.json} and write temp/<slug>/draft.md per your instructions."`

### Stage 3 — Audit + Prereq (병렬, 한 메시지에 두 Agent 호출)
- `summary-auditor` — prompt: `"slug = <slug>. Audit draft.md against text.md."`
- `prereq-recommender` — prompt: `"slug = <slug>. Generate prereq.md."`

두 agent 결과를 모두 받은 뒤 다음 단계.

만약 auditor가 `block: true`를 반환하면 → 사용자에게 audit 요약 보여주고 진행 여부 확인.

### Stage 4 — Format & Place (직렬, 마지막)
`format-checker` subagent를 호출.
- prompt: `"slug = <slug>. Integrate draft + audit + prereq, place figures, decide folder + filename per repo convention, save."`

폴더 선택이 애매하다고 응답하면 → 사용자에게 선택지 제시 후 재호출.

### Stage 5 — 최종 보고
format-checker의 ✅ 블록을 사용자에게 그대로 출력하고, 끝에 한 줄 추가:
```
다음 단계: 파일 검토 후 `git add` / commit은 사용자 명시 요청 시에만 진행합니다.
```

## 주의

- 절대 자동 `git add` / `git commit` / `git push` 하지 말 것. CLAUDE.md global rule.
- 5개 agent의 raw 출력을 사용자에게 dump하지 말 것. 각 단계 1-2줄 진행 상황만 보고.
- 중간 파일(`temp/<slug>/`)은 그대로 둬도 무방. 사용자가 정리하라고 하면 `rm -rf temp/<slug>/` 만 진행.
- 입력이 모호하면 (`title only`인 경우 후보 여러 개) paper-fetcher에서 user에게 질문 위임.
