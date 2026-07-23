# Adaptive, Time-Budgeted Research Test Results

Run date: 2026-07-23

## Adaptive duration-controller gate

The final focused backend suite passed in the exact built application
environment:

```text
92 passed, 4 warnings in 1.55s
4 passed, 4 warnings in 1.56s
```

The first line covers duration validation and every cold-start band, matching
stack calibration and bounded upward/downward adjustment, structured aspect
planning, failure-specific repair, focused concurrency, source selection and
integrity, adaptive Jina compression rescue, fallback corroboration,
cross-aspect verified-evidence reuse, trajectory metrics, and the final report
citation allowlist. The second line covers the MCP duration contract and
response utilities.

The registered live MCP schema exposes only:

```text
deep_research(query: string, research_duration_seconds: integer = 60)
```

`breadth` and `depth` remain internal rollback controls and are absent from the
public tool schema. Durations outside 15–600 seconds return a structured error.

## Warm practical duration runs

Report generation was disabled for the research-only rows so the measured time
matches `research_duration_seconds`.

| Target | Prompt class | Policy (aspects / repairs / deepened) | Coverage | Verified sources | Actual research |
|---:|---|---:|---:|---:|---:|
| 15s | Direct URL, Mage-Flow | 1 / 0 / 0 | 1/1 (100%) | 1 | 1.25s |
| 60s | Broad embedding/retrieval explanation | 3 / 2 / 1 | 3/3 (100%) | 8 | 55.36s |
| 120s | Jina v5 vs Qwen3-0.6B, versioned comparison | 4 / 3 / 2 | 4/4 (100%) | 7 | 45.97s |
| 300s | Recent vLLM/Jina/Qwen deployment research | 5 / 5 / 3 | 5/5 (100%) | 13 | 65.42s |

The 60-second run repaired two missing aspects before deepening and completed
within 7.7% of its target. The 120- and 300-second jobs exhausted their bounded
quality policies much earlier than the cold-start time estimates. The
controller now learns in both directions after ten same-stack samples:
over-budget stacks lose optional deepening/repair/coverage, while fast stacks
gain coverage, recovery, cards, and deepening up to the global 6 / 8 / 4 caps.
It never sleeps to fill time and never creates nested planner trees.

The 300-second prompt initially scored 3/5 coverage. Entity-specific comparison
planning raised this to 4/5; reusing already-verified primary evidence across
aspects without a duplicate scrape raised the final run to 5/5.

## Full-report and citation containment smoke

A 15-second Mage-Flow direct-URL job was run with report generation enabled:

```text
research: 1.597s
report generation: 16.301s
report length: 9,918 characters
verified sources: 1
distinct report citation URLs: 1
unverified citation URLs: 0
```

The report writer originally followed related links embedded in the verified
model card. The final implementation gives it an explicit verified-URL
allowlist and applies a hard post-generation guard. The guarded report cited
only `https://huggingface.co/microsoft/Mage-Flow`, and the trajectory recorded
`report_citation_guard`.

The production controller remains in `shadow` mode as planned. Calibration has
not yet reached ten representative samples in each duration band, so enabling
the controller by default and the five-prompt pairwise report-quality
acceptance decision remain rollout gates rather than claimed results.

## Automated quality gate

Commands run in the exact built backend and MCP images:

```text
61 passed, 2 warnings in 1.52s
2 passed, 2 warnings in 0.02s
```

Covered areas:

- Jina query/document formatting and independent embedding endpoint routing.
- Evidence-grounded planner prompts for versioned, niche questions.
- Only selector-approved URLs reach Crawl4AI.
- Invalid selector JSON uses deterministic relevance/source-quality/domain-diversity fallback.
- A model-selected off-topic card is blocked by the query-anchor guard.
- Job-scoped trajectory recording, focused/legacy policy combinations, branch
  ranking, empty-evidence deepening guards, and llama-swap `:gptr` alias
  discovery.
- MCP exposes only verified evidence URLs, while attempted URLs remain internal
  for concurrent deduplication.
- Adversarial source pools for EU CRA policy, OpenTelemetry semantic-convention migration, and Linux 6.12 io_uring. These include official sources, independent coverage, dictionaries, text-comparison utilities, duplicate-domain candidates, and generic off-topic pages.

The warnings are pytest configuration warnings and a PyMuPDF deprecation
warning; neither is a test failure.

An additional broad suite run reached `222 passed / 25 failed`. The remaining
failures are outside this change's quality gate and include an unrelated
`secure_filename` import failure, combined-suite module pollution, read-only
logging tests, a missing async marker, stale SearX metadata expectations, and
retriever mocks that leak between tests. The full suite is therefore not
represented as green.

## Deployed-path checks

- Backend returned HTTP 200 from `/`; MCP `/health` returned healthy after deployment.
- The running backend imports `QueryInstructionEmbeddings` and exposes `ResearchConductor._select_source_candidates`.
- Runtime model resolution is `laguna-s-2.1-nvfp4:gptr` for all LLM roles and
  `jina-v5-retrieval` for embeddings.
- Production `auto` policy resolves to `ranked/focused`; source standards and
  trajectory recording are enabled.
- The Jina route returned model identity `jina-v5-retrieval` and
  1024-dimensional vectors through the application network path.

## Practical focused-versus-upstream-like comparison

Both runs used the same warm services, models, source standards, prompt, and
MCP depth/breadth (`2/3`). Report generation was disabled so timings isolate
research execution.

| Policy | Branches | Unique queries | Search passes | Verified sources | Context | Research time |
|---|---:|---:|---:|---:|---:|---:|
| `ranked/focused` | 5 | 5 | 5 | 6 | 4,277 chars | 33.4s |
| `legacy_all/standard` | 7 | 18 | 21 | 15 | 66,065 chars | 64.0s |

Trajectories:

- `28a45817-e95f-4969-9fb2-060b7be30ba2` — ranked/focused
- `505bc36d-9599-413b-b12e-376527de6d44` — legacy_all/standard

The focused path was 47.8% faster and used 76.2% fewer search passes. Its first
level covered training, deployment, and evaluation, but weak SERPs caused two
branches and one child to yield no usable context; only the deployment branch
was deepened. It retained Jina's official site and the Jina v5 paper, but also
used broader-web fallback pages for deployment questions.

The upstream-like path produced much more context and stronger Jina-specific
primary evidence, but it did not preserve broad topical coverage. Recursive
planning concentrated on Jina and then drifted into CARE/CMedTEB, Chinese
medical retrieval, and annotation pipelines. More breadth therefore increased
volume and niche detail, not proportional relevance to the original question.

Current recommendation: retain `ranked/focused` as the production default, but
make first-level coverage resilient before adding more recursion. A dynamic
duration controller should spend additional budget first on retrying an empty
high-value aspect with a reformulated query, then on deepening the next
best distinct branch—not on restoring planner expansion inside every branch.
