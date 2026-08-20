# Lab 20 — Multi-Agent Research System: Báo cáo triển khai

**Ngày:** 2026-08-20
**Người thực hiện:** Tống Duy An (2A202601995)
**Repo:** K4-Track3-Lab20 — starter skeleton, điền toàn bộ `TODO(student)` trong `src/`

## Tóm tắt

Hoàn thành end-to-end bài lab Multi-Agent Research System: implement LLM client, offline search client, 5 agents (Supervisor, Researcher, Analyst, Writer, Critic), LangGraph workflow hub-spoke, benchmark single-agent vs multi-agent trên 3 queries, tracing 2 tầng (local JSON + LangSmith wiring). 14 unit tests pass, ruff check + format clean.

## Các thay đổi chính

### Services

- `services/llm_client.py` — OpenAI Chat Completions client, retry ×3 với exponential backoff, timeout theo `TIMEOUT_SECONDS`, tracking token + ước tính cost theo bảng giá per-model (gpt-4o-mini, gpt-4o, gpt-4.1-mini, gpt-4.1). Retry/timeout/cost đặt tập trung ở client, agents không import SDK trực tiếp.
- `services/search_client.py` — offline search trên `ai_agent_offline_research_corpus_v2/` (30 topics): chọn topic file tốt nhất qua keyword overlap với `manifest.csv`, rồi rank `source_documents` + `knowledge_articles` trong topic đó (title ×2 + body). Không cần Tavily key, không cần mạng.

### Agents

- `agents/supervisor.py` — routing **rule-based** (không LLM call: deterministic, rẻ, dễ trace) theo field còn thiếu trong state: `sources`/`research_notes` → researcher; `analysis_notes` → analyst; `final_answer` → writer; critic chạy đúng 1 lần sau writer; đủ hết → done. **3 guard chống loop/treo:** `max_iterations` (6), `timeout_seconds` wall-clock (60s, qua `state.started_at`), error-count fallback (dừng sau 3 lỗi worker).
- `agents/researcher.py` — search corpus, LLM tóm tắt thành `research_notes` chỉ dựa trên sources, mọi bullet cite nhãn `[S#]`, flag nguồn synthetic là lower-trust.
- `agents/analyst.py` — trích key claims kèm nhãn nguồn, so điểm đồng thuận/mâu thuẫn giữa các nguồn, flag evidence yếu. Fallback: nếu lỗi thì pass `research_notes` thô cho Writer để pipeline không gãy.
- `agents/writer.py` — tổng hợp `final_answer` ~500 từ, mọi claim cite `[S#]`, kết bằng danh sách Sources.
- `agents/critic.py` — **rule-based** (không LLM nên không tự hallucinate): tính citation coverage (số nguồn được cite / tổng nguồn) và bắt **dangling citation** (cite `[S9]` khi chỉ có 5 nguồn) → ghi vào `state.errors`.
- `agents/base.py` — thêm helper `format_sources` / `source_label` dùng chung, nhãn `[S1]..[Sn]` nhất quán xuyên suốt pipeline.

### Orchestration & state

- `graph/workflow.py` — LangGraph `StateGraph(ResearchState)` hình sao (hub-spoke): mọi worker edge về supervisor; supervisor là điểm quyết định duy nhất qua conditional edges; `recursion_limit = 2×max_iterations + 4`. Một `LLMClient` share cho mọi worker.
- `core/state.py` — thêm field `started_at` phục vụ timeout guard.

### Evaluation & CLI

- `evaluation/benchmark.py` — đủ 5 metric: latency (wall-clock), cost (tổng `cost_usd` từ `agent_results`), quality (LLM-as-judge 0–10, best-effort), citation coverage, failure rate. Run crash được ghi nhận là data point thay vì làm sập benchmark.
- `cli.py` — 3 lệnh: `baseline` (single-agent: cùng retrieval, 1 LLM call làm hết), `multi-agent`, `benchmark` (nhiều query, chạy cả 2 mode, xuất `reports/benchmark_report.md` + trace JSON per run).

### Observability

- `observability/tracing.py` — 2 tầng: (1) `trace_span` local JSON luôn bật, lưu vào `state.trace`, export ra `reports/traces/*.json`; (2) LangSmith qua `configure_tracing()` — set env cho langchain-core/langgraph auto-trace graph run + `wrap_openai` để LLM call lồng dưới graph run. Degrade an toàn: LangSmith lỗi thì chỉ warning, pipeline vẫn chạy.

### Tests

- `tests/test_agents_todo.py` — thay test TODO-guard cũ bằng 14 test thật, chạy offline không gọi API: routing policy đủ nhánh (researcher/analyst/writer/critic/done), max_iterations guard, error fallback, citation coverage, dangling citation, offline search ranking.

## Kết quả benchmark (gpt-4o-mini, 3 queries × 2 modes)

| Metric | Single-agent | Multi-agent |
|---|---|---|
| Latency | ~8s | ~15–17s (≈2×) |
| Cost/query | ~$0.0005 | ~$0.0012 (≈2.4×) |
| Quality (judge 0–10) | 8.0 / 8.0 / 8.0 | 8.0 / 6.0 / 6.0 |
| Citation coverage | 100% | 100% |
| Failure rate | 0% | 0% |

Kết luận trung thực: **single-agent thắng trên task cỡ này** — khớp decision rule "single agent >80% thì đừng thêm agents". Multi-agent chỉ đáng khi task cần decomposition thật, verifier độc lập, hoặc subtask parallel được. Chi tiết + failure mode phân tích trong `reports/benchmark_report.md`.

## Failure mode đã gặp/chặn

1. **Infinite routing loop** — worker fail lặp làm supervisor route mãi; chặn bằng 3 guard ở supervisor.
2. **Dangling citation** — Writer cite nhãn không tồn tại; Critic bắt và ghi error.
3. **Ruff E501/SIM905** — sửa format, `ruff check` + `ruff format` sạch.

## LangSmith — đã hoạt động (APAC endpoint)

- Lỗi 403 ban đầu do sai region: key thuộc workspace **APAC**, còn SDK mặc định trỏ US (`api.smith.langchain.com`). Probe xác nhận `https://apac.api.smith.langchain.com` trả 200 cho `/sessions`.
- Fix: thêm setting `langsmith_endpoint` (`LANGSMITH_ENDPOINT` trong `.env`, đã thêm cả `.env.example`), `configure_tracing()` set thêm `LANGSMITH_ENDPOINT`/`LANGCHAIN_ENDPOINT`.
- Verify bằng API: 10 runs ingest thành công vào project `multi-agent-research-lab` (spans `agent.supervisor`, `agent.critic`, các node LangGraph, status=success).
- **Còn lại (manual):** mở project trên smith.langchain.com (workspace APAC) chụp screenshot trace cho deliverable. Local JSON traces vẫn song song tại `reports/traces/*.json`.
