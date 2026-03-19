"""
BiasGuard Orchestrator
=======================
LangGraph-powered agentic orchestration pipeline.

Agent DAG:
    retrieve → analyze → mitigate → score → finalize

Each node is a stateful agent that reads from and writes to
the shared BiasGuardState TypedDict.
"""

from __future__ import annotations

import time
import uuid
from typing import Annotated, Any

import structlog
from langchain_core.messages import BaseMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from agents.analyzer_agent import AnalyzerAgent
from agents.mitigator_agent import MitigatorAgent
from agents.retriever_agent import RetrieverAgent
from agents.rewrite_utils import build_full_document_rewrite
from agents.scorer_agent import ScorerAgent
from config.llm_router import build_llm
from config.settings import get_settings
from monitoring.prometheus_metrics import LLM_CALLS_TOTAL, LLM_LATENCY

logger = structlog.get_logger(__name__)


class BiasGuardState(TypedDict):
    """Shared state object passed through the LangGraph pipeline."""

    # Input
    run_id: str
    text: str
    doc_type: str
    llm_provider: str | None

    # Intermediate state
    messages: Annotated[list[BaseMessage], add_messages]
    retrieved_patterns: list[dict]
    raw_analysis: dict | None
    bias_instances: list[dict]
    rewrites: list[dict]

    # Output
    overall_score: float
    severity: str
    report: dict | None
    error: str | None

    # Timing
    started_at: float
    retrieval_duration_ms: float
    analysis_duration_ms: float
    mitigation_duration_ms: float
    scoring_duration_ms: float


def _create_initial_state(
    text: str,
    doc_type: str,
    llm_provider: str | None = None,
) -> BiasGuardState:
    return BiasGuardState(
        run_id=str(uuid.uuid4()),
        text=text,
        doc_type=doc_type,
        llm_provider=llm_provider,
        messages=[],
        retrieved_patterns=[],
        raw_analysis=None,
        bias_instances=[],
        rewrites=[],
        overall_score=0.0,
        severity="UNKNOWN",
        report=None,
        error=None,
        started_at=time.time(),
        retrieval_duration_ms=0.0,
        analysis_duration_ms=0.0,
        mitigation_duration_ms=0.0,
        scoring_duration_ms=0.0,
    )


class BiasGuardOrchestrator:
    """
    Main orchestrator using LangGraph StateGraph.

    Nodes:
        retrieve  → Retriever Agent: RAG over bias lexicon
        analyze   → Analyzer Agent: structured bias detection
        mitigate  → Mitigator Agent: neutral rewrite suggestions
        score     → Scorer Agent: severity scoring + ranking
        finalize  → Compiles final BiasGuardReport

    Edges:
        retrieve → analyze → mitigate → score → finalize → END
    """

    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self._graph = self._build_graph()

    def _build_graph(self) -> Any:
        """Construct the LangGraph StateGraph."""

        graph = StateGraph(BiasGuardState)

        # Register nodes
        graph.add_node("retrieve", self._retrieve_node)
        graph.add_node("analyze", self._analyze_node)
        graph.add_node("mitigate", self._mitigate_node)
        graph.add_node("score", self._score_node)
        graph.add_node("finalize", self._finalize_node)

        # Entry point
        graph.set_entry_point("retrieve")

        # Linear pipeline edges
        graph.add_edge("retrieve", "analyze")
        graph.add_edge("analyze", "mitigate")
        graph.add_edge("mitigate", "score")
        graph.add_edge("score", "finalize")
        graph.add_edge("finalize", END)

        return graph.compile()

    # ─── Agent Nodes ───────────────────────────────────────────────────────

    def _retrieve_node(self, state: BiasGuardState) -> dict:
        """Retriever Agent: fetch relevant bias patterns via RAG."""
        logger.info("node_retrieve_start", run_id=state["run_id"])
        t0 = time.time()

        try:
            agent = RetrieverAgent(settings=self.settings)
            patterns = agent.retrieve(
                text=state["text"],
                doc_type=state["doc_type"],
            )
            duration = (time.time() - t0) * 1000
            logger.info(
                "node_retrieve_complete",
                run_id=state["run_id"],
                patterns_found=len(patterns),
                duration_ms=round(duration, 2),
            )
            return {
                "retrieved_patterns": patterns,
                "retrieval_duration_ms": duration,
            }
        except Exception as e:
            logger.error("node_retrieve_failed", error=str(e))
            return {"error": f"Retrieval failed: {e}", "retrieved_patterns": []}

    def _analyze_node(self, state: BiasGuardState) -> dict:
        """Analyzer Agent: structured multi-category bias detection."""
        if state.get("error"):
            return {}

        logger.info("node_analyze_start", run_id=state["run_id"])
        t0 = time.time()

        provider = state.get("llm_provider") or self.settings.llm_provider.value
        model = self.settings.llm_model

        try:
            llm = build_llm(
                provider=state.get("llm_provider"),
                settings=self.settings,
            )
            agent = AnalyzerAgent(llm=llm, settings=self.settings)
            analysis = agent.analyze(
                text=state["text"],
                doc_type=state["doc_type"],
                retrieved_patterns=state["retrieved_patterns"],
            )
            duration = (time.time() - t0) * 1000
            logger.info(
                "node_analyze_complete",
                run_id=state["run_id"],
                bias_instances=len(analysis.get("bias_instances", [])),
                duration_ms=round(duration, 2),
            )
            return {
                "raw_analysis": analysis,
                "bias_instances": analysis.get("bias_instances", []),
                "analysis_duration_ms": duration,
            }
        except Exception as e:
            logger.error("node_analyze_failed", error=str(e))
            return {"error": f"Analysis failed: {e}"}
        finally:
            elapsed_seconds = max((time.time() - t0), 0.0)
            LLM_CALLS_TOTAL.labels(
                provider=str(provider),
                model=str(model),
                agent="analyzer",
            ).inc()
            LLM_LATENCY.labels(
                provider=str(provider),
                agent="analyzer",
            ).observe(elapsed_seconds)

    def _mitigate_node(self, state: BiasGuardState) -> dict:
        """Mitigator Agent: generate neutral rewrites for each bias instance."""
        if state.get("error") or not state.get("bias_instances"):
            return {"rewrites": []}

        logger.info("node_mitigate_start", run_id=state["run_id"])
        t0 = time.time()

        provider = self.settings.llm_provider.value
        model = self.settings.llm_model

        try:
            llm = build_llm(settings=self.settings)
            agent = MitigatorAgent(llm=llm, settings=self.settings)
            rewrites = agent.mitigate(
                text=state["text"],
                bias_instances=state["bias_instances"],
                doc_type=state["doc_type"],
            )
            duration = (time.time() - t0) * 1000
            logger.info(
                "node_mitigate_complete",
                run_id=state["run_id"],
                rewrites_generated=len(rewrites),
                duration_ms=round(duration, 2),
            )
            return {
                "rewrites": rewrites,
                "mitigation_duration_ms": duration,
            }
        except Exception as e:
            logger.error("node_mitigate_failed", error=str(e))
            return {"rewrites": [], "error": f"Mitigation failed: {e}"}
        finally:
            elapsed_seconds = max((time.time() - t0), 0.0)
            LLM_CALLS_TOTAL.labels(
                provider=str(provider),
                model=str(model),
                agent="mitigator",
            ).inc()
            LLM_LATENCY.labels(
                provider=str(provider),
                agent="mitigator",
            ).observe(elapsed_seconds)

    def _score_node(self, state: BiasGuardState) -> dict:
        """Scorer Agent: calculate overall bias score and severity."""
        logger.info("node_score_start", run_id=state["run_id"])
        t0 = time.time()

        try:
            agent = ScorerAgent(settings=self.settings)
            score, severity = agent.score(
                bias_instances=state["bias_instances"],
                text_length=len(state["text"].split()),
            )
            duration = (time.time() - t0) * 1000
            logger.info(
                "node_score_complete",
                run_id=state["run_id"],
                score=score,
                severity=severity,
                duration_ms=round(duration, 2),
            )
            return {
                "overall_score": score,
                "severity": severity,
                "scoring_duration_ms": duration,
            }
        except Exception as e:
            logger.error("node_score_failed", error=str(e))
            return {"overall_score": 0.0, "severity": "UNKNOWN"}

    def _finalize_node(self, state: BiasGuardState) -> dict:
        """Compile all agent outputs into final BiasGuardReport."""
        logger.info("node_finalize_start", run_id=state["run_id"])

        total_duration = (time.time() - state["started_at"]) * 1000

        # Merge rewrites into bias instances
        rewrite_map = {r["instance_id"]: r for r in state.get("rewrites", [])}
        enriched_instances = []
        for instance in state.get("bias_instances", []):
            instance_id = instance.get("id", "")
            if instance_id in rewrite_map:
                instance["rewrite_suggestion"] = rewrite_map[instance_id].get(
                    "rewrite", ""
                )
                instance["rewrite_explanation"] = rewrite_map[instance_id].get(
                    "explanation", ""
                )
            enriched_instances.append(instance)

        report = {
            "run_id": state["run_id"],
            "doc_type": state["doc_type"],
            "overall_bias_score": round(state["overall_score"], 3),
            "severity": state["severity"],
            "bias_instance_count": len(enriched_instances),
            "bias_instances": enriched_instances,
            "category_summary": _summarize_by_category(enriched_instances),
            "full_document_rewrite": build_full_document_rewrite(
                text=state["text"],
                bias_instances=enriched_instances,
                rewrites=state.get("rewrites", []),
            ),
            "performance": {
                "total_duration_ms": round(total_duration, 2),
                "retrieval_duration_ms": round(state["retrieval_duration_ms"], 2),
                "analysis_duration_ms": round(state["analysis_duration_ms"], 2),
                "mitigation_duration_ms": round(state["mitigation_duration_ms"], 2),
                "scoring_duration_ms": round(state["scoring_duration_ms"], 2),
            },
            "error": state.get("error"),
        }

        logger.info(
            "pipeline_complete",
            run_id=state["run_id"],
            score=state["overall_score"],
            severity=state["severity"],
            instances=len(enriched_instances),
            total_ms=round(total_duration, 2),
        )

        return {"report": report}

    def _error_node(self, state: BiasGuardState) -> dict:
        """Error handler node — logs and returns partial report."""
        logger.error("pipeline_error", error=state.get("error"), run_id=state["run_id"])
        return {
            "report": {
                "run_id": state["run_id"],
                "error": state.get("error"),
                "overall_bias_score": 0.0,
                "severity": "ERROR",
                "bias_instances": [],
            }
        }

    # ─── Public Interface ──────────────────────────────────────────────────

    def run(
        self,
        text: str,
        doc_type: str = "job_description",
        llm_provider: str | None = None,
    ) -> dict:
        """
        Run the full BiasGuard pipeline on a text document.

        Args:
            text: Input text to analyze
            doc_type: Type of document ('job_description', 'resume', 'interview_transcript')
            llm_provider: Override LLM provider for this run

        Returns:
            Complete BiasGuardReport as dict
        """
        initial_state = _create_initial_state(
            text=text,
            doc_type=doc_type,
            llm_provider=llm_provider,
        )

        logger.info(
            "pipeline_started",
            run_id=initial_state["run_id"],
            doc_type=doc_type,
            text_length=len(text.split()),
        )

        final_state = self._graph.invoke(initial_state)
        return final_state.get("report", {})

    async def arun(
        self,
        text: str,
        doc_type: str = "job_description",
        llm_provider: str | None = None,
    ) -> dict:
        """Async version of run()."""
        initial_state = _create_initial_state(
            text=text,
            doc_type=doc_type,
            llm_provider=llm_provider,
        )
        final_state = await self._graph.ainvoke(initial_state)
        return final_state.get("report", {})


# ─── Utilities ─────────────────────────────────────────────────────────────

def _summarize_by_category(instances: list[dict]) -> dict:
    """Aggregate bias instances by category for summary statistics."""
    summary: dict[str, dict] = {}
    for inst in instances:
        cat = inst.get("category", "UNKNOWN")
        if cat not in summary:
            summary[cat] = {"count": 0, "high": 0, "medium": 0, "low": 0}
        summary[cat]["count"] += 1
        severity = inst.get("severity", "LOW").lower()
        if severity in summary[cat]:
            summary[cat][severity] += 1
    return summary


# Module-level singleton
_orchestrator: BiasGuardOrchestrator | None = None


def get_orchestrator() -> BiasGuardOrchestrator:
    """Return singleton orchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = BiasGuardOrchestrator()
    return _orchestrator
