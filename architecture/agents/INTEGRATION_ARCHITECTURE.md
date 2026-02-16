# Integration Architecture: Composing GAIA's 18 Architectural Frameworks

**Date**: February 7, 2026
**Version**: 1.0
**Status**: Specification
**Purpose**: Show how all 18 GAIA architecture documents compose into a unified, production-ready agent system
**Audience**: Implementers, architects, and contributors building on the GAIA SDK

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture Catalog](#2-architecture-catalog)
3. [Dependency Graph](#3-dependency-graph)
4. [End-to-End Examples](#4-end-to-end-examples)
5. [Integration Patterns](#5-integration-patterns)
6. [Shared Infrastructure](#6-shared-infrastructure)
7. [Composition Guidelines](#7-composition-guidelines)
8. [Migration Strategy](#8-migration-strategy)
9. [Complete Working Example](#9-complete-working-example)
10. [Appendix: Cross-Reference Matrix](#10-appendix-cross-reference-matrix)

---

## 1. Executive Summary

### 1.1 The Vision

GAIA's 18 architecture documents are not isolated specifications. They form a **layered, composable system** where each architecture provides a distinct capability that integrates cleanly with the others. Together, they transform the base `Agent` class from a stateless, single-query processor into a production-grade platform for persistent, observable, secure, self-healing autonomous agents.

### 1.2 The 18 Architectures at a Glance

The architectures fall into four natural tiers:

```
TIER 4: DOMAIN CAPABILITIES (what agents do)
+------------------------------------------------------------------+
|  Computer Use    Email Integration    Multi-Document Synthesis    |
|  (UI automation) (inbox management)   (cross-doc analysis)       |
+------------------------------------------------------------------+

TIER 3: ORCHESTRATION (how work gets coordinated)
+------------------------------------------------------------------+
|  Workflow Orchestration    Task-Centric Interface    Adaptive     |
|  (triggers, pipelines)    (task > chat paradigm)    Prompts      |
|                                                     (state       |
|                                                      machine)    |
+------------------------------------------------------------------+

TIER 2: INTELLIGENCE (how agents learn and grow)
+------------------------------------------------------------------+
|  Persistent Memory    Learning & Adaptation    Dynamic Tools     |
|  (4-tier memory)      (feedback loops)         (runtime tools)   |
|                                                                   |
|  Architecture Manifest    Anthropic Skills Standard              |
|  (project tracking)      (skill interoperability)                |
+------------------------------------------------------------------+

TIER 1: INFRASTRUCTURE (foundational cross-cutting concerns)
+------------------------------------------------------------------+
|  Error Recovery      Observability        Security & Secrets     |
|  (retry, circuit     (tracing, metrics,   (vault, permissions,   |
|   breaker, self-     time-travel debug,    audit, resource       |
|   healing)           cost tracking)        governance)           |
+------------------------------------------------------------------+

TIER 0: EXISTING GAIA SDK
+------------------------------------------------------------------+
|  Agent base class    @tool decorator    LemonadeClient           |
|  AgentConsole        ChatSDK / RAGSDK   MCP Bridge               |
+------------------------------------------------------------------+
```

### 1.3 Key Insight: Infrastructure Wraps Everything

The three Tier 1 architectures -- Error Recovery, Observability, and Security -- are **cross-cutting concerns** that wrap every operation across all other tiers. They do not add domain capability; they make every capability production-ready.

```
Every external call:
  Security.check_permission()
    -> Observability.start_span()
      -> ErrorRecovery.with_retry()
        -> actual_operation()
      -> ErrorRecovery.handle_result()
    -> Observability.end_span()
  -> Security.audit_log()
```

### 1.4 Key Insight: Memory is the Foundation

Persistent Memory (Tier 2) is the single foundational dependency. Every other architecture reads from or writes to the memory system:

- **Manifest** stores project state in memory
- **Learning Loop** writes patterns to semantic memory
- **Workflow** checkpoints execution state via memory
- **Observability** records traces to the universal knowledge DB
- **Security** logs audit events that persist across sessions

### 1.5 How to Read This Document

- **Section 2**: Catalog of all 18 architectures with their core classes
- **Section 3**: Full dependency graph showing what depends on what
- **Section 4**: Four complete end-to-end examples showing 5-8 architectures working together
- **Section 5**: Integration patterns for the three cross-cutting concerns
- **Section 6**: Shared infrastructure (SQLite, async, config, logging)
- **Section 7**: Guidelines for when and how to combine architectures
- **Section 8**: Implementation order and migration strategy
- **Section 9**: A complete, production-ready agent using 6+ architectures

---

## 2. Architecture Catalog

### 2.1 Tier 1 -- Infrastructure (Cross-Cutting)

| # | Architecture | Document | Core Classes | Purpose |
|---|-------------|----------|-------------|---------|
| 1 | **Error Recovery** | `ERROR_RECOVERY_ARCHITECTURE.md` | `RetryEngine`, `CircuitBreaker`, `DegradationManager`, `SelfHealingEngine` | Retry, failover, graceful degradation, self-healing |
| 2 | **Observability** | `OBSERVABILITY_ARCHITECTURE.md` | `TracerProvider`, `MeterProvider`, `SessionRecorder`, `CostCalculator` | Tracing, metrics, time-travel debug, cost tracking |
| 3 | **Security** | `SECURITY_SECRETS_ARCHITECTURE.md` | `SecretManager`, `PermissionEnforcer`, `AuditLogger`, `ResourceGovernor` | Vault, permissions, audit, resource limits |

### 2.2 Tier 2 -- Intelligence (Learning and Knowledge)

| # | Architecture | Document | Core Classes | Purpose |
|---|-------------|----------|-------------|---------|
| 4 | **Persistent Memory** | `PERSISTENT_MEMORY_FRAMEWORK.md` | `WorkingMemory`, `EpisodicMemory`, `SemanticMemory`, `UniversalKnowledgeDB` | 4-tier memory with timestamps and CRUD |
| 5 | **Learning & Adaptation** | `LEARNING_ADAPTATION_FRAMEWORK.md` | `LearningLoop`, `FeedbackCollector`, `OutcomeTracker`, `KnowledgeConsolidator` | Feedback loops, pattern extraction, self-improvement |
| 6 | **Dynamic Tools** | `DYNAMIC_TOOLS_FRAMEWORK.md` | `DynamicToolManager`, `ToolBuilderAgent`, `SkillRegistry` | Runtime tool creation, SKILLS system |
| 7 | **Architecture Manifest** | `ARCHITECTURE_MANIFEST_FRAMEWORK.md` | `ProjectManifest`, `ManifestToolsMixin`, `DependencyTracker` | Real-time project tracking |
| 8 | **Anthropic Skills Standard** | `ANTHROPIC_SKILLS_STANDARD.md` | `SkillSync`, `MarkdownSkillParser` | Bidirectional skill sync with .claude/skills/ |

### 2.3 Tier 3 -- Orchestration (Coordination and Flow)

| # | Architecture | Document | Core Classes | Purpose |
|---|-------------|----------|-------------|---------|
| 9 | **Workflow Orchestration** | `WORKFLOW_ORCHESTRATION_ARCHITECTURE.md` | `WorkflowEngine`, `TriggerSystem`, `StepExecutor`, `StateManager` | Scheduled pipelines, triggers, branching |
| 10 | **Task-Centric Interface** | `TASK_CENTRIC_INTERFACE_PARADIGM.md` | `Task`, `TaskQueue`, `TaskManager` | Tasks as first-class citizens |
| 11 | **Adaptive Prompts** | `ADAPTIVE_PROMPTS_FRAMEWORK.md` | `AgentState`, `StateMachine`, `PromptComposer` | State machine, dynamic prompts |

### 2.4 Tier 4 -- Domain Capabilities

| # | Architecture | Document | Core Classes | Purpose |
|---|-------------|----------|-------------|---------|
| 12 | **Computer Use** | `COMPUTER_USE_ARCHITECTURE.md` | `ComputerUseController`, `ScreenCaptureEngine`, `MouseController`, `BrowserAutomation` | UI automation, desktop control |
| 13 | **Email Integration** | `EMAIL_INTEGRATION_ARCHITECTURE.md` | `EmailAgent`, `GmailProvider`, `EmailClassifier`, `CalendarBridge` | Inbox triage, auto-response, scheduling |
| 14 | **Multi-Document Synthesis** | `MULTI_DOCUMENT_SYNTHESIS_ARCHITECTURE.md` | `DocumentStore`, `EntityResolver`, `ContradictionDetector`, `CitationTracker` | Cross-document analysis, citations |

### 2.5 Supporting Documents (Non-Architecture)

| # | Document | Purpose |
|---|----------|---------|
| 15 | `TUI_DESIGN_SPECIFICATION.md` | Terminal UI (Textual) |
| 16 | `AGENT_DASHBOARD_DESIGN.md` | Web dashboard (React + FastAPI) |
| 17 | `REQUIREMENTS_GATHERING_STATE.md` | Initial clarification state |
| 18 | `MISSING_ARCHITECTURES_ASSESSMENT.md` | Gap analysis |

---

## 3. Dependency Graph

### 3.1 Complete Dependency Diagram

```
                    DEPENDENCY GRAPH (arrows = "depends on")
                    =========================================

                         ┌──────────────┐
                         │  GAIA SDK    │
                         │  (Agent,     │
                         │   @tool,     │
                         │   ChatSDK)   │
                         └──────┬───────┘
                                │
               ┌────────────────┼────────────────┐
               │                │                │
               v                v                v
    ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
    │   Error      │ │ Observability│ │  Security    │
    │   Recovery   │ │              │ │  & Secrets   │
    │              │ │ (traces,     │ │              │
    │ (retry,      │ │  metrics,    │ │ (vault,      │
    │  circuit     │ │  cost)       │ │  perms,      │
    │  breaker)    │ │              │ │  audit)      │
    └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
           │                │                │
           └────────────────┼────────────────┘
                            │
                            v
                  ┌──────────────────┐
                  │   Persistent     │
                  │   Memory         │
                  │   (4-tier)       │
                  └────────┬─────────┘
                           │
          ┌────────────────┼────────────────┬──────────────┐
          │                │                │              │
          v                v                v              v
   ┌────────────┐  ┌────────────┐  ┌────────────┐ ┌────────────┐
   │ Learning & │  │ Manifest   │  │ Adaptive   │ │ Anthropic  │
   │ Adaptation │  │ Framework  │  │ Prompts    │ │ Skills     │
   │            │  │            │  │ (state     │ │ Standard   │
   │ (feedback  │  │ (project   │  │  machine)  │ │            │
   │  loops)    │  │  tracking) │  │            │ │            │
   └─────┬──────┘  └─────┬──────┘  └─────┬──────┘ └─────┬──────┘
         │               │               │              │
         │        ┌──────┘               │              │
         v        v                      v              v
   ┌────────────────┐            ┌─────────────┐ ┌───────────┐
   │ Dynamic Tools  │            │ Task-Centric│ │ (merged   │
   │ & Skills       │            │ Interface   │ │  into     │
   │                │            │             │ │  Dynamic  │
   │ (runtime tool  │            │ (task queue │ │  Tools)   │
   │  creation)     │            │  paradigm)  │ └───────────┘
   └────────┬───────┘            └──────┬──────┘
            │                           │
            └─────────────┬─────────────┘
                          │
                          v
               ┌──────────────────┐
               │    Workflow      │
               │    Orchestration │
               │                  │
               │ (triggers,       │
               │  pipelines,      │
               │  scheduling)     │
               └────────┬─────────┘
                        │
       ┌────────────────┼────────────────┐
       │                │                │
       v                v                v
┌────────────┐  ┌────────────┐  ┌────────────────┐
│ Computer   │  │   Email    │  │ Multi-Document │
│ Use        │  │ Integration│  │ Synthesis      │
│            │  │            │  │                │
│ (UI auto)  │  │ (inbox,    │  │ (cross-doc     │
│            │  │  calendar) │  │  analysis)     │
└────────────┘  └────────────┘  └────────────────┘
```

### 3.2 Dependency Rules

**Hard Dependencies** (must be present):

| Architecture | Hard Dependencies |
|-------------|-------------------|
| Error Recovery | GAIA SDK (Agent base) |
| Observability | GAIA SDK (Agent base) |
| Security | GAIA SDK (Agent base) |
| Persistent Memory | GAIA SDK (Agent base) |
| Learning & Adaptation | Persistent Memory |
| Dynamic Tools | Persistent Memory, Learning & Adaptation |
| Manifest | Persistent Memory |
| Adaptive Prompts | Persistent Memory |
| Task-Centric Interface | Adaptive Prompts |
| Workflow Orchestration | Task-Centric Interface |
| Computer Use | GAIA SDK (VLM client) |
| Email Integration | GAIA SDK (Agent base) |
| Multi-Document Synthesis | GAIA SDK (RAGSDK) |

**Soft Dependencies** (enhances but not required):

| Architecture | Soft Dependencies |
|-------------|-------------------|
| Every architecture | Error Recovery, Observability, Security |
| Workflow Orchestration | Email (for notifications), Manifest (for state) |
| Computer Use | Error Recovery (for UI retry), Security (for action validation) |
| Email Integration | Security (for credential vault), Workflow (for scheduling) |
| Multi-Document Synthesis | Persistent Memory (for caching), Learning (for pattern reuse) |

### 3.3 Circular Dependency Prevention

Two potential circular dependencies exist and must be resolved:

**1. Observability <-> Error Recovery**

Observability traces errors. Error Recovery retries observed operations. Resolution: Observability instruments Error Recovery, but Error Recovery does NOT depend on Observability. If Observability fails, Error Recovery continues without tracing.

```python
# CORRECT: Observability wraps Error Recovery
with tracer.start_span("retry_operation"):
    result = retry_engine.execute_with_retry(operation)

# INCORRECT: Error Recovery depends on Observability
result = retry_engine.execute_with_retry(
    operation,
    tracer=tracer  # DO NOT pass tracer into retry engine
)
```

**2. Security <-> Observability**

Security audit logs are observable. Observability traces security checks. Resolution: Security writes audit logs to its own store. Observability can read audit logs but Security never calls Observability.

```python
# Security writes its own audit log (no Observability dependency)
class AuditLogger:
    def log(self, event: AuditEvent) -> None:
        self._db.insert(event)  # Direct SQLite write

# Observability can instrument security checks
with tracer.start_span("permission_check"):
    allowed = permission_enforcer.check(agent_id, permission)
```

---

## 4. End-to-End Examples

### 4.1 Example 1: "Triage My Emails"

**Architectures involved**: Computer Use + Email Integration + Workflow Orchestration + Observability + Security + Error Recovery (6 architectures)

**User command**: `gaia workflow run email-triage`

#### Step-by-Step Flow

```
Step 1: WORKFLOW TRIGGER
========================
WorkflowEngine loads "email-triage" workflow definition.
CronTrigger fires at 8:00 AM (or ManualTrigger from CLI).

    workflow = WorkflowEngine.load("email-triage")
    workflow.trigger(TriggerType.MANUAL)


Step 2: SECURITY -- Credential Retrieval
==========================================
Before accessing email, Security provides scoped credentials.

    with PermissionContext(agent="EmailAgent", scope=["email:read", "email:send"]):
        gmail_creds = SecretManager.get("gmail_oauth_token")
        # AuditLogger: "EmailAgent accessed gmail_oauth_token"


Step 3: ERROR RECOVERY -- Fetch Emails with Retry
===================================================
Email fetch wraps in retry with circuit breaker.

    @with_retry(strategy=ExponentialBackoff(base=1.0, max_retries=5))
    @circuit_breaker(service="gmail_api", threshold=3)
    async def fetch_emails():
        provider = GmailProvider(credentials=gmail_creds)
        return await provider.fetch_unread(limit=50)

    emails = await fetch_emails()
    # If Gmail API is down: retry 5 times with backoff
    # If still failing: circuit breaker opens, fallback to IMAP
    # If IMAP fails: degrade to cached emails from last sync


Step 4: OBSERVABILITY -- Trace the Classification
===================================================
Each email classification is traced for debugging and cost tracking.

    with tracer.start_span("email_triage", kind=SpanKind.AGENT) as span:
        span.set_attribute("email_count", len(emails))

        for email in emails:
            with tracer.start_span("classify_email") as child:
                classification = classifier.classify(email)
                child.set_attribute("category", classification.category)
                child.set_attribute("urgency", classification.urgency)
                meter.record("emails_classified", 1, {"category": classification.category})


Step 5: EMAIL INTEGRATION -- Classify and Act
===============================================
EmailClassifier uses LLM to classify each email.

    classifier = EmailClassifier(llm_client=lemonade_client)

    for email in emails:
        result = classifier.classify(email)
        # result.category: "urgent" | "action_required" | "informational" | "spam"
        # result.urgency: 0.0 - 1.0
        # result.suggested_action: "reply" | "archive" | "forward" | "flag"

    urgent = [e for e in results if e.urgency > 0.8]
    action_required = [e for e in results if e.category == "action_required"]


Step 6: COMPUTER USE -- Handle UI-Only Actions (Optional)
==========================================================
If an email requires interaction with a web app that has no API:

    controller = ComputerUseController()

    # Open the internal HR system (no API available)
    await controller.execute_action_sequence([
        BrowserAction(action="navigate", url="https://hr.internal.com"),
        BrowserAction(action="click", selector="#timeoff-requests"),
        BrowserAction(action="screenshot"),  # Verify we are on correct page
        # VLM analyzes screenshot to confirm navigation succeeded
    ])


Step 7: WORKFLOW -- Conditional Branching
==========================================
Workflow engine handles branching based on classification results.

    # If urgent emails found:
    if urgent:
        workflow.branch("notify_user")
        # Step 7a: Send desktop notification
        # Step 7b: Draft response summaries
    else:
        workflow.branch("archive_and_summarize")
        # Step 7c: Archive newsletters
        # Step 7d: Generate daily summary


Step 8: OBSERVABILITY -- Record Costs and Metrics
===================================================
After completion, record full trace and cost breakdown.

    cost_tracker.record_query_cost(
        agent="EmailAgent",
        input_tokens=total_input_tokens,
        output_tokens=total_output_tokens,
        model="Qwen3-Coder-30B",
    )
    # Trace available: gaia observe traces --last 1
    # Cost report: gaia observe costs --today
```

#### Sequence Diagram

```
User          Workflow      Security      ErrorRecov    Email         Observ.
 |               |             |              |            |             |
 |--run triage-->|             |              |            |             |
 |               |--get cred-->|              |            |             |
 |               |<--token-----|              |            |             |
 |               |             |--audit log-->|            |             |
 |               |                            |            |             |
 |               |----------with_retry------->|            |             |
 |               |                            |--fetch---->|             |
 |               |                            |<--emails---|             |
 |               |<--------emails-------------|            |             |
 |               |                                         |             |
 |               |---start_span("classify")--------------->|             |
 |               |                                         |             |
 |               |--classify(email)----------------------->|             |
 |               |<--result--------------------------------|             |
 |               |                                         |             |
 |               |---end_span, record_cost---------------->|             |
 |               |                                                       |
 |<--summary-----|                                                       |
```

---

### 4.2 Example 2: "Analyze These 10 Research Papers"

**Architectures involved**: Multi-Document Synthesis + Persistent Memory + Learning & Adaptation + Observability + Error Recovery (5 architectures)

**User command**: `gaia chat --docs paper1.pdf paper2.pdf ... paper10.pdf "Compare findings on transformer efficiency"`

#### Step-by-Step Flow

```
Step 1: MULTI-DOC -- Parallel Ingestion
=========================================
DocumentStore ingests all 10 papers in parallel.

    doc_store = DocumentStore(db_path="~/.gaia/multidoc.db")
    collection = doc_store.create_collection("transformer_efficiency_review")

    for pdf_path in pdf_paths:
        doc = await ingestion_pipeline.ingest(
            path=pdf_path,
            extract_tables=True,
            extract_citations=True,
            preserve_structure=True,
        )
        collection.add_document(doc)
    # Result: 10 documents indexed with chunks, tables, citations, metadata


Step 2: OBSERVABILITY -- Instrument the Analysis
==================================================
Trace the entire multi-document analysis for debugging.

    with tracer.start_span("multi_doc_analysis", kind=SpanKind.AGENT) as span:
        span.set_attribute("document_count", 10)
        span.set_attribute("query", user_query)


Step 3: MULTI-DOC -- Cross-Document Entity Resolution
=======================================================
Link entities that appear across multiple papers.

    resolver = EntityResolver(llm_client=lemonade_client)
    entity_graph = await resolver.resolve_across_documents(
        collection=collection,
        entity_types=["method", "metric", "dataset", "finding"],
    )
    # Result: "Flash Attention" in paper 1 linked to "FlashAttention-2" in paper 5
    # Result: "FLOPS" in paper 3 linked to "FLOPs" in paper 7 (normalized)


Step 4: MULTI-DOC -- Contradiction Detection
==============================================
Find where papers agree or disagree.

    detector = ContradictionDetector(llm_client=lemonade_client)
    contradictions = await detector.detect(
        collection=collection,
        entity_graph=entity_graph,
    )
    # Result: Paper 2 claims "linear attention scales better"
    #         Paper 6 claims "quadratic attention with Flash is faster in practice"
    #         Confidence: 0.85, Pages: [2:14, 6:8]


Step 5: ERROR RECOVERY -- Handle LLM Failures During Analysis
===============================================================
Cross-document analysis requires many LLM calls. Retry on failure.

    @with_retry(strategy=ExponentialBackoff(base=0.5, max_retries=3))
    async def synthesize_findings(collection, query):
        synthesis_engine = SynthesisEngine(llm_client=lemonade_client)
        return await synthesis_engine.synthesize(
            collection=collection,
            query=query,
            citation_mode=CitationMode.PAGE_LEVEL,
        )

    report = await synthesize_findings(collection, user_query)
    # If LLM times out during synthesis: retry with smaller chunk size
    # If retries exhausted: degrade to per-document summaries (no cross-doc)


Step 6: PERSISTENT MEMORY -- Cache and Learn
==============================================
Store the analysis results for future retrieval.

    # Episodic memory: store this analysis session
    episodic_memory.store_session(
        session_type="multi_doc_analysis",
        summary=report.executive_summary,
        metadata={
            "documents": [d.title for d in collection.documents],
            "query": user_query,
            "contradictions_found": len(contradictions),
        },
    )

    # Semantic memory: extract durable knowledge
    for finding in report.key_findings:
        semantic_memory.store_knowledge(
            fact=finding.statement,
            source=finding.citations,
            confidence=finding.confidence,
            domain="transformer_efficiency",
        )


Step 7: LEARNING -- Extract Patterns for Future Use
=====================================================
LearningLoop identifies reusable analysis patterns.

    learning_loop.record_outcome(
        task="multi_doc_analysis",
        input_description="10 papers on transformer efficiency",
        output_quality=0.92,  # User feedback or automated score
        strategy_used="entity_resolution_then_contradiction_detection",
    )
    # Future runs: if similar query detected, suggest same strategy
    # Knowledge: "For comparative analysis, entity resolution before
    #             contradiction detection yields 15% better results"


Step 8: MULTI-DOC -- Generate Cited Report
============================================
Final output with page-level citations.

    report_output = report.render(format="markdown")
    # Output includes:
    # - Executive summary
    # - Key findings with [Paper 3, p.14] style citations
    # - Contradiction matrix
    # - Table comparisons (normalized across papers)
    # - Bibliography
```

#### Data Flow Diagram

```
  [10 PDFs]
      |
      v
  ┌─────────────────────┐      ┌─────────────────┐
  │  Ingestion Pipeline  │----->│  Document Store  │
  │  (parse, extract,    │      │  (chunks, tables │
  │   embed)             │      │   citations)     │
  └─────────────────────┘      └────────┬──────────┘
                                        |
                          ┌─────────────┼─────────────┐
                          |             |             |
                          v             v             v
                   ┌───────────┐ ┌───────────┐ ┌──────────┐
                   │ Entity    │ │Contradict.│ │Synthesis │
                   │ Resolver  │ │ Detector  │ │ Engine   │
                   └─────┬─────┘ └─────┬─────┘ └────┬─────┘
                         |             |             |
                         v             v             v
                   ┌──────────────────────────────────────┐
                   │        Citation Tracker               │
                   │  (page-level, paragraph-level refs)   │
                   └──────────────────┬───────────────────┘
                                      |
                         ┌────────────┼────────────┐
                         |            |            |
                         v            v            v
                   ┌──────────┐ ┌──────────┐ ┌──────────┐
                   │ Episodic │ │ Semantic │ │ Learning │
                   │ Memory   │ │ Memory   │ │ Loop     │
                   │ (session)│ │ (facts)  │ │(patterns)│
                   └──────────┘ └──────────┘ └──────────┘
```

---

### 4.3 Example 3: "Build and Test a Web App"

**Architectures involved**: Adaptive Prompts (State Machine) + Architecture Manifest + Error Recovery + Observability + Persistent Memory + Task-Centric Interface + Learning & Adaptation (7 architectures)

**User command**: `gaia code "Build a FastAPI todo app with auth, tests, and Docker deployment"`

#### Step-by-Step Flow

```
Step 1: TASK-CENTRIC -- Create Task Hierarchy
===============================================
TaskManager creates a structured task tree, not a chat message.

    task = TaskManager.create_task(
        description="Build a FastAPI todo app with auth, tests, Docker",
        agent="CodeAgent",
        sub_tasks=[
            Task("Set up project structure"),
            Task("Implement user authentication"),
            Task("Implement todo CRUD endpoints"),
            Task("Write unit tests"),
            Task("Write integration tests"),
            Task("Create Dockerfile and docker-compose"),
        ],
    )
    task_queue.enqueue(task)
    # Task ID: task_abc123
    # Status: PENDING
    # Progress: 0/6 sub-tasks


Step 2: ADAPTIVE PROMPTS -- Enter Planning State
==================================================
State machine transitions to PLANNING with planning-specific prompt and tools.

    state_machine.enter_state("planning", context={
        "task": task,
        "available_tools": ["analyze_requirements", "list_files", "search_code"],
        # Note: write_file, run_command NOT available in planning state
    })

    # Planning state system prompt:
    # "You are in PLANNING mode. Analyze the requirements and create a
    #  detailed implementation plan. Do NOT write code yet. Focus on
    #  architecture decisions, file structure, and dependency choices."

    plan = agent.execute_in_state("planning")
    # Output: Detailed plan with file list, architecture decisions


Step 3: MANIFEST -- Initialize Project Manifest
=================================================
ManifestToolsMixin creates a manifest from the plan.

    manifest = ProjectManifest.create(
        project_id="todo-app-abc123",
        task_description=task.description,
        architecture={
            "pattern": "monolith",
            "framework": "FastAPI",
            "database": "SQLite",
            "auth": "JWT",
        },
        planned_files=[
            FileEntry(path="app/main.py", purpose="FastAPI app entry point"),
            FileEntry(path="app/models.py", purpose="SQLAlchemy models"),
            FileEntry(path="app/auth.py", purpose="JWT authentication"),
            FileEntry(path="app/routes/todos.py", purpose="Todo CRUD routes"),
            FileEntry(path="tests/test_auth.py", purpose="Auth unit tests"),
            FileEntry(path="tests/test_todos.py", purpose="Todo unit tests"),
            FileEntry(path="Dockerfile", purpose="Container image"),
            FileEntry(path="docker-compose.yml", purpose="Multi-service setup"),
        ],
    )


Step 4: ADAPTIVE PROMPTS -- Enter Implementation State
========================================================
State machine transitions with implementation-specific tools.

    state_machine.enter_state("implementation", context={
        "task": task,
        "manifest": manifest,
        "available_tools": ["write_file", "read_file", "run_command",
                            "update_manifest", "run_tests"],
        # Note: planning tools removed, code tools added
    })

    # Implementation state system prompt:
    # "You are in IMPLEMENTATION mode. Write production-quality code following
    #  the plan. After each file, update the manifest. Follow existing patterns
    #  in the manifest. Run tests after implementing each module."


Step 5: ERROR RECOVERY -- Handle Build Failures
=================================================
Each file write and test run is wrapped in error recovery.

    @with_retry(strategy=FixedBackoff(delay=2.0, max_retries=2))
    async def write_and_verify(file_path, content):
        write_file(file_path, content)
        # Run syntax check
        result = run_command(f"python -m py_compile {file_path}")
        if result.returncode != 0:
            raise TransientError(f"Syntax error in {file_path}: {result.stderr}")
        return result

    # If syntax error: agent gets error, fixes code, retries
    # After 2 retries: transition to DEBUG state


Step 6: ADAPTIVE PROMPTS -- Auto-Transition to Debug State
============================================================
On test failure, state machine enters nested debug state.

    # Tests fail for auth module
    test_result = run_command("pytest tests/test_auth.py -v")
    if test_result.returncode != 0:
        state_machine.enter_state("debug", context={
            "error": test_result.stderr,
            "failing_tests": parse_test_failures(test_result),
            "available_tools": ["read_file", "write_file", "run_tests",
                                "inspect_variable", "add_breakpoint"],
        })
        # Debug state prompt: "Analyze the test failure. Form a hypothesis.
        #                      Fix the issue. Re-run tests to verify."

        # After fix: return to implementation state
        state_machine.return_to_state("implementation")


Step 7: MANIFEST -- Track Progress in Real-Time
=================================================
Manifest updates automatically after each operation.

    # After implementing auth module:
    manifest.update_file("app/auth.py", status="complete",
                         quality_check={"syntax": "pass", "tests": "pass"})
    manifest.update_progress()
    # Progress: 2/8 files complete, 25% overall


Step 8: OBSERVABILITY -- Full Execution Trace
===============================================
Every state transition, tool call, and LLM invocation is traced.

    # Trace output (gaia observe traces --last 1):
    # Trace ID: trace_xyz789
    #   |-- CodeAgent.process_query (total: 342s)
    #       |-- State: planning (18s, 2 LLM calls)
    #       |-- State: implementation (280s, 24 LLM calls)
    #       |   |-- write_file: app/main.py (0.2s)
    #       |   |-- write_file: app/models.py (0.3s)
    #       |   |-- write_file: app/auth.py (0.2s)
    #       |   |-- run_tests: test_auth.py (4.1s, FAILED)
    #       |   |-- State: debug (45s, 3 LLM calls)
    #       |   |   |-- read_file: app/auth.py (0.1s)
    #       |   |   |-- write_file: app/auth.py (0.2s, fix applied)
    #       |   |   |-- run_tests: test_auth.py (3.8s, PASSED)
    #       |   |-- write_file: app/routes/todos.py (0.3s)
    #       |   |-- ...
    #       |-- State: review (44s, 2 LLM calls)
    #
    # Cost: $0.0287 (14,200 input + 8,900 output tokens)


Step 9: PERSISTENT MEMORY -- Store for Future Sessions
========================================================
Session results persisted for learning.

    episodic_memory.store_session(
        session_type="code_generation",
        summary="Built FastAPI todo app with JWT auth, 8 files, all tests passing",
        artifacts=manifest.to_dict(),
    )

    # Semantic memory: learned patterns
    semantic_memory.store_knowledge(
        fact="FastAPI + SQLAlchemy + JWT auth pattern: use python-jose for JWT tokens",
        confidence=0.95,
        source="task_abc123",
    )


Step 10: LEARNING -- Record Outcome
=====================================

    learning_loop.record_outcome(
        task_id="task_abc123",
        success=True,
        metrics={
            "total_time": 342,
            "files_created": 8,
            "test_failures_recovered": 1,
            "states_visited": ["planning", "implementation", "debug", "review"],
        },
    )
```

---

### 4.4 Example 4: "Schedule Weekly Report Generation"

**Architectures involved**: Workflow Orchestration + Email Integration + Task-Centric Interface + Persistent Memory + Error Recovery + Security + Observability (7 architectures)

**User command**: `gaia workflow create weekly-report --schedule "0 17 * * 5"`

#### Workflow Definition (YAML)

```yaml
# ~/.gaia/workflows/weekly-report.yaml
name: weekly-report
description: Generate and distribute weekly team status report
schedule: "0 17 * * 5"  # Every Friday at 5 PM

triggers:
  - type: cron
    expression: "0 17 * * 5"
  - type: manual
    allowed_users: ["team-lead"]

steps:
  - id: fetch_jira_tickets
    type: agent_step
    agent: JiraAgent
    action: "List all tickets completed this week in project GAIA"
    output: completed_tickets
    retry:
      strategy: exponential
      max_retries: 3

  - id: fetch_git_commits
    type: function_step
    function: git_log_this_week
    output: weekly_commits

  - id: summarize
    type: agent_step
    agent: ChatAgent
    action: |
      Summarize these completed tickets and commits into a weekly
      status report. Include: accomplishments, blockers, next week plans.
      Tickets: {{ completed_tickets }}
      Commits: {{ weekly_commits }}
    output: report_text

  - id: check_urgency
    type: conditional
    condition: "{{ completed_tickets | length }} < 3"
    if_true: flag_low_productivity
    if_false: send_report

  - id: flag_low_productivity
    type: agent_step
    agent: ChatAgent
    action: "Add a note about low ticket completion this week"
    output: flagged_report
    next: send_report

  - id: send_report
    type: agent_step
    agent: EmailAgent
    action: |
      Send the weekly report to team@company.com.
      Subject: "Weekly Status Report - {{ current_date }}"
      Body: {{ report_text or flagged_report }}

  - id: archive
    type: function_step
    function: archive_report
    args:
      report: "{{ report_text or flagged_report }}"
      date: "{{ current_date }}"

security:
  required_permissions:
    - "jira:read"
    - "email:send"
    - "git:read"
  credential_scope:
    - "jira_api_token"
    - "gmail_oauth_token"

observability:
  trace_all_steps: true
  alert_on_failure: true
  cost_budget: 0.50  # USD per run
```

#### Execution Flow

```
Step 1: WORKFLOW -- Cron Trigger Fires
========================================
Friday 5:00 PM. CronTrigger fires.

    scheduler = WorkflowScheduler()
    workflow = scheduler.get_due_workflows()
    # Returns: [weekly-report]

    runtime = WorkflowRuntime()
    run_id = runtime.start(workflow="weekly-report", trigger="cron")
    # run_id: run_20260207_170000_abc


Step 2: SECURITY -- Verify Permissions and Retrieve Credentials
================================================================

    with PermissionContext(workflow="weekly-report"):
        # Check: does this workflow have jira:read, email:send, git:read?
        enforcer.check_permissions(["jira:read", "email:send", "git:read"])

        # Retrieve scoped credentials
        jira_token = SecretManager.get("jira_api_token", scope="weekly-report")
        gmail_creds = SecretManager.get("gmail_oauth_token", scope="weekly-report")

        # Audit: "workflow:weekly-report accessed jira_api_token at 17:00:01"


Step 3: WORKFLOW + ERROR RECOVERY -- Execute Steps with Retry
===============================================================

    # Step: fetch_jira_tickets
    with tracer.start_span("fetch_jira_tickets"):
        @with_retry(strategy=ExponentialBackoff(max_retries=3))
        async def fetch_tickets():
            jira_agent = JiraAgent(token=jira_token)
            return await jira_agent.process_query(
                "List all tickets completed this week in project GAIA"
            )
        completed_tickets = await fetch_tickets()

    # Step: fetch_git_commits (parallel with above if independent)
    with tracer.start_span("fetch_git_commits"):
        weekly_commits = git_log_this_week()

    # Step: summarize
    with tracer.start_span("summarize_report"):
        chat_agent = ChatAgent()
        report_text = await chat_agent.process_query(
            f"Summarize: {completed_tickets}\n{weekly_commits}"
        )

    # Step: conditional check
    if len(completed_tickets) < 3:
        report_text = await chat_agent.process_query(
            "Add low productivity note to: " + report_text
        )

    # Step: send_report
    with tracer.start_span("send_email"):
        @with_retry(strategy=ExponentialBackoff(max_retries=3))
        async def send():
            email_agent = EmailAgent(credentials=gmail_creds)
            return await email_agent.send_email(
                to="team@company.com",
                subject=f"Weekly Status Report - {current_date}",
                body=report_text,
            )
        await send()


Step 4: PERSISTENT MEMORY -- Archive Run Results
==================================================

    episodic_memory.store_session(
        session_type="workflow_run",
        summary=f"Weekly report generated and sent. {len(completed_tickets)} tickets.",
        metadata={
            "workflow": "weekly-report",
            "run_id": run_id,
            "tickets_count": len(completed_tickets),
            "report_sent_to": "team@company.com",
        },
    )

    # Universal Knowledge DB: store full report for historical analysis
    universal_db.store_artifact(
        type="weekly_report",
        content=report_text,
        timestamp=datetime.now(),
        tags=["weekly-report", "automated"],
    )


Step 5: OBSERVABILITY -- Record Trace and Cost
================================================

    # Full trace available via CLI:
    # $ gaia observe traces --workflow weekly-report --last 1
    #
    # Trace ID: trace_wr_20260207
    #   |-- workflow:weekly-report (total: 28s)
    #       |-- security:check_permissions (0.1s)
    #       |-- step:fetch_jira_tickets (8.2s)
    #       |   |-- JiraAgent.process_query (7.9s)
    #       |   |   |-- LLM: plan (1.1s, 300 tokens)
    #       |   |   |-- Tool: jira_search (6.5s)
    #       |-- step:fetch_git_commits (1.2s)
    #       |-- step:summarize_report (12.4s)
    #       |   |-- LLM: summarize (12.1s, 2400 tokens in, 800 out)
    #       |-- step:send_email (5.8s)
    #       |   |-- GmailProvider.send (5.5s)
    #       |-- step:archive (0.3s)
    #
    # Cost: $0.0089 (2700 input + 800 output tokens)
    # Budget: $0.0089 / $0.50 (1.8% of budget)
```

#### State Transition Diagram

```
  PENDING ──trigger──> RUNNING
                          |
            ┌─────────────┼─────────────┐
            |             |             |
            v             v             v
     [fetch_jira]   [fetch_git]   (parallel)
            |             |
            └──────┬──────┘
                   |
                   v
            [summarize]
                   |
                   v
            [conditional]
              /        \
          true          false
            |             |
            v             v
     [flag_low]     (skip)
            |             |
            └──────┬──────┘
                   |
                   v
            [send_report]
                   |
                   v
             [archive]
                   |
                   v
             COMPLETED
```

---

## 5. Integration Patterns

### 5.1 How Error Recovery Wraps All External Calls

Error Recovery is the innermost wrapper around any operation that can fail. It provides three mechanisms that compose together.

#### Pattern: Retry + Circuit Breaker + Degradation Chain

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Error recovery integration pattern.
All external calls in GAIA should follow this pattern.
"""

from gaia.recovery import (
    CircuitBreaker,
    DegradationChain,
    ErrorClassifier,
    RetryEngine,
    with_retry,
    circuit_breaker,
)


# === Pattern 1: Decorator-Based (Preferred for Simple Cases) ===

@with_retry(
    strategy="exponential",
    base_delay=1.0,
    max_retries=3,
    retryable_errors=[ConnectionError, TimeoutError],
)
@circuit_breaker(service="lemonade_server", failure_threshold=5, recovery_timeout=30)
def call_llm(prompt: str) -> str:
    """Call LLM with automatic retry and circuit breaker."""
    return lemonade_client.complete(prompt)


# === Pattern 2: Context Manager (For Complex Error Handling) ===

async def process_with_fallback(query: str) -> str:
    """Process query with full degradation chain."""
    chain = DegradationChain(
        name="llm_completion",
        levels=[
            # Level 0: Full quality (local LLM on NPU)
            DegradationLevel(
                name="local_npu",
                operation=lambda: lemonade_client.complete(query),
                quality=1.0,
            ),
            # Level 1: Reduced quality (smaller local model)
            DegradationLevel(
                name="local_small_model",
                operation=lambda: lemonade_client.complete(query, model="Qwen3-0.6B"),
                quality=0.7,
            ),
            # Level 2: Cloud fallback
            DegradationLevel(
                name="cloud_fallback",
                operation=lambda: claude_client.complete(query),
                quality=0.9,
            ),
            # Level 3: Cached response
            DegradationLevel(
                name="cached",
                operation=lambda: cache.get_similar(query),
                quality=0.3,
            ),
        ],
    )
    result = await chain.execute()
    return result.value


# === Pattern 3: Error Classification ===

classifier = ErrorClassifier()

try:
    result = some_operation()
except Exception as e:
    error_type = classifier.classify(e)
    # error_type: TRANSIENT | PERMANENT | UNKNOWN

    if error_type == ErrorType.TRANSIENT:
        # Retry with backoff
        result = retry_engine.execute(some_operation)
    elif error_type == ErrorType.PERMANENT:
        # Skip to fallback immediately
        result = degradation_chain.execute()
    else:
        # Dead letter queue for investigation
        dead_letter_queue.enqueue(e, context={"operation": "some_operation"})
        raise
```

#### Where Error Recovery Applies

| Component | Wrapped Operations | Retry Strategy |
|-----------|-------------------|----------------|
| LLM Client | `complete()`, `stream()` | Exponential backoff, circuit breaker, cloud fallback |
| Email Provider | `fetch()`, `send()` | Exponential backoff, provider failover (Gmail -> IMAP) |
| Jira Client | API calls | Linear backoff, rate-limit aware |
| Computer Use | Browser actions | Fixed retry (UI can be flaky), screenshot verification |
| RAG Pipeline | Embedding, search | Retry with smaller batch, fallback to keyword search |
| Workflow Steps | Any step | Configurable per-step via YAML |
| MCP Bridge | Tool calls | Circuit breaker per MCP server |

### 5.2 How Observability Instruments All Components

Observability wraps the outermost layer. It sees everything but never blocks execution.

#### Pattern: Auto-Instrumentation via Mixins

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Observability integration pattern.
Auto-instruments Agent subclasses without code changes.
"""

from gaia.observe import TracerProvider, MeterProvider, tracer, meter


class ObservableMixin:
    """
    Mixin that auto-instruments any Agent subclass.

    Add to any agent class to get automatic tracing and metrics.
    Zero code changes required in the agent itself.
    """

    def process_query(self, query: str, **kwargs) -> str:
        """Wrap process_query with tracing."""
        with tracer.start_span(
            name=f"{self.__class__.__name__}.process_query",
            kind=SpanKind.AGENT,
            attributes={
                "agent.class": self.__class__.__name__,
                "agent.query_length": len(query),
                "agent.model": getattr(self, "model_id", "unknown"),
            },
        ) as span:
            try:
                result = super().process_query(query, **kwargs)
                span.set_attribute("agent.success", True)
                span.set_attribute("agent.result_length", len(str(result)))
                meter.record("agent_queries_total", 1, {"agent": self.__class__.__name__})
                return result
            except Exception as e:
                span.set_attribute("agent.success", False)
                span.set_attribute("agent.error", str(e))
                meter.record("agent_errors_total", 1, {"agent": self.__class__.__name__})
                raise

    def _call_tool(self, tool_name: str, tool_args: dict) -> str:
        """Wrap tool calls with tracing."""
        with tracer.start_span(
            name=f"tool:{tool_name}",
            kind=SpanKind.TOOL,
            attributes={"tool.name": tool_name},
        ) as span:
            start = time.monotonic()
            try:
                result = super()._call_tool(tool_name, tool_args)
                span.set_attribute("tool.success", True)
                return result
            except Exception as e:
                span.set_attribute("tool.success", False)
                span.set_attribute("tool.error", str(e))
                raise
            finally:
                duration = time.monotonic() - start
                meter.record_histogram(
                    "tool_duration_seconds",
                    duration,
                    {"tool": tool_name},
                )


# Usage: just add the mixin to any agent
class ObservableCodeAgent(ObservableMixin, CodeAgent):
    """CodeAgent with automatic observability."""
    pass


# Or apply globally via monkey-patching (for production deployments)
def enable_observability():
    """Enable observability for all agents without code changes."""
    original_process_query = Agent.process_query
    original_call_tool = Agent._call_tool

    def instrumented_process_query(self, query, **kwargs):
        with tracer.start_span(f"{self.__class__.__name__}.process_query"):
            return original_process_query(self, query, **kwargs)

    def instrumented_call_tool(self, tool_name, tool_args):
        with tracer.start_span(f"tool:{tool_name}"):
            return original_call_tool(self, tool_name, tool_args)

    Agent.process_query = instrumented_process_query
    Agent._call_tool = instrumented_call_tool
```

#### Observability Coverage Map

```
┌─────────────────────────────────────────────────────────────────┐
│                   What Gets Traced                               │
│                                                                   │
│  Agent Layer:                                                    │
│    [x] process_query() start/end/duration                        │
│    [x] State transitions (PLANNING -> EXECUTING -> ...)          │
│    [x] Error recovery attempts                                   │
│                                                                   │
│  LLM Layer:                                                      │
│    [x] Every LLM call (prompt tokens, completion tokens, model)  │
│    [x] Streaming chunk timing                                    │
│    [x] Token costs (per-query, per-agent, per-user)             │
│                                                                   │
│  Tool Layer:                                                     │
│    [x] Every tool invocation (name, args, result, duration)      │
│    [x] Tool success/failure rates                                │
│    [x] File I/O operations                                       │
│                                                                   │
│  Workflow Layer:                                                 │
│    [x] Workflow run start/end/duration                           │
│    [x] Per-step execution traces                                 │
│    [x] Trigger events                                            │
│    [x] Conditional branch paths taken                            │
│                                                                   │
│  External Service Layer:                                         │
│    [x] Email API calls (fetch, send, classify)                   │
│    [x] Jira API calls                                            │
│    [x] Computer Use actions (click, type, screenshot)            │
│    [x] MCP bridge calls                                          │
└─────────────────────────────────────────────────────────────────┘
```

### 5.3 How Security Enforces Permissions Across All Operations

Security is enforced at two levels: **gateway** (before operation) and **audit** (after operation).

#### Pattern: Permission Context with Capability Checking

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Security integration pattern.
All capability-sensitive operations must go through the Security Gateway.
"""

from gaia.security import (
    AuditLogger,
    Permission,
    PermissionContext,
    PermissionEnforcer,
    ResourceGovernor,
    SecretManager,
)


# === Pattern 1: Permission Context (Scoped Operations) ===

async def triage_emails(agent_id: str) -> dict:
    """Email triage with security enforcement."""
    # Create a security context that scopes all operations
    with PermissionContext(
        agent=agent_id,
        scope=[Permission.EMAIL_READ, Permission.EMAIL_SEND],
        resource_limits={"max_tokens": 10000, "max_api_calls": 100},
    ) as ctx:
        # Inside this context:
        # - Only email:read and email:send permissions are available
        # - Token usage is tracked against the limit
        # - Every operation is audit-logged

        emails = await fetch_emails()  # Allowed: email:read
        classified = classify_emails(emails)  # Allowed: internal operation

        # This would raise PermissionDenied:
        # jira_client.create_issue(...)  # Denied: jira:write not in scope

        return classified


# === Pattern 2: Secret Scoping ===

class SecureEmailAgent(Agent):
    """EmailAgent with credential isolation."""

    def __init__(self):
        super().__init__()
        # Agent can only access secrets scoped to it
        self._secret_scope = "email_agent"

    def _get_credentials(self) -> dict:
        # SecretManager enforces: this agent can only read email-scoped secrets
        gmail_token = SecretManager.get(
            "gmail_oauth_token",
            scope=self._secret_scope,
        )
        # Audit log entry: "EmailAgent accessed gmail_oauth_token"

        # This would raise SecretAccessDenied:
        # SecretManager.get("jira_api_token", scope=self._secret_scope)
        # "EmailAgent attempted to access jira_api_token (not in scope)"

        return {"gmail": gmail_token}


# === Pattern 3: Resource Governance ===

governor = ResourceGovernor()

@governor.enforce_limits(
    max_tokens_per_query=50000,
    max_queries_per_hour=100,
    max_concurrent=3,
)
async def run_agent_query(agent: Agent, query: str) -> str:
    """Run agent query with resource limits."""
    return await agent.process_query(query)

# If limits exceeded:
# ResourceLimitExceeded: "EmailAgent exceeded token limit (50000/50000)"
# Query is rejected before execution, not mid-stream
```

#### Security Enforcement Points

```
                    Security Enforcement Map
                    ========================

  ┌─────────────────┐
  │  User Request    │
  └────────┬────────┘
           │
           v
  ┌─────────────────┐     ┌──────────────────┐
  │ AUTHENTICATION  │────>│ Who is asking?    │
  │                 │     │ - User identity   │
  └────────┬────────┘     │ - Agent identity  │
           │              │ - Workflow context │
           v              └──────────────────┘
  ┌─────────────────┐     ┌──────────────────┐
  │ AUTHORIZATION   │────>│ Is this allowed?  │
  │                 │     │ - Check perms     │
  └────────┬────────┘     │ - Validate scope  │
           │              └──────────────────┘
           v
  ┌─────────────────┐     ┌──────────────────┐
  │ SECRET RETRIEVAL│────>│ Credential access │
  │                 │     │ - Scoped vault    │
  └────────┬────────┘     │ - Auto-decrypt    │
           │              └──────────────────┘
           v
  ┌─────────────────┐     ┌──────────────────┐
  │ RESOURCE CHECK  │────>│ Within limits?    │
  │                 │     │ - Token budget    │
  └────────┬────────┘     │ - Rate limit      │
           │              └──────────────────┘
           v
  ┌─────────────────┐
  │  EXECUTE        │ <-- Actual operation happens here
  └────────┬────────┘
           │
           v
  ┌─────────────────┐     ┌──────────────────┐
  │ AUDIT LOG       │────>│ Record everything │
  │                 │     │ - Action taken    │
  └────────┬────────┘     │ - Result/error    │
           │              │ - Timestamp       │
           v              └──────────────────┘
  ┌─────────────────┐     ┌──────────────────┐
  │ SECRET SCAN     │────>│ Output safe?      │
  │                 │     │ - No leaked creds │
  └─────────────────┘     │ - No PII exposure │
                          └──────────────────┘
```

### 5.4 How Memory and Manifest Work with Workflow State

Workflows are long-running and must survive restarts. Memory provides the persistence layer.

#### Pattern: Workflow State Backed by Memory

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Workflow-Memory integration pattern.
Workflow state is persisted via Persistent Memory for crash recovery.
"""


class WorkflowMemoryBridge:
    """
    Connects Workflow Orchestration with Persistent Memory.

    Responsibilities:
    - Checkpoint workflow state to episodic memory after each step
    - Store workflow results in universal knowledge DB
    - Resume workflows from last checkpoint on restart
    - Track cross-workflow patterns in semantic memory
    """

    def __init__(self, memory: PersistentMemory, manifest: ProjectManifest):
        self._memory = memory
        self._manifest = manifest

    def checkpoint_workflow(self, workflow_run: WorkflowRun) -> None:
        """Save workflow state after each step completes."""
        # Episodic memory: full state snapshot
        self._memory.episodic.store_session(
            session_type="workflow_checkpoint",
            summary=f"Workflow '{workflow_run.name}' step {workflow_run.current_step}",
            metadata={
                "workflow_id": workflow_run.id,
                "current_step": workflow_run.current_step,
                "completed_steps": workflow_run.completed_steps,
                "step_outputs": workflow_run.step_outputs,
                "state": workflow_run.state.value,
            },
        )

        # Manifest: update project state if workflow modifies files
        if workflow_run.current_step_type == "agent_step":
            for file_changed in workflow_run.files_modified:
                self._manifest.update_file(
                    file_changed.path,
                    status="modified",
                    modified_by=f"workflow:{workflow_run.name}",
                )

    def resume_workflow(self, workflow_id: str) -> WorkflowRun:
        """Resume a workflow from its last checkpoint."""
        # Find latest checkpoint in episodic memory
        checkpoints = self._memory.episodic.search(
            query=f"workflow_checkpoint workflow_id:{workflow_id}",
            limit=1,
            sort_by="timestamp_desc",
        )
        if not checkpoints:
            raise WorkflowNotFound(f"No checkpoint found for {workflow_id}")

        checkpoint = checkpoints[0]
        return WorkflowRun.from_checkpoint(checkpoint.metadata)

    def store_workflow_result(self, workflow_run: WorkflowRun) -> None:
        """Store completed workflow results in universal DB."""
        self._memory.universal.store_artifact(
            type="workflow_result",
            content={
                "workflow": workflow_run.name,
                "run_id": workflow_run.id,
                "duration": workflow_run.duration,
                "steps_completed": len(workflow_run.completed_steps),
                "outputs": workflow_run.step_outputs,
            },
            tags=[f"workflow:{workflow_run.name}"],
        )

    def learn_from_workflow(self, workflow_run: WorkflowRun) -> None:
        """Extract patterns from workflow execution for future optimization."""
        # Semantic memory: store patterns about workflow performance
        if workflow_run.state == WorkflowState.COMPLETED:
            self._memory.semantic.store_knowledge(
                fact=(
                    f"Workflow '{workflow_run.name}' completes in "
                    f"~{workflow_run.duration}s with {len(workflow_run.completed_steps)} steps"
                ),
                confidence=0.8,
                source=f"workflow_run:{workflow_run.id}",
            )
```

---

## 6. Shared Infrastructure

### 6.1 SQLite Usage Patterns Across All Architectures

Every architecture uses SQLite for persistence. To avoid conflicts, each uses its own database file within a shared directory structure.

#### Database File Layout

```
~/.gaia/
├── db/
│   ├── memory.db              # Persistent Memory (all 4 tiers)
│   ├── manifest.db            # Architecture Manifest
│   ├── observability.db       # Traces, metrics, sessions
│   ├── security.db            # Audit logs, resource usage
│   ├── error_recovery.db      # Dead letter queue, circuit states, incident log
│   ├── workflows.db           # Workflow definitions, run history, state
│   ├── email.db               # Email cache, thread state, classification history
│   ├── multidoc.db            # Document store, entity graph, citations
│   └── learning.db            # Feedback, outcomes, patterns
├── vault/
│   └── secrets.enc            # Encrypted secrets vault (AES-256)
├── skills/
│   ├── registry.json          # Skill metadata
│   └── *.py                   # Generated skill files
├── workflows/
│   └── *.yaml                 # Workflow definitions
└── config.json                # Global configuration
```

#### Shared Database Access Pattern

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Shared database access pattern for all GAIA architectures.
Each architecture gets its own database file to avoid contention.
"""

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator


GAIA_HOME = Path(os.environ.get("GAIA_HOME", Path.home() / ".gaia"))
DB_DIR = GAIA_HOME / "db"


def get_db_path(component: str) -> Path:
    """
    Get the database path for a specific component.

    Args:
        component: One of "memory", "manifest", "observability",
                   "security", "error_recovery", "workflows",
                   "email", "multidoc", "learning"

    Returns:
        Path to the SQLite database file.
    """
    DB_DIR.mkdir(parents=True, exist_ok=True)
    return DB_DIR / f"{component}.db"


@contextmanager
def get_connection(component: str) -> Generator[sqlite3.Connection, None, None]:
    """
    Get a database connection for a specific component.

    Uses WAL mode for concurrent reads during long-running operations.
    Enforces foreign keys for data integrity.

    Args:
        component: Component name (determines which .db file to use).

    Yields:
        sqlite3.Connection with WAL mode and foreign keys enabled.
    """
    db_path = get_db_path(component)
    conn = sqlite3.connect(str(db_path), timeout=30.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# Usage in any architecture:
#
#   with get_connection("observability") as conn:
#       conn.execute("INSERT INTO traces ...")
#
#   with get_connection("security") as conn:
#       conn.execute("INSERT INTO audit_log ...")
```

#### Schema Migration Pattern

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Schema migration pattern shared across all architectures.
Each component manages its own migrations.
"""


class SchemaMigrator:
    """
    Manages database schema migrations for a component.

    Each component defines its migrations as a list of SQL statements.
    Migrations are idempotent and versioned.
    """

    def __init__(self, component: str):
        self._component = component

    def migrate(self, migrations: list[str]) -> None:
        """Apply pending migrations."""
        with get_connection(self._component) as conn:
            # Create migration tracking table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS _migrations (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT DEFAULT (datetime('now')),
                    description TEXT
                )
            """)

            # Get current version
            row = conn.execute(
                "SELECT COALESCE(MAX(version), 0) FROM _migrations"
            ).fetchone()
            current_version = row[0]

            # Apply pending migrations
            for i, sql in enumerate(migrations, 1):
                if i > current_version:
                    conn.executescript(sql)
                    conn.execute(
                        "INSERT INTO _migrations (version, description) VALUES (?, ?)",
                        (i, sql[:100]),
                    )
```

### 6.2 Async/Sync Considerations

GAIA's base Agent class is synchronous (`process_query` returns `str`). Several architectures introduce async operations. The bridge pattern handles this.

#### Sync-Async Bridge Pattern

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Sync-async bridge for GAIA architectures.
Allows async architectures to work with the sync Agent base class.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Awaitable, Callable, TypeVar

T = TypeVar("T")

# Shared thread pool for running async code from sync context
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="gaia-async")


def run_async(coro: Awaitable[T]) -> T:
    """
    Run an async coroutine from synchronous code.

    Uses a dedicated event loop in a thread pool to avoid
    blocking the main thread or conflicting with existing loops.

    Args:
        coro: An awaitable coroutine to execute.

    Returns:
        The result of the coroutine.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # We are inside an existing event loop (e.g., Jupyter, API server)
        # Run in a separate thread to avoid deadlock
        import concurrent.futures

        with ThreadPoolExecutor(1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result(timeout=300)
    else:
        # No existing event loop -- create one
        return asyncio.run(coro)


# Usage in sync Agent code:
class EmailAgent(Agent):
    def _process_email(self, query: str) -> str:
        # Async email operations called from sync process_query
        emails = run_async(self._provider.fetch_unread())
        classified = run_async(self._classifier.classify_batch(emails))
        return self._format_results(classified)
```

#### Which Architectures Are Async

| Architecture | Sync/Async | Rationale |
|-------------|-----------|-----------|
| Error Recovery | **Async** | Retry delays, concurrent fallbacks |
| Observability | **Async** | Non-blocking telemetry export |
| Security | **Sync** | Must block on permission check |
| Persistent Memory | **Sync** | SQLite is sync, reads are fast |
| Learning | **Sync** | Background consolidation via thread |
| Dynamic Tools | **Sync** | Tool execution is sync in GAIA |
| Manifest | **Sync** | SQLite reads/writes |
| Workflow | **Async** | Parallel steps, cron scheduling |
| Task-Centric | **Sync** | Task CRUD is fast |
| Computer Use | **Async** | Browser automation, UI waits |
| Email | **Async** | Network I/O (IMAP, API calls) |
| Multi-Document | **Async** | Parallel document processing |

### 6.3 Configuration Management

All architectures share a unified configuration system rooted in `~/.gaia/config.json`.

#### Configuration Schema

```json
{
    "$schema": "gaia-config-v1",
    "version": "0.15.3",

    "model": "Qwen3-Coder-30B-A3B-Instruct-GGUF",
    "temperature": 0.7,
    "ctx_size": 32768,

    "error_recovery": {
        "enabled": true,
        "default_retry_strategy": "exponential",
        "default_max_retries": 3,
        "circuit_breaker_threshold": 5,
        "circuit_breaker_recovery_seconds": 30,
        "dead_letter_queue_enabled": true,
        "self_healing_enabled": true
    },

    "observability": {
        "enabled": true,
        "trace_sampling_rate": 1.0,
        "metrics_enabled": true,
        "session_recording": true,
        "cost_tracking": true,
        "exporter": "sqlite",
        "retention_days": 30
    },

    "security": {
        "vault_enabled": true,
        "vault_encryption": "AES-256-GCM",
        "audit_logging": true,
        "permission_enforcement": true,
        "secret_scanning": true,
        "resource_limits": {
            "max_tokens_per_day": 10000000,
            "max_concurrent_agents": 5,
            "max_memory_mb": 2048
        }
    },

    "memory": {
        "enabled": true,
        "episodic_retention_days": 90,
        "semantic_confidence_decay": 0.01,
        "universal_db_enabled": true,
        "embedding_model": "all-MiniLM-L6-v2"
    },

    "workflow": {
        "enabled": true,
        "max_concurrent_workflows": 3,
        "default_timeout_seconds": 3600,
        "checkpoint_interval_steps": 5
    },

    "email": {
        "provider": "gmail",
        "auto_response_enabled": false,
        "classification_model": "Qwen3-Coder-30B-A3B-Instruct-GGUF",
        "send_confirmation_required": true
    },

    "computer_use": {
        "enabled": false,
        "safety_confirmation_required": true,
        "allowed_domains": ["*.company.com"],
        "screenshot_retention_hours": 24
    },

    "multidoc": {
        "max_documents_per_collection": 100,
        "table_extraction_enabled": true,
        "citation_mode": "page_level"
    }
}
```

#### Configuration Access Pattern

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Unified configuration access for all GAIA architectures.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional


class GaiaConfig:
    """
    Centralized configuration for all GAIA architectures.

    Loads from ~/.gaia/config.json with environment variable overrides.
    Each architecture accesses its section via get_section().
    """

    _instance: Optional["GaiaConfig"] = None

    def __init__(self):
        self._config = self._load_config()

    @classmethod
    def get(cls) -> "GaiaConfig":
        """Get singleton config instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_config(self) -> Dict[str, Any]:
        config_path = Path.home() / ".gaia" / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                return json.load(f)
        return {}

    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get configuration for a specific architecture.

        Args:
            section: Architecture name (e.g., "error_recovery", "observability")

        Returns:
            Configuration dict for that section, with defaults applied.
        """
        return self._config.get(section, {})

    def is_enabled(self, section: str) -> bool:
        """Check if an architecture is enabled."""
        return self.get_section(section).get("enabled", False)


# Usage in any architecture:
#
#   config = GaiaConfig.get()
#   if config.is_enabled("observability"):
#       tracer = TracerProvider(config.get_section("observability"))
```

### 6.4 Logging Patterns

All architectures use GAIA's existing logger with structured context.

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Structured logging pattern for GAIA architectures.
Builds on gaia.logger with architecture-specific context.
"""

from gaia.logger import get_logger


# Each architecture creates a namespaced logger
log = get_logger("gaia.recovery")      # Error Recovery
log = get_logger("gaia.observe")       # Observability
log = get_logger("gaia.security")      # Security
log = get_logger("gaia.memory")        # Persistent Memory
log = get_logger("gaia.workflow")      # Workflow
log = get_logger("gaia.email")         # Email
log = get_logger("gaia.computeruse")   # Computer Use
log = get_logger("gaia.multidoc")      # Multi-Document

# Structured logging with context
log.info(
    "Workflow step completed",
    extra={
        "workflow_id": "weekly-report",
        "step_id": "fetch_jira",
        "duration_ms": 8200,
        "success": True,
    },
)

# Error logging with recovery context
log.warning(
    "LLM call failed, retrying",
    extra={
        "service": "lemonade_server",
        "attempt": 2,
        "max_retries": 3,
        "backoff_seconds": 2.0,
        "error_type": "ConnectionError",
    },
)
```

---

## 7. Composition Guidelines

### 7.1 When to Use Which Architecture

| You Need To... | Use These Architectures | Skip These |
|----------------|------------------------|------------|
| Build a simple chatbot | Agent base + Memory | Everything else |
| Add email automation | Email + Security + Error Recovery | Computer Use, Multi-Doc |
| Create scheduled jobs | Workflow + Error Recovery | Computer Use |
| Analyze document sets | Multi-Doc + Memory + Learning | Email, Workflow, Computer Use |
| Automate desktop tasks | Computer Use + Security + Error Recovery | Email, Multi-Doc |
| Build a production API | Agent + Observability + Security + Error Recovery | Computer Use |
| Track long coding tasks | Manifest + Adaptive Prompts + Memory + Task-Centric | Email |
| Full autonomous agent | All of them | None |

### 7.2 Recommended Architecture Combinations

#### Combination A: "Reliable Agent" (Minimum Production Setup)

```
Agent Base + Error Recovery + Observability + Security
```

This is the minimum for any production deployment. Every agent should have retry logic, tracing, and permission controls.

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Minimum production agent: Error Recovery + Observability + Security.
"""

from gaia.agents.base import Agent
from gaia.agents.base.tools import tool
from gaia.observe import ObservableMixin
from gaia.recovery import ResilientMixin
from gaia.security import SecureMixin


class ReliableAgent(SecureMixin, ObservableMixin, ResilientMixin, Agent):
    """
    Agent with production-grade infrastructure.

    Mixin order matters (MRO):
    1. SecureMixin: Permission check FIRST (outermost)
    2. ObservableMixin: Trace SECOND (wraps everything inside security)
    3. ResilientMixin: Retry THIRD (wraps actual operations)
    4. Agent: Base class LAST
    """

    required_permissions = ["llm:query"]

    def _get_system_prompt(self) -> str:
        return "You are a reliable production agent."

    def _register_tools(self):
        @tool
        def greet(name: str) -> str:
            """Greet a user by name."""
            return f"Hello, {name}!"
```

#### Combination B: "Knowledge Agent" (Learning + Memory)

```
Agent Base + Memory + Learning + Manifest + Adaptive Prompts
```

For agents that improve over time and track complex projects.

#### Combination C: "Automation Agent" (Workflow + External Services)

```
Agent Base + Workflow + Email + Security + Error Recovery + Observability
```

For agents that run scheduled tasks involving external services.

#### Combination D: "Full Autonomous Agent" (Everything)

```
All 14 architectures (excluding TUI, Dashboard, Requirements, Assessment)
```

For the most capable agent possible. See Section 9 for the complete example.

### 7.3 Performance Implications

Each architecture adds overhead. Understand the costs before composing.

| Architecture | Overhead per Query | Memory Footprint | Disk I/O |
|-------------|-------------------|-------------------|----------|
| Error Recovery | ~0ms (only on failure) | ~1 MB (circuit state) | Minimal |
| Observability | ~2-5ms (span creation) | ~5 MB (trace buffer) | Periodic flush |
| Security | ~1-3ms (permission check) | ~2 MB (policy cache) | Audit log writes |
| Persistent Memory | ~5-10ms (memory lookup) | ~50 MB (FAISS index) | Per-session writes |
| Learning | ~0ms (background) | ~10 MB (pattern store) | Background consolidation |
| Manifest | ~1ms (manifest update) | ~5 MB (file tracking) | Per-file-change writes |
| Adaptive Prompts | ~0ms (state lookup) | ~1 MB (state stack) | State transition writes |
| Workflow | ~2ms (state machine) | ~10 MB (scheduler) | Per-step checkpoint |
| Task-Centric | ~1ms (task CRUD) | ~2 MB (task queue) | Task state writes |

**Total overhead for "Reliable Agent" (3 architectures)**: ~3-8ms per query
**Total overhead for "Full Agent" (all architectures)**: ~12-22ms per query

For context, a typical LLM call takes 500-5000ms, so the architecture overhead is negligible (0.2-4% of total latency).

### 7.4 Trade-Offs

| Decision | Option A | Option B | Recommendation |
|----------|---------|---------|----------------|
| Memory tier depth | 2 tiers (working + episodic) | 4 tiers (all) | Start with 2, add semantic when learning is needed |
| Observability sampling | 100% (all traces) | 10% (sampled) | 100% for dev/staging, 10% for production |
| Security enforcement | Soft (log violations) | Hard (block violations) | Hard in production, soft in development |
| Error Recovery fallback | Local-only | Local + cloud fallback | Local + cloud for reliability, local-only for privacy |
| Workflow persistence | In-memory | SQLite checkpoints | Always SQLite (survives restarts) |
| Computer Use safety | Auto-approve actions | Confirm destructive actions | Confirm in production, auto in testing |

---

## 8. Migration Strategy

### 8.1 Implementation Order (Respecting Dependencies)

```
Phase 0: FOUNDATION (Week 1-2)
================================
  [1] Error Recovery
      - No dependencies on other new architectures
      - Immediately improves reliability of existing agents
      - Implement: RetryEngine, CircuitBreaker, ErrorClassifier

  [2] Observability (Core)
      - No dependencies on other new architectures
      - Provides visibility needed for debugging everything else
      - Implement: TracerProvider, Span, SQLiteExporter

  [3] Security (Core)
      - No dependencies on other new architectures
      - Needed before any external service integration
      - Implement: SecretManager (vault), PermissionEnforcer, AuditLogger


Phase 1: MEMORY + INTELLIGENCE (Week 3-6)
===========================================
  [4] Persistent Memory
      - Depends on: SQLite infrastructure from Phase 0
      - Foundation for all learning and state management
      - Implement: WorkingMemory, EpisodicMemory, SemanticMemory

  [5] Architecture Manifest
      - Depends on: Persistent Memory
      - Needed for long-running code generation tasks
      - Implement: ProjectManifest, ManifestToolsMixin

  [6] Adaptive Prompts / State Machine
      - Depends on: Persistent Memory
      - Enables multi-mode execution
      - Implement: AgentState, StateMachine, PromptComposer


Phase 2: ORCHESTRATION (Week 7-10)
====================================
  [7] Task-Centric Interface
      - Depends on: Adaptive Prompts, Persistent Memory
      - Required for Workflow Orchestration
      - Implement: Task, TaskQueue, TaskManager

  [8] Learning & Adaptation
      - Depends on: Persistent Memory
      - Enables self-improvement
      - Implement: LearningLoop, FeedbackCollector, OutcomeTracker

  [9] Dynamic Tools
      - Depends on: Persistent Memory, Learning
      - Enables runtime tool creation
      - Implement: DynamicToolManager, ToolBuilderAgent

  [10] Workflow Orchestration
       - Depends on: Task-Centric, Error Recovery
       - Enables scheduled and event-driven pipelines
       - Implement: WorkflowEngine, TriggerSystem, StepExecutor


Phase 3: DOMAIN CAPABILITIES (Week 11-16)
============================================
  [11] Email Integration
       - Depends on: Security (credentials), Error Recovery (API retries)
       - Soft dependency on Workflow (for scheduling)
       - Implement: EmailAgent, GmailProvider, EmailClassifier

  [12] Multi-Document Synthesis
       - Depends on: RAGSDK (existing), Persistent Memory
       - Implement: DocumentStore, EntityResolver, CitationTracker

  [13] Computer Use
       - Depends on: VLM client (existing), Security (safety)
       - Implement: ComputerUseController, ScreenCaptureEngine, BrowserAutomation


Phase 4: POLISH (Week 17+)
============================
  [14] Observability (Advanced)
       - Time-travel debugging, cost dashboards, alerting
       - Depends on: Core observability from Phase 0

  [15] Security (Advanced)
       - Resource Governor, anomaly detection, compliance reports
       - Depends on: Core security from Phase 0

  [16] Error Recovery (Advanced)
       - Self-healing engine, recovery playbooks
       - Depends on: Core error recovery from Phase 0
```

### 8.2 How to Adopt Incrementally

Each architecture is designed as a **mixin** that can be added to any existing Agent subclass without modifying the base class.

#### Step-by-Step Adoption

```python
# === Step 1: Start with existing agent (no changes) ===

class MyAgent(Agent):
    def _get_system_prompt(self) -> str:
        return "You are a helpful agent."


# === Step 2: Add Error Recovery (Phase 0) ===

from gaia.recovery import ResilientMixin

class MyAgent(ResilientMixin, Agent):
    def _get_system_prompt(self) -> str:
        return "You are a helpful agent."
    # Now: all LLM calls retry on failure automatically


# === Step 3: Add Observability (Phase 0) ===

from gaia.observe import ObservableMixin

class MyAgent(ObservableMixin, ResilientMixin, Agent):
    def _get_system_prompt(self) -> str:
        return "You are a helpful agent."
    # Now: all operations traced, metrics collected


# === Step 4: Add Security (Phase 0) ===

from gaia.security import SecureMixin

class MyAgent(SecureMixin, ObservableMixin, ResilientMixin, Agent):
    required_permissions = ["llm:query"]
    def _get_system_prompt(self) -> str:
        return "You are a helpful agent."
    # Now: permission checks, audit logging, secret scanning


# === Step 5: Add Memory (Phase 1) ===

from gaia.memory import MemoryMixin

class MyAgent(SecureMixin, ObservableMixin, ResilientMixin, MemoryMixin, Agent):
    required_permissions = ["llm:query"]
    def _get_system_prompt(self) -> str:
        # Memory mixin injects relevant memories into prompt
        memories = self.recall_relevant(self._current_query)
        return f"You are a helpful agent.\n\nRelevant context:\n{memories}"
    # Now: persistent memory across sessions


# === Step 6: Add whatever domain capabilities you need ===
# Each step is independent and backward-compatible
```

### 8.3 Backward Compatibility

All architectures follow these compatibility rules:

1. **No breaking changes to Agent base class**. All new functionality is added via mixins.
2. **All architectures are opt-in**. If not configured, they are no-ops.
3. **Existing agents work unchanged**. Adding a mixin does not require modifying existing tool functions.
4. **Configuration defaults to safe values**. Unconfigured architectures default to disabled or minimal overhead.
5. **Graceful feature detection**. Architectures check for dependencies before using them.

```python
# Example: Observability gracefully handles missing tracer
class ObservableMixin:
    def process_query(self, query, **kwargs):
        if GaiaConfig.get().is_enabled("observability"):
            with tracer.start_span("process_query"):
                return super().process_query(query, **kwargs)
        else:
            return super().process_query(query, **kwargs)
```

---

## 9. Complete Working Example

### 9.1 Production Agent Using 6 Architectures

This example shows a complete, production-ready `InboxAssistantAgent` that composes Error Recovery, Observability, Security, Persistent Memory, Email Integration, and Workflow Orchestration.

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
InboxAssistantAgent: Production agent combining 6 architectures.

Architectures used:
1. Error Recovery   - Retry email API calls, circuit breaker for providers
2. Observability    - Trace every classification, track email processing costs
3. Security         - Credential vault for OAuth tokens, permission scoping
4. Persistent Memory - Remember email patterns, user preferences across sessions
5. Email Integration - Gmail/IMAP provider, classification, auto-response
6. Workflow          - Scheduled triage, event-driven responses

Usage:
    # One-shot triage
    agent = InboxAssistantAgent()
    result = agent.process_query("Triage my inbox and summarize urgent emails")

    # Scheduled workflow
    gaia workflow run email-triage
"""

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from gaia.agents.base import Agent
from gaia.agents.base.tools import tool
from gaia.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Architecture imports (each is a self-contained module)
# ---------------------------------------------------------------------------

# 1. Error Recovery
from gaia.recovery import (
    CircuitBreaker,
    CircuitState,
    DegradationChain,
    DegradationLevel,
    ErrorClassifier,
    ErrorType,
    ExponentialBackoff,
    RetryEngine,
    circuit_breaker,
    with_retry,
)

# 2. Observability
from gaia.observe import (
    CostCalculator,
    MeterProvider,
    SessionRecorder,
    SpanKind,
    TracerProvider,
    meter,
    tracer,
)

# 3. Security
from gaia.security import (
    AuditLogger,
    Permission,
    PermissionContext,
    PermissionEnforcer,
    ResourceGovernor,
    SecretManager,
)

# 4. Persistent Memory
from gaia.memory import (
    EpisodicMemory,
    PersistentMemory,
    SemanticMemory,
    WorkingMemory,
)

# 5. Email Integration
from gaia.email import (
    CalendarBridge,
    Email,
    EmailClassification,
    EmailClassifier,
    GmailProvider,
    IMAPProvider,
    ResponseDrafter,
    SendGuard,
)

# 6. Workflow Orchestration
from gaia.workflow import (
    StateManager,
    StepExecutor,
    TriggerSystem,
    WorkflowEngine,
    WorkflowRun,
    WorkflowState,
)


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class TriageResult:
    """Result of email triage operation."""
    total_processed: int
    urgent: List[Dict[str, Any]]
    action_required: List[Dict[str, Any]]
    informational: List[Dict[str, Any]]
    spam: List[Dict[str, Any]]
    auto_responses_drafted: int
    errors: List[str]
    duration_seconds: float
    cost_usd: float


# ---------------------------------------------------------------------------
# Agent Implementation
# ---------------------------------------------------------------------------

class InboxAssistantAgent(Agent):
    """
    Production email triage agent using 6 GAIA architectures.

    Capabilities:
    - Connects to Gmail via OAuth2 (Security: credential vault)
    - Fetches and classifies emails (Email: classification engine)
    - Retries on API failures (Error Recovery: retry + circuit breaker)
    - Traces all operations (Observability: distributed tracing)
    - Remembers user preferences (Memory: semantic memory)
    - Runs on schedule (Workflow: cron trigger)
    """

    # Security: declare required permissions
    REQUIRED_PERMISSIONS = [
        Permission.EMAIL_READ,
        Permission.EMAIL_SEND,
        Permission.LLM_QUERY,
    ]

    def __init__(
        self,
        debug: bool = False,
        show_prompts: bool = False,
        streaming: bool = False,
        **kwargs,
    ):
        super().__init__(
            debug=debug,
            show_prompts=show_prompts,
            streaming=streaming,
            **kwargs,
        )

        # --- Architecture 1: Error Recovery ---
        self._retry_engine = RetryEngine(
            default_strategy=ExponentialBackoff(
                base_delay=1.0,
                max_delay=30.0,
                max_retries=3,
            ),
        )
        self._circuit_breakers = {
            "gmail": CircuitBreaker(
                service="gmail_api",
                failure_threshold=5,
                recovery_timeout=60,
            ),
            "imap": CircuitBreaker(
                service="imap_fallback",
                failure_threshold=3,
                recovery_timeout=120,
            ),
        }

        # --- Architecture 2: Observability ---
        self._tracer = TracerProvider.get_tracer("inbox_assistant")
        self._meter = MeterProvider.get_meter("inbox_assistant")
        self._session_recorder = SessionRecorder()
        self._cost_calculator = CostCalculator()

        # --- Architecture 3: Security ---
        self._enforcer = PermissionEnforcer()
        self._audit = AuditLogger()
        self._secrets = SecretManager()

        # --- Architecture 4: Persistent Memory ---
        self._memory = PersistentMemory(agent_id="inbox_assistant")

        # --- Architecture 5: Email Integration ---
        self._classifier = None  # Initialized after credential retrieval
        self._drafter = None
        self._send_guard = SendGuard(confirmation_required=True)

        # --- Architecture 6: Workflow ---
        self._workflow_engine = WorkflowEngine()

        # Register tools
        self._register_tools()

    def _get_system_prompt(self) -> str:
        """Build system prompt with memory-injected context."""
        # Retrieve learned preferences from semantic memory
        preferences = self._memory.semantic.search(
            query="email triage preferences",
            limit=5,
        )
        pref_text = "\n".join(
            f"- {p.fact} (confidence: {p.confidence:.0%})"
            for p in preferences
        ) if preferences else "No learned preferences yet."

        return f"""You are an intelligent email assistant. You help users manage
their inbox by classifying, triaging, and responding to emails.

Learned User Preferences:
{pref_text}

Available tools: triage_inbox, classify_email, draft_response,
send_email, check_calendar, search_emails

Always classify emails before taking action. Never send emails
without user confirmation unless auto-response is explicitly enabled.
"""

    def _register_tools(self):
        """Register email-related tools."""

        @tool
        def triage_inbox(
            max_emails: int = 50,
            time_window_hours: int = 24,
        ) -> str:
            """
            Triage the user's inbox by fetching, classifying, and
            organizing recent emails.

            Args:
                max_emails: Maximum number of emails to process.
                time_window_hours: How far back to look for emails.

            Returns:
                JSON summary of triage results.
            """
            result = self._triage_inbox(max_emails, time_window_hours)
            return json.dumps({
                "total_processed": result.total_processed,
                "urgent_count": len(result.urgent),
                "action_required_count": len(result.action_required),
                "informational_count": len(result.informational),
                "spam_count": len(result.spam),
                "auto_responses_drafted": result.auto_responses_drafted,
                "duration_seconds": result.duration_seconds,
                "cost_usd": result.cost_usd,
                "urgent_subjects": [
                    e["subject"] for e in result.urgent
                ],
            }, indent=2)

        @tool
        def draft_response(email_id: str, tone: str = "professional") -> str:
            """
            Draft a response to a specific email.

            Args:
                email_id: The ID of the email to respond to.
                tone: Response tone (professional, casual, formal).

            Returns:
                Drafted response text for review.
            """
            return self._draft_response(email_id, tone)

        @tool
        def search_emails(query: str, limit: int = 10) -> str:
            """
            Search emails by keyword or criteria.

            Args:
                query: Search query (subject, sender, content).
                limit: Maximum results to return.

            Returns:
                JSON list of matching emails.
            """
            return self._search_emails(query, limit)

    # -------------------------------------------------------------------
    # Core Operations (where architectures compose)
    # -------------------------------------------------------------------

    def _triage_inbox(
        self,
        max_emails: int,
        time_window_hours: int,
    ) -> TriageResult:
        """
        Core triage operation demonstrating architecture composition.

        Composition order:
        1. Security: check permissions, retrieve credentials
        2. Observability: start trace span
        3. Error Recovery: fetch emails with retry + circuit breaker
        4. Email: classify each email
        5. Memory: store results and learn preferences
        """
        start_time = time.monotonic()
        errors = []
        total_input_tokens = 0
        total_output_tokens = 0

        # --- SECURITY: Permission check and credential retrieval ---
        self._enforcer.check_permissions(
            agent_id="inbox_assistant",
            permissions=self.REQUIRED_PERMISSIONS,
        )
        self._audit.log_action(
            agent="inbox_assistant",
            action="triage_inbox",
            details={"max_emails": max_emails},
        )

        gmail_token = self._secrets.get(
            "gmail_oauth_token",
            scope="inbox_assistant",
        )

        # --- OBSERVABILITY: Start trace ---
        with self._tracer.start_span(
            "triage_inbox",
            kind=SpanKind.AGENT,
            attributes={
                "max_emails": max_emails,
                "time_window_hours": time_window_hours,
            },
        ) as root_span:

            # --- ERROR RECOVERY: Fetch emails with retry + circuit breaker ---
            emails = self._fetch_emails_with_recovery(
                gmail_token, max_emails, time_window_hours
            )
            root_span.set_attribute("emails_fetched", len(emails))

            # --- EMAIL + OBSERVABILITY: Classify each email ---
            classified = {"urgent": [], "action_required": [],
                          "informational": [], "spam": []}

            for email in emails:
                with self._tracer.start_span(
                    "classify_email",
                    kind=SpanKind.LLM,
                ) as classify_span:
                    try:
                        classification = self._classifier.classify(email)
                        classify_span.set_attribute(
                            "category", classification.category
                        )
                        classify_span.set_attribute(
                            "urgency", classification.urgency
                        )

                        classified[classification.category].append({
                            "id": email.id,
                            "subject": email.subject,
                            "sender": email.sender,
                            "urgency": classification.urgency,
                            "summary": classification.summary,
                        })

                        # Track tokens for cost calculation
                        total_input_tokens += classification.input_tokens
                        total_output_tokens += classification.output_tokens

                        # Metric: classification counter
                        self._meter.record(
                            "emails_classified",
                            1,
                            {"category": classification.category},
                        )

                    except Exception as e:
                        log.warning(
                            "Failed to classify email %s: %s",
                            email.id,
                            str(e),
                        )
                        errors.append(f"Classification failed for {email.id}: {e}")

            # --- MEMORY: Store results and learn ---
            duration = time.monotonic() - start_time
            cost = self._cost_calculator.calculate(
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                model="Qwen3-Coder-30B",
            )

            # Episodic: remember this triage session
            self._memory.episodic.store_session(
                session_type="email_triage",
                summary=(
                    f"Triaged {len(emails)} emails: "
                    f"{len(classified['urgent'])} urgent, "
                    f"{len(classified['action_required'])} action required, "
                    f"{len(classified['informational'])} informational, "
                    f"{len(classified['spam'])} spam"
                ),
                metadata={
                    "total_emails": len(emails),
                    "categories": {k: len(v) for k, v in classified.items()},
                    "cost_usd": cost,
                },
            )

            # Semantic: learn sender patterns
            for email_data in classified["urgent"]:
                self._memory.semantic.store_knowledge(
                    fact=f"Emails from {email_data['sender']} are often urgent",
                    confidence=0.7,
                    source="email_triage_pattern",
                )

            # Observability: record cost
            self._cost_calculator.record(
                agent="inbox_assistant",
                operation="triage_inbox",
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                model="Qwen3-Coder-30B",
                cost_usd=cost,
            )

            root_span.set_attribute("cost_usd", cost)
            root_span.set_attribute("duration_seconds", duration)

        return TriageResult(
            total_processed=len(emails),
            urgent=classified["urgent"],
            action_required=classified["action_required"],
            informational=classified["informational"],
            spam=classified["spam"],
            auto_responses_drafted=0,
            errors=errors,
            duration_seconds=duration,
            cost_usd=cost,
        )

    def _fetch_emails_with_recovery(
        self,
        gmail_token: str,
        max_emails: int,
        time_window_hours: int,
    ) -> List[Email]:
        """
        Fetch emails with full error recovery chain.

        Degradation chain:
        1. Gmail API (primary)
        2. IMAP (fallback)
        3. Cached emails from last sync (last resort)
        """
        # Initialize provider with credentials
        gmail_provider = GmailProvider(oauth_token=gmail_token)
        imap_provider = IMAPProvider(
            host="imap.gmail.com",
            credentials=gmail_token,
        )

        # Check circuit breaker state
        if self._circuit_breakers["gmail"].state == CircuitState.OPEN:
            log.warning("Gmail circuit breaker is OPEN, trying IMAP fallback")

        # Build degradation chain
        chain = DegradationChain(
            name="fetch_emails",
            levels=[
                DegradationLevel(
                    name="gmail_api",
                    operation=lambda: self._retry_engine.execute(
                        lambda: gmail_provider.fetch_unread(
                            limit=max_emails,
                            since_hours=time_window_hours,
                        ),
                        strategy=ExponentialBackoff(
                            base_delay=1.0,
                            max_retries=3,
                        ),
                    ),
                    circuit_breaker=self._circuit_breakers["gmail"],
                    quality=1.0,
                ),
                DegradationLevel(
                    name="imap_fallback",
                    operation=lambda: self._retry_engine.execute(
                        lambda: imap_provider.fetch_unread(
                            limit=max_emails,
                            since_hours=time_window_hours,
                        ),
                        strategy=ExponentialBackoff(
                            base_delay=2.0,
                            max_retries=2,
                        ),
                    ),
                    circuit_breaker=self._circuit_breakers["imap"],
                    quality=0.9,
                ),
                DegradationLevel(
                    name="cached_emails",
                    operation=lambda: self._memory.episodic.search(
                        query="email_cache",
                        limit=max_emails,
                    ),
                    quality=0.3,
                ),
            ],
        )

        result = chain.execute()

        if result.degraded:
            log.warning(
                "Email fetch degraded to level '%s' (quality: %.0f%%)",
                result.level_used,
                result.quality * 100,
            )
            self._meter.record(
                "email_fetch_degraded",
                1,
                {"level": result.level_used},
            )

        return result.value

    def _draft_response(self, email_id: str, tone: str) -> str:
        """Draft a response with security and observability."""
        with self._tracer.start_span("draft_response", kind=SpanKind.LLM):
            # Retrieve the original email from cache
            email = self._memory.episodic.search(
                query=f"email_id:{email_id}",
                limit=1,
            )
            if not email:
                return f"Email {email_id} not found in cache."

            # Draft using LLM
            drafter = ResponseDrafter(llm_client=self.chat_sdk)
            draft = drafter.draft(
                email=email[0],
                tone=tone,
                user_preferences=self._memory.semantic.search(
                    "email response preferences", limit=3
                ),
            )

            return draft.text

    def _search_emails(self, query: str, limit: int) -> str:
        """Search emails in local cache."""
        results = self._memory.episodic.search(
            query=f"email: {query}",
            limit=limit,
        )
        return json.dumps([
            {"subject": r.metadata.get("subject", ""), "sender": r.metadata.get("sender", "")}
            for r in results
        ], indent=2)
```

### 9.2 Running the Agent

```bash
# One-shot triage
gaia inbox "Triage my inbox and summarize urgent emails"

# View the trace
gaia observe traces --agent inbox_assistant --last 1

# View costs
gaia observe costs --agent inbox_assistant --today

# Schedule as workflow
gaia workflow create email-triage --schedule "0 8 * * *"
gaia workflow status email-triage

# Check audit log
gaia security audit --agent inbox_assistant --last 24h
```

### 9.3 Architecture Interaction Diagram for the Complete Example

```
                    InboxAssistantAgent
                    ===================

  User: "Triage my inbox"
    |
    v
  ┌──────────────────────────────────────────────────────┐
  │  process_query("Triage my inbox")                     │
  │                                                       │
  │  1. SECURITY: check_permissions([email:read, ...])    │
  │     |                                                 │
  │     v                                                 │
  │  2. SECURITY: SecretManager.get("gmail_oauth_token")  │
  │     |                                                 │
  │     v                                                 │
  │  3. OBSERVABILITY: tracer.start_span("triage_inbox")  │
  │     |                                                 │
  │     v                                                 │
  │  4. ERROR RECOVERY: DegradationChain.execute()        │
  │     |                                                 │
  │     |  ┌─────────────────────────────────────────┐    │
  │     |  │ Level 1: Gmail API (with retry x3)      │    │
  │     |  │ Level 2: IMAP fallback (with retry x2)  │    │
  │     |  │ Level 3: Cached emails (last resort)    │    │
  │     |  └─────────────────────────────────────────┘    │
  │     |                                                 │
  │     v                                                 │
  │  5. EMAIL: classifier.classify(email) x N             │
  │     |   (each wrapped in observability span)          │
  │     |                                                 │
  │     v                                                 │
  │  6. MEMORY: episodic.store_session(triage_results)    │
  │     |       semantic.store_knowledge(sender_patterns) │
  │     |                                                 │
  │     v                                                 │
  │  7. OBSERVABILITY: record_cost, end_span              │
  │     |                                                 │
  │     v                                                 │
  │  8. SECURITY: audit.log_action(triage_complete)       │
  │                                                       │
  └──────────────────────────────────────────────────────┘
    |
    v
  Result: TriageResult with classified emails
```

---

## 10. Appendix: Cross-Reference Matrix

### 10.1 Architecture-to-Architecture Integration Points

This matrix shows which architectures directly interact with each other. An "X" means the row architecture directly calls or is called by the column architecture.

```
                 ErrRec  Obsrv  Secur  Memry  Learn  DynTl  Manif  Adapt  Task   Wkflw  CmpUse Email  MDoc
Error Recovery     -       .      .      .      .      .      .      .      .      X      X      X      X
Observability      X       -      .      X      .      .      .      .      .      X      X      X      X
Security           .       .      -      X      .      .      .      .      .      X      X      X      .
Pers. Memory       .       X      X      -      X      X      X      X      X      X      .      X      X
Learning           .       .      .      X      -      X      .      .      .      .      .      .      .
Dynamic Tools      .       .      .      X      X      -      .      .      .      .      .      .      .
Manifest           .       .      .      X      .      .      -      .      X      X      .      .      .
Adaptive Prompts   .       .      .      X      .      .      .      -      X      .      .      .      .
Task-Centric       .       .      .      X      .      .      X      X      -      X      .      .      .
Workflow           X       X      X      X      .      .      X      .      X      -      .      X      .
Computer Use       X       X      X      .      .      .      .      .      .      .      -      .      .
Email              X       X      X      X      .      .      .      .      .      X      .      -      .
Multi-Doc          X       X      .      X      X      .      .      .      .      .      .      .      -

Legend:
  X = Direct integration (calls or is called by)
  . = No direct integration (may interact indirectly via Memory)
  - = Self
```

### 10.2 Shared Resource Usage

| Resource | Used By |
|----------|---------|
| SQLite (WAL mode) | Memory, Manifest, Observability, Security, Error Recovery, Workflow, Email, Multi-Doc, Learning |
| FAISS vector index | Memory (episodic), Dynamic Tools (SKILLS), Multi-Doc (chunks) |
| Sentence-Transformers | Memory (embeddings), Multi-Doc (embeddings), Learning (similarity) |
| LLM (Lemonade) | Email (classification), Multi-Doc (synthesis), Learning (consolidation), Adaptive Prompts (state execution) |
| File system | Manifest (project files), Dynamic Tools (skill files), Workflow (YAML definitions), Security (vault) |
| Network I/O | Email (API), Computer Use (browser), Workflow (webhooks), Error Recovery (health probes) |

### 10.3 Event Flow Between Architectures

```
EVENT: "Agent processes a query"
=====================================

1. Security.check_permissions()         --> Audit log entry
2. Observability.start_span()           --> Trace created
3. Memory.recall_relevant()             --> Context injected into prompt
4. Adaptive Prompts.get_current_state() --> State-specific prompt selected
5. LLM.complete()                       --> Wrapped in Error Recovery retry
6. Tool execution                       --> Traced by Observability
7. Manifest.update_file()               --> If file was modified
8. Memory.store_working()               --> Current context updated
9. Learning.record_execution()          --> Execution trace stored
10. Observability.end_span()            --> Trace completed
11. Security.audit_log()                --> Final audit entry
12. Cost.record()                       --> Token cost calculated


EVENT: "Workflow step fails"
=============================

1. Error Recovery.classify_error()      --> TRANSIENT
2. Error Recovery.retry()               --> Attempt 1, 2, 3
3. Observability.record_retry()         --> Retry spans created
4. Error Recovery.circuit_breaker()     --> Check state (CLOSED/OPEN)
5. Error Recovery.degrade()             --> Fallback to next level
6. Workflow.checkpoint()                --> State saved via Memory
7. Security.audit_log()                 --> Failure recorded
8. Observability.alert()                --> Alert if SLO breached


EVENT: "New email arrives (webhook trigger)"
=============================================

1. Workflow.trigger(webhook)            --> Workflow run created
2. Security.check_permissions()         --> Verify workflow permissions
3. Email.fetch_new()                    --> Wrapped in Error Recovery
4. Email.classify()                     --> LLM classification (traced)
5. Memory.store_episodic()              --> Email cached
6. Workflow.conditional()               --> Branch based on classification
7. Email.draft_response()               --> If action required (traced)
8. Email.send()                         --> Via SendGuard (security)
9. Memory.store_semantic()              --> Learn sender patterns
10. Workflow.complete()                 --> Run marked complete
11. Observability.record_cost()         --> Total cost calculated
```

### 10.4 File Map: Where Each Architecture Lives

```
src/gaia/
├── agents/base/
│   ├── agent.py                    # Base Agent (EXISTING)
│   ├── tools.py                    # @tool decorator (EXISTING)
│   ├── console.py                  # AgentConsole (EXISTING)
│   └── errors.py                   # Error formatting (EXISTING)
│
├── recovery/                       # ERROR RECOVERY (NEW)
│   ├── __init__.py
│   ├── retry.py                    # RetryEngine, ExponentialBackoff
│   ├── circuit_breaker.py          # CircuitBreaker, CircuitState
│   ├── degradation.py              # DegradationChain, DegradationLevel
│   ├── classifier.py               # ErrorClassifier, ErrorType
│   ├── self_healing.py             # SelfHealingEngine
│   ├── dead_letter.py              # DeadLetterQueue
│   └── mixin.py                    # ResilientMixin
│
├── observe/                        # OBSERVABILITY (NEW)
│   ├── __init__.py
│   ├── tracer.py                   # TracerProvider, Span, SpanKind
│   ├── metrics.py                  # MeterProvider, Counter, Histogram
│   ├── session.py                  # SessionRecorder
│   ├── cost.py                     # CostCalculator
│   ├── exporters/
│   │   ├── sqlite.py               # SQLiteExporter
│   │   ├── otlp.py                 # OTLPExporter
│   │   └── console.py              # ConsoleExporter
│   └── mixin.py                    # ObservableMixin
│
├── security/                       # SECURITY (NEW)
│   ├── __init__.py
│   ├── vault.py                    # SecretManager, EncryptedVault
│   ├── permissions.py              # PermissionEnforcer, Permission
│   ├── audit.py                    # AuditLogger, AuditEvent
│   ├── governor.py                 # ResourceGovernor
│   ├── scanner.py                  # SecretScanner
│   └── mixin.py                    # SecureMixin
│
├── memory/                         # PERSISTENT MEMORY (NEW)
│   ├── __init__.py
│   ├── working.py                  # WorkingMemory
│   ├── episodic.py                 # EpisodicMemory
│   ├── semantic.py                 # SemanticMemory
│   ├── universal.py                # UniversalKnowledgeDB
│   ├── manager.py                  # PersistentMemory (facade)
│   └── mixin.py                    # MemoryMixin
│
├── learning/                       # LEARNING (NEW)
│   ├── __init__.py
│   ├── loop.py                     # LearningLoop
│   ├── feedback.py                 # FeedbackCollector
│   ├── outcomes.py                 # OutcomeTracker
│   ├── consolidation.py            # KnowledgeConsolidator
│   └── mixin.py                    # LearningMixin
│
├── manifest/                       # MANIFEST (NEW)
│   ├── __init__.py
│   ├── project.py                  # ProjectManifest
│   ├── tracker.py                  # DependencyTracker
│   └── mixin.py                    # ManifestToolsMixin
│
├── workflow/                       # WORKFLOW (NEW)
│   ├── __init__.py
│   ├── engine.py                   # WorkflowEngine
│   ├── triggers.py                 # CronTrigger, WebhookTrigger, etc.
│   ├── steps.py                    # AgentStep, FunctionStep, etc.
│   ├── state.py                    # StateManager, WorkflowState
│   ├── scheduler.py                # WorkflowScheduler
│   └── dsl.py                      # YAML/Python DSL parser
│
├── email_integration/              # EMAIL (NEW)
│   ├── __init__.py
│   ├── providers/
│   │   ├── gmail.py                # GmailProvider
│   │   ├── msgraph.py              # MSGraphProvider
│   │   └── imap.py                 # IMAPProvider
│   ├── classifier.py               # EmailClassifier
│   ├── drafter.py                  # ResponseDrafter
│   ├── calendar.py                 # CalendarBridge
│   ├── thread.py                   # ThreadTracker
│   └── guard.py                    # SendGuard, ContentFilter
│
├── computer_use/                   # COMPUTER USE (NEW)
│   ├── __init__.py
│   ├── controller.py               # ComputerUseController
│   ├── vision.py                   # ScreenCaptureEngine, VLMAnalyzer
│   ├── input.py                    # MouseController, KeyboardController
│   ├── browser.py                  # BrowserAutomation (Playwright)
│   ├── desktop.py                  # DesktopAutomation
│   └── safety.py                   # SafetyGuard, ActionValidator
│
├── multidoc/                       # MULTI-DOCUMENT (NEW)
│   ├── __init__.py
│   ├── ingestion.py                # IngestionPipeline
│   ├── store.py                    # DocumentStore
│   ├── entity.py                   # EntityResolver
│   ├── contradiction.py            # ContradictionDetector
│   ├── synthesis.py                # SynthesisEngine
│   ├── citation.py                 # CitationTracker
│   └── tables.py                   # TableExtractor, TableComparator
│
└── shared/                         # SHARED INFRASTRUCTURE (NEW)
    ├── __init__.py
    ├── database.py                 # get_connection, SchemaMigrator
    ├── config.py                   # GaiaConfig
    ├── async_bridge.py             # run_async, sync-async bridge
    └── types.py                    # Shared type definitions
```

---

## Summary

The 18 GAIA architecture documents form a coherent, layered system:

1. **Tier 1 (Infrastructure)**: Error Recovery, Observability, and Security are cross-cutting concerns that wrap every operation. They make any agent production-ready.

2. **Tier 2 (Intelligence)**: Persistent Memory is the foundation. Learning, Tools, and Manifest build on it to create agents that remember, improve, and track their work.

3. **Tier 3 (Orchestration)**: Task-Centric Interface, Adaptive Prompts, and Workflow Orchestration coordinate how work flows through the system.

4. **Tier 4 (Domain)**: Computer Use, Email, and Multi-Document Synthesis are domain-specific capabilities that compose with all lower tiers.

**Key principles**:
- Each architecture is a **mixin** -- add it to any Agent subclass
- Each architecture uses its own **SQLite database** -- no contention
- Each architecture is **opt-in** -- disabled by default, zero overhead when off
- The **dependency graph is acyclic** -- implement bottom-up
- The **migration path is incremental** -- add one architecture at a time

**Start with**: Error Recovery + Observability + Security (immediate production value).
**Build toward**: the full 18-architecture stack for autonomous, self-healing, observable, secure agents.

---

*Integration Architecture for GAIA's 18 Architectural Frameworks.*
*All frameworks compose together to enable production-grade autonomous AI agents.*
