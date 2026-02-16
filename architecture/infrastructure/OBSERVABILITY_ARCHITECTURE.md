# Observability Architecture for GAIA

**Date**: February 7, 2026
**Version**: 1.0
**Status**: Specification
**Priority**: CRITICAL
**Estimated Effort**: 10-12 weeks (2 engineers)
**Target**: Enable distributed tracing, metrics collection, time-travel debugging, and cost tracking

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Problem Statement](#problem-statement)
3. [Architecture Overview](#architecture-overview)
4. [Component Specifications](#component-specifications)
5. [Data Models and Schemas](#data-models-and-schemas)
6. [Distributed Tracing](#distributed-tracing)
7. [Metrics Collection](#metrics-collection)
8. [Time-Travel Debugging](#time-travel-debugging)
9. [Cost Tracking](#cost-tracking)
10. [Integration with GAIA](#integration-with-gaia)
11. [Safety and Security Considerations](#safety-and-security-considerations)
12. [Implementation Plan](#implementation-plan)
13. [Testing Strategy](#testing-strategy)
14. [Success Metrics](#success-metrics)
15. [Complete Code](#complete-code)

---

## Executive Summary

### The Gap

GAIA currently has:
- Basic Python logging via `gaia.logger`
- Console output via `AgentConsole`
- Simple token/timing stats in `ChatSDK`

GAIA **lacks**:
- Distributed tracing across agent-to-agent calls
- Structured metrics collection and aggregation
- Time-travel debugging to replay past agent sessions
- Cost tracking for LLM token usage and API calls
- Performance profiling of tool execution pipelines
- Anomaly detection and alerting
- Dashboard visualization of system health

### The Solution

A comprehensive **Observability Architecture** that provides:
- OpenTelemetry-compatible distributed tracing across all GAIA components
- Structured metrics (counters, gauges, histograms) with dimensional labels
- Session recording and time-travel replay for debugging
- Per-query, per-agent, and per-user cost tracking
- Real-time dashboards and alerting
- Zero-overhead when disabled (sampling-based)

### Impact

**Unlocks entire category**: Production observability for:
- Multi-agent pipeline debugging (trace across agent hops)
- Performance optimization (identify bottlenecks)
- Cost management (budget enforcement, usage reports)
- Reliability engineering (SLO tracking, error budgets)
- User experience analysis (latency percentiles)

**Before**: 10% observability coverage (basic logging only)
**After**: 90% observability coverage

---

## Problem Statement

### Current Limitations

**Example scenario**: User reports slow response from CodeAgent. Developer needs to investigate.

**What debugging looks like today**:
```python
# Developer must manually add print statements
# No structured tracing, no timing breakdown
agent = CodeAgent(debug=True, show_prompts=True)
result = agent.process_query("Generate a REST API")
# Output: wall of unstructured text
# No way to see: LLM latency vs tool execution vs network time
```

**What debugging should look like**:
```
Trace ID: abc-123
  |-- CodeAgent.process_query (total: 8.2s)
      |-- LLM: plan generation (1.2s, 450 tokens in, 380 tokens out)
      |-- Tool: write_code (0.3s)
      |-- LLM: validation prompt (2.1s, 1200 tokens in, 890 tokens out)
      |-- Tool: run_tests (4.1s)
      |-- LLM: final summary (0.5s, 200 tokens in, 150 tokens out)

Cost: $0.0034 (1850 input tokens + 1420 output tokens @ Qwen3-Coder-30B)
```

### User Stories

**Story 1: Performance Debugging**
```
As a developer, I want to:
- See a trace of every step in an agent's execution
- Identify which step took the most time
- Compare traces across similar queries
- Set up alerts when latency exceeds thresholds
So that I can optimize agent performance
```

**Story 2: Cost Management**
```
As a team lead, I want to:
- Track LLM token usage per agent, per user, per query
- Set budget limits and get alerts before exceeding them
- Generate monthly cost reports
- Compare local vs cloud LLM costs
So that I can manage AI spending effectively
```

**Story 3: Time-Travel Debugging**
```
As a developer, I want to:
- Record complete agent sessions (prompts, responses, tool calls)
- Replay any past session step by step
- Compare "before" and "after" sessions after code changes
- Share session recordings with teammates
So that I can reproduce and fix complex bugs
```

**Story 4: Production Monitoring**
```
As an ops engineer, I want to:
- Monitor agent health in real-time (latency, errors, throughput)
- Set up SLOs (e.g., 95th percentile latency < 5s)
- Get alerted when error rates spike
- View dashboards showing system-wide metrics
So that I can maintain service reliability
```

---

## Architecture Overview

### High-Level Design

```
+------------------------------------------------------------------+
|                     GAIA Application Layer                        |
|  [ChatAgent]  [CodeAgent]  [JiraAgent]  [EmailAgent]  [API]     |
+------+------------+------------+-----------+----------+----------+
       |            |            |           |          |
       v            v            v           v          v
+------------------------------------------------------------------+
|                  Observability SDK (Auto-Instrumented)            |
|                                                                   |
|  +-------------------+  +------------------+  +----------------+ |
|  | Trace Collector   |  | Metrics Recorder |  | Session        | |
|  |                   |  |                  |  | Recorder       | |
|  | - Span creation   |  | - Counters       |  | - Prompt log   | |
|  | - Context prop.   |  | - Gauges         |  | - Response log | |
|  | - Sampling        |  | - Histograms     |  | - Tool calls   | |
|  | - Baggage         |  | - Labels         |  | - State changes| |
|  +--------+----------+  +--------+---------+  +--------+-------+ |
|           |                      |                      |         |
|           v                      v                      v         |
|  +-----------------------------------------------------------+   |
|  |              Telemetry Pipeline                             |   |
|  |                                                             |   |
|  |  [Buffer] --> [Process] --> [Export]                        |   |
|  |                                                             |   |
|  |  Exporters:                                                 |   |
|  |  - SQLite (local, default)                                  |   |
|  |  - OTLP (OpenTelemetry Protocol)                           |   |
|  |  - JSON file                                                |   |
|  |  - Console (debug)                                          |   |
|  +-----------------------------------------------------------+   |
+------------------------------------------------------------------+
           |                      |                      |
           v                      v                      v
+------------------+  +-------------------+  +-------------------+
| Trace Storage    |  | Metrics Storage   |  | Session Storage   |
|                  |  |                   |  |                   |
| SQLite / OTLP    |  | SQLite / Prom.   |  | SQLite + Files    |
+------------------+  +-------------------+  +-------------------+
           |                      |                      |
           +----------------------+----------------------+
                                  |
                                  v
                    +----------------------------+
                    |     Query & Dashboard      |
                    |                            |
                    | - gaia observe traces      |
                    | - gaia observe metrics     |
                    | - gaia observe replay      |
                    | - gaia observe costs       |
                    | - Web dashboard (optional) |
                    +----------------------------+
```

### Component Layers

| Layer | Components | Responsibility |
|-------|-----------|----------------|
| **Instrumentation** | TracerProvider, MeterProvider, SessionRecorder | Auto-instrument GAIA components |
| **Collection** | SpanProcessor, MetricReader, SessionBuffer | Buffer and batch telemetry |
| **Export** | SQLiteExporter, OTLPExporter, ConsoleExporter | Persist telemetry data |
| **Storage** | TraceStore, MetricStore, SessionStore | Query-optimized storage |
| **Query** | TraceQuery, MetricQuery, CostCalculator | Analysis and reporting |
| **Presentation** | CLI commands, web dashboard, alerts | User-facing observability |

---

## Component Specifications

### 1. Core Telemetry Types

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Core observability types for GAIA.
"""

import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Generator, List, Optional

from gaia.logger import get_logger

log = get_logger(__name__)


class SpanKind(Enum):
    """Type of traced operation."""
    INTERNAL = "internal"
    AGENT = "agent"
    LLM = "llm"
    TOOL = "tool"
    HTTP = "http"
    DATABASE = "database"


class SpanStatus(Enum):
    """Span completion status."""
    OK = "ok"
    ERROR = "error"
    TIMEOUT = "timeout"


class MetricKind(Enum):
    """Type of metric."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"


@dataclass
class SpanContext:
    """Distributed trace context."""
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    parent_span_id: Optional[str] = None
    baggage: Dict[str, str] = field(default_factory=dict)

    def child(self) -> "SpanContext":
        """Create a child context."""
        return SpanContext(
            trace_id=self.trace_id,
            span_id=uuid.uuid4().hex[:16],
            parent_span_id=self.span_id,
            baggage=dict(self.baggage),
        )


@dataclass
class Span:
    """A single trace span representing an operation."""
    context: SpanContext
    name: str
    kind: SpanKind
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    status: SpanStatus = SpanStatus.OK
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def duration_ms(self) -> float:
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return 0.0

    @property
    def trace_id(self) -> str:
        return self.context.trace_id

    @property
    def span_id(self) -> str:
        return self.context.span_id

    @property
    def parent_span_id(self) -> Optional[str]:
        return self.context.parent_span_id

    def set_attribute(self, key: str, value: Any) -> None:
        """Set a span attribute."""
        self.attributes[key] = value

    def add_event(self, name: str, attributes: Optional[Dict] = None) -> None:
        """Add a timestamped event to the span."""
        self.events.append({
            "name": name,
            "timestamp": time.time(),
            "attributes": attributes or {},
        })

    def set_error(self, error: str) -> None:
        """Mark span as error."""
        self.status = SpanStatus.ERROR
        self.error = error
        self.add_event("exception", {"message": error})

    def end(self) -> None:
        """End the span."""
        self.end_time = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize span to dictionary."""
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "name": self.name,
            "kind": self.kind.value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "status": self.status.value,
            "attributes": self.attributes,
            "events": self.events,
            "error": self.error,
        }


@dataclass
class MetricDataPoint:
    """A single metric measurement."""
    name: str
    kind: MetricKind
    value: float
    timestamp: float = field(default_factory=time.time)
    labels: Dict[str, str] = field(default_factory=dict)
    unit: str = ""


@dataclass
class SessionEvent:
    """A recorded event in an agent session."""
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    event_type: str = ""  # prompt, response, tool_call, tool_result, state_change, error
    timestamp: float = field(default_factory=time.time)
    data: Dict[str, Any] = field(default_factory=dict)
    span_id: Optional[str] = None


@dataclass
class SessionRecording:
    """Complete recording of an agent session."""
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    agent_type: str = ""
    query: str = ""
    started_at: float = field(default_factory=time.time)
    ended_at: Optional[float] = None
    events: List[SessionEvent] = field(default_factory=list)
    trace_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    total_cost: float = 0.0
    final_result: Optional[str] = None

    @property
    def duration_seconds(self) -> float:
        if self.ended_at:
            return self.ended_at - self.started_at
        return 0.0

    def add_event(self, event_type: str, data: Dict[str, Any], span_id: Optional[str] = None) -> None:
        """Add an event to the recording."""
        self.events.append(
            SessionEvent(
                event_type=event_type,
                data=data,
                span_id=span_id,
            )
        )
```

---

## Distributed Tracing

### 2.1 Tracer Provider

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Distributed tracing system for GAIA.
"""

import contextvars
import threading
from contextlib import contextmanager
from typing import Any, Callable, Dict, Generator, List, Optional

from gaia.logger import get_logger

log = get_logger(__name__)


# Context variable for current span context (works with both threads and async)
_context_var: contextvars.ContextVar[Optional[SpanContext]] = contextvars.ContextVar(
    "gaia_span_context", default=None
)


def get_current_context() -> Optional[SpanContext]:
    """Get the current span context (async-safe via contextvars)."""
    return _context_var.get()


def set_current_context(context: Optional[SpanContext]) -> Optional[contextvars.Token]:
    """Set the current span context (async-safe via contextvars).

    Returns a token that can be used to reset to the previous value.
    """
    return _context_var.set(context)


class TracerProvider:
    """
    Central tracer provider for GAIA.

    Creates tracers and manages span lifecycle.
    Thread-safe and supports both sync and async code.

    Usage:
        tracer = TracerProvider.get_tracer("gaia.agents.code")

        with tracer.start_span("process_query", kind=SpanKind.AGENT) as span:
            span.set_attribute("query", user_query)
            result = do_work()
            span.set_attribute("result_length", len(result))
    """

    _instance: Optional["TracerProvider"] = None
    _lock = threading.Lock()

    def __init__(
        self,
        service_name: str = "gaia",
        sample_rate: float = 1.0,
        exporters: Optional[List["SpanExporter"]] = None,
        enabled: bool = True,
    ):
        self.service_name = service_name
        self.sample_rate = sample_rate
        self.exporters = exporters or []
        self.enabled = enabled
        self._tracers: Dict[str, "Tracer"] = {}
        self._span_buffer: List[Span] = []
        self._buffer_lock = threading.Lock()

    @classmethod
    def initialize(cls, **kwargs) -> "TracerProvider":
        """Initialize the global tracer provider (singleton)."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(**kwargs)
                log.info("TracerProvider initialized")
            return cls._instance

    @classmethod
    def get_tracer(cls, name: str) -> "Tracer":
        """Get a named tracer instance."""
        if cls._instance is None:
            cls.initialize()
        provider = cls._instance

        if name not in provider._tracers:
            provider._tracers[name] = Tracer(name, provider)

        return provider._tracers[name]

    def _should_sample(self) -> bool:
        """Determine if this trace should be sampled."""
        if not self.enabled:
            return False
        if self.sample_rate >= 1.0:
            return True
        import random
        return random.random() < self.sample_rate

    def _on_span_end(self, span: Span) -> None:
        """Called when a span ends. Buffers for export."""
        with self._buffer_lock:
            self._span_buffer.append(span)

            # Flush when buffer is large enough
            if len(self._span_buffer) >= 100:
                self._flush()

    def _flush(self) -> None:
        """Export buffered spans."""
        if not self._span_buffer:
            return

        spans = self._span_buffer.copy()
        self._span_buffer.clear()

        for exporter in self.exporters:
            try:
                exporter.export(spans)
            except Exception as e:
                log.error(f"Span export failed: {e}")

    def shutdown(self) -> None:
        """Flush remaining spans and shutdown."""
        with self._buffer_lock:
            self._flush()
        for exporter in self.exporters:
            try:
                exporter.shutdown()
            except Exception:
                pass


class Tracer:
    """
    Named tracer for creating spans within a component.

    Each GAIA component (agent, LLM client, tool) gets its own tracer.
    """

    def __init__(self, name: str, provider: TracerProvider):
        self.name = name
        self.provider = provider

    @contextmanager
    def start_span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Generator[Span, None, None]:
        """
        Start a new span as a context manager.

        Automatically:
        - Creates parent-child relationship
        - Sets start/end times
        - Propagates context
        - Exports on completion

        Args:
            name: Span name (e.g., "process_query", "llm_call")
            kind: Type of operation
            attributes: Initial attributes

        Yields:
            Span object for adding attributes/events
        """
        if not self.provider._should_sample():
            # Return a no-op span
            yield Span(
                context=SpanContext(),
                name=name,
                kind=kind,
            )
            return

        # Get or create parent context
        parent_context = get_current_context()
        if parent_context:
            context = parent_context.child()
        else:
            context = SpanContext()

        span = Span(
            context=context,
            name=f"{self.name}.{name}",
            kind=kind,
            attributes=attributes or {},
        )

        # Set as current context; save token for proper reset
        token = set_current_context(context)

        try:
            yield span
        except Exception as e:
            span.set_error(str(e))
            raise
        finally:
            span.end()
            _context_var.reset(token)
            self.provider._on_span_end(span)


class NoOpTracer:
    """No-op tracer for when observability is disabled."""

    @contextmanager
    def start_span(self, name: str, **kwargs):
        yield Span(
            context=SpanContext(),
            name=name,
            kind=SpanKind.INTERNAL,
        )
```

### 2.2 Span Exporters

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Span exporters for persisting trace data.
"""

import abc
import json
import sqlite3
import threading
from typing import List

from gaia.logger import get_logger

log = get_logger(__name__)


class SpanExporter(abc.ABC):
    """Base class for span exporters."""

    @abc.abstractmethod
    def export(self, spans: List[Span]) -> None:
        """Export a batch of spans."""
        pass

    def shutdown(self) -> None:
        """Cleanup resources."""
        pass


class SQLiteConnectionPool:
    """
    Thread-safe SQLite connection pool.

    Reuses connections per thread to avoid repeated open/close overhead.
    Connections are stored in a thread-local to respect SQLite's
    threading constraints.
    """

    def __init__(self, db_path: str, max_idle_seconds: float = 300.0):
        self.db_path = db_path
        self.max_idle_seconds = max_idle_seconds
        self._local = threading.local()
        self._lock = threading.Lock()

    def get_connection(self) -> sqlite3.Connection:
        """Get a connection for the current thread (reused if available)."""
        conn = getattr(self._local, "conn", None)
        last_used = getattr(self._local, "last_used", 0)

        # Recycle stale connections
        if conn is not None and (time.time() - last_used) > self.max_idle_seconds:
            try:
                conn.close()
            except Exception:
                pass
            conn = None

        if conn is None:
            import time as _time
            conn = sqlite3.connect(self.db_path, timeout=10)
            conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn = conn

        self._local.last_used = time.time()
        return conn

    def close_all(self) -> None:
        """Close the connection for the current thread."""
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
            self._local.conn = None


class SQLiteSpanExporter(SpanExporter):
    """
    Export spans to SQLite for local query and analysis.

    Default exporter for development and single-machine deployment.
    Uses connection pooling to avoid repeated open/close overhead.
    """

    def __init__(self, db_path: str = "gaia_traces.db"):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._pool = SQLiteConnectionPool(db_path)
        self._init_db()

    def _init_db(self) -> None:
        """Create trace tables if they don't exist."""
        conn = self._pool.get_connection()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS spans (
                trace_id TEXT NOT NULL,
                span_id TEXT NOT NULL,
                parent_span_id TEXT,
                name TEXT NOT NULL,
                kind TEXT NOT NULL,
                start_time REAL NOT NULL,
                end_time REAL,
                duration_ms REAL,
                status TEXT NOT NULL,
                attributes TEXT,
                events TEXT,
                error TEXT,
                PRIMARY KEY (trace_id, span_id)
            );

            CREATE INDEX IF NOT EXISTS idx_spans_trace ON spans(trace_id);
            CREATE INDEX IF NOT EXISTS idx_spans_name ON spans(name);
            CREATE INDEX IF NOT EXISTS idx_spans_start ON spans(start_time);
            CREATE INDEX IF NOT EXISTS idx_spans_status ON spans(status);
        """)
        conn.commit()

    def export(self, spans: List[Span]) -> None:
        """Export spans to SQLite."""
        with self._lock:
            conn = self._pool.get_connection()
            cursor = conn.cursor()

            for span in spans:
                cursor.execute("""
                    INSERT OR REPLACE INTO spans
                    (trace_id, span_id, parent_span_id, name, kind,
                     start_time, end_time, duration_ms, status,
                     attributes, events, error)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    span.trace_id,
                    span.span_id,
                    span.parent_span_id,
                    span.name,
                    span.kind.value,
                    span.start_time,
                    span.end_time,
                    span.duration_ms,
                    span.status.value,
                    json.dumps(span.attributes, default=str),
                    json.dumps(span.events, default=str),
                    span.error,
                ))

            conn.commit()

    def shutdown(self) -> None:
        """Close pooled connections."""
        self._pool.close_all()


class ConsoleSpanExporter(SpanExporter):
    """Print spans to console for debugging."""

    def export(self, spans: List[Span]) -> None:
        for span in spans:
            indent = "  " if span.parent_span_id else ""
            status_icon = "OK" if span.status == SpanStatus.OK else "ERR"
            print(
                f"{indent}[{status_icon}] {span.name} "
                f"({span.duration_ms:.1f}ms) "
                f"{json.dumps(span.attributes, default=str)}"
            )


class JSONFileSpanExporter(SpanExporter):
    """Export spans to a JSON file for offline analysis."""

    def __init__(self, file_path: str = "gaia_traces.jsonl"):
        self.file_path = file_path
        self._lock = threading.Lock()

    def export(self, spans: List[Span]) -> None:
        with self._lock:
            with open(self.file_path, "a") as f:
                for span in spans:
                    f.write(json.dumps(span.to_dict(), default=str) + "\n")
```

---

## Metrics Collection

### 3.1 Meter Provider

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Metrics collection system for GAIA.
"""

import threading
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

from gaia.logger import get_logger

log = get_logger(__name__)


class Counter:
    """Monotonically increasing counter metric."""

    def __init__(self, name: str, description: str = "", unit: str = ""):
        self.name = name
        self.description = description
        self.unit = unit
        self._values: Dict[Tuple, float] = defaultdict(float)
        self._lock = threading.Lock()

    def add(self, value: float = 1, labels: Optional[Dict[str, str]] = None) -> None:
        """Increment counter."""
        key = tuple(sorted((labels or {}).items()))
        with self._lock:
            self._values[key] += value

    def get(self, labels: Optional[Dict[str, str]] = None) -> float:
        """Get current counter value."""
        key = tuple(sorted((labels or {}).items()))
        return self._values.get(key, 0.0)

    def collect(self) -> List[MetricDataPoint]:
        """Collect all data points."""
        points = []
        with self._lock:
            for key, value in self._values.items():
                points.append(
                    MetricDataPoint(
                        name=self.name,
                        kind=MetricKind.COUNTER,
                        value=value,
                        labels=dict(key),
                        unit=self.unit,
                    )
                )
        return points


class Gauge:
    """Point-in-time gauge metric."""

    def __init__(self, name: str, description: str = "", unit: str = ""):
        self.name = name
        self.description = description
        self.unit = unit
        self._values: Dict[Tuple, float] = {}
        self._lock = threading.Lock()

    def set(self, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Set gauge value."""
        key = tuple(sorted((labels or {}).items()))
        with self._lock:
            self._values[key] = value

    def get(self, labels: Optional[Dict[str, str]] = None) -> float:
        """Get current gauge value."""
        key = tuple(sorted((labels or {}).items()))
        return self._values.get(key, 0.0)

    def collect(self) -> List[MetricDataPoint]:
        """Collect all data points."""
        points = []
        with self._lock:
            for key, value in self._values.items():
                points.append(
                    MetricDataPoint(
                        name=self.name,
                        kind=MetricKind.GAUGE,
                        value=value,
                        labels=dict(key),
                        unit=self.unit,
                    )
                )
        return points


class Histogram:
    """Distribution metric with configurable buckets.

    Uses a rolling window to prevent unbounded memory growth.
    Only the most recent `max_observations` values are retained per label set.
    """

    DEFAULT_BUCKETS = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0]
    DEFAULT_MAX_OBSERVATIONS = 10_000

    def __init__(
        self,
        name: str,
        description: str = "",
        unit: str = "",
        buckets: Optional[List[float]] = None,
        max_observations: int = DEFAULT_MAX_OBSERVATIONS,
    ):
        self.name = name
        self.description = description
        self.unit = unit
        self.buckets = sorted(buckets or self.DEFAULT_BUCKETS)
        self.max_observations = max_observations
        self._observations: Dict[Tuple, List[float]] = defaultdict(list)
        # Running counters preserved across evictions for accurate totals
        self._total_count: Dict[Tuple, int] = defaultdict(int)
        self._total_sum: Dict[Tuple, float] = defaultdict(float)
        self._lock = threading.Lock()

    def observe(self, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Record an observation. Evicts oldest values when window is full."""
        key = tuple(sorted((labels or {}).items()))
        with self._lock:
            obs = self._observations[key]
            obs.append(value)
            self._total_count[key] += 1
            self._total_sum[key] += value
            # Evict oldest observations when exceeding the rolling window
            if len(obs) > self.max_observations:
                self._observations[key] = obs[len(obs) - self.max_observations:]

    def get_statistics(
        self, labels: Optional[Dict[str, str]] = None
    ) -> Dict[str, float]:
        """Get statistical summary (percentiles are over the rolling window)."""
        key = tuple(sorted((labels or {}).items()))
        values = self._observations.get(key, [])
        if not values:
            return {"count": 0, "sum": 0, "min": 0, "max": 0, "mean": 0, "p50": 0, "p95": 0, "p99": 0}

        sorted_vals = sorted(values)
        n = len(sorted_vals)
        total_count = self._total_count.get(key, n)
        total_sum = self._total_sum.get(key, sum(sorted_vals))
        return {
            "count": total_count,
            "sum": total_sum,
            "min": sorted_vals[0],
            "max": sorted_vals[-1],
            "mean": total_sum / total_count if total_count else 0,
            "p50": sorted_vals[int(n * 0.50)],
            "p95": sorted_vals[min(int(n * 0.95), n - 1)],
            "p99": sorted_vals[min(int(n * 0.99), n - 1)],
        }

    def collect(self) -> List[MetricDataPoint]:
        """Collect histogram as summary statistics."""
        points = []
        with self._lock:
            for key, values in self._observations.items():
                if values:
                    stats = self.get_statistics(dict(key) if key else None)
                    for stat_name, stat_value in stats.items():
                        labels = dict(key)
                        labels["statistic"] = stat_name
                        points.append(
                            MetricDataPoint(
                                name=f"{self.name}_{stat_name}",
                                kind=MetricKind.HISTOGRAM,
                                value=stat_value,
                                labels=labels,
                                unit=self.unit,
                            )
                        )
        return points


class MeterProvider:
    """
    Central provider for metrics instruments.

    Usage:
        meter = MeterProvider.get_meter("gaia.llm")
        tokens_counter = meter.create_counter("llm_tokens_total", unit="tokens")
        latency_hist = meter.create_histogram("llm_latency_seconds", unit="s")

        tokens_counter.add(150, labels={"model": "Qwen3", "direction": "input"})
        latency_hist.observe(1.23, labels={"model": "Qwen3"})
    """

    _instance: Optional["MeterProvider"] = None
    _lock = threading.Lock()

    def __init__(self, export_interval_seconds: float = 60.0):
        self.export_interval = export_interval_seconds
        self._meters: Dict[str, "Meter"] = {}
        self._exporters: List["MetricExporter"] = []
        self._running = False
        self._export_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    @classmethod
    def initialize(cls, **kwargs) -> "MeterProvider":
        """Initialize the global meter provider."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(**kwargs)
            return cls._instance

    @classmethod
    def get_meter(cls, name: str) -> "Meter":
        """Get a named meter instance."""
        if cls._instance is None:
            cls.initialize()
        provider = cls._instance

        if name not in provider._meters:
            provider._meters[name] = Meter(name)

        return provider._meters[name]

    def add_exporter(self, exporter: "MetricExporter") -> None:
        """Add a metric exporter."""
        self._exporters.append(exporter)
        # Auto-start background export thread when first exporter is added
        if not self._running and self._exporters:
            self.start_export_thread()

    def start_export_thread(self) -> None:
        """Start background thread that periodically exports collected metrics."""
        if self._running:
            return
        self._running = True
        self._stop_event.clear()
        self._export_thread = threading.Thread(
            target=self._export_loop,
            daemon=True,
            name="gaia-metrics-exporter",
        )
        self._export_thread.start()
        log.info(
            f"Metrics export thread started (interval={self.export_interval}s)"
        )

    def stop_export_thread(self) -> None:
        """Stop the background metrics export thread and flush remaining data."""
        self._running = False
        self._stop_event.set()
        if self._export_thread:
            self._export_thread.join(timeout=10)
        # Final flush
        self._export_metrics()
        log.info("Metrics export thread stopped")

    def _export_loop(self) -> None:
        """Background loop that exports metrics at the configured interval."""
        while self._running and not self._stop_event.is_set():
            self._stop_event.wait(timeout=self.export_interval)
            if self._running:
                self._export_metrics()

    def _export_metrics(self) -> None:
        """Collect and export all metrics to registered exporters."""
        try:
            points = self.collect_all()
            if not points:
                return
            for exporter in self._exporters:
                try:
                    exporter.export(points)
                except Exception as e:
                    log.error(f"Metric export failed: {e}")
        except Exception as e:
            log.error(f"Metric collection failed: {e}")

    def collect_all(self) -> List[MetricDataPoint]:
        """Collect all metrics from all meters."""
        points = []
        for meter in self._meters.values():
            points.extend(meter.collect_all())
        return points


class Meter:
    """Named meter for creating metric instruments."""

    def __init__(self, name: str):
        self.name = name
        self._counters: Dict[str, Counter] = {}
        self._gauges: Dict[str, Gauge] = {}
        self._histograms: Dict[str, Histogram] = {}

    def create_counter(
        self, name: str, description: str = "", unit: str = ""
    ) -> Counter:
        """Create or get a counter."""
        full_name = f"{self.name}.{name}"
        if full_name not in self._counters:
            self._counters[full_name] = Counter(full_name, description, unit)
        return self._counters[full_name]

    def create_gauge(
        self, name: str, description: str = "", unit: str = ""
    ) -> Gauge:
        """Create or get a gauge."""
        full_name = f"{self.name}.{name}"
        if full_name not in self._gauges:
            self._gauges[full_name] = Gauge(full_name, description, unit)
        return self._gauges[full_name]

    def create_histogram(
        self, name: str, description: str = "", unit: str = "", buckets: List[float] = None
    ) -> Histogram:
        """Create or get a histogram."""
        full_name = f"{self.name}.{name}"
        if full_name not in self._histograms:
            self._histograms[full_name] = Histogram(full_name, description, unit, buckets)
        return self._histograms[full_name]

    def collect_all(self) -> List[MetricDataPoint]:
        """Collect all metrics from this meter."""
        points = []
        for c in self._counters.values():
            points.extend(c.collect())
        for g in self._gauges.values():
            points.extend(g.collect())
        for h in self._histograms.values():
            points.extend(h.collect())
        return points
```

### 3.2 Pre-Defined GAIA Metrics

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Pre-defined metrics for GAIA components.
"""


class GAIAMetrics:
    """Standard metrics used across GAIA."""

    def __init__(self):
        # LLM metrics
        llm_meter = MeterProvider.get_meter("gaia.llm")
        self.llm_requests = llm_meter.create_counter(
            "requests_total", "Total LLM requests"
        )
        self.llm_tokens_in = llm_meter.create_counter(
            "tokens_input_total", "Total input tokens", "tokens"
        )
        self.llm_tokens_out = llm_meter.create_counter(
            "tokens_output_total", "Total output tokens", "tokens"
        )
        self.llm_latency = llm_meter.create_histogram(
            "latency_seconds", "LLM request latency", "s"
        )
        self.llm_errors = llm_meter.create_counter(
            "errors_total", "Total LLM errors"
        )
        self.llm_tokens_per_second = llm_meter.create_gauge(
            "tokens_per_second", "Current generation speed", "tokens/s"
        )

        # Agent metrics
        agent_meter = MeterProvider.get_meter("gaia.agent")
        self.agent_queries = agent_meter.create_counter(
            "queries_total", "Total agent queries"
        )
        self.agent_steps = agent_meter.create_counter(
            "steps_total", "Total agent steps executed"
        )
        self.agent_duration = agent_meter.create_histogram(
            "query_duration_seconds", "Agent query duration", "s"
        )
        self.agent_tool_calls = agent_meter.create_counter(
            "tool_calls_total", "Total tool calls"
        )
        self.agent_errors = agent_meter.create_counter(
            "errors_total", "Total agent errors"
        )
        self.active_agents = agent_meter.create_gauge(
            "active_count", "Currently active agents"
        )

        # Tool metrics
        tool_meter = MeterProvider.get_meter("gaia.tool")
        self.tool_executions = tool_meter.create_counter(
            "executions_total", "Total tool executions"
        )
        self.tool_duration = tool_meter.create_histogram(
            "duration_seconds", "Tool execution duration", "s"
        )
        self.tool_errors = tool_meter.create_counter(
            "errors_total", "Total tool errors"
        )

        # RAG metrics
        rag_meter = MeterProvider.get_meter("gaia.rag")
        self.rag_queries = rag_meter.create_counter(
            "queries_total", "Total RAG queries"
        )
        self.rag_chunks_retrieved = rag_meter.create_histogram(
            "chunks_retrieved", "Chunks retrieved per query"
        )
        self.rag_relevance_score = rag_meter.create_histogram(
            "relevance_score", "RAG result relevance score"
        )


# Global metrics instance
_metrics: Optional[GAIAMetrics] = None


def get_metrics() -> GAIAMetrics:
    """Get the global GAIA metrics instance."""
    global _metrics
    if _metrics is None:
        _metrics = GAIAMetrics()
    return _metrics
```

---

## Time-Travel Debugging

### 4.1 Session Recorder

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Session recording and time-travel replay for agent debugging.
"""

import json
import sqlite3
import threading
from typing import Any, Dict, List, Optional

from gaia.logger import get_logger

log = get_logger(__name__)


class SessionRecorder:
    """
    Records complete agent sessions for later replay and debugging.

    Records:
    - User queries
    - System prompts
    - LLM prompts and responses
    - Tool calls and results
    - Agent state transitions
    - Error events
    - Timing information

    Usage:
        recorder = SessionRecorder()
        session = recorder.start_session("code_agent", "Write a REST API")

        # During agent execution, events are recorded automatically
        recorder.record_event(session.session_id, "prompt", {"text": "..."})
        recorder.record_event(session.session_id, "response", {"text": "..."})

        recorder.end_session(session.session_id, result="API code generated")
    """

    def __init__(self, db_path: str = "gaia_sessions.db"):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._active_sessions: Dict[str, SessionRecording] = {}
        self._init_db()

    def _init_db(self) -> None:
        """Initialize session storage."""
        conn = sqlite3.connect(self.db_path)
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                agent_type TEXT NOT NULL,
                query TEXT,
                started_at REAL NOT NULL,
                ended_at REAL,
                trace_id TEXT,
                total_tokens_in INTEGER DEFAULT 0,
                total_tokens_out INTEGER DEFAULT 0,
                total_cost REAL DEFAULT 0.0,
                final_result TEXT,
                metadata TEXT
            );

            CREATE TABLE IF NOT EXISTS session_events (
                event_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                timestamp REAL NOT NULL,
                data TEXT,
                span_id TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            );

            CREATE INDEX IF NOT EXISTS idx_events_session ON session_events(session_id);
            CREATE INDEX IF NOT EXISTS idx_events_type ON session_events(event_type);
            CREATE INDEX IF NOT EXISTS idx_sessions_agent ON sessions(agent_type);
            CREATE INDEX IF NOT EXISTS idx_sessions_time ON sessions(started_at);
        """)
        conn.commit()
        conn.close()

    def start_session(
        self,
        agent_type: str,
        query: str,
        trace_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SessionRecording:
        """Start recording a new session."""
        session = SessionRecording(
            agent_type=agent_type,
            query=query,
            trace_id=trace_id,
            metadata=metadata or {},
        )
        self._active_sessions[session.session_id] = session
        log.debug(f"Session recording started: {session.session_id}")
        return session

    def record_event(
        self,
        session_id: str,
        event_type: str,
        data: Dict[str, Any],
        span_id: Optional[str] = None,
    ) -> None:
        """Record an event in an active session."""
        session = self._active_sessions.get(session_id)
        if session:
            session.add_event(event_type, data, span_id)

            # Track token usage
            if event_type == "llm_response":
                session.total_tokens_in += data.get("tokens_in", 0)
                session.total_tokens_out += data.get("tokens_out", 0)

    def end_session(
        self,
        session_id: str,
        result: Optional[str] = None,
    ) -> Optional[SessionRecording]:
        """End and persist a recording session."""
        import time

        session = self._active_sessions.pop(session_id, None)
        if not session:
            return None

        session.ended_at = time.time()
        session.final_result = result

        # Persist to database
        self._persist_session(session)

        log.debug(
            f"Session recording ended: {session_id} "
            f"({session.duration_seconds:.1f}s, {len(session.events)} events)"
        )
        return session

    def _persist_session(self, session: SessionRecording) -> None:
        """Write session to SQLite."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT OR REPLACE INTO sessions
                (session_id, agent_type, query, started_at, ended_at,
                 trace_id, total_tokens_in, total_tokens_out, total_cost,
                 final_result, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session.session_id,
                session.agent_type,
                session.query,
                session.started_at,
                session.ended_at,
                session.trace_id,
                session.total_tokens_in,
                session.total_tokens_out,
                session.total_cost,
                session.final_result,
                json.dumps(session.metadata, default=str),
            ))

            for event in session.events:
                cursor.execute("""
                    INSERT OR REPLACE INTO session_events
                    (event_id, session_id, event_type, timestamp, data, span_id)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    event.event_id,
                    session.session_id,
                    event.event_type,
                    event.timestamp,
                    json.dumps(event.data, default=str),
                    event.span_id,
                ))

            conn.commit()
            conn.close()

    def replay_session(self, session_id: str) -> Optional[SessionRecording]:
        """Load and return a complete session recording for replay."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
        row = cursor.fetchone()
        if not row:
            conn.close()
            return None

        session = SessionRecording(
            session_id=row[0],
            agent_type=row[1],
            query=row[2],
            started_at=row[3],
            ended_at=row[4],
            trace_id=row[5],
            total_tokens_in=row[6],
            total_tokens_out=row[7],
            total_cost=row[8],
            final_result=row[9],
            metadata=json.loads(row[10]) if row[10] else {},
        )

        cursor.execute(
            "SELECT * FROM session_events WHERE session_id = ? ORDER BY timestamp",
            (session_id,),
        )
        for event_row in cursor.fetchall():
            session.events.append(
                SessionEvent(
                    event_id=event_row[0],
                    event_type=event_row[2],
                    timestamp=event_row[3],
                    data=json.loads(event_row[4]) if event_row[4] else {},
                    span_id=event_row[5],
                )
            )

        conn.close()
        return session

    def list_sessions(
        self,
        agent_type: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """List recorded sessions."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = "SELECT session_id, agent_type, query, started_at, ended_at, total_tokens_in, total_tokens_out FROM sessions"
        params = []

        if agent_type:
            query += " WHERE agent_type = ?"
            params.append(agent_type)

        query += " ORDER BY started_at DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        return [
            {
                "session_id": r[0],
                "agent_type": r[1],
                "query": r[2][:100] if r[2] else "",
                "started_at": r[3],
                "ended_at": r[4],
                "tokens_in": r[5],
                "tokens_out": r[6],
                "duration_s": (r[4] - r[3]) if r[4] and r[3] else 0,
            }
            for r in rows
        ]
```

---

## Cost Tracking

### 5.1 Cost Calculator

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
LLM cost tracking and budget management.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from gaia.logger import get_logger

log = get_logger(__name__)


@dataclass
class ModelPricing:
    """Pricing for an LLM model (per 1M tokens)."""
    model_name: str
    input_cost_per_million: float  # USD
    output_cost_per_million: float  # USD
    is_local: bool = False  # Local models have zero API cost

    @property
    def input_cost_per_token(self) -> float:
        return self.input_cost_per_million / 1_000_000

    @property
    def output_cost_per_token(self) -> float:
        return self.output_cost_per_million / 1_000_000


# Pricing registry
MODEL_PRICING = {
    # Local models (AMD NPU/GPU) - zero API cost, compute cost only
    "Qwen3-Coder-30B-A3B-Instruct-GGUF": ModelPricing(
        model_name="Qwen3-Coder-30B-A3B-Instruct-GGUF",
        input_cost_per_million=0.0,
        output_cost_per_million=0.0,
        is_local=True,
    ),
    "Qwen3-0.6B-GGUF": ModelPricing(
        model_name="Qwen3-0.6B-GGUF",
        input_cost_per_million=0.0,
        output_cost_per_million=0.0,
        is_local=True,
    ),
    # Cloud models
    "claude-sonnet-4-20250514": ModelPricing(
        model_name="claude-sonnet-4-20250514",
        input_cost_per_million=3.00,
        output_cost_per_million=15.00,
    ),
    "claude-opus-4-20250514": ModelPricing(
        model_name="claude-opus-4-20250514",
        input_cost_per_million=15.00,
        output_cost_per_million=75.00,
    ),
    "gpt-4o": ModelPricing(
        model_name="gpt-4o",
        input_cost_per_million=2.50,
        output_cost_per_million=10.00,
    ),
    "gpt-4o-mini": ModelPricing(
        model_name="gpt-4o-mini",
        input_cost_per_million=0.15,
        output_cost_per_million=0.60,
    ),
}


@dataclass
class CostEntry:
    """A single cost entry."""
    timestamp: float
    model: str
    tokens_in: int
    tokens_out: int
    cost_usd: float
    agent_type: str = ""
    session_id: str = ""
    query_preview: str = ""


@dataclass
class CostReport:
    """Aggregated cost report."""
    period_start: datetime
    period_end: datetime
    total_cost_usd: float
    total_tokens_in: int
    total_tokens_out: int
    total_requests: int
    by_model: Dict[str, Dict[str, float]] = field(default_factory=dict)
    by_agent: Dict[str, Dict[str, float]] = field(default_factory=dict)
    entries: List[CostEntry] = field(default_factory=list)

    def to_markdown(self) -> str:
        """Format report as markdown."""
        lines = [
            f"# Cost Report: {self.period_start.strftime('%Y-%m-%d')} to {self.period_end.strftime('%Y-%m-%d')}\n",
            f"**Total Cost**: ${self.total_cost_usd:.4f}",
            f"**Total Requests**: {self.total_requests}",
            f"**Total Tokens**: {self.total_tokens_in + self.total_tokens_out:,} "
            f"({self.total_tokens_in:,} in + {self.total_tokens_out:,} out)\n",
        ]

        if self.by_model:
            lines.append("## By Model\n")
            lines.append("| Model | Requests | Tokens In | Tokens Out | Cost |")
            lines.append("|-------|----------|-----------|------------|------|")
            for model, stats in self.by_model.items():
                lines.append(
                    f"| {model} | {stats.get('requests', 0)} | "
                    f"{stats.get('tokens_in', 0):,} | "
                    f"{stats.get('tokens_out', 0):,} | "
                    f"${stats.get('cost', 0):.4f} |"
                )

        if self.by_agent:
            lines.append("\n## By Agent\n")
            lines.append("| Agent | Requests | Cost |")
            lines.append("|-------|----------|------|")
            for agent, stats in self.by_agent.items():
                lines.append(
                    f"| {agent} | {stats.get('requests', 0)} | "
                    f"${stats.get('cost', 0):.4f} |"
                )

        return "\n".join(lines)


class CostTracker:
    """
    Track and report LLM usage costs.

    Features:
    - Per-request cost calculation
    - Budget limits with alerts
    - Aggregated reporting (by model, agent, time period)
    - Local model compute cost estimation
    """

    def __init__(
        self,
        db_path: str = "gaia_costs.db",
        budget_limit_usd: Optional[float] = None,
        alert_callback: Optional[Any] = None,
    ):
        self.db_path = db_path
        self.budget_limit = budget_limit_usd
        self.alert_callback = alert_callback
        self._init_db()
        self._current_period_cost = 0.0

    def _init_db(self) -> None:
        """Initialize cost tracking database."""
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS cost_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                model TEXT NOT NULL,
                tokens_in INTEGER NOT NULL,
                tokens_out INTEGER NOT NULL,
                cost_usd REAL NOT NULL,
                agent_type TEXT,
                session_id TEXT,
                query_preview TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_cost_time ON cost_entries(timestamp);
            CREATE INDEX IF NOT EXISTS idx_cost_model ON cost_entries(model);
            CREATE INDEX IF NOT EXISTS idx_cost_agent ON cost_entries(agent_type);
        """)
        conn.commit()
        conn.close()

    def record_usage(
        self,
        model: str,
        tokens_in: int,
        tokens_out: int,
        agent_type: str = "",
        session_id: str = "",
        query_preview: str = "",
    ) -> CostEntry:
        """
        Record a single LLM usage event and calculate cost.

        Args:
            model: Model name (must be in MODEL_PRICING)
            tokens_in: Input tokens used
            tokens_out: Output tokens generated
            agent_type: Agent that made the call
            session_id: Associated session ID
            query_preview: First 100 chars of query

        Returns:
            CostEntry with calculated cost
        """
        import time

        pricing = MODEL_PRICING.get(model)
        if pricing:
            cost = (
                tokens_in * pricing.input_cost_per_token
                + tokens_out * pricing.output_cost_per_token
            )
        else:
            log.warning(f"No pricing for model: {model}, estimating at $0")
            cost = 0.0

        entry = CostEntry(
            timestamp=time.time(),
            model=model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost,
            agent_type=agent_type,
            session_id=session_id,
            query_preview=query_preview[:100],
        )

        # Persist
        self._persist_entry(entry)

        # Budget check
        self._current_period_cost += cost
        if self.budget_limit and self._current_period_cost > self.budget_limit:
            self._trigger_budget_alert(self._current_period_cost)

        return entry

    def _persist_entry(self, entry: CostEntry) -> None:
        """Persist cost entry to database."""
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT INTO cost_entries
            (timestamp, model, tokens_in, tokens_out, cost_usd,
             agent_type, session_id, query_preview)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            entry.timestamp, entry.model, entry.tokens_in, entry.tokens_out,
            entry.cost_usd, entry.agent_type, entry.session_id, entry.query_preview,
        ))
        conn.commit()
        conn.close()

    def _trigger_budget_alert(self, current_cost: float) -> None:
        """Trigger budget alert."""
        log.warning(
            f"Budget alert: ${current_cost:.4f} exceeds limit ${self.budget_limit:.4f}"
        )
        if self.alert_callback:
            self.alert_callback(current_cost, self.budget_limit)

    def generate_report(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> CostReport:
        """Generate a cost report for a time period."""
        import sqlite3

        if start_date is None:
            start_date = datetime.now() - timedelta(days=30)
        if end_date is None:
            end_date = datetime.now()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        start_ts = start_date.timestamp()
        end_ts = end_date.timestamp()

        # Overall totals
        cursor.execute("""
            SELECT COUNT(*), SUM(tokens_in), SUM(tokens_out), SUM(cost_usd)
            FROM cost_entries WHERE timestamp BETWEEN ? AND ?
        """, (start_ts, end_ts))
        row = cursor.fetchone()

        total_requests = row[0] or 0
        total_tokens_in = row[1] or 0
        total_tokens_out = row[2] or 0
        total_cost = row[3] or 0.0

        # By model
        cursor.execute("""
            SELECT model, COUNT(*), SUM(tokens_in), SUM(tokens_out), SUM(cost_usd)
            FROM cost_entries WHERE timestamp BETWEEN ? AND ?
            GROUP BY model ORDER BY SUM(cost_usd) DESC
        """, (start_ts, end_ts))

        by_model = {}
        for r in cursor.fetchall():
            by_model[r[0]] = {
                "requests": r[1],
                "tokens_in": r[2],
                "tokens_out": r[3],
                "cost": r[4],
            }

        # By agent
        cursor.execute("""
            SELECT agent_type, COUNT(*), SUM(cost_usd)
            FROM cost_entries WHERE timestamp BETWEEN ? AND ?
            AND agent_type != ''
            GROUP BY agent_type ORDER BY SUM(cost_usd) DESC
        """, (start_ts, end_ts))

        by_agent = {}
        for r in cursor.fetchall():
            by_agent[r[0]] = {"requests": r[1], "cost": r[2]}

        conn.close()

        return CostReport(
            period_start=start_date,
            period_end=end_date,
            total_cost_usd=total_cost,
            total_tokens_in=total_tokens_in,
            total_tokens_out=total_tokens_out,
            total_requests=total_requests,
            by_model=by_model,
            by_agent=by_agent,
        )
```

---

## Integration with GAIA

### 6.1 Auto-Instrumentation

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Auto-instrumentation hooks for GAIA components.

Patches Agent base class, ChatSDK, and tool execution
to automatically emit traces, metrics, and session events.
"""

import functools
import time
from typing import Any, Callable

from gaia.logger import get_logger

log = get_logger(__name__)


def instrument_agent(agent_class):
    """
    Decorator to auto-instrument a GAIA Agent subclass.

    Wraps process_query() to emit:
    - Trace spans for query processing
    - Metrics for query count, latency, errors
    - Session recording events
    """
    original_process_query = agent_class.process_query

    @functools.wraps(original_process_query)
    def instrumented_process_query(self, query: str, *args, **kwargs):
        tracer = TracerProvider.get_tracer(f"gaia.agent.{self.__class__.__name__}")
        metrics = get_metrics()
        recorder = get_session_recorder()

        agent_type = self.__class__.__name__
        labels = {"agent": agent_type}

        # Start session recording
        session = recorder.start_session(
            agent_type=agent_type,
            query=query,
            metadata={"model": getattr(self, "model_id", "unknown")},
        )

        with tracer.start_span("process_query", kind=SpanKind.AGENT) as span:
            span.set_attribute("agent.type", agent_type)
            span.set_attribute("agent.query", query[:200])
            span.set_attribute("session.id", session.session_id)

            metrics.agent_queries.add(1, labels)
            metrics.active_agents.set(
                metrics.active_agents.get(labels) + 1, labels
            )

            start = time.time()

            try:
                result = original_process_query(self, query, *args, **kwargs)

                span.set_attribute("agent.result_length", len(str(result)) if result else 0)
                recorder.record_event(
                    session.session_id, "result",
                    {"result": str(result)[:1000]}
                )

                return result

            except Exception as e:
                span.set_error(str(e))
                metrics.agent_errors.add(1, labels)
                recorder.record_event(
                    session.session_id, "error",
                    {"error": str(e)}
                )
                raise

            finally:
                duration = time.time() - start
                metrics.agent_duration.observe(duration, labels)
                metrics.active_agents.set(
                    max(0, metrics.active_agents.get(labels) - 1), labels
                )
                recorder.end_session(session.session_id, result=str(result)[:500] if "result" in dir() else None)

    agent_class.process_query = instrumented_process_query
    return agent_class


def instrument_tool(tool_func: Callable) -> Callable:
    """
    Decorator to auto-instrument a GAIA tool function.

    Wraps tool execution to emit:
    - Trace spans for tool calls
    - Metrics for tool execution count and latency
    """
    @functools.wraps(tool_func)
    def instrumented(*args, **kwargs):
        tracer = TracerProvider.get_tracer("gaia.tool")
        metrics = get_metrics()
        tool_name = tool_func.__name__
        labels = {"tool": tool_name}

        with tracer.start_span(tool_name, kind=SpanKind.TOOL) as span:
            span.set_attribute("tool.name", tool_name)
            span.set_attribute("tool.args", str(kwargs)[:500])

            metrics.tool_executions.add(1, labels)
            start = time.time()

            try:
                result = tool_func(*args, **kwargs)
                span.set_attribute("tool.result_status", result.get("status", "unknown") if isinstance(result, dict) else "ok")
                return result

            except Exception as e:
                span.set_error(str(e))
                metrics.tool_errors.add(1, labels)
                raise

            finally:
                duration = time.time() - start
                metrics.tool_duration.observe(duration, labels)

    return instrumented


# Global session recorder
_session_recorder: Optional[SessionRecorder] = None


def get_session_recorder() -> SessionRecorder:
    """Get the global session recorder."""
    global _session_recorder
    if _session_recorder is None:
        _session_recorder = SessionRecorder()
    return _session_recorder
```

### 6.2 CLI Integration

```python
# CLI commands for 'gaia observe'

def observe_traces_command(args):
    """Show recent traces."""
    import sqlite3
    import json

    conn = sqlite3.connect("gaia_traces.db")
    cursor = conn.cursor()

    if args.trace_id:
        cursor.execute(
            "SELECT * FROM spans WHERE trace_id = ? ORDER BY start_time",
            (args.trace_id,)
        )
    else:
        cursor.execute(
            "SELECT DISTINCT trace_id, MIN(start_time), MAX(end_time) "
            "FROM spans GROUP BY trace_id ORDER BY MIN(start_time) DESC LIMIT ?",
            (args.limit or 10,)
        )

    for row in cursor.fetchall():
        print(row)

    conn.close()


def observe_costs_command(args):
    """Show cost report."""
    tracker = CostTracker()
    report = tracker.generate_report()
    print(report.to_markdown())


def observe_replay_command(args):
    """Replay a recorded session."""
    recorder = SessionRecorder()
    session = recorder.replay_session(args.session_id)

    if not session:
        print(f"Session not found: {args.session_id}")
        return

    print(f"Replaying session: {session.session_id}")
    print(f"Agent: {session.agent_type}")
    print(f"Query: {session.query}")
    print(f"Duration: {session.duration_seconds:.1f}s")
    print(f"Tokens: {session.total_tokens_in} in / {session.total_tokens_out} out")
    print("---")

    for event in session.events:
        timestamp = event.timestamp - session.started_at
        print(f"  [{timestamp:.1f}s] {event.event_type}: {json.dumps(event.data, default=str)[:200]}")
```

---

## Safety and Security Considerations

1. **Sensitive Data Filtering**: Never record full prompts containing PII; redact by default
2. **Configurable Sampling**: Production should use sampling (1-10%) to reduce overhead
3. **Storage Limits**: Auto-rotate trace/session databases (default: 7 days retention)
4. **Access Control**: Session recordings contain LLM interactions; restrict access
5. **Cost Alert Thresholds**: Configurable budget alerts to prevent runaway spending
6. **Performance Impact**: Zero overhead when disabled; <1% overhead when enabled with sampling
7. **Data Encryption**: Session recordings encrypted at rest (optional)

---

## Implementation Plan

| Week | Phase | Tasks | Deliverables |
|------|-------|-------|-------------|
| 1-2 | Core Types | Span, SpanContext, MetricDataPoint, SessionEvent models | Core data types |
| 3-4 | Tracing | TracerProvider, Tracer, context propagation, exporters | Distributed tracing |
| 5-6 | Metrics | MeterProvider, Counter/Gauge/Histogram, GAIAMetrics | Metrics collection |
| 7-8 | Sessions | SessionRecorder, time-travel replay, storage | Session recording |
| 9 | Cost | CostTracker, pricing registry, budget alerts, reports | Cost management |
| 10 | Instrumentation | Auto-instrumentation hooks, Agent/Tool/LLM patches | Zero-config setup |
| 11 | CLI | `gaia observe` commands (traces, metrics, costs, replay) | CLI interface |
| 12 | Testing | Unit tests, integration tests, performance benchmarks | Production-ready |

---

## Testing Strategy

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Unit tests for observability framework."""

import pytest
import time


class TestTracing:
    def test_span_creation(self):
        provider = TracerProvider(enabled=True, exporters=[])
        tracer = Tracer("test", provider)

        with tracer.start_span("test_op", kind=SpanKind.INTERNAL) as span:
            span.set_attribute("key", "value")
            time.sleep(0.01)

        assert span.duration_ms > 0
        assert span.attributes["key"] == "value"
        assert span.status == SpanStatus.OK

    def test_parent_child_spans(self):
        provider = TracerProvider(enabled=True, exporters=[])
        tracer = Tracer("test", provider)

        with tracer.start_span("parent") as parent:
            with tracer.start_span("child") as child:
                assert child.parent_span_id == parent.span_id
                assert child.trace_id == parent.trace_id

    def test_error_span(self):
        provider = TracerProvider(enabled=True, exporters=[])
        tracer = Tracer("test", provider)

        with pytest.raises(ValueError):
            with tracer.start_span("error_op") as span:
                raise ValueError("test error")

        assert span.status == SpanStatus.ERROR
        assert span.error == "test error"


class TestMetrics:
    def test_counter(self):
        counter = Counter("test_counter")
        counter.add(1, labels={"env": "test"})
        counter.add(5, labels={"env": "test"})
        assert counter.get(labels={"env": "test"}) == 6

    def test_histogram(self):
        hist = Histogram("test_hist")
        for v in [0.1, 0.2, 0.5, 1.0, 2.0]:
            hist.observe(v)
        stats = hist.get_statistics()
        assert stats["count"] == 5
        assert stats["min"] == 0.1
        assert stats["max"] == 2.0

    def test_gauge(self):
        gauge = Gauge("test_gauge")
        gauge.set(42.0, labels={"host": "local"})
        assert gauge.get(labels={"host": "local"}) == 42.0
        gauge.set(0.0, labels={"host": "local"})
        assert gauge.get(labels={"host": "local"}) == 0.0


class TestCostTracker:
    def test_local_model_zero_cost(self):
        tracker = CostTracker(db_path=":memory:")
        entry = tracker.record_usage(
            model="Qwen3-Coder-30B-A3B-Instruct-GGUF",
            tokens_in=1000,
            tokens_out=500,
        )
        assert entry.cost_usd == 0.0

    def test_cloud_model_cost(self):
        tracker = CostTracker(db_path=":memory:")
        entry = tracker.record_usage(
            model="claude-sonnet-4-20250514",
            tokens_in=1000,
            tokens_out=500,
        )
        expected = (1000 * 3.0 / 1_000_000) + (500 * 15.0 / 1_000_000)
        assert abs(entry.cost_usd - expected) < 0.0001


class TestSessionRecorder:
    def test_record_and_replay(self):
        recorder = SessionRecorder(db_path=":memory:")
        session = recorder.start_session("test_agent", "test query")
        recorder.record_event(session.session_id, "prompt", {"text": "hello"})
        recorder.record_event(session.session_id, "response", {"text": "world"})
        recorder.end_session(session.session_id, result="done")

        # Note: replay from :memory: DB requires same connection
        # In production, use file-based DB
        assert session.duration_seconds >= 0
        assert len(session.events) == 2
```

---

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Trace overhead (enabled) | <1% latency increase | A/B benchmark |
| Trace overhead (disabled) | 0% latency increase | A/B benchmark |
| Metric collection interval | <1s for real-time metrics | Timer verification |
| Session recording fidelity | 100% event capture | Compare with manual logging |
| Cost calculation accuracy | <0.1% error vs provider bill | Monthly reconciliation |
| Storage efficiency | <100MB/day at 100 queries/day | Storage monitoring |
| Query latency (trace lookup) | <100ms for single trace | Benchmark |

---

## Complete Code

```
src/gaia/observe/
    __init__.py
    types.py              # Core types (Span, MetricDataPoint, SessionEvent)
    tracer.py             # TracerProvider, Tracer, context propagation
    metrics.py            # MeterProvider, Counter, Gauge, Histogram
    session.py            # SessionRecorder, time-travel replay
    cost.py               # CostTracker, pricing, budget management
    exporters/
        __init__.py
        sqlite.py         # SQLiteSpanExporter, SQLiteMetricExporter
        console.py        # ConsoleSpanExporter
        json_file.py      # JSONFileSpanExporter
        otlp.py           # OTLPExporter (OpenTelemetry Protocol)
    instrument.py         # Auto-instrumentation hooks
    predefined.py         # GAIAMetrics, pre-defined metric definitions
    cli.py                # CLI commands for 'gaia observe'

tests/unit/observe/
    test_tracer.py
    test_metrics.py
    test_session.py
    test_cost.py
    test_exporters.py

tests/integration/observe/
    test_agent_instrumentation.py
    test_end_to_end_tracing.py
```
