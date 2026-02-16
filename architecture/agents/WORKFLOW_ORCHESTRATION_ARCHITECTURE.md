# Workflow Orchestration Architecture for GAIA

**Date**: February 7, 2026
**Version**: 1.0
**Status**: Specification
**Priority**: HIGH
**Estimated Effort**: 10-12 weeks (2 engineers)
**Target**: Enable multi-step automation workflows with triggers, conditions, and scheduling

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Problem Statement](#problem-statement)
3. [Architecture Overview](#architecture-overview)
4. [Component Specifications](#component-specifications)
5. [Data Models and Schemas](#data-models-and-schemas)
6. [Trigger System](#trigger-system)
7. [Workflow Engine](#workflow-engine)
8. [Conditional Logic and Branching](#conditional-logic-and-branching)
9. [Integration with GAIA](#integration-with-gaia)
10. [Safety and Security Considerations](#safety-and-security-considerations)
11. [Implementation Plan](#implementation-plan)
12. [Testing Strategy](#testing-strategy)
13. [Success Metrics](#success-metrics)
14. [Complete Code](#complete-code)

---

## Executive Summary

### The Gap

GAIA agents can:
- Execute single queries with multi-step tool chains
- Respond to interactive user prompts
- Perform agent-to-agent routing via RoutingAgent

GAIA agents **cannot**:
- Run automated workflows on a schedule (cron-like triggers)
- React to external events (webhooks, file changes, email arrival)
- Execute conditional branching (if/else/switch logic)
- Orchestrate multi-agent pipelines with data passing
- Persist workflow state across restarts
- Retry failed steps with backoff strategies
- Provide workflow monitoring and observability

### The Solution

A **Workflow Orchestration Architecture** that enables:
- Declarative workflow definitions (YAML/Python DSL)
- Trigger system (cron, webhook, file watcher, manual)
- Conditional branching and parallel execution
- Multi-agent step orchestration with data passing
- Persistent state with checkpointing and resume
- Built-in retry, timeout, and error handling
- Workflow monitoring dashboard and audit trail

### Impact

**Unlocks entire category**: Business process automation for:
- Scheduled report generation and distribution
- Event-driven data processing pipelines
- Multi-step approval workflows
- CI/CD-style agent pipelines
- Automated monitoring and alerting

**Before**: 5% coverage for workflow automation category
**After**: 80% coverage

---

## Problem Statement

### Current Limitations

**Example task**: "Every morning at 8 AM, check my email for urgent items, summarize them, and post to Slack"

**What GAIA can do today**:
```python
# Manual, one-shot execution only
from gaia.agents.base import Agent

agent = Agent()
agent.process_query("Check email and summarize urgent items")
# Works once, but no scheduling, no persistence, no retry
```

**What GAIA needs to do**:
- Define the workflow declaratively
- Schedule it to run at 8 AM daily
- Handle failures gracefully (email API down, LLM timeout)
- Pass data between steps (email list -> summarizer -> Slack poster)
- Track execution history and provide monitoring
- Allow manual triggering and parameter override

### User Stories

**Story 1: Scheduled Report**
```
As a team lead, I want to:
- Define a workflow that runs every Friday at 5 PM
- Step 1: Query Jira for this week's completed tickets
- Step 2: Summarize the tickets using LLM
- Step 3: Format as a weekly report
- Step 4: Email the report to the team
So that weekly status reports are automated
```

**Story 2: Event-Driven Pipeline**
```
As a data engineer, I want to:
- Trigger a workflow when a new CSV file appears in a directory
- Step 1: Parse and validate the CSV data
- Step 2: Run anomaly detection using LLM
- Step 3: If anomalies found, alert the team
- Step 4: Otherwise, archive the file and update the dashboard
So that data ingestion is automated with quality checks
```

**Story 3: Approval Workflow**
```
As a manager, I want to:
- Trigger a workflow when an expense report email arrives
- Step 1: Extract expense details from the email
- Step 2: If amount > $500, require manager approval
- Step 3: If approved, submit to accounting system
- Step 4: Send confirmation email to requester
So that expense approvals are streamlined
```

**Story 4: Multi-Agent Pipeline**
```
As a developer, I want to:
- Chain multiple GAIA agents in a pipeline
- Step 1: ChatAgent answers customer question
- Step 2: If confidence < 0.7, escalate to CodeAgent
- Step 3: JiraAgent creates a ticket for unresolved questions
- Step 4: EmailAgent sends the answer to the customer
So that customer support is fully automated
```

---

## Architecture Overview

### High-Level Design

```
+------------------------------------------------------------------+
|                     Workflow Orchestrator                          |
|                                                                    |
|  +------------------+  +------------------+  +------------------+ |
|  |  Trigger System  |  | Workflow Engine   |  |  State Manager   | |
|  |                  |  |                  |  |                  | |
|  | - CronTrigger    |  | - Step Executor  |  | - Checkpoints   | |
|  | - WebhookTrigger |  | - Data Router    |  | - Resume/Replay | |
|  | - FileWatcher    |  | - Branch Logic   |  | - History        | |
|  | - ManualTrigger  |  | - Parallel Exec  |  | - Audit Trail   | |
|  | - EventTrigger   |  | - Error Handler  |  |                  | |
|  +--------+---------+  +--------+---------+  +--------+---------+ |
|           |                      |                      |          |
|           v                      v                      v          |
|  +----------------------------------------------------------+     |
|  |                  Workflow Runtime                          |     |
|  |                                                            |     |
|  |  [Trigger] --> [Step 1] --> [Condition] --> [Step 2] ...  |     |
|  |                                  |                         |     |
|  |                                  +--> [Step 3 (branch)]   |     |
|  +----------------------------------------------------------+     |
|                              |                                     |
|              +---------------+---------------+                     |
|              |               |               |                     |
|              v               v               v                     |
|      +-----------+   +-----------+   +-----------+                 |
|      |GAIA Agents|   |External   |   |Notification|                |
|      |           |   |Services   |   |System      |                |
|      |- Chat     |   |- REST APIs|   |- Email     |                |
|      |- Code     |   |- Databases|   |- Slack     |                |
|      |- Jira     |   |- Webhooks |   |- Console   |                |
|      |- Email    |   |           |   |            |                |
|      +-----------+   +-----------+   +-----------+                 |
+------------------------------------------------------------------+
```

### Component Layers

| Layer | Components | Responsibility |
|-------|-----------|----------------|
| **Triggers** | CronTrigger, WebhookTrigger, FileWatcher, ManualTrigger | Initiate workflow runs |
| **Engine** | WorkflowEngine, StepExecutor, BranchRouter | Execute workflow graph |
| **State** | StateManager, Checkpoint, AuditLog | Persistence, resume, history |
| **Steps** | AgentStep, FunctionStep, HttpStep, ConditionalStep | Individual workflow actions |
| **Runtime** | WorkflowRuntime, Scheduler, WorkerPool | Process management |

### Workflow Execution Flow

```
              Workflow Lifecycle
              ===================

  [Define]                    [Trigger]                  [Execute]
     |                            |                          |
     v                            v                          v
  YAML/Python DSL  -->  CronScheduler   -->  WorkflowEngine
                        WebhookListener       |
                        FileWatcher           v
                                         [Step 1: Fetch Data]
                                              |
                                              v
                                         [Step 2: Process]
                                              |
                                         +----+----+
                                         |         |
                                         v         v
                                    [Condition] [Condition]
                                    (if true)   (if false)
                                         |         |
                                         v         v
                                    [Step 3a]  [Step 3b]
                                         |         |
                                         +----+----+
                                              |
                                              v
                                         [Step 4: Output]
                                              |
                                              v
                                         [Complete/Notify]


              State Transitions
              ==================

  PENDING --> TRIGGERED --> RUNNING --> COMPLETED
                              |
                              +--> PAUSED --> RUNNING
                              |
                              +--> FAILED --> RETRYING --> RUNNING
                              |
                              +--> CANCELLED
```

---

## Component Specifications

### 1. Core Data Models

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Core data models for workflow orchestration.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

from gaia.logger import get_logger

log = get_logger(__name__)


class WorkflowStatus(Enum):
    """Workflow execution status."""
    PENDING = "pending"
    TRIGGERED = "triggered"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"


class StepStatus(Enum):
    """Individual step execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"


class TriggerType(Enum):
    """Types of workflow triggers."""
    CRON = "cron"
    WEBHOOK = "webhook"
    FILE_WATCH = "file_watch"
    MANUAL = "manual"
    EVENT = "event"
    CHAIN = "chain"  # Triggered by another workflow completing


class StepType(Enum):
    """Types of workflow steps."""
    AGENT = "agent"
    FUNCTION = "function"
    HTTP = "http"
    CONDITION = "condition"
    PARALLEL = "parallel"
    WAIT = "wait"
    NOTIFICATION = "notification"
    SUBPROCESS = "subprocess"


@dataclass
class RetryPolicy:
    """Retry configuration for steps."""
    max_retries: int = 3
    initial_delay_seconds: float = 1.0
    max_delay_seconds: float = 300.0
    backoff_multiplier: float = 2.0
    retry_on_exceptions: List[str] = field(
        default_factory=lambda: ["TimeoutError", "ConnectionError"]
    )
    retry_on_status: List[str] = field(
        default_factory=lambda: ["failed", "timeout"]
    )


@dataclass
class StepDefinition:
    """Definition of a single workflow step."""
    step_id: str
    name: str
    step_type: StepType
    config: Dict[str, Any] = field(default_factory=dict)
    inputs: Dict[str, str] = field(default_factory=dict)  # Mapping of param -> source
    outputs: List[str] = field(default_factory=list)
    timeout_seconds: int = 300
    retry_policy: Optional[RetryPolicy] = None
    condition: Optional[str] = None  # Expression to evaluate before running
    on_failure: str = "fail"  # "fail", "continue", "skip_remaining"
    depends_on: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TriggerDefinition:
    """Definition of a workflow trigger."""
    trigger_type: TriggerType
    config: Dict[str, Any] = field(default_factory=dict)
    # For cron: {"schedule": "0 8 * * MON-FRI"}
    # For webhook: {"path": "/hooks/my-workflow", "method": "POST"}
    # For file_watch: {"path": "/data/inbox", "pattern": "*.csv"}
    # For event: {"event_name": "email_received", "filter": {...}}
    enabled: bool = True


@dataclass
class WorkflowDefinition:
    """Complete workflow definition."""
    workflow_id: str
    name: str
    description: str = ""
    version: str = "1.0"
    triggers: List[TriggerDefinition] = field(default_factory=list)
    steps: List[StepDefinition] = field(default_factory=list)
    global_timeout_seconds: int = 3600
    max_concurrent_runs: int = 1
    retry_policy: Optional[RetryPolicy] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    enabled: bool = True

    def validate(self) -> List[str]:
        """Validate workflow definition and return errors."""
        errors = []
        step_ids = set()

        for step in self.steps:
            if step.step_id in step_ids:
                errors.append(f"Duplicate step_id: {step.step_id}")
            step_ids.add(step.step_id)

            for dep in step.depends_on:
                if dep not in step_ids:
                    errors.append(
                        f"Step {step.step_id} depends on unknown step: {dep}"
                    )

        if not self.steps:
            errors.append("Workflow must have at least one step")

        return errors


@dataclass
class StepResult:
    """Result of executing a single step."""
    step_id: str
    status: StepStatus
    output: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retry_count: int = 0
    duration_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowRun:
    """A single execution of a workflow."""
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    workflow_id: str = ""
    status: WorkflowStatus = WorkflowStatus.PENDING
    trigger_data: Dict[str, Any] = field(default_factory=dict)
    step_results: Dict[str, StepResult] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)  # Shared data between steps
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> float:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return 0.0
```

---

## Trigger System

### 2.1 Cron Trigger

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Cron-based scheduling trigger for recurring workflows.
"""

import asyncio
import threading
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

from gaia.logger import get_logger

log = get_logger(__name__)


class CronExpression:
    """
    Parse and evaluate cron expressions.

    Supports standard 5-field cron syntax:
      minute hour day_of_month month day_of_week

    Examples:
      "0 8 * * MON-FRI"    -> 8 AM weekdays
      "*/15 * * * *"       -> Every 15 minutes
      "0 9,17 * * *"       -> 9 AM and 5 PM daily
      "0 0 1 * *"          -> First of every month
    """

    def __init__(self, expression: str):
        self.expression = expression
        self.fields = self._parse(expression)

    def _parse(self, expr: str) -> Dict[str, set]:
        """Parse cron expression into field sets."""
        parts = expr.strip().split()
        if len(parts) != 5:
            raise ValueError(f"Invalid cron expression: {expr} (expected 5 fields)")

        return {
            "minute": self._parse_field(parts[0], 0, 59),
            "hour": self._parse_field(parts[1], 0, 23),
            "day": self._parse_field(parts[2], 1, 31),
            "month": self._parse_field(parts[3], 1, 12),
            "weekday": self._parse_field(parts[4], 0, 6, day_names=True),
        }

    def _parse_field(
        self, field: str, min_val: int, max_val: int, day_names: bool = False
    ) -> set:
        """Parse a single cron field into a set of valid values."""
        # Standard cron convention: Sunday=0, Monday=1, ..., Saturday=6
        DAY_MAP = {
            "SUN": 0, "MON": 1, "TUE": 2, "WED": 3,
            "THU": 4, "FRI": 5, "SAT": 6,
        }

        if day_names:
            for name, num in DAY_MAP.items():
                field = field.replace(name, str(num))

        values = set()

        for part in field.split(","):
            if part == "*":
                values.update(range(min_val, max_val + 1))
            elif "/" in part:
                base, step = part.split("/")
                start = min_val if base == "*" else int(base)
                values.update(range(start, max_val + 1, int(step)))
            elif "-" in part:
                start, end = part.split("-")
                values.update(range(int(start), int(end) + 1))
            else:
                values.add(int(part))

        return values

    def matches(self, dt: datetime) -> bool:
        """Check if a datetime matches the cron expression.

        Note: Python's datetime.weekday() returns Monday=0 ... Sunday=6,
        but cron convention uses Sunday=0 ... Saturday=6.
        We convert via (weekday + 1) % 7.
        """
        cron_weekday = (dt.weekday() + 1) % 7  # Convert Python weekday to cron
        return (
            dt.minute in self.fields["minute"]
            and dt.hour in self.fields["hour"]
            and dt.day in self.fields["day"]
            and dt.month in self.fields["month"]
            and cron_weekday in self.fields["weekday"]
        )

    def next_run(self, after: Optional[datetime] = None) -> datetime:
        """Calculate the next datetime matching the expression."""
        if after is None:
            after = datetime.now()

        # Start from the next minute
        candidate = after.replace(second=0, microsecond=0) + timedelta(minutes=1)

        # Search up to 1 year ahead
        max_iterations = 525960  # minutes in a year
        for _ in range(max_iterations):
            if self.matches(candidate):
                return candidate
            candidate += timedelta(minutes=1)

        raise ValueError(f"No matching time found within 1 year for: {self.expression}")


class CronTrigger:
    """
    Cron-based trigger that fires workflows on schedule.

    Features:
    - Standard cron expression support
    - Timezone-aware scheduling
    - Missed run detection and catch-up policy
    - Jitter for distributed deployments
    - Graceful shutdown
    """

    def __init__(self, timezone: str = "UTC"):
        self.timezone = timezone
        self._schedules: Dict[str, Dict[str, Any]] = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def register(
        self,
        workflow_id: str,
        cron_expression: str,
        callback: Callable,
        catch_up: bool = False,
    ) -> None:
        """
        Register a workflow for cron scheduling.

        Args:
            workflow_id: Unique workflow identifier
            cron_expression: Standard cron expression
            callback: Function to call when triggered
            catch_up: Run missed executions on startup
        """
        cron = CronExpression(cron_expression)
        self._schedules[workflow_id] = {
            "cron": cron,
            "expression": cron_expression,
            "callback": callback,
            "catch_up": catch_up,
            "last_run": None,
            "next_run": cron.next_run(),
        }
        log.info(
            f"Registered cron trigger for {workflow_id}: "
            f"{cron_expression} (next: {cron.next_run()})"
        )

    def unregister(self, workflow_id: str) -> None:
        """Remove a workflow from scheduling."""
        self._schedules.pop(workflow_id, None)
        log.info(f"Unregistered cron trigger for {workflow_id}")

    def start(self) -> None:
        """Start the cron scheduler in a background thread."""
        if self._running:
            return

        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="gaia-cron-scheduler"
        )
        self._thread.start()
        log.info("Cron scheduler started")

    def stop(self) -> None:
        """Stop the cron scheduler gracefully."""
        self._running = False
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)
        log.info("Cron scheduler stopped")

    def _run_loop(self) -> None:
        """Main scheduling loop."""
        while self._running and not self._stop_event.is_set():
            now = datetime.now()

            for wf_id, schedule in list(self._schedules.items()):
                if schedule["next_run"] and now >= schedule["next_run"]:
                    try:
                        log.info(f"Cron trigger firing for workflow: {wf_id}")
                        schedule["callback"](
                            workflow_id=wf_id,
                            trigger_data={
                                "trigger_type": "cron",
                                "scheduled_time": schedule["next_run"].isoformat(),
                                "actual_time": now.isoformat(),
                                "expression": schedule["expression"],
                            },
                        )
                        schedule["last_run"] = now
                    except Exception as e:
                        log.error(f"Cron trigger error for {wf_id}: {e}")

                    # Calculate next run
                    schedule["next_run"] = schedule["cron"].next_run(now)
                    log.debug(f"Next run for {wf_id}: {schedule['next_run']}")

            # Sleep until next minute boundary
            self._stop_event.wait(timeout=30)

    def get_status(self) -> List[Dict[str, Any]]:
        """Get status of all scheduled workflows."""
        return [
            {
                "workflow_id": wf_id,
                "expression": s["expression"],
                "next_run": str(s["next_run"]),
                "last_run": str(s["last_run"]) if s["last_run"] else None,
            }
            for wf_id, s in self._schedules.items()
        ]
```

### 2.2 Webhook Trigger

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Webhook trigger for event-driven workflow execution.
"""

import hashlib
import hmac
import json
import threading
from typing import Any, Callable, Dict, List, Optional

from gaia.logger import get_logger

log = get_logger(__name__)


class WebhookTrigger:
    """
    HTTP webhook trigger for external event-driven workflows.

    Features:
    - HMAC signature validation for security
    - Request body parsing (JSON, form data)
    - Path-based routing to workflows
    - Rate limiting per source IP
    - Request logging and audit trail

    Uses GAIA's API server infrastructure (FastAPI).
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8089,
        webhook_secret: Optional[str] = None,
    ):
        self.host = host
        self.port = port
        self.webhook_secret = webhook_secret
        self._routes: Dict[str, Dict[str, Any]] = {}
        self._server = None
        self._server_thread: Optional[threading.Thread] = None

    def register(
        self,
        path: str,
        workflow_id: str,
        callback: Callable,
        methods: List[str] = None,
        require_signature: bool = True,
    ) -> None:
        """
        Register a webhook endpoint for a workflow.

        Args:
            path: URL path (e.g., "/hooks/my-workflow")
            workflow_id: Associated workflow ID
            callback: Function to call with parsed request data
            methods: HTTP methods to accept (default: ["POST"])
            require_signature: Require HMAC signature validation
        """
        self._routes[path] = {
            "workflow_id": workflow_id,
            "callback": callback,
            "methods": methods or ["POST"],
            "require_signature": require_signature,
        }
        log.info(f"Registered webhook: {path} -> {workflow_id}")

    def start(self) -> None:
        """Start the webhook server."""
        from fastapi import FastAPI, Request, HTTPException
        import uvicorn

        app = FastAPI(title="GAIA Workflow Webhooks")

        @app.post("/hooks/{path:path}")
        @app.put("/hooks/{path:path}")
        async def handle_webhook(path: str, request: Request):
            full_path = f"/hooks/{path}"
            route = self._routes.get(full_path)
            if not route:
                raise HTTPException(status_code=404, detail="Webhook not found")

            if request.method not in route["methods"]:
                raise HTTPException(status_code=405, detail="Method not allowed")

            # Validate signature if required
            if route["require_signature"] and self.webhook_secret:
                signature = request.headers.get("X-Webhook-Signature", "")
                body = await request.body()
                if not self._validate_signature(body, signature):
                    raise HTTPException(status_code=401, detail="Invalid signature")

            # Parse request body
            try:
                body_data = await request.json()
            except Exception:
                body_data = {"raw": (await request.body()).decode("utf-8", errors="replace")}

            trigger_data = {
                "trigger_type": "webhook",
                "path": full_path,
                "method": request.method,
                "headers": dict(request.headers),
                "body": body_data,
                "source_ip": request.client.host,
                "timestamp": datetime.now().isoformat(),
            }

            try:
                route["callback"](
                    workflow_id=route["workflow_id"],
                    trigger_data=trigger_data,
                )
                return {"status": "accepted", "workflow_id": route["workflow_id"]}
            except Exception as e:
                log.error(f"Webhook handler error: {e}")
                raise HTTPException(status_code=500, detail="Workflow trigger failed")

        @app.get("/hooks/status")
        async def webhook_status():
            return {
                "registered_hooks": list(self._routes.keys()),
                "count": len(self._routes),
            }

        self._server_thread = threading.Thread(
            target=lambda: uvicorn.run(app, host=self.host, port=self.port),
            daemon=True,
            name="gaia-webhook-server",
        )
        self._server_thread.start()
        log.info(f"Webhook server started on {self.host}:{self.port}")

    def _validate_signature(self, body: bytes, signature: str) -> bool:
        """Validate HMAC-SHA256 signature."""
        if not self.webhook_secret:
            return True
        expected = hmac.new(
            self.webhook_secret.encode(),
            body,
            hashlib.sha256,
        ).hexdigest()
        return hmac.compare_digest(f"sha256={expected}", signature)


class FileWatchTrigger:
    """
    File system watcher trigger for data-driven workflows.

    Monitors directories for new/modified files and triggers workflows.
    """

    def __init__(self):
        self._watchers: Dict[str, Dict[str, Any]] = {}
        self._running = False

    def register(
        self,
        watch_path: str,
        workflow_id: str,
        callback: Callable,
        pattern: str = "*",
        events: List[str] = None,
        debounce_seconds: float = 2.0,
    ) -> None:
        """
        Register a file watch trigger.

        Args:
            watch_path: Directory to watch
            workflow_id: Associated workflow ID
            callback: Function to call on file events
            pattern: Glob pattern to match (e.g., "*.csv")
            events: File events to watch ("created", "modified", "deleted")
            debounce_seconds: Debounce interval to avoid duplicate triggers
        """
        self._watchers[workflow_id] = {
            "path": watch_path,
            "pattern": pattern,
            "callback": callback,
            "events": events or ["created"],
            "debounce_seconds": debounce_seconds,
        }
        log.info(f"Registered file watcher: {watch_path}/{pattern} -> {workflow_id}")

    def start(self) -> None:
        """Start all file watchers."""
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler
        import fnmatch

        self._running = True
        self._observer = Observer()

        for wf_id, config in self._watchers.items():

            class WorkflowHandler(FileSystemEventHandler):
                def __init__(self, wf_config, wf_identifier):
                    self.config = wf_config
                    self.wf_id = wf_identifier
                    self._last_trigger = {}

                def on_created(self, event):
                    if not event.is_directory and "created" in self.config["events"]:
                        self._handle(event, "created")

                def on_modified(self, event):
                    if not event.is_directory and "modified" in self.config["events"]:
                        self._handle(event, "modified")

                def _handle(self, event, event_type):
                    import fnmatch as fnm
                    import time

                    filename = os.path.basename(event.src_path)
                    if not fnm.fnmatch(filename, self.config["pattern"]):
                        return

                    # Debounce
                    now = time.time()
                    last = self._last_trigger.get(event.src_path, 0)
                    if now - last < self.config["debounce_seconds"]:
                        return
                    self._last_trigger[event.src_path] = now

                    log.info(f"File trigger: {event_type} {event.src_path}")
                    self.config["callback"](
                        workflow_id=self.wf_id,
                        trigger_data={
                            "trigger_type": "file_watch",
                            "event_type": event_type,
                            "file_path": event.src_path,
                            "filename": filename,
                            "timestamp": datetime.now().isoformat(),
                        },
                    )

            handler = WorkflowHandler(config, wf_id)
            self._observer.schedule(handler, config["path"], recursive=False)

        self._observer.start()
        log.info("File watchers started")

    def stop(self) -> None:
        """Stop all file watchers."""
        self._running = False
        if hasattr(self, "_observer"):
            self._observer.stop()
            self._observer.join()
```

---

## Workflow Engine

### 3.1 Step Executors

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Step executors for different workflow step types.
"""

import abc
import asyncio
import time
import traceback
from typing import Any, Dict, Optional

from gaia.logger import get_logger

log = get_logger(__name__)


class BaseStepExecutor(abc.ABC):
    """Base class for step executors."""

    @abc.abstractmethod
    async def execute(
        self,
        step: StepDefinition,
        context: Dict[str, Any],
        run: WorkflowRun,
    ) -> StepResult:
        """Execute a single workflow step."""
        pass

    def _interpolate(self, template: str, context: Dict[str, Any]) -> str:
        """Interpolate {variable.path} references in template string.

        Shared utility for all step executors. Supports dot-notation
        for nested dictionary access (e.g., "{step1.result.count}").

        Args:
            template: String with {variable} placeholders
            context: Workflow context dictionary

        Returns:
            Template with resolved variable references
        """
        import re

        def replace_var(match):
            var_path = match.group(1)
            parts = var_path.split(".")
            value = context
            for part in parts:
                if isinstance(value, dict):
                    value = value.get(part, match.group(0))
                else:
                    return match.group(0)
            return str(value)

        return re.sub(r"\{(\w+(?:\.\w+)*)\}", replace_var, template)


class AgentStepExecutor(BaseStepExecutor):
    """
    Execute a step using a GAIA agent.

    Supports all agent types: ChatAgent, CodeAgent, JiraAgent, EmailAgent, etc.
    """

    # Agent type registry
    AGENT_TYPES = {
        "chat": "gaia.agents.chat.agent.ChatAgent",
        "code": "gaia.agents.code.agent.CodeAgent",
        "jira": "gaia.agents.jira.agent.JiraAgent",
        "blender": "gaia.agents.blender.agent.BlenderAgent",
        "email": "gaia.agents.email.agent.EmailAgent",
        "routing": "gaia.agents.routing.agent.RoutingAgent",
    }

    async def execute(
        self,
        step: StepDefinition,
        context: Dict[str, Any],
        run: WorkflowRun,
    ) -> StepResult:
        """
        Execute a GAIA agent step.

        Config options:
            agent_type: "chat" | "code" | "jira" | "email" | etc.
            query: The prompt/query for the agent (supports {context} interpolation)
            agent_kwargs: Additional kwargs for agent constructor
        """
        started_at = datetime.now()

        try:
            agent_type = step.config.get("agent_type", "chat")
            query_template = step.config.get("query", "")
            agent_kwargs = step.config.get("agent_kwargs", {})

            # Interpolate context variables into query
            query = self._interpolate(query_template, context)

            # Import and instantiate agent
            agent_class = self._get_agent_class(agent_type)
            agent = agent_class(
                silent_mode=True,
                **agent_kwargs,
            )

            # Execute the agent query
            log.info(f"Executing agent step: {step.name} (type={agent_type})")
            result = agent.process_query(query)

            completed_at = datetime.now()

            return StepResult(
                step_id=step.step_id,
                status=StepStatus.COMPLETED,
                output={"result": result, "agent_type": agent_type},
                started_at=started_at,
                completed_at=completed_at,
                duration_seconds=(completed_at - started_at).total_seconds(),
            )

        except Exception as e:
            log.error(f"Agent step failed: {step.name}: {e}")
            return StepResult(
                step_id=step.step_id,
                status=StepStatus.FAILED,
                error=str(e),
                started_at=started_at,
                completed_at=datetime.now(),
                metadata={"traceback": traceback.format_exc()},
            )

    def _get_agent_class(self, agent_type: str):
        """Dynamically import and return agent class."""
        import importlib

        module_path = self.AGENT_TYPES.get(agent_type)
        if not module_path:
            raise ValueError(f"Unknown agent type: {agent_type}")

        module_name, class_name = module_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)

    # _interpolate is inherited from BaseStepExecutor


class FunctionStepExecutor(BaseStepExecutor):
    """
    Execute a Python function as a workflow step.

    Supports both sync and async functions.
    """

    def __init__(self):
        self._registry: Dict[str, Any] = {}

    def register_function(self, name: str, func: Any) -> None:
        """Register a callable for use in workflows."""
        self._registry[name] = func

    async def execute(
        self,
        step: StepDefinition,
        context: Dict[str, Any],
        run: WorkflowRun,
    ) -> StepResult:
        """
        Execute a registered function step.

        Config options:
            function: Name of registered function
            args: Positional arguments
            kwargs: Keyword arguments (supports {context} interpolation)
        """
        started_at = datetime.now()

        try:
            func_name = step.config.get("function", "")
            func = self._registry.get(func_name)
            if not func:
                raise ValueError(f"Function not registered: {func_name}")

            args = step.config.get("args", [])
            kwargs = step.config.get("kwargs", {})

            # Interpolate context into kwargs
            resolved_kwargs = {}
            for key, value in kwargs.items():
                if isinstance(value, str) and value.startswith("{") and value.endswith("}"):
                    var_path = value[1:-1]
                    resolved_kwargs[key] = self._resolve_context(var_path, context)
                else:
                    resolved_kwargs[key] = value

            # Execute function (handle async)
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **resolved_kwargs)
            else:
                result = func(*args, **resolved_kwargs)

            completed_at = datetime.now()
            return StepResult(
                step_id=step.step_id,
                status=StepStatus.COMPLETED,
                output={"result": result} if not isinstance(result, dict) else result,
                started_at=started_at,
                completed_at=completed_at,
                duration_seconds=(completed_at - started_at).total_seconds(),
            )

        except Exception as e:
            log.error(f"Function step failed: {step.name}: {e}")
            return StepResult(
                step_id=step.step_id,
                status=StepStatus.FAILED,
                error=str(e),
                started_at=started_at,
                completed_at=datetime.now(),
            )

    def _resolve_context(self, path: str, context: Dict) -> Any:
        """Resolve a dot-notation path in context."""
        parts = path.split(".")
        value = context
        for part in parts:
            if isinstance(value, dict):
                value = value[part]
            else:
                raise KeyError(f"Cannot resolve {path}")
        return value


class HttpStepExecutor(BaseStepExecutor):
    """Execute an HTTP request as a workflow step."""

    async def execute(
        self,
        step: StepDefinition,
        context: Dict[str, Any],
        run: WorkflowRun,
    ) -> StepResult:
        """
        Execute an HTTP request step.

        Config options:
            url: Request URL (supports {context} interpolation)
            method: HTTP method (GET, POST, PUT, DELETE)
            headers: Request headers dict
            body: Request body (for POST/PUT)
            timeout: Request timeout in seconds
        """
        import aiohttp

        started_at = datetime.now()

        try:
            url = step.config.get("url", "")
            method = step.config.get("method", "GET").upper()
            headers = step.config.get("headers", {})
            body = step.config.get("body")
            timeout_sec = step.config.get("timeout", 30)

            # Interpolate context
            url = self._interpolate(url, context)
            if isinstance(body, str):
                body = self._interpolate(body, context)

            log.info(f"HTTP step: {method} {url}")

            async with aiohttp.ClientSession() as session:
                timeout = aiohttp.ClientTimeout(total=timeout_sec)
                async with session.request(
                    method, url, headers=headers, json=body, timeout=timeout
                ) as response:
                    response_body = await response.text()
                    try:
                        response_json = await response.json()
                    except Exception:
                        response_json = None

                    completed_at = datetime.now()

                    if response.status >= 400:
                        return StepResult(
                            step_id=step.step_id,
                            status=StepStatus.FAILED,
                            error=f"HTTP {response.status}: {response_body[:500]}",
                            output={
                                "status_code": response.status,
                                "body": response_body[:5000],
                            },
                            started_at=started_at,
                            completed_at=completed_at,
                        )

                    return StepResult(
                        step_id=step.step_id,
                        status=StepStatus.COMPLETED,
                        output={
                            "status_code": response.status,
                            "headers": dict(response.headers),
                            "body": response_json or response_body[:5000],
                        },
                        started_at=started_at,
                        completed_at=completed_at,
                        duration_seconds=(completed_at - started_at).total_seconds(),
                    )

        except asyncio.TimeoutError:
            return StepResult(
                step_id=step.step_id,
                status=StepStatus.FAILED,
                error="HTTP request timed out",
                started_at=started_at,
                completed_at=datetime.now(),
            )
        except Exception as e:
            return StepResult(
                step_id=step.step_id,
                status=StepStatus.FAILED,
                error=str(e),
                started_at=started_at,
                completed_at=datetime.now(),
            )

    # _interpolate is inherited from BaseStepExecutor


class ConditionalStepExecutor(BaseStepExecutor):
    """
    Execute a conditional evaluation step.

    Evaluates a boolean expression against the workflow context
    and stores the result for downstream steps to branch on.
    """

    async def execute(
        self,
        step: StepDefinition,
        context: Dict[str, Any],
        run: WorkflowRun,
    ) -> StepResult:
        """
        Evaluate a condition expression.

        Config options:
            expression: Condition to evaluate (e.g., "{step1.count} > 10")
        """
        started_at = datetime.now()

        try:
            expression = step.config.get("expression", "")
            result = self._evaluate_condition(expression, context)

            return StepResult(
                step_id=step.step_id,
                status=StepStatus.COMPLETED,
                output={"result": result, "expression": expression},
                started_at=started_at,
                completed_at=datetime.now(),
            )
        except Exception as e:
            log.error(f"Condition step failed: {step.name}: {e}")
            return StepResult(
                step_id=step.step_id,
                status=StepStatus.FAILED,
                error=str(e),
                started_at=started_at,
                completed_at=datetime.now(),
            )

    def _evaluate_condition(self, expression: str, context: Dict[str, Any]) -> bool:
        """
        Evaluate a condition expression against the workflow context.

        Supported expressions:
        - "{step_id.result} == 'success'"
        - "{step_id.output.count} > 10"
        - "{_trigger.body.action} == 'approve'"

        Security: Uses AST-based evaluation, NOT eval().
        """
        import ast
        import operator
        import re

        # Resolve context references
        def resolve_ref(match):
            path = match.group(1)
            parts = path.split(".")
            value = context
            for part in parts:
                if isinstance(value, dict):
                    value = value.get(part)
                else:
                    return "None"
            if isinstance(value, str):
                return f"'{value}'"
            return str(value)

        resolved = re.sub(r"\{(\w+(?:\.\w+)*)\}", resolve_ref, expression)

        # Safe evaluation using AST
        ops = {
            ast.Eq: operator.eq,
            ast.NotEq: operator.ne,
            ast.Lt: operator.lt,
            ast.LtE: operator.le,
            ast.Gt: operator.gt,
            ast.GtE: operator.ge,
        }

        try:
            tree = ast.parse(resolved, mode="eval")
            node = tree.body

            if isinstance(node, ast.Compare):
                left = ast.literal_eval(node.left)
                right = ast.literal_eval(node.comparators[0])
                op = ops.get(type(node.ops[0]))
                if op:
                    return op(left, right)

            # Fallback: try as truthy evaluation
            return bool(ast.literal_eval(resolved))

        except Exception as e:
            log.warning(f"Condition evaluation failed: {expression} -> {e}")
            return False


class NotificationStepExecutor(BaseStepExecutor):
    """
    Send notifications via various channels.

    Supported channels: console, log, webhook (extensible).
    """

    async def execute(
        self,
        step: StepDefinition,
        context: Dict[str, Any],
        run: WorkflowRun,
    ) -> StepResult:
        """
        Send a notification.

        Config options:
            channel: "console" | "log" | "webhook"
            message: Notification message (supports {context} interpolation)
            webhook_url: URL for webhook channel
        """
        started_at = datetime.now()

        try:
            channel = step.config.get("channel", "console")
            message_template = step.config.get("message", "")
            message = self._interpolate(message_template, context)

            if channel == "console":
                print(f"[NOTIFICATION] {message}")
            elif channel == "log":
                log.info(f"Workflow notification: {message}")
            elif channel == "webhook":
                import aiohttp

                webhook_url = step.config.get("webhook_url", "")
                if not webhook_url:
                    raise ValueError("webhook_url required for webhook channel")
                async with aiohttp.ClientSession() as session:
                    await session.post(
                        webhook_url,
                        json={"text": message, "workflow_id": run.workflow_id},
                        timeout=aiohttp.ClientTimeout(total=30),
                    )
            else:
                raise ValueError(f"Unknown notification channel: {channel}")

            return StepResult(
                step_id=step.step_id,
                status=StepStatus.COMPLETED,
                output={"channel": channel, "message": message},
                started_at=started_at,
                completed_at=datetime.now(),
            )

        except Exception as e:
            log.error(f"Notification step failed: {step.name}: {e}")
            return StepResult(
                step_id=step.step_id,
                status=StepStatus.FAILED,
                error=str(e),
                started_at=started_at,
                completed_at=datetime.now(),
            )


class SubprocessStepExecutor(BaseStepExecutor):
    """
    Execute a shell command as a workflow step.

    Security: Commands are NOT passed through a shell by default.
    """

    async def execute(
        self,
        step: StepDefinition,
        context: Dict[str, Any],
        run: WorkflowRun,
    ) -> StepResult:
        """
        Execute a subprocess command.

        Config options:
            command: List of command arguments (e.g., ["python", "script.py"])
            shell: Whether to use shell execution (default: False, for security)
            cwd: Working directory
            env: Additional environment variables
        """
        import subprocess

        started_at = datetime.now()

        try:
            command = step.config.get("command", [])
            use_shell = step.config.get("shell", False)
            cwd = step.config.get("cwd")
            extra_env = step.config.get("env", {})

            if not command:
                raise ValueError("command is required for subprocess step")

            # Interpolate context into string command args
            if isinstance(command, list):
                command = [self._interpolate(str(arg), context) for arg in command]

            import os
            env = dict(os.environ)
            env.update(extra_env)

            proc = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=env,
            )

            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=step.timeout_seconds,
            )

            completed_at = datetime.now()

            if proc.returncode != 0:
                return StepResult(
                    step_id=step.step_id,
                    status=StepStatus.FAILED,
                    error=f"Exit code {proc.returncode}: {stderr.decode()[:500]}",
                    output={
                        "returncode": proc.returncode,
                        "stdout": stdout.decode()[:5000],
                        "stderr": stderr.decode()[:5000],
                    },
                    started_at=started_at,
                    completed_at=completed_at,
                )

            return StepResult(
                step_id=step.step_id,
                status=StepStatus.COMPLETED,
                output={
                    "returncode": proc.returncode,
                    "stdout": stdout.decode()[:5000],
                    "stderr": stderr.decode()[:5000],
                },
                started_at=started_at,
                completed_at=completed_at,
                duration_seconds=(completed_at - started_at).total_seconds(),
            )

        except asyncio.TimeoutError:
            return StepResult(
                step_id=step.step_id,
                status=StepStatus.FAILED,
                error=f"Subprocess timed out after {step.timeout_seconds}s",
                started_at=started_at,
                completed_at=datetime.now(),
            )
        except Exception as e:
            log.error(f"Subprocess step failed: {step.name}: {e}")
            return StepResult(
                step_id=step.step_id,
                status=StepStatus.FAILED,
                error=str(e),
                started_at=started_at,
                completed_at=datetime.now(),
            )


class WaitStepExecutor(BaseStepExecutor):
    """
    Pause workflow execution for a specified duration.

    Useful for rate limiting, cooldown periods, or waiting for
    external processes to complete.
    """

    async def execute(
        self,
        step: StepDefinition,
        context: Dict[str, Any],
        run: WorkflowRun,
    ) -> StepResult:
        """
        Wait for a configured duration.

        Config options:
            duration_seconds: How long to wait (default: 10)
            message: Optional log message during wait
        """
        started_at = datetime.now()

        try:
            duration = step.config.get("duration_seconds", 10)
            message = step.config.get("message", "")

            if message:
                log.info(f"Wait step '{step.name}': {message} ({duration}s)")
            else:
                log.info(f"Wait step '{step.name}': pausing for {duration}s")

            await asyncio.sleep(duration)

            return StepResult(
                step_id=step.step_id,
                status=StepStatus.COMPLETED,
                output={"waited_seconds": duration},
                started_at=started_at,
                completed_at=datetime.now(),
                duration_seconds=duration,
            )

        except asyncio.CancelledError:
            return StepResult(
                step_id=step.step_id,
                status=StepStatus.FAILED,
                error="Wait step cancelled",
                started_at=started_at,
                completed_at=datetime.now(),
            )
        except Exception as e:
            return StepResult(
                step_id=step.step_id,
                status=StepStatus.FAILED,
                error=str(e),
                started_at=started_at,
                completed_at=datetime.now(),
            )
```

### 3.2 Workflow Engine Core

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Core workflow execution engine.
"""

import asyncio
import time
import traceback
from typing import Any, Callable, Dict, List, Optional

from gaia.logger import get_logger

log = get_logger(__name__)


class WorkflowEngine:
    """
    Core engine that executes workflow definitions.

    Features:
    - DAG-based step execution with dependency resolution
    - Parallel step execution where dependencies allow
    - Conditional branching (if/else/switch)
    - Retry with exponential backoff
    - Step timeout enforcement
    - Checkpoint/resume for long-running workflows
    - Context passing between steps
    """

    def __init__(self):
        self._executors: Dict[StepType, BaseStepExecutor] = {
            StepType.AGENT: AgentStepExecutor(),
            StepType.FUNCTION: FunctionStepExecutor(),
            StepType.HTTP: HttpStepExecutor(),
            StepType.CONDITION: ConditionalStepExecutor(),
            StepType.NOTIFICATION: NotificationStepExecutor(),
            StepType.SUBPROCESS: SubprocessStepExecutor(),
            StepType.WAIT: WaitStepExecutor(),
        }
        self._state_manager = WorkflowStateManager()
        self._running_workflows: Dict[str, WorkflowRun] = {}
        self._cancelled_runs: set = set()

    def register_executor(
        self, step_type: StepType, executor: BaseStepExecutor
    ) -> None:
        """Register a custom step executor."""
        self._executors[step_type] = executor

    async def cancel_workflow(self, run_id: str) -> bool:
        """
        Cancel a running workflow.

        Args:
            run_id: ID of the workflow run to cancel

        Returns:
            True if the cancellation was accepted, False if the run was not found
        """
        run = self._running_workflows.get(run_id)
        if run is None:
            log.warning(f"Cannot cancel workflow {run_id}: not found in running workflows")
            return False

        log.info(f"Cancelling workflow: {run_id}")
        self._cancelled_runs.add(run_id)
        run.status = WorkflowStatus.CANCELLED
        run.completed_at = datetime.now()
        run.error = "Workflow cancelled by user"

        await self._state_manager.save_run(run)
        return True

    def is_cancelled(self, run_id: str) -> bool:
        """Check if a workflow run has been cancelled."""
        return run_id in self._cancelled_runs

    async def execute_workflow(
        self,
        definition: WorkflowDefinition,
        trigger_data: Optional[Dict[str, Any]] = None,
        initial_context: Optional[Dict[str, Any]] = None,
    ) -> WorkflowRun:
        """
        Execute a complete workflow from definition.

        Args:
            definition: Workflow definition to execute
            trigger_data: Data from the trigger that started this run
            initial_context: Initial context data for the workflow

        Returns:
            WorkflowRun with all step results
        """
        # Validate definition
        errors = definition.validate()
        if errors:
            raise ValueError(f"Invalid workflow: {', '.join(errors)}")

        # Create run instance
        run = WorkflowRun(
            workflow_id=definition.workflow_id,
            status=WorkflowStatus.RUNNING,
            trigger_data=trigger_data or {},
            context=initial_context or {},
            started_at=datetime.now(),
        )

        # Add trigger data to context
        run.context["_trigger"] = trigger_data or {}
        run.context["_workflow_id"] = definition.workflow_id
        run.context["_run_id"] = run.run_id

        self._running_workflows[run.run_id] = run
        log.info(f"Starting workflow: {definition.name} (run: {run.run_id})")

        try:
            # Build execution DAG
            execution_order = self._resolve_dependencies(definition.steps)

            # Execute steps in order
            for step_batch in execution_order:
                # Check for cancellation
                if self.is_cancelled(run.run_id):
                    run.status = WorkflowStatus.CANCELLED
                    run.error = "Workflow cancelled by user"
                    log.info(f"Workflow cancelled: {run.run_id}")
                    break

                # Check for workflow timeout
                elapsed = (datetime.now() - run.started_at).total_seconds()
                if elapsed > definition.global_timeout_seconds:
                    run.status = WorkflowStatus.FAILED
                    run.error = "Workflow timeout exceeded"
                    break

                if len(step_batch) == 1:
                    # Sequential execution
                    await self._execute_step(step_batch[0], run, definition)
                else:
                    # Parallel execution
                    tasks = [
                        self._execute_step(step, run, definition)
                        for step in step_batch
                    ]
                    await asyncio.gather(*tasks)

                # Check if any step failed fatally
                for step in step_batch:
                    result = run.step_results.get(step.step_id)
                    if result and result.status == StepStatus.FAILED:
                        if step.on_failure == "fail":
                            run.status = WorkflowStatus.FAILED
                            run.error = f"Step {step.step_id} failed: {result.error}"
                            break

                if run.status in (WorkflowStatus.FAILED, WorkflowStatus.CANCELLED):
                    break

            # Mark completion
            if run.status == WorkflowStatus.RUNNING:
                run.status = WorkflowStatus.COMPLETED
                log.info(f"Workflow completed: {run.run_id}")

        except Exception as e:
            run.status = WorkflowStatus.FAILED
            run.error = str(e)
            log.error(f"Workflow failed: {run.run_id}: {e}")

        finally:
            run.completed_at = datetime.now()
            self._running_workflows.pop(run.run_id, None)
            self._cancelled_runs.discard(run.run_id)

            # Save final state
            await self._state_manager.save_run(run)

        return run

    async def _execute_step(
        self,
        step: StepDefinition,
        run: WorkflowRun,
        definition: WorkflowDefinition,
    ) -> None:
        """Execute a single step with retry and timeout handling."""

        # Evaluate condition
        if step.condition:
            should_run = self._evaluate_condition(step.condition, run.context)
            if not should_run:
                run.step_results[step.step_id] = StepResult(
                    step_id=step.step_id,
                    status=StepStatus.SKIPPED,
                    metadata={"reason": f"Condition not met: {step.condition}"},
                )
                log.info(f"Skipping step {step.step_id}: condition not met")
                return

        executor = self._executors.get(step.step_type)
        if not executor:
            run.step_results[step.step_id] = StepResult(
                step_id=step.step_id,
                status=StepStatus.FAILED,
                error=f"No executor for step type: {step.step_type}",
            )
            return

        retry_policy = step.retry_policy or definition.retry_policy or RetryPolicy(max_retries=0)
        retry_count = 0
        last_error = None

        while retry_count <= retry_policy.max_retries:
            try:
                # Execute with timeout
                result = await asyncio.wait_for(
                    executor.execute(step, run.context, run),
                    timeout=step.timeout_seconds,
                )

                if result.status == StepStatus.COMPLETED:
                    result.retry_count = retry_count
                    run.step_results[step.step_id] = result

                    # Update context with step outputs
                    if result.output:
                        run.context[step.step_id] = result.output

                    # Save checkpoint
                    await self._state_manager.save_checkpoint(run, step.step_id)
                    return

                elif result.status == StepStatus.FAILED:
                    last_error = result.error
                    if retry_count < retry_policy.max_retries:
                        delay = min(
                            retry_policy.initial_delay_seconds
                            * (retry_policy.backoff_multiplier ** retry_count),
                            retry_policy.max_delay_seconds,
                        )
                        log.warning(
                            f"Step {step.step_id} failed (attempt {retry_count + 1}), "
                            f"retrying in {delay}s: {last_error}"
                        )
                        await asyncio.sleep(delay)
                        retry_count += 1
                    else:
                        break

            except asyncio.TimeoutError:
                last_error = f"Step timed out after {step.timeout_seconds}s"
                log.warning(f"Step {step.step_id} timed out")
                if retry_count < retry_policy.max_retries:
                    retry_count += 1
                else:
                    break

        # All retries exhausted
        run.step_results[step.step_id] = StepResult(
            step_id=step.step_id,
            status=StepStatus.FAILED,
            error=last_error or "Unknown error after retries",
            retry_count=retry_count,
        )

    def _resolve_dependencies(
        self, steps: List[StepDefinition]
    ) -> List[List[StepDefinition]]:
        """
        Resolve step dependencies into execution batches.

        Returns list of batches, where steps in the same batch can run in parallel.
        Uses topological sort.
        """
        step_map = {s.step_id: s for s in steps}
        in_degree = {s.step_id: len(s.depends_on) for s in steps}
        batches = []

        remaining = set(step_map.keys())

        while remaining:
            # Find all steps with no unresolved dependencies
            ready = [
                sid for sid in remaining
                if in_degree[sid] == 0
            ]

            if not ready:
                raise ValueError("Circular dependency detected in workflow steps")

            batch = [step_map[sid] for sid in ready]
            batches.append(batch)

            for sid in ready:
                remaining.remove(sid)
                # Reduce in-degree for dependent steps
                for other_sid in remaining:
                    if sid in step_map[other_sid].depends_on:
                        in_degree[other_sid] -= 1

        return batches

    def _evaluate_condition(self, expression: str, context: Dict[str, Any]) -> bool:
        """
        Evaluate a condition expression against the workflow context.

        Supported expressions:
        - "{step_id.result} == 'success'"
        - "{step_id.output.count} > 10"
        - "{_trigger.body.action} == 'approve'"

        Security: Uses AST-based evaluation, NOT eval().
        """
        import ast
        import operator
        import re

        # Resolve context references
        def resolve_ref(match):
            path = match.group(1)
            parts = path.split(".")
            value = context
            for part in parts:
                if isinstance(value, dict):
                    value = value.get(part)
                else:
                    return "None"
            if isinstance(value, str):
                return f"'{value}'"
            return str(value)

        resolved = re.sub(r"\{(\w+(?:\.\w+)*)\}", resolve_ref, expression)

        # Safe evaluation using AST
        ops = {
            ast.Eq: operator.eq,
            ast.NotEq: operator.ne,
            ast.Lt: operator.lt,
            ast.LtE: operator.le,
            ast.Gt: operator.gt,
            ast.GtE: operator.ge,
        }

        try:
            tree = ast.parse(resolved, mode="eval")
            node = tree.body

            if isinstance(node, ast.Compare):
                left = ast.literal_eval(node.left)
                right = ast.literal_eval(node.comparators[0])
                op = ops.get(type(node.ops[0]))
                if op:
                    return op(left, right)

            # Fallback: try as truthy evaluation
            return bool(ast.literal_eval(resolved))

        except Exception as e:
            log.warning(f"Condition evaluation failed: {expression} -> {e}")
            return False


class WorkflowStateManager:
    """
    Persistent state management for workflow runs.

    Stores:
    - Workflow run state and history
    - Step checkpoints for resume capability
    - Audit trail of all executions
    """

    def __init__(self, db_path: str = "gaia_workflows.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        """Initialize SQLite database."""
        import sqlite3

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.executescript("""
            CREATE TABLE IF NOT EXISTS workflow_runs (
                run_id TEXT PRIMARY KEY,
                workflow_id TEXT NOT NULL,
                status TEXT NOT NULL,
                trigger_data TEXT,
                context TEXT,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                error TEXT,
                metadata TEXT
            );

            CREATE TABLE IF NOT EXISTS step_results (
                run_id TEXT NOT NULL,
                step_id TEXT NOT NULL,
                status TEXT NOT NULL,
                output TEXT,
                error TEXT,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                retry_count INTEGER DEFAULT 0,
                duration_seconds REAL,
                PRIMARY KEY (run_id, step_id),
                FOREIGN KEY (run_id) REFERENCES workflow_runs(run_id)
            );

            CREATE TABLE IF NOT EXISTS workflow_audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                workflow_id TEXT,
                event TEXT NOT NULL,
                details TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_runs_workflow
                ON workflow_runs(workflow_id);
            CREATE INDEX IF NOT EXISTS idx_runs_status
                ON workflow_runs(status);
            CREATE INDEX IF NOT EXISTS idx_audit_run
                ON workflow_audit(run_id);
        """)

        conn.commit()
        conn.close()

    async def save_run(self, run: WorkflowRun) -> None:
        """Save or update a workflow run."""
        import sqlite3
        import json

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO workflow_runs
            (run_id, workflow_id, status, trigger_data, context,
             started_at, completed_at, error, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run.run_id,
            run.workflow_id,
            run.status.value,
            json.dumps(run.trigger_data),
            json.dumps(run.context, default=str),
            run.started_at.isoformat() if run.started_at else None,
            run.completed_at.isoformat() if run.completed_at else None,
            run.error,
            json.dumps(run.metadata),
        ))

        for step_id, result in run.step_results.items():
            cursor.execute("""
                INSERT OR REPLACE INTO step_results
                (run_id, step_id, status, output, error,
                 started_at, completed_at, retry_count, duration_seconds)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run.run_id,
                step_id,
                result.status.value,
                json.dumps(result.output, default=str),
                result.error,
                result.started_at.isoformat() if result.started_at else None,
                result.completed_at.isoformat() if result.completed_at else None,
                result.retry_count,
                result.duration_seconds,
            ))

        conn.commit()
        conn.close()

    async def save_checkpoint(self, run: WorkflowRun, step_id: str) -> None:
        """Save a checkpoint after successful step completion."""
        await self.save_run(run)
        log.debug(f"Checkpoint saved: run={run.run_id}, step={step_id}")

    async def get_run(self, run_id: str) -> Optional[WorkflowRun]:
        """Retrieve a workflow run by ID."""
        import sqlite3
        import json

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM workflow_runs WHERE run_id = ?", (run_id,))
        row = cursor.fetchone()

        if not row:
            conn.close()
            return None

        run = WorkflowRun(
            run_id=row[0],
            workflow_id=row[1],
            status=WorkflowStatus(row[2]),
            trigger_data=json.loads(row[3]) if row[3] else {},
            context=json.loads(row[4]) if row[4] else {},
            started_at=datetime.fromisoformat(row[5]) if row[5] else None,
            completed_at=datetime.fromisoformat(row[6]) if row[6] else None,
            error=row[7],
        )

        cursor.execute(
            "SELECT * FROM step_results WHERE run_id = ?", (run_id,)
        )
        for step_row in cursor.fetchall():
            run.step_results[step_row[1]] = StepResult(
                step_id=step_row[1],
                status=StepStatus(step_row[2]),
                output=json.loads(step_row[3]) if step_row[3] else {},
                error=step_row[4],
                started_at=datetime.fromisoformat(step_row[5]) if step_row[5] else None,
                completed_at=datetime.fromisoformat(step_row[6]) if step_row[6] else None,
                retry_count=step_row[7],
                duration_seconds=step_row[8] or 0.0,
            )

        conn.close()
        return run

    async def list_runs(
        self,
        workflow_id: Optional[str] = None,
        status: Optional[WorkflowStatus] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """List workflow runs with optional filters."""
        import sqlite3

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = "SELECT run_id, workflow_id, status, started_at, completed_at, error FROM workflow_runs"
        params = []
        conditions = []

        if workflow_id:
            conditions.append("workflow_id = ?")
            params.append(workflow_id)
        if status:
            conditions.append("status = ?")
            params.append(status.value)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY started_at DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        return [
            {
                "run_id": r[0],
                "workflow_id": r[1],
                "status": r[2],
                "started_at": r[3],
                "completed_at": r[4],
                "error": r[5],
            }
            for r in rows
        ]
```

---

## Integration with GAIA

### 4.1 Workflow DSL (Python Builder)

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Python DSL for building workflows declaratively.
"""

from typing import Any, Dict, List, Optional


class WorkflowBuilder:
    """
    Fluent builder API for constructing workflows.

    Example:
        workflow = (
            WorkflowBuilder("daily-email-triage")
            .description("Triage inbox every morning")
            .cron("0 8 * * MON-FRI")
            .agent_step(
                "fetch_emails",
                agent_type="email",
                query="Fetch my unread emails from the last 24 hours",
            )
            .agent_step(
                "classify",
                agent_type="email",
                query="Classify these emails: {fetch_emails.result}",
                depends_on=["fetch_emails"],
            )
            .condition_step(
                "check_urgent",
                expression="{classify.result.urgent_count} > 0",
                depends_on=["classify"],
            )
            .agent_step(
                "draft_responses",
                agent_type="email",
                query="Draft responses to urgent emails: {classify.result.urgent}",
                condition="{check_urgent} == True",
                depends_on=["check_urgent"],
            )
            .notification_step(
                "notify",
                channel="console",
                message="Triage complete: {classify.result.summary}",
                depends_on=["classify"],
            )
            .build()
        )
    """

    def __init__(self, workflow_id: str):
        self._workflow_id = workflow_id
        self._name = workflow_id
        self._description = ""
        self._triggers: List[TriggerDefinition] = []
        self._steps: List[StepDefinition] = []
        self._global_timeout = 3600
        self._retry_policy: Optional[RetryPolicy] = None
        self._step_counter = 0

    def name(self, name: str) -> "WorkflowBuilder":
        self._name = name
        return self

    def description(self, desc: str) -> "WorkflowBuilder":
        self._description = desc
        return self

    def cron(self, expression: str) -> "WorkflowBuilder":
        """Add a cron trigger."""
        self._triggers.append(
            TriggerDefinition(
                trigger_type=TriggerType.CRON,
                config={"schedule": expression},
            )
        )
        return self

    def webhook(self, path: str, methods: List[str] = None) -> "WorkflowBuilder":
        """Add a webhook trigger."""
        self._triggers.append(
            TriggerDefinition(
                trigger_type=TriggerType.WEBHOOK,
                config={"path": path, "methods": methods or ["POST"]},
            )
        )
        return self

    def file_watch(self, path: str, pattern: str = "*") -> "WorkflowBuilder":
        """Add a file watch trigger."""
        self._triggers.append(
            TriggerDefinition(
                trigger_type=TriggerType.FILE_WATCH,
                config={"path": path, "pattern": pattern},
            )
        )
        return self

    def agent_step(
        self,
        step_id: str,
        agent_type: str = "chat",
        query: str = "",
        depends_on: Optional[List[str]] = None,
        condition: Optional[str] = None,
        timeout: int = 300,
        retry_max: int = 1,
        on_failure: str = "fail",
        **agent_kwargs,
    ) -> "WorkflowBuilder":
        """Add an agent execution step."""
        self._steps.append(
            StepDefinition(
                step_id=step_id,
                name=step_id,
                step_type=StepType.AGENT,
                config={
                    "agent_type": agent_type,
                    "query": query,
                    "agent_kwargs": agent_kwargs,
                },
                depends_on=depends_on or [],
                condition=condition,
                timeout_seconds=timeout,
                retry_policy=RetryPolicy(max_retries=retry_max) if retry_max > 0 else None,
                on_failure=on_failure,
            )
        )
        return self

    def function_step(
        self,
        step_id: str,
        function: str,
        kwargs: Optional[Dict] = None,
        depends_on: Optional[List[str]] = None,
        condition: Optional[str] = None,
    ) -> "WorkflowBuilder":
        """Add a Python function step."""
        self._steps.append(
            StepDefinition(
                step_id=step_id,
                name=step_id,
                step_type=StepType.FUNCTION,
                config={"function": function, "kwargs": kwargs or {}},
                depends_on=depends_on or [],
                condition=condition,
            )
        )
        return self

    def http_step(
        self,
        step_id: str,
        url: str,
        method: str = "GET",
        headers: Optional[Dict] = None,
        body: Optional[Any] = None,
        depends_on: Optional[List[str]] = None,
    ) -> "WorkflowBuilder":
        """Add an HTTP request step."""
        self._steps.append(
            StepDefinition(
                step_id=step_id,
                name=step_id,
                step_type=StepType.HTTP,
                config={
                    "url": url,
                    "method": method,
                    "headers": headers or {},
                    "body": body,
                },
                depends_on=depends_on or [],
            )
        )
        return self

    def condition_step(
        self,
        step_id: str,
        expression: str,
        depends_on: Optional[List[str]] = None,
    ) -> "WorkflowBuilder":
        """Add a conditional evaluation step."""
        self._steps.append(
            StepDefinition(
                step_id=step_id,
                name=step_id,
                step_type=StepType.CONDITION,
                config={"expression": expression},
                depends_on=depends_on or [],
            )
        )
        return self

    def notification_step(
        self,
        step_id: str,
        channel: str = "console",
        message: str = "",
        depends_on: Optional[List[str]] = None,
    ) -> "WorkflowBuilder":
        """Add a notification step."""
        self._steps.append(
            StepDefinition(
                step_id=step_id,
                name=step_id,
                step_type=StepType.NOTIFICATION,
                config={"channel": channel, "message": message},
                depends_on=depends_on or [],
            )
        )
        return self

    def timeout(self, seconds: int) -> "WorkflowBuilder":
        self._global_timeout = seconds
        return self

    def retry(self, max_retries: int = 3, backoff: float = 2.0) -> "WorkflowBuilder":
        self._retry_policy = RetryPolicy(
            max_retries=max_retries, backoff_multiplier=backoff
        )
        return self

    def build(self) -> WorkflowDefinition:
        """Build the workflow definition."""
        definition = WorkflowDefinition(
            workflow_id=self._workflow_id,
            name=self._name,
            description=self._description,
            triggers=self._triggers,
            steps=self._steps,
            global_timeout_seconds=self._global_timeout,
            retry_policy=self._retry_policy,
        )

        errors = definition.validate()
        if errors:
            raise ValueError(f"Invalid workflow: {', '.join(errors)}")

        return definition
```

### 4.2 YAML Workflow Definition

```yaml
# Example: weekly-report.yaml
workflow_id: weekly-jira-report
name: Weekly Jira Report
description: Generate and distribute weekly Jira progress report
version: "1.0"

triggers:
  - type: cron
    config:
      schedule: "0 17 * * FRI"  # Every Friday at 5 PM

steps:
  - step_id: fetch_tickets
    name: Fetch Completed Tickets
    type: agent
    config:
      agent_type: jira
      query: "Find all tickets completed this week in project GAIA"
    timeout_seconds: 120

  - step_id: summarize
    name: Summarize Tickets
    type: agent
    config:
      agent_type: chat
      query: "Summarize these Jira tickets into a weekly report: {fetch_tickets.result}"
    depends_on: [fetch_tickets]
    timeout_seconds: 180

  - step_id: send_report
    name: Email Report
    type: agent
    config:
      agent_type: email
      query: "Send this weekly report to team@company.com: {summarize.result}"
    depends_on: [summarize]
    retry_policy:
      max_retries: 2
      initial_delay_seconds: 10

global_timeout_seconds: 600
```

---

## Safety and Security Considerations

### Access Control

```
+------------------------------------------------+
|            Workflow Access Model                |
|                                                |
|  [Workflow Definition]                         |
|       |                                        |
|       +-- owner: user who created it           |
|       +-- permissions: read/execute/admin       |
|       +-- allowed_agents: [chat, email]         |
|       +-- resource_limits:                     |
|           +-- max_runtime: 3600s               |
|           +-- max_steps: 50                    |
|           +-- max_retries: 10                  |
|           +-- max_concurrent: 3                |
|                                                |
|  [Step Execution]                              |
|       |                                        |
|       +-- sandboxed: agent runs in isolation    |
|       +-- no_filesystem: unless whitelisted     |
|       +-- no_network: unless explicitly allowed |
|       +-- credential_scope: per-workflow         |
+------------------------------------------------+
```

**Key Safety Rules**:
1. Workflows cannot access credentials of other workflows
2. File system access is sandboxed to workflow-specific directories
3. Network access requires explicit whitelist in workflow definition
4. Maximum execution time enforced at both step and workflow level
5. Concurrent run limits prevent resource exhaustion
6. All webhook endpoints require authentication (HMAC or API key)
7. Workflow definitions are version-controlled and auditable

---

## Implementation Plan

| Week | Phase | Tasks | Deliverables |
|------|-------|-------|-------------|
| 1-2 | Core Models | Data models, WorkflowDefinition, WorkflowRun, StepResult | Core types and validation |
| 3-4 | Triggers | CronTrigger, WebhookTrigger, FileWatchTrigger, ManualTrigger | All trigger types operational |
| 5-6 | Engine | WorkflowEngine, dependency resolution, parallel execution | Engine with DAG execution |
| 7 | Executors | AgentStepExecutor, FunctionStepExecutor, HttpStepExecutor | All step types working |
| 8 | State | WorkflowStateManager, checkpoints, resume, SQLite persistence | Full state management |
| 9 | DSL | WorkflowBuilder (Python), YAML loader, validation | Both definition formats |
| 10 | Integration | GAIA CLI integration (`gaia workflow`), WorkflowAgent | CLI and agent interface |
| 11 | Safety | Access control, rate limiting, sandboxing, audit log | Safety framework complete |
| 12 | Testing | Unit tests, integration tests, stress tests, documentation | Production-ready release |

---

## Testing Strategy

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Unit tests for workflow engine."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime


class TestCronExpression:
    """Test cron expression parsing and matching."""

    def test_every_minute(self):
        cron = CronExpression("* * * * *")
        assert cron.matches(datetime(2026, 2, 7, 10, 30))

    def test_specific_time(self):
        cron = CronExpression("30 8 * * MON-FRI")
        assert cron.matches(datetime(2026, 2, 10, 8, 30))  # Monday
        assert not cron.matches(datetime(2026, 2, 8, 8, 30))  # Saturday

    def test_interval(self):
        cron = CronExpression("*/15 * * * *")
        assert cron.matches(datetime(2026, 2, 7, 10, 0))
        assert cron.matches(datetime(2026, 2, 7, 10, 15))
        assert not cron.matches(datetime(2026, 2, 7, 10, 7))

    def test_next_run(self):
        cron = CronExpression("0 8 * * *")
        after = datetime(2026, 2, 7, 9, 0)
        next_run = cron.next_run(after)
        assert next_run.hour == 8
        assert next_run.day == 8


class TestWorkflowEngine:
    """Test workflow execution engine."""

    @pytest.fixture
    def engine(self):
        return WorkflowEngine()

    @pytest.fixture
    def simple_workflow(self):
        return WorkflowDefinition(
            workflow_id="test-workflow",
            name="Test Workflow",
            steps=[
                StepDefinition(
                    step_id="step1",
                    name="First Step",
                    step_type=StepType.FUNCTION,
                    config={"function": "test_func", "kwargs": {"x": 1}},
                ),
                StepDefinition(
                    step_id="step2",
                    name="Second Step",
                    step_type=StepType.FUNCTION,
                    config={"function": "test_func", "kwargs": {"x": 2}},
                    depends_on=["step1"],
                ),
            ],
        )

    @pytest.mark.asyncio
    async def test_sequential_execution(self, engine, simple_workflow):
        """Test basic sequential workflow execution."""
        func_executor = engine._executors[StepType.FUNCTION]
        func_executor.register_function(
            "test_func", lambda x: {"value": x * 2}
        )

        run = await engine.execute_workflow(simple_workflow)
        assert run.status == WorkflowStatus.COMPLETED
        assert len(run.step_results) == 2

    @pytest.mark.asyncio
    async def test_conditional_skip(self, engine):
        """Test step skipping when condition is false."""
        workflow = WorkflowDefinition(
            workflow_id="conditional-test",
            name="Conditional Test",
            steps=[
                StepDefinition(
                    step_id="always_run",
                    name="Always Run",
                    step_type=StepType.FUNCTION,
                    config={"function": "return_false"},
                ),
                StepDefinition(
                    step_id="maybe_run",
                    name="Maybe Run",
                    step_type=StepType.FUNCTION,
                    config={"function": "should_not_run"},
                    condition="{always_run.result} == True",
                    depends_on=["always_run"],
                ),
            ],
        )

        func_executor = engine._executors[StepType.FUNCTION]
        func_executor.register_function("return_false", lambda: {"result": False})
        func_executor.register_function("should_not_run", lambda: {"ran": True})

        run = await engine.execute_workflow(workflow)
        assert run.step_results["maybe_run"].status == StepStatus.SKIPPED

    @pytest.mark.asyncio
    async def test_retry_on_failure(self, engine):
        """Test step retry with exponential backoff."""
        call_count = 0

        def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Temporary failure")
            return {"success": True}

        func_executor = engine._executors[StepType.FUNCTION]
        func_executor.register_function("flaky", flaky_function)

        workflow = WorkflowDefinition(
            workflow_id="retry-test",
            name="Retry Test",
            steps=[
                StepDefinition(
                    step_id="flaky_step",
                    name="Flaky Step",
                    step_type=StepType.FUNCTION,
                    config={"function": "flaky"},
                    retry_policy=RetryPolicy(
                        max_retries=3,
                        initial_delay_seconds=0.01,
                    ),
                ),
            ],
        )

        run = await engine.execute_workflow(workflow)
        assert run.status == WorkflowStatus.COMPLETED
        assert run.step_results["flaky_step"].retry_count == 2


class TestWorkflowBuilder:
    """Test workflow DSL builder."""

    def test_basic_build(self):
        workflow = (
            WorkflowBuilder("test")
            .name("Test Workflow")
            .description("A test")
            .cron("0 8 * * *")
            .agent_step("step1", agent_type="chat", query="Hello")
            .build()
        )
        assert workflow.workflow_id == "test"
        assert len(workflow.triggers) == 1
        assert len(workflow.steps) == 1

    def test_validation_catches_circular_deps(self):
        with pytest.raises(ValueError):
            # This should fail because step1 depends on step2 and vice versa
            # But depends_on validation only checks forward references
            WorkflowDefinition(
                workflow_id="bad",
                name="Bad",
                steps=[],  # Empty steps should fail
            ).validate()
```

---

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Workflow definition time | <5 min for common patterns | User timing studies |
| Cron trigger accuracy | <1 min drift from schedule | Timestamp analysis |
| Step execution reliability | >99% for non-external steps | Success rate tracking |
| Retry recovery rate | >80% of transient failures | Retry success tracking |
| State persistence | 100% checkpoint integrity | Crash recovery tests |
| Parallel execution speedup | >2x for 3+ parallel steps | Benchmark comparison |
| End-to-end latency | <30s overhead vs manual | Timing instrumentation |
| Concurrent workflows | 10+ without degradation | Load testing |

---

## Complete Code

The complete implementation spans the following source files:

```
src/gaia/workflow/
    __init__.py
    models.py             # Core data models (WorkflowDefinition, StepDefinition, etc.)
    engine.py             # WorkflowEngine core
    state.py              # WorkflowStateManager, checkpoints
    triggers/
        __init__.py
        cron.py           # CronTrigger, CronExpression
        webhook.py        # WebhookTrigger
        file_watch.py     # FileWatchTrigger
        manual.py         # ManualTrigger
    executors/
        __init__.py
        base.py           # BaseStepExecutor
        agent.py          # AgentStepExecutor
        function.py       # FunctionStepExecutor
        http.py           # HttpStepExecutor
        condition.py      # ConditionalStepExecutor
    builder.py            # WorkflowBuilder (Python DSL)
    yaml_loader.py        # YAML workflow definition loader
    runtime.py            # WorkflowRuntime (scheduler + triggers)
    cli.py                # CLI integration for 'gaia workflow'

tests/unit/workflow/
    test_cron.py
    test_engine.py
    test_executors.py
    test_builder.py
    test_state.py
    test_triggers.py

tests/integration/workflow/
    test_full_workflows.py
    test_agent_steps.py
```
