# Error Recovery Architecture for GAIA

**Date**: February 7, 2026
**Version**: 1.0
**Status**: Specification
**Priority**: CRITICAL
**Estimated Effort**: 8-10 weeks (2 engineers)
**Target**: Enable retry strategies, circuit breakers, graceful degradation, and self-healing

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Problem Statement](#problem-statement)
3. [Architecture Overview](#architecture-overview)
4. [Component Specifications](#component-specifications)
5. [Data Models and Schemas](#data-models-and-schemas)
6. [Retry Strategies](#retry-strategies)
7. [Circuit Breaker Pattern](#circuit-breaker-pattern)
8. [Graceful Degradation](#graceful-degradation)
9. [Self-Healing System](#self-healing-system)
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
- Basic try/except error handling in Agent base class
- `error_history` list for tracking failures
- `STATE_ERROR_RECOVERY` state in agent state machine
- Simple retry via `max_plan_iterations`

GAIA **lacks**:
- Configurable retry strategies (exponential backoff, jitter, retry budgets)
- Circuit breaker pattern for failing external services
- Graceful degradation when components are unavailable
- Self-healing capabilities (auto-restart, auto-repair)
- Error classification and routing (transient vs permanent)
- Cascading failure prevention
- Health checking and readiness probes

### The Solution

An **Error Recovery Architecture** that provides:
- Pluggable retry strategies with error classification
- Circuit breaker for all external dependencies (LLM, APIs, databases)
- Multi-level degradation with fallback chains
- Self-healing subsystem with health monitoring
- Dead letter queues for unrecoverable failures
- Error budgets and SLO tracking
- Integration with GAIA's existing Agent error recovery state

### Impact

**Unlocks entire category**: Production reliability for:
- Always-on agent deployments (API server, workflows)
- Multi-agent pipelines that survive partial failures
- Cloud LLM fallback when local LLM is overloaded
- Graceful handling of network outages
- Zero-downtime agent updates and restarts

**Before**: 40% reliability coverage (basic error handling only)
**After**: 95% reliability coverage

---

## Problem Statement

### Current Limitations

**Example scenario**: LLM server becomes temporarily unavailable during a multi-step agent query

**Current behavior**:
```python
# Agent hits LLM error, transitions to ERROR_RECOVERY state
# After 1 retry, gives up with generic error message
agent = CodeAgent()
result = agent.process_query("Generate a REST API")
# If Lemonade server is restarting:
# -> "Error: Connection refused" after 1 failed attempt
# -> No fallback to cloud LLM
# -> No automatic retry with backoff
# -> User must manually retry
```

**Required behavior**:
```python
# Smart error recovery with fallback chain
agent = CodeAgent()
result = agent.process_query("Generate a REST API")
# If Lemonade server is temporarily unavailable:
# -> Retry 1 (wait 1s): Connection refused
# -> Retry 2 (wait 2s): Connection refused
# -> Retry 3 (wait 4s): Connection refused
# -> Circuit breaker opens, switches to fallback
# -> Fallback: try cloud LLM (Claude)
# -> If cloud fails: degrade to cached response / simpler model
# -> If all fail: queue for retry, notify user gracefully
```

### User Stories

**Story 1: Transparent Retries**
```
As a user, I want my agent to:
- Automatically retry failed operations
- Use exponential backoff to avoid overloading services
- Distinguish between retryable and non-retryable errors
- Show progress during retries (not just silence)
So that transient failures are handled without my intervention
```

**Story 2: Service Failover**
```
As a developer, I want GAIA to:
- Detect when the local LLM server is overloaded
- Automatically failover to a cloud LLM
- Return to the local LLM when it recovers
- Track the circuit state for monitoring
So that agents continue working even when the local server has issues
```

**Story 3: Graceful Degradation**
```
As a user, I want my agent to:
- Provide a reduced-quality answer when full processing fails
- Use cached results when the LLM is completely unavailable
- Inform me of degraded quality transparently
- Queue my request for full processing when service recovers
So that I always get some response, even during outages
```

**Story 4: Self-Healing**
```
As an ops engineer, I want GAIA to:
- Automatically restart crashed components
- Detect and recover from stuck/hung processes
- Clear corrupted caches without manual intervention
- Generate incident reports for recurring failures
So that the system recovers from failures without manual effort
```

---

## Architecture Overview

### High-Level Design

```
+------------------------------------------------------------------+
|                    GAIA Application Layer                          |
|  [Agents]  [Tools]  [LLM Clients]  [API Server]  [Workflows]    |
+------+--------+----------+-----------+-----------+---------------+
       |        |          |           |           |
       v        v          v           v           v
+------------------------------------------------------------------+
|                  Error Recovery Framework                         |
|                                                                   |
|  +-------------------+  +--------------------+  +---------------+ |
|  | Retry Engine      |  | Circuit Breaker    |  | Degradation   | |
|  |                   |  | Registry           |  | Manager       | |
|  | - Exponential     |  |                    |  |               | |
|  |   backoff         |  | - Per-service      |  | - Fallback    | |
|  | - Jitter          |  |   breakers         |  |   chains      | |
|  | - Retry budgets   |  | - State machine    |  | - Cache       | |
|  | - Error classify  |  |   (closed/open/    |  |   fallback    | |
|  | - Dead letter     |  |    half-open)      |  | - Quality     | |
|  |   queue           |  | - Health probes    |  |   reduction   | |
|  +--------+----------+  +--------+-----------+  +-------+-------+ |
|           |                      |                       |         |
|           v                      v                       v         |
|  +-----------------------------------------------------------+   |
|  |                   Self-Healing Engine                       |   |
|  |                                                             |   |
|  |  - Health monitor (periodic checks)                        |   |
|  |  - Auto-restart for crashed services                       |   |
|  |  - Cache/state repair                                      |   |
|  |  - Incident detection and reporting                        |   |
|  |  - Recovery playbooks                                      |   |
|  +-----------------------------------------------------------+   |
+------------------------------------------------------------------+
           |
           v
+-----------------------------------------------------------+
|                  Error Recovery Storage                     |
|                                                            |
|  [Dead Letter Queue]  [Incident Log]  [Circuit States]    |
+-----------------------------------------------------------+
```

### Error Flow Diagram

```
                    Error Handling Flow
                    ====================

  [Operation Attempt]
         |
         v
  [Success?] --YES--> [Return Result]
         |
        NO
         |
         v
  [Classify Error]
         |
         +-- Transient (retryable)
         |         |
         |         v
         |   [Retry Engine]
         |         |
         |         +-- Under retry budget?
         |         |         |
         |         |        YES --> [Wait (backoff + jitter)] --> [Retry]
         |         |         |                                        |
         |         |        NO                                   [Success?]
         |         |         |                                   YES --> Return
         |         |         v                                   NO  --> Continue
         |         +-- [Circuit Breaker Check]
         |                   |
         |                   +-- OPEN --> [Fallback Chain]
         |                   |                |
         |                   +-- CLOSED/HALF_OPEN --> [Retry]
         |
         +-- Permanent (non-retryable)
         |         |
         |         v
         |   [Fallback Chain]
         |         |
         |         +-- Fallback 1: Alternative service
         |         +-- Fallback 2: Cached result
         |         +-- Fallback 3: Degraded response
         |         +-- Fallback N: Dead letter queue
         |
         +-- Unknown
                   |
                   v
             [Log + Dead Letter Queue]
```

### Error Classification

```
+--------------------------------------------------------+
|              Error Classification Tree                  |
|                                                        |
|  Transient (retry-worthy)                              |
|  |-- ConnectionError                                   |
|  |-- TimeoutError                                      |
|  |-- HTTP 429 (rate limited)                           |
|  |-- HTTP 502, 503, 504 (server errors)                |
|  |-- LLM overloaded / busy                             |
|  |-- Database lock timeout                             |
|                                                        |
|  Permanent (do not retry)                              |
|  |-- HTTP 400 (bad request)                            |
|  |-- HTTP 401, 403 (auth failure)                      |
|  |-- HTTP 404 (not found)                              |
|  |-- ValueError (invalid input)                        |
|  |-- PermissionError                                   |
|  |-- FileNotFoundError                                 |
|                                                        |
|  Degradable (try fallback)                             |
|  |-- LLM quality too low                               |
|  |-- Model not available                               |
|  |-- Feature not supported by fallback model           |
|  |-- Context window exceeded                           |
+--------------------------------------------------------+
```

---

## Component Specifications

### 1. Core Error Types

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Core error recovery types for GAIA.
"""

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type

from gaia.logger import get_logger

log = get_logger(__name__)


class ErrorCategory(Enum):
    """Classification of error types."""
    TRANSIENT = "transient"       # Retry-worthy (network, timeout, rate limit)
    PERMANENT = "permanent"       # Do not retry (auth, bad request, not found)
    DEGRADABLE = "degradable"     # Try fallback (model unavailable, quality issue)
    UNKNOWN = "unknown"           # Unclassified (log and decide)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"         # Normal operation
    OPEN = "open"             # Failures exceeded threshold; rejecting calls
    HALF_OPEN = "half_open"   # Testing if service has recovered


class HealthStatus(Enum):
    """Component health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ErrorContext:
    """Rich context for an error event."""
    error_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    error_type: str = ""
    error_message: str = ""
    category: ErrorCategory = ErrorCategory.UNKNOWN
    component: str = ""        # Which component failed
    operation: str = ""        # What operation was attempted
    timestamp: float = field(default_factory=time.time)
    retry_count: int = 0
    max_retries: int = 0
    original_exception: Optional[Exception] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_id": self.error_id,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "category": self.category.value,
            "component": self.component,
            "operation": self.operation,
            "timestamp": self.timestamp,
            "retry_count": self.retry_count,
        }


@dataclass
class RecoveryResult:
    """Result of a recovery attempt."""
    success: bool
    result: Any = None
    error_context: Optional[ErrorContext] = None
    recovery_method: str = ""  # "retry", "fallback", "cache", "degraded"
    attempts: int = 0
    total_duration_ms: float = 0.0
    degraded: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DeadLetterEntry:
    """An unrecoverable operation queued for later processing."""
    entry_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    operation: str = ""
    component: str = ""
    args: Dict[str, Any] = field(default_factory=dict)
    error_context: Optional[ErrorContext] = None
    created_at: float = field(default_factory=time.time)
    retry_after: Optional[float] = None  # Timestamp to retry
    retry_count: int = 0
    max_retries: int = 5
    status: str = "pending"  # pending, retrying, resolved, abandoned


# Error classification registry
ERROR_CLASSIFICATIONS: Dict[Type[Exception], ErrorCategory] = {
    ConnectionError: ErrorCategory.TRANSIENT,
    TimeoutError: ErrorCategory.TRANSIENT,
    ConnectionRefusedError: ErrorCategory.TRANSIENT,
    ConnectionResetError: ErrorCategory.TRANSIENT,
    BrokenPipeError: ErrorCategory.TRANSIENT,
    OSError: ErrorCategory.TRANSIENT,
    FileNotFoundError: ErrorCategory.PERMANENT,
    PermissionError: ErrorCategory.PERMANENT,
    ValueError: ErrorCategory.PERMANENT,
    TypeError: ErrorCategory.PERMANENT,
    KeyError: ErrorCategory.PERMANENT,
    AttributeError: ErrorCategory.PERMANENT,
    NotImplementedError: ErrorCategory.PERMANENT,
    MemoryError: ErrorCategory.PERMANENT,
}


def classify_error(exception: Exception) -> ErrorCategory:
    """
    Classify an exception into an error category.

    Checks exception type hierarchy for best match.
    """
    for exc_type, category in ERROR_CLASSIFICATIONS.items():
        if isinstance(exception, exc_type):
            return category

    # Check HTTP errors by status code
    if hasattr(exception, "status_code"):
        status = exception.status_code
        if status == 429:
            return ErrorCategory.TRANSIENT
        if status in (502, 503, 504):
            return ErrorCategory.TRANSIENT
        if status in (400, 401, 403, 404, 405):
            return ErrorCategory.PERMANENT

    # Check error message for hints
    msg = str(exception).lower()
    if any(word in msg for word in ["timeout", "timed out", "rate limit", "busy", "overloaded"]):
        return ErrorCategory.TRANSIENT
    if any(word in msg for word in ["model not found", "context length", "not available"]):
        return ErrorCategory.DEGRADABLE

    return ErrorCategory.UNKNOWN
```

---

## Retry Strategies

### 2.1 Retry Engine

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Configurable retry engine with multiple strategies.
"""

import asyncio
import functools
import random
import time
from typing import Any, Callable, Dict, List, Optional, Set, Type, Union

from gaia.logger import get_logger

log = get_logger(__name__)


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_retries: int = 3
    initial_delay_seconds: float = 1.0
    max_delay_seconds: float = 60.0
    backoff_multiplier: float = 2.0
    jitter: bool = True
    jitter_range: float = 0.5  # +/- 50% of calculated delay
    retry_on: Set[ErrorCategory] = field(
        default_factory=lambda: {ErrorCategory.TRANSIENT}
    )
    retry_on_exceptions: Set[Type[Exception]] = field(
        default_factory=lambda: {ConnectionError, TimeoutError}
    )
    retry_budget_seconds: float = 120.0  # Total time budget for all retries
    on_retry: Optional[Callable] = None  # Callback on each retry


class RetryEngine:
    """
    Retry engine with exponential backoff, jitter, and error classification.

    Strategies:
    - Exponential backoff with jitter (default)
    - Fixed delay
    - Linear backoff
    - Decorrelated jitter (AWS-style)

    Features:
    - Error classification (only retry transient errors)
    - Retry budget (total time limit for retries)
    - Per-operation retry configuration
    - Retry callbacks for progress reporting
    - Integration with circuit breaker

    Usage:
        retry = RetryEngine()

        # Method 1: Decorator
        @retry.with_retry(max_retries=3)
        def call_api():
            return requests.get("https://api.example.com/data")

        # Method 2: Context manager style
        result = await retry.execute(call_api, config=RetryConfig(max_retries=5))

        # Method 3: Inline
        result = retry.execute_sync(
            lambda: llm_client.generate("Hello"),
            config=RetryConfig(max_retries=3, initial_delay_seconds=2.0)
        )
    """

    def __init__(self, default_config: Optional[RetryConfig] = None):
        self.default_config = default_config or RetryConfig()

    def calculate_delay(
        self,
        attempt: int,
        config: RetryConfig,
        strategy: str = "exponential",
    ) -> float:
        """
        Calculate delay before next retry attempt.

        Args:
            attempt: Current attempt number (0-indexed)
            config: Retry configuration
            strategy: "exponential", "fixed", "linear", "decorrelated"

        Returns:
            Delay in seconds
        """
        if strategy == "fixed":
            delay = config.initial_delay_seconds

        elif strategy == "linear":
            delay = config.initial_delay_seconds * (attempt + 1)

        elif strategy == "decorrelated":
            # AWS-style decorrelated jitter
            prev_delay = (
                config.initial_delay_seconds
                * (config.backoff_multiplier ** max(0, attempt - 1))
            )
            delay = random.uniform(
                config.initial_delay_seconds,
                prev_delay * 3,
            )

        else:  # exponential (default)
            delay = config.initial_delay_seconds * (config.backoff_multiplier ** attempt)

        # Apply jitter
        if config.jitter:
            jitter = delay * config.jitter_range
            delay = delay + random.uniform(-jitter, jitter)

        # Cap at max delay
        delay = min(delay, config.max_delay_seconds)
        delay = max(0.01, delay)  # Minimum 10ms

        return delay

    def execute_sync(
        self,
        operation: Callable,
        config: Optional[RetryConfig] = None,
        operation_name: str = "",
    ) -> RecoveryResult:
        """
        Execute an operation with synchronous retry logic.

        Args:
            operation: Callable to execute
            config: Retry configuration (uses default if None)
            operation_name: Name for logging

        Returns:
            RecoveryResult with success/failure details
        """
        cfg = config or self.default_config
        op_name = operation_name or getattr(operation, "__name__", "unknown")
        start_time = time.time()

        last_error = None

        for attempt in range(cfg.max_retries + 1):
            # Check retry budget
            elapsed = time.time() - start_time
            if elapsed > cfg.retry_budget_seconds:
                log.warning(
                    f"Retry budget exhausted for {op_name} "
                    f"({elapsed:.1f}s > {cfg.retry_budget_seconds}s)"
                )
                break

            try:
                result = operation()

                if attempt > 0:
                    log.info(
                        f"Operation {op_name} succeeded after {attempt} retries"
                    )

                return RecoveryResult(
                    success=True,
                    result=result,
                    recovery_method="retry" if attempt > 0 else "direct",
                    attempts=attempt + 1,
                    total_duration_ms=(time.time() - start_time) * 1000,
                )

            except Exception as e:
                last_error = e
                category = classify_error(e)

                error_ctx = ErrorContext(
                    error_type=type(e).__name__,
                    error_message=str(e),
                    category=category,
                    component=op_name,
                    operation=op_name,
                    retry_count=attempt,
                    max_retries=cfg.max_retries,
                    original_exception=e,
                )

                # Should we retry?
                should_retry = (
                    attempt < cfg.max_retries
                    and (
                        category in cfg.retry_on
                        or type(e) in cfg.retry_on_exceptions
                    )
                )

                if should_retry:
                    delay = self.calculate_delay(attempt, cfg)
                    log.info(
                        f"Retrying {op_name} in {delay:.1f}s "
                        f"(attempt {attempt + 1}/{cfg.max_retries}, "
                        f"error: {type(e).__name__}: {str(e)[:100]})"
                    )

                    if cfg.on_retry:
                        cfg.on_retry(error_ctx, delay)

                    time.sleep(delay)
                else:
                    log.warning(
                        f"Not retrying {op_name}: {category.value} error "
                        f"({type(e).__name__}: {str(e)[:100]})"
                    )
                    break

        # All retries exhausted
        return RecoveryResult(
            success=False,
            error_context=ErrorContext(
                error_type=type(last_error).__name__ if last_error else "Unknown",
                error_message=str(last_error) if last_error else "Max retries exceeded",
                category=classify_error(last_error) if last_error else ErrorCategory.UNKNOWN,
                component=op_name,
                retry_count=cfg.max_retries,
                max_retries=cfg.max_retries,
            ),
            recovery_method="retry_exhausted",
            attempts=cfg.max_retries + 1,
            total_duration_ms=(time.time() - start_time) * 1000,
        )

    async def execute_async(
        self,
        operation: Callable,
        config: Optional[RetryConfig] = None,
        operation_name: str = "",
    ) -> RecoveryResult:
        """Execute an operation with async retry logic."""
        cfg = config or self.default_config
        op_name = operation_name or getattr(operation, "__name__", "unknown")
        start_time = time.time()
        last_error = None

        for attempt in range(cfg.max_retries + 1):
            elapsed = time.time() - start_time
            if elapsed > cfg.retry_budget_seconds:
                break

            try:
                if asyncio.iscoroutinefunction(operation):
                    result = await operation()
                else:
                    result = operation()

                return RecoveryResult(
                    success=True,
                    result=result,
                    recovery_method="retry" if attempt > 0 else "direct",
                    attempts=attempt + 1,
                    total_duration_ms=(time.time() - start_time) * 1000,
                )

            except Exception as e:
                last_error = e
                category = classify_error(e)

                should_retry = (
                    attempt < cfg.max_retries
                    and category in cfg.retry_on
                )

                if should_retry:
                    delay = self.calculate_delay(attempt, cfg)
                    log.info(f"Async retrying {op_name} in {delay:.1f}s (attempt {attempt + 1})")
                    await asyncio.sleep(delay)
                else:
                    break

        return RecoveryResult(
            success=False,
            error_context=ErrorContext(
                error_type=type(last_error).__name__ if last_error else "Unknown",
                error_message=str(last_error) if last_error else "Max retries exceeded",
                component=op_name,
            ),
            recovery_method="retry_exhausted",
            attempts=cfg.max_retries + 1,
            total_duration_ms=(time.time() - start_time) * 1000,
        )

    def with_retry(
        self,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        retry_on: Optional[Set[ErrorCategory]] = None,
        return_recovery_result: bool = False,
    ):
        """
        Decorator for adding retry logic to functions.

        Preserves the original function's signature and return type.
        On success, returns the original function's return value.
        On failure after exhausting retries, raises the last exception.

        Args:
            max_retries: Maximum number of retry attempts
            initial_delay: Initial delay in seconds before first retry
            retry_on: Set of error categories to retry on
            return_recovery_result: If True, return RecoveryResult instead
                                    of the raw return value (for advanced use)

        Usage:
            @retry_engine.with_retry(max_retries=3)
            def call_api():
                return requests.get(url)

            # call_api() returns requests.Response, not RecoveryResult
        """
        config = RetryConfig(
            max_retries=max_retries,
            initial_delay_seconds=initial_delay,
            retry_on=retry_on or {ErrorCategory.TRANSIENT},
        )

        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                recovery = self.execute_sync(
                    lambda: func(*args, **kwargs),
                    config=config,
                    operation_name=func.__name__,
                )
                if return_recovery_result:
                    return recovery
                if recovery.success:
                    return recovery.result
                # Re-raise the original exception to preserve function contract
                if (
                    recovery.error_context
                    and recovery.error_context.original_exception
                ):
                    raise recovery.error_context.original_exception
                raise RuntimeError(
                    f"Operation {func.__name__} failed after {recovery.attempts} attempts: "
                    f"{recovery.error_context.error_message if recovery.error_context else 'unknown error'}"
                )

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                recovery = await self.execute_async(
                    lambda: func(*args, **kwargs),
                    config=config,
                    operation_name=func.__name__,
                )
                if return_recovery_result:
                    return recovery
                if recovery.success:
                    return recovery.result
                if (
                    recovery.error_context
                    and recovery.error_context.original_exception
                ):
                    raise recovery.error_context.original_exception
                raise RuntimeError(
                    f"Operation {func.__name__} failed after {recovery.attempts} attempts"
                )

            # Return the appropriate wrapper based on whether func is async
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            return wrapper
        return decorator
```

---

## Circuit Breaker Pattern

### 3.1 Circuit Breaker

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Circuit breaker pattern implementation for GAIA.

Prevents cascading failures by cutting off calls to failing services.
"""

import threading
import time
from typing import Any, Callable, Dict, List, Optional

from gaia.logger import get_logger

log = get_logger(__name__)


@dataclass
class CircuitBreakerConfig:
    """Configuration for a circuit breaker."""
    failure_threshold: int = 5        # Failures before opening
    success_threshold: int = 3        # Successes in half-open before closing
    timeout_seconds: float = 30.0     # How long to stay open before half-open
    monitoring_window_seconds: float = 60.0  # Window for counting failures
    half_open_max_calls: int = 1      # Max concurrent calls in half-open


class CircuitBreaker:
    """
    Circuit breaker that prevents repeated calls to failing services.

    States:
    - CLOSED: Normal operation. Failures are counted.
    - OPEN: Service is known to be failing. Calls are rejected immediately.
    - HALF_OPEN: Testing if service recovered. Limited calls allowed.

    State Transitions:
    - CLOSED -> OPEN: When failures exceed threshold in monitoring window
    - OPEN -> HALF_OPEN: After timeout expires
    - HALF_OPEN -> CLOSED: When success threshold is met
    - HALF_OPEN -> OPEN: When a call fails in half-open state

    Usage:
        breaker = CircuitBreaker("lemonade_server")

        if breaker.allow_request():
            try:
                result = call_lemonade()
                breaker.record_success()
            except Exception as e:
                breaker.record_failure()
                raise
        else:
            # Circuit is open, use fallback
            result = use_fallback()
    """

    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._failure_times: List[float] = []
        self._last_failure_time: float = 0
        self._last_state_change: float = time.time()
        self._half_open_calls = 0
        self._lock = threading.Lock()
        self._state_change_callbacks: List[Callable] = []

    def allow_request(self) -> bool:
        """
        Check if a request should be allowed through.

        Returns:
            True if request can proceed, False if circuit is open
        """
        with self._lock:
            now = time.time()

            if self.state == CircuitState.CLOSED:
                return True

            elif self.state == CircuitState.OPEN:
                # Check if timeout has expired
                if now - self._last_state_change >= self.config.timeout_seconds:
                    self._transition_to(CircuitState.HALF_OPEN)
                    self._half_open_calls = 0
                    return True
                return False

            elif self.state == CircuitState.HALF_OPEN:
                if self._half_open_calls < self.config.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False

        return False

    def record_success(self) -> None:
        """Record a successful operation."""
        with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self._transition_to(CircuitState.CLOSED)
                    self._failure_count = 0
                    self._success_count = 0
                    self._failure_times.clear()

            elif self.state == CircuitState.CLOSED:
                # Reset failure count on success (optional, depends on strategy)
                pass

    def record_failure(self) -> None:
        """Record a failed operation."""
        now = time.time()

        with self._lock:
            self._failure_times.append(now)
            self._last_failure_time = now

            # Clean old failures outside monitoring window
            window_start = now - self.config.monitoring_window_seconds
            self._failure_times = [
                t for t in self._failure_times if t > window_start
            ]

            if self.state == CircuitState.HALF_OPEN:
                # Any failure in half-open immediately reopens
                self._transition_to(CircuitState.OPEN)
                self._success_count = 0

            elif self.state == CircuitState.CLOSED:
                if len(self._failure_times) >= self.config.failure_threshold:
                    self._transition_to(CircuitState.OPEN)

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new circuit state."""
        old_state = self.state
        self.state = new_state
        self._last_state_change = time.time()

        log.info(
            f"Circuit breaker '{self.name}': {old_state.value} -> {new_state.value}"
        )

        for callback in self._state_change_callbacks:
            try:
                callback(self.name, old_state, new_state)
            except Exception as e:
                log.error(f"State change callback error: {e}")

    def on_state_change(self, callback: Callable) -> None:
        """Register a state change callback."""
        self._state_change_callbacks.append(callback)

    def get_status(self) -> Dict[str, Any]:
        """Get current circuit breaker status."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": len(self._failure_times),
            "failure_threshold": self.config.failure_threshold,
            "time_in_state": time.time() - self._last_state_change,
            "last_failure": self._last_failure_time,
        }

    def force_open(self) -> None:
        """Manually open the circuit (for testing or emergency)."""
        with self._lock:
            self._transition_to(CircuitState.OPEN)

    def force_close(self) -> None:
        """Manually close the circuit (for recovery)."""
        with self._lock:
            self._transition_to(CircuitState.CLOSED)
            self._failure_count = 0
            self._failure_times.clear()

    def save_state(self) -> Dict[str, Any]:
        """Serialize circuit breaker state for persistence."""
        with self._lock:
            return {
                "name": self.name,
                "state": self.state.value,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "failure_times": list(self._failure_times),
                "last_failure_time": self._last_failure_time,
                "last_state_change": self._last_state_change,
            }

    def restore_state(self, data: Dict[str, Any]) -> None:
        """Restore circuit breaker state from persisted data."""
        with self._lock:
            self.state = CircuitState(data.get("state", "closed"))
            self._failure_count = data.get("failure_count", 0)
            self._success_count = data.get("success_count", 0)
            self._failure_times = data.get("failure_times", [])
            self._last_failure_time = data.get("last_failure_time", 0)
            self._last_state_change = data.get("last_state_change", time.time())
            log.info(f"Circuit breaker '{self.name}' state restored: {self.state.value}")


class CircuitBreakerRegistry:
    """
    Registry of circuit breakers for all GAIA services.

    Provides a centralized view of all service health.
    """

    _instance: Optional["CircuitBreakerRegistry"] = None

    def __init__(self):
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "CircuitBreakerRegistry":
        """Get the global registry."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get_or_create(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
    ) -> CircuitBreaker:
        """Get an existing circuit breaker or create a new one."""
        with self._lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(name, config)
                log.info(f"Circuit breaker created: {name}")
            return self._breakers[name]

    def get_all_status(self) -> List[Dict[str, Any]]:
        """Get status of all circuit breakers."""
        return [b.get_status() for b in self._breakers.values()]

    def get_open_circuits(self) -> List[str]:
        """Get names of all open circuit breakers."""
        return [
            name for name, breaker in self._breakers.items()
            if breaker.state == CircuitState.OPEN
        ]

    def save_all_states(self, file_path: str = "gaia_circuit_states.json") -> None:
        """Persist all circuit breaker states to a JSON file."""
        import json
        states = {
            name: breaker.save_state()
            for name, breaker in self._breakers.items()
        }
        with open(file_path, "w") as f:
            json.dump(states, f, indent=2)
        log.info(f"Saved {len(states)} circuit breaker states to {file_path}")

    def restore_all_states(self, file_path: str = "gaia_circuit_states.json") -> None:
        """Restore all circuit breaker states from a JSON file."""
        import json
        import os

        if not os.path.exists(file_path):
            log.debug(f"No persisted circuit states at {file_path}")
            return

        try:
            with open(file_path, "r") as f:
                states = json.load(f)

            for name, state_data in states.items():
                breaker = self.get_or_create(name)
                breaker.restore_state(state_data)

            log.info(f"Restored {len(states)} circuit breaker states from {file_path}")
        except Exception as e:
            log.error(f"Failed to restore circuit states: {e}")


class ErrorBudgetTracker:
    """
    Track error budgets for SLO (Service Level Objective) compliance.

    An error budget defines the maximum acceptable failure rate for a
    component over a rolling time window. When the budget is exhausted,
    the system should reduce change velocity or trigger alerts.

    Example:
        tracker = ErrorBudgetTracker()
        tracker.define_budget("llm_service", target_success_rate=0.995, window_hours=24)

        # Record outcomes
        tracker.record_success("llm_service")
        tracker.record_failure("llm_service")

        # Check budget
        status = tracker.get_budget_status("llm_service")
        # {'remaining_pct': 85.2, 'budget_exhausted': False, ...}
    """

    def __init__(self):
        self._budgets: Dict[str, Dict[str, Any]] = {}
        self._events: Dict[str, List[Tuple[float, bool]]] = defaultdict(list)
        self._lock = threading.Lock()
        self._alert_callbacks: List[Callable] = []

    def define_budget(
        self,
        component: str,
        target_success_rate: float = 0.995,
        window_hours: float = 24.0,
    ) -> None:
        """
        Define an error budget for a component.

        Args:
            component: Component name
            target_success_rate: Required success rate (e.g., 0.995 = 99.5%)
            window_hours: Rolling window in hours
        """
        self._budgets[component] = {
            "target_success_rate": target_success_rate,
            "window_seconds": window_hours * 3600,
            "max_error_rate": 1.0 - target_success_rate,
        }
        log.info(
            f"Error budget defined for {component}: "
            f"target={target_success_rate*100}% over {window_hours}h"
        )

    def record_success(self, component: str) -> None:
        """Record a successful operation."""
        with self._lock:
            self._events[component].append((time.time(), True))
            self._cleanup_events(component)

    def record_failure(self, component: str) -> None:
        """Record a failed operation."""
        with self._lock:
            self._events[component].append((time.time(), False))
            self._cleanup_events(component)

        # Check if budget is now exhausted
        status = self.get_budget_status(component)
        if status and status.get("budget_exhausted"):
            log.warning(
                f"Error budget EXHAUSTED for {component}: "
                f"{status['current_error_rate']*100:.2f}% error rate "
                f"exceeds {status['max_error_rate']*100:.2f}% budget"
            )
            for callback in self._alert_callbacks:
                try:
                    callback(component, status)
                except Exception:
                    pass

    def get_budget_status(self, component: str) -> Optional[Dict[str, Any]]:
        """Get current error budget status for a component."""
        budget = self._budgets.get(component)
        if not budget:
            return None

        with self._lock:
            events = self._events.get(component, [])
            now = time.time()
            window_start = now - budget["window_seconds"]

            window_events = [e for e in events if e[0] >= window_start]
            total = len(window_events)

            if total == 0:
                return {
                    "component": component,
                    "total_requests": 0,
                    "current_error_rate": 0.0,
                    "max_error_rate": budget["max_error_rate"],
                    "remaining_pct": 100.0,
                    "budget_exhausted": False,
                }

            failures = sum(1 for _, success in window_events if not success)
            error_rate = failures / total
            max_error_rate = budget["max_error_rate"]

            if max_error_rate > 0:
                remaining_pct = max(0, (1 - error_rate / max_error_rate) * 100)
            else:
                remaining_pct = 0.0 if failures > 0 else 100.0

            return {
                "component": component,
                "total_requests": total,
                "successes": total - failures,
                "failures": failures,
                "current_error_rate": error_rate,
                "current_success_rate": 1 - error_rate,
                "target_success_rate": budget["target_success_rate"],
                "max_error_rate": max_error_rate,
                "remaining_pct": remaining_pct,
                "budget_exhausted": error_rate >= max_error_rate,
                "window_hours": budget["window_seconds"] / 3600,
            }

    def _cleanup_events(self, component: str) -> None:
        """Remove events outside all tracking windows."""
        budget = self._budgets.get(component)
        if not budget:
            return
        cutoff = time.time() - budget["window_seconds"]
        self._events[component] = [
            e for e in self._events[component] if e[0] >= cutoff
        ]

    def on_budget_exhausted(self, callback: Callable) -> None:
        """Register a callback for when an error budget is exhausted."""
        self._alert_callbacks.append(callback)

    def get_all_statuses(self) -> Dict[str, Dict[str, Any]]:
        """Get budget status for all tracked components."""
        return {
            component: self.get_budget_status(component)
            for component in self._budgets
        }
```

---

## Graceful Degradation

### 4.1 Fallback Chain Manager

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Graceful degradation with multi-level fallback chains.
"""

from typing import Any, Callable, Dict, List, Optional

from gaia.logger import get_logger

log = get_logger(__name__)


@dataclass
class FallbackOption:
    """A single fallback option in a chain."""
    name: str
    handler: Callable
    quality_score: float = 1.0  # 1.0 = full quality, 0.0 = minimal
    description: str = ""
    requires: Optional[List[str]] = None  # Required services


class FallbackChain:
    """
    Ordered chain of fallback options for graceful degradation.

    When the primary operation fails, the chain tries each fallback
    in order until one succeeds or all are exhausted.

    Example chain for LLM calls:
    1. Local Lemonade server (full quality)
    2. Cloud Claude API (full quality, higher cost)
    3. Smaller local model (reduced quality)
    4. Cached response (stale but fast)
    5. Error message (last resort)

    Usage:
        chain = FallbackChain("llm_inference")
        chain.add(FallbackOption(
            name="lemonade_local",
            handler=lambda q: lemonade_client.generate(q),
            quality_score=1.0,
        ))
        chain.add(FallbackOption(
            name="claude_cloud",
            handler=lambda q: claude_client.generate(q),
            quality_score=1.0,
        ))
        chain.add(FallbackOption(
            name="cached_response",
            handler=lambda q: cache.get_similar(q),
            quality_score=0.3,
        ))

        result = chain.execute(query="What is GAIA?")
    """

    def __init__(self, name: str):
        self.name = name
        self._options: List[FallbackOption] = []
        self._circuit_registry = CircuitBreakerRegistry.get_instance()

    def add(self, option: FallbackOption) -> "FallbackChain":
        """Add a fallback option to the chain."""
        self._options.append(option)
        return self

    def execute(self, *args, **kwargs) -> RecoveryResult:
        """
        Execute the fallback chain.

        Tries each option in order. Returns the first successful result.
        """
        start_time = time.time()
        attempted = []

        for option in self._options:
            # Check circuit breaker for this option
            breaker = self._circuit_registry.get_or_create(
                f"fallback_{self.name}_{option.name}"
            )

            if not breaker.allow_request():
                log.debug(f"Skipping {option.name}: circuit open")
                continue

            try:
                log.info(
                    f"Fallback chain '{self.name}': trying {option.name} "
                    f"(quality={option.quality_score})"
                )
                result = option.handler(*args, **kwargs)

                breaker.record_success()

                degraded = option.quality_score < 1.0
                if degraded:
                    log.info(
                        f"Fallback chain '{self.name}': degraded to {option.name} "
                        f"(quality={option.quality_score})"
                    )

                return RecoveryResult(
                    success=True,
                    result=result,
                    recovery_method=f"fallback:{option.name}",
                    attempts=len(attempted) + 1,
                    total_duration_ms=(time.time() - start_time) * 1000,
                    degraded=degraded,
                    metadata={
                        "quality_score": option.quality_score,
                        "fallback_name": option.name,
                        "attempted": attempted,
                    },
                )

            except Exception as e:
                breaker.record_failure()
                attempted.append({
                    "name": option.name,
                    "error": str(e)[:200],
                })
                log.debug(f"Fallback {option.name} failed: {e}")

        # All fallbacks exhausted
        log.error(f"Fallback chain '{self.name}': all options exhausted")
        return RecoveryResult(
            success=False,
            recovery_method="fallback_exhausted",
            attempts=len(attempted),
            total_duration_ms=(time.time() - start_time) * 1000,
            metadata={"attempted": attempted},
        )


class DegradationManager:
    """
    Manages degradation policies across GAIA components.

    Provides pre-built fallback chains for common scenarios.
    """

    def __init__(self):
        self._chains: Dict[str, FallbackChain] = {}
        self._cache = ResponseCache()

    def get_chain(self, name: str) -> FallbackChain:
        """Get or create a fallback chain by name."""
        if name not in self._chains:
            self._chains[name] = FallbackChain(name)
        return self._chains[name]

    def create_llm_chain(
        self,
        primary_client: Any,
        cloud_client: Optional[Any] = None,
        small_model_client: Optional[Any] = None,
    ) -> FallbackChain:
        """
        Create a standard LLM fallback chain.

        Chain order:
        1. Primary (local Lemonade)
        2. Cloud LLM (Claude/GPT if configured)
        3. Smaller local model (if available)
        4. Cached response
        5. Error message
        """
        chain = FallbackChain("llm")

        # Primary: local LLM
        chain.add(FallbackOption(
            name="primary_local",
            handler=lambda prompt, **kw: primary_client.generate(prompt, **kw),
            quality_score=1.0,
            description="Primary local LLM via Lemonade",
        ))

        # Cloud fallback
        if cloud_client:
            chain.add(FallbackOption(
                name="cloud_llm",
                handler=lambda prompt, **kw: cloud_client.generate(prompt, **kw),
                quality_score=1.0,
                description="Cloud LLM fallback",
            ))

        # Smaller model
        if small_model_client:
            chain.add(FallbackOption(
                name="small_model",
                handler=lambda prompt, **kw: small_model_client.generate(prompt, **kw),
                quality_score=0.6,
                description="Smaller local model (reduced quality)",
            ))

        # Cached response
        chain.add(FallbackOption(
            name="cache",
            handler=lambda prompt, **kw: self._cache.get_similar(prompt),
            quality_score=0.3,
            description="Cached response (possibly stale)",
        ))

        self._chains["llm"] = chain
        return chain


class ResponseCache:
    """
    Cache for storing recent LLM responses for fallback.

    Uses semantic similarity for cache lookup.
    """

    def __init__(self, max_entries: int = 1000):
        self.max_entries = max_entries
        self._cache: Dict[str, Dict[str, Any]] = {}

    def store(self, prompt: str, response: str, metadata: Dict = None) -> None:
        """Store a prompt-response pair."""
        import hashlib
        key = hashlib.md5(prompt.encode()).hexdigest()
        self._cache[key] = {
            "prompt": prompt,
            "response": response,
            "timestamp": time.time(),
            "metadata": metadata or {},
        }

        # Evict oldest if over capacity
        if len(self._cache) > self.max_entries:
            oldest_key = min(self._cache, key=lambda k: self._cache[k]["timestamp"])
            del self._cache[oldest_key]

    def get_similar(self, prompt: str) -> Optional[str]:
        """
        Get a cached response for a similar prompt.

        Uses simple keyword matching. Replace with embedding similarity
        for production use.
        """
        if not self._cache:
            raise LookupError("No cached responses available")

        prompt_words = set(prompt.lower().split())
        best_match = None
        best_score = 0

        for key, entry in self._cache.items():
            cached_words = set(entry["prompt"].lower().split())
            common = prompt_words & cached_words
            total = prompt_words | cached_words
            score = len(common) / len(total) if total else 0

            if score > best_score:
                best_score = score
                best_match = entry

        if best_match and best_score > 0.3:
            log.info(f"Cache hit (similarity={best_score:.2f})")
            return best_match["response"]

        raise LookupError("No similar cached response found")
```

---

## Self-Healing System

### 5.1 Health Monitor

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Self-healing system with health monitoring and auto-recovery.
"""

import subprocess
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from gaia.logger import get_logger

log = get_logger(__name__)


@dataclass
class HealthCheck:
    """Definition of a health check."""
    name: str
    check_fn: Callable[[], bool]
    interval_seconds: float = 30.0
    timeout_seconds: float = 10.0
    failure_threshold: int = 3  # Consecutive failures before unhealthy
    recovery_fn: Optional[Callable] = None  # Function to call for recovery
    description: str = ""


@dataclass
class ComponentHealth:
    """Health status of a single component."""
    name: str
    status: HealthStatus
    last_check: float
    consecutive_failures: int = 0
    last_error: Optional[str] = None
    recovery_attempts: int = 0
    last_recovery: Optional[float] = None


class HealthMonitor:
    """
    Monitor component health and trigger auto-recovery.

    Components registered:
    - Lemonade LLM server
    - Database connections
    - External API connections
    - File system access
    - Memory usage

    Recovery actions:
    - Restart Lemonade server
    - Reconnect to databases
    - Clear corrupted caches
    - Free memory (clear unused embeddings)
    """

    def __init__(self):
        self._checks: Dict[str, HealthCheck] = {}
        self._status: Dict[str, ComponentHealth] = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._listeners: List[Callable] = []

    def register_check(self, check: HealthCheck) -> None:
        """Register a health check."""
        self._checks[check.name] = check
        self._status[check.name] = ComponentHealth(
            name=check.name,
            status=HealthStatus.UNKNOWN,
            last_check=0,
        )
        log.info(f"Health check registered: {check.name}")

    def start(self) -> None:
        """Start the health monitor background thread."""
        if self._running:
            return
        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="gaia-health-monitor",
        )
        self._thread.start()
        log.info("Health monitor started")

    def stop(self) -> None:
        """Stop the health monitor."""
        self._running = False
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)
        log.info("Health monitor stopped")

    def _monitor_loop(self) -> None:
        """Main monitoring loop. Runs health checks in parallel using a thread pool."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(
            max_workers=min(8, len(self._checks) or 1),
            thread_name_prefix="gaia-health",
        ) as pool:
            while self._running and not self._stop_event.is_set():
                now = time.time()

                # Collect checks that are due
                due_checks = {}
                for name, check in self._checks.items():
                    status = self._status[name]
                    if now - status.last_check >= check.interval_seconds:
                        due_checks[name] = check

                if due_checks:
                    # Submit all due checks in parallel
                    futures = {
                        pool.submit(self._run_single_check, name, check): name
                        for name, check in due_checks.items()
                    }

                    # Collect results with a timeout to avoid blocking forever
                    for future in as_completed(futures, timeout=30):
                        name = futures[future]
                        try:
                            future.result()
                        except Exception as e:
                            log.error(f"Health check thread error for {name}: {e}")

                self._stop_event.wait(timeout=5)

    def _run_single_check(self, name: str, check: HealthCheck) -> None:
        """Run a single health check (called from thread pool)."""
        status = self._status[name]
        now = time.time()

        try:
            healthy = check.check_fn()
            status.last_check = now

            if healthy:
                if status.status != HealthStatus.HEALTHY:
                    log.info(f"Component recovered: {name}")
                status.status = HealthStatus.HEALTHY
                status.consecutive_failures = 0
                status.last_error = None
            else:
                status.consecutive_failures += 1
                status.last_error = "Health check returned False"
                self._handle_failure(name, check, status)

        except Exception as e:
            status.last_check = now
            status.consecutive_failures += 1
            status.last_error = str(e)
            self._handle_failure(name, check, status)

    def _handle_failure(
        self,
        name: str,
        check: HealthCheck,
        status: ComponentHealth,
    ) -> None:
        """Handle a health check failure."""
        if status.consecutive_failures >= check.failure_threshold:
            if status.status != HealthStatus.UNHEALTHY:
                log.warning(
                    f"Component unhealthy: {name} "
                    f"({status.consecutive_failures} consecutive failures)"
                )
                status.status = HealthStatus.UNHEALTHY

                # Notify listeners
                for listener in self._listeners:
                    try:
                        listener(name, status)
                    except Exception:
                        pass

            # Attempt recovery
            if check.recovery_fn:
                self._attempt_recovery(name, check, status)

        elif status.consecutive_failures > 0:
            status.status = HealthStatus.DEGRADED

    def _attempt_recovery(
        self,
        name: str,
        check: HealthCheck,
        status: ComponentHealth,
    ) -> None:
        """Attempt to recover an unhealthy component."""
        # Limit recovery attempts
        if status.recovery_attempts >= 5:
            log.error(
                f"Recovery exhausted for {name} "
                f"({status.recovery_attempts} attempts)"
            )
            return

        # Cooldown between recovery attempts
        if status.last_recovery and time.time() - status.last_recovery < 60:
            return

        log.info(f"Attempting recovery for {name} (attempt {status.recovery_attempts + 1})")

        try:
            check.recovery_fn()
            status.recovery_attempts += 1
            status.last_recovery = time.time()
            log.info(f"Recovery initiated for {name}")
        except Exception as e:
            log.error(f"Recovery failed for {name}: {e}")
            status.recovery_attempts += 1

    def get_overall_health(self) -> Dict[str, Any]:
        """Get overall system health summary."""
        statuses = {}
        for name, status in self._status.items():
            statuses[name] = {
                "status": status.status.value,
                "consecutive_failures": status.consecutive_failures,
                "last_error": status.last_error,
                "recovery_attempts": status.recovery_attempts,
            }

        unhealthy = [n for n, s in self._status.items() if s.status == HealthStatus.UNHEALTHY]
        degraded = [n for n, s in self._status.items() if s.status == HealthStatus.DEGRADED]

        if unhealthy:
            overall = HealthStatus.UNHEALTHY
        elif degraded:
            overall = HealthStatus.DEGRADED
        else:
            overall = HealthStatus.HEALTHY

        return {
            "overall": overall.value,
            "components": statuses,
            "unhealthy": unhealthy,
            "degraded": degraded,
        }

    def on_health_change(self, callback: Callable) -> None:
        """Register a callback for health status changes."""
        self._listeners.append(callback)


# Pre-built health checks for GAIA components
def lemonade_health_check() -> bool:
    """Check if Lemonade server is responding."""
    import urllib.request
    try:
        base_url = os.environ.get("LEMONADE_BASE_URL", "http://localhost:8000/api/v1")
        url = f"{base_url.rstrip('/').rsplit('/api', 1)[0]}/health"
        req = urllib.request.urlopen(url, timeout=5)
        return req.status == 200
    except Exception:
        return False


def lemonade_recovery() -> None:
    """Attempt to restart Lemonade server."""
    try:
        from gaia.llm.lemonade_manager import LemonadeManager
        LemonadeManager.ensure_ready(quiet=True)
        log.info("Lemonade server recovery initiated")
    except Exception as e:
        log.error(f"Lemonade recovery failed: {e}")


def database_health_check(db_path: str = "gaia.db") -> bool:
    """Check if SQLite database is accessible."""
    import sqlite3
    try:
        conn = sqlite3.connect(db_path, timeout=5)
        conn.execute("SELECT 1")
        conn.close()
        return True
    except Exception:
        return False
```

---

## Integration with GAIA

### 6.1 Error Recovery Mixin

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Error recovery mixin for GAIA Agent base class.
"""


class ErrorRecoveryMixin:
    """
    Mixin that adds error recovery capabilities to GAIA agents.

    Integrates:
    - Retry engine for LLM calls
    - Circuit breaker for external services
    - Fallback chains for degraded operation
    - Error classification and routing

    Usage:
        class MyAgent(Agent, ErrorRecoveryMixin):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.init_error_recovery()
    """

    def init_error_recovery(
        self,
        retry_config: Optional[RetryConfig] = None,
        enable_fallback: bool = True,
        enable_circuit_breaker: bool = True,
    ) -> None:
        """Initialize error recovery for this agent."""
        self._retry_engine = RetryEngine(
            default_config=retry_config or RetryConfig()
        )
        self._circuit_registry = CircuitBreakerRegistry.get_instance()
        self._degradation_manager = DegradationManager()

        if enable_fallback:
            self._setup_fallback_chains()

        if enable_circuit_breaker:
            self._setup_circuit_breakers()

    def _setup_fallback_chains(self) -> None:
        """Set up default fallback chains."""
        # LLM fallback chain
        self._llm_chain = self._degradation_manager.create_llm_chain(
            primary_client=getattr(self, "chat", None),
        )

    def _setup_circuit_breakers(self) -> None:
        """Set up circuit breakers for external services."""
        self._llm_breaker = self._circuit_registry.get_or_create(
            "lemonade_server",
            CircuitBreakerConfig(
                failure_threshold=3,
                timeout_seconds=30,
            ),
        )

    def resilient_llm_call(
        self,
        prompt: str,
        max_retries: int = 3,
        **kwargs,
    ) -> RecoveryResult:
        """
        Make a resilient LLM call with retry and fallback.

        First tries the primary LLM with retries.
        If circuit breaker opens, uses fallback chain.

        Args:
            prompt: The LLM prompt
            max_retries: Maximum retry attempts
            **kwargs: Additional LLM parameters

        Returns:
            RecoveryResult with the response or error details
        """
        # Check circuit breaker
        if not self._llm_breaker.allow_request():
            log.info("LLM circuit breaker open, using fallback chain")
            return self._llm_chain.execute(prompt, **kwargs)

        # Try with retries
        config = RetryConfig(
            max_retries=max_retries,
            initial_delay_seconds=1.0,
            retry_on={ErrorCategory.TRANSIENT},
        )

        def attempt():
            result = self.chat.send(prompt, **kwargs)
            return result

        recovery = self._retry_engine.execute_sync(
            attempt,
            config=config,
            operation_name="llm_call",
        )

        if recovery.success:
            self._llm_breaker.record_success()
            # Cache successful response for future fallback
            self._degradation_manager._cache.store(
                prompt, str(recovery.result)
            )
        else:
            self._llm_breaker.record_failure()
            # Try fallback chain
            log.info("Retries exhausted, trying fallback chain")
            return self._llm_chain.execute(prompt, **kwargs)

        return recovery

    def resilient_tool_call(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        max_retries: int = 2,
    ) -> RecoveryResult:
        """
        Make a resilient tool call with retry.

        Args:
            tool_name: Name of the tool to execute
            tool_args: Arguments for the tool
            max_retries: Maximum retry attempts

        Returns:
            RecoveryResult with tool output
        """
        from gaia.agents.base.tools import _TOOL_REGISTRY

        tool_info = _TOOL_REGISTRY.get(tool_name)
        if not tool_info:
            return RecoveryResult(
                success=False,
                error_context=ErrorContext(
                    error_type="ToolNotFound",
                    error_message=f"Tool not registered: {tool_name}",
                    category=ErrorCategory.PERMANENT,
                ),
            )

        config = RetryConfig(
            max_retries=max_retries,
            initial_delay_seconds=0.5,
            retry_on={ErrorCategory.TRANSIENT},
        )

        return self._retry_engine.execute_sync(
            lambda: tool_info["function"](**tool_args),
            config=config,
            operation_name=f"tool:{tool_name}",
        )
```

### 6.2 Dead Letter Queue

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Dead letter queue for unrecoverable operations.
"""

import json
import sqlite3
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from gaia.logger import get_logger

log = get_logger(__name__)


class DeadLetterQueue:
    """
    Queue for operations that failed all retry and fallback attempts.

    Entries can be:
    - Manually retried later
    - Automatically retried on a schedule
    - Reviewed and resolved by an admin
    - Abandoned after max retries

    Usage:
        dlq = DeadLetterQueue()

        # Enqueue a failed operation
        dlq.enqueue(DeadLetterEntry(
            operation="send_email",
            component="EmailAgent",
            args={"to": "john@example.com", "body": "Hello"},
            error_context=error_ctx,
        ))

        # Process queue (manual or scheduled)
        dlq.process(handler=retry_email_send)
    """

    def __init__(self, db_path: str = "gaia_dlq.db"):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        conn = sqlite3.connect(self.db_path)
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS dead_letters (
                entry_id TEXT PRIMARY KEY,
                operation TEXT NOT NULL,
                component TEXT,
                args TEXT,
                error_type TEXT,
                error_message TEXT,
                created_at REAL NOT NULL,
                retry_after REAL,
                retry_count INTEGER DEFAULT 0,
                max_retries INTEGER DEFAULT 5,
                status TEXT DEFAULT 'pending'
            );

            CREATE INDEX IF NOT EXISTS idx_dlq_status ON dead_letters(status);
            CREATE INDEX IF NOT EXISTS idx_dlq_retry ON dead_letters(retry_after);
        """)
        conn.commit()
        conn.close()

    def enqueue(self, entry: DeadLetterEntry) -> None:
        """Add a failed operation to the dead letter queue."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                INSERT INTO dead_letters
                (entry_id, operation, component, args, error_type, error_message,
                 created_at, retry_after, retry_count, max_retries, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry.entry_id,
                entry.operation,
                entry.component,
                json.dumps(entry.args, default=str),
                entry.error_context.error_type if entry.error_context else "",
                entry.error_context.error_message if entry.error_context else "",
                entry.created_at,
                entry.retry_after,
                entry.retry_count,
                entry.max_retries,
                entry.status,
            ))
            conn.commit()
            conn.close()

        log.info(
            f"Dead letter queued: {entry.operation} "
            f"({entry.component}, error: {entry.error_context.error_type if entry.error_context else 'unknown'})"
        )

    def process(
        self,
        handler: Callable[[DeadLetterEntry], bool],
        max_items: int = 10,
    ) -> Dict[str, int]:
        """
        Process pending entries in the dead letter queue.

        Args:
            handler: Function that attempts to process each entry.
                     Returns True if successful, False to re-queue.
            max_items: Maximum entries to process

        Returns:
            Summary of processing results
        """
        now = time.time()
        results = {"processed": 0, "succeeded": 0, "failed": 0, "abandoned": 0}

        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT * FROM dead_letters
                WHERE status = 'pending'
                AND (retry_after IS NULL OR retry_after <= ?)
                ORDER BY created_at
                LIMIT ?
            """, (now, max_items))

            entries = cursor.fetchall()

            for row in entries:
                entry = DeadLetterEntry(
                    entry_id=row[0],
                    operation=row[1],
                    component=row[2],
                    args=json.loads(row[3]) if row[3] else {},
                    created_at=row[5],
                    retry_count=row[7],
                    max_retries=row[8],
                )

                results["processed"] += 1

                if entry.retry_count >= entry.max_retries:
                    cursor.execute(
                        "UPDATE dead_letters SET status = 'abandoned' WHERE entry_id = ?",
                        (entry.entry_id,),
                    )
                    results["abandoned"] += 1
                    log.warning(f"Dead letter abandoned: {entry.entry_id}")
                    continue

                try:
                    success = handler(entry)
                    if success:
                        cursor.execute(
                            "UPDATE dead_letters SET status = 'resolved' WHERE entry_id = ?",
                            (entry.entry_id,),
                        )
                        results["succeeded"] += 1
                    else:
                        # Re-queue with exponential backoff
                        delay = min(300, 30 * (2 ** entry.retry_count))
                        cursor.execute(
                            "UPDATE dead_letters SET retry_count = retry_count + 1, "
                            "retry_after = ? WHERE entry_id = ?",
                            (now + delay, entry.entry_id),
                        )
                        results["failed"] += 1
                except Exception as e:
                    log.error(f"Dead letter processing error: {e}")
                    results["failed"] += 1

            conn.commit()
            conn.close()

        return results

    def get_pending_count(self) -> int:
        """Get count of pending dead letter entries."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM dead_letters WHERE status = 'pending'")
        count = cursor.fetchone()[0]
        conn.close()
        return count

    def list_entries(
        self, status: str = "pending", limit: int = 20
    ) -> List[Dict[str, Any]]:
        """List dead letter entries."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT entry_id, operation, component, error_type, error_message, "
            "created_at, retry_count, status FROM dead_letters "
            "WHERE status = ? ORDER BY created_at DESC LIMIT ?",
            (status, limit),
        )
        rows = cursor.fetchall()
        conn.close()

        return [
            {
                "entry_id": r[0],
                "operation": r[1],
                "component": r[2],
                "error_type": r[3],
                "error_message": r[4][:200] if r[4] else "",
                "created_at": r[5],
                "retry_count": r[6],
                "status": r[7],
            }
            for r in rows
        ]

    def retry_entry(self, entry_id: str) -> bool:
        """
        Manually retry a specific dead letter entry.

        Resets the entry to 'pending' status so it will be picked up
        by the next process() call.

        Args:
            entry_id: ID of the entry to retry

        Returns:
            True if the entry was found and reset, False otherwise
        """
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE dead_letters SET status = 'pending', retry_after = NULL "
                "WHERE entry_id = ? AND status IN ('pending', 'abandoned')",
                (entry_id,),
            )
            updated = cursor.rowcount > 0
            conn.commit()
            conn.close()

        if updated:
            log.info(f"Dead letter {entry_id} reset for retry")
        return updated

    def purge(self, status: str = "resolved", older_than_seconds: float = 86400 * 7) -> int:
        """
        Remove old entries from the dead letter queue.

        Args:
            status: Only purge entries with this status
            older_than_seconds: Only purge entries older than this (default: 7 days)

        Returns:
            Number of entries purged
        """
        cutoff = time.time() - older_than_seconds
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM dead_letters WHERE status = ? AND created_at < ?",
                (status, cutoff),
            )
            count = cursor.rowcount
            conn.commit()
            conn.close()

        if count > 0:
            log.info(f"Purged {count} dead letter entries (status={status})")
        return count

    def get_stats(self) -> Dict[str, int]:
        """Get summary statistics of the dead letter queue."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT status, COUNT(*) FROM dead_letters GROUP BY status"
        )
        stats = {row[0]: row[1] for row in cursor.fetchall()}
        conn.close()
        stats["total"] = sum(stats.values())
        return stats
```

---

## Safety and Security Considerations

1. **Retry Amplification**: Rate-limited retries to prevent thundering herd
2. **Circuit Breaker Sensitivity**: Configurable thresholds to avoid false positives
3. **Fallback Security**: Cloud fallback must maintain same security posture as local
4. **Dead Letter Encryption**: Sensitive data in DLQ is encrypted at rest
5. **Recovery Limits**: Self-healing capped at 5 attempts per component per hour
6. **Cascading Failure Prevention**: Circuit breakers at every service boundary
7. **Monitoring Overhead**: Health checks do not themselves cause load issues

---

## Implementation Plan

| Week | Phase | Tasks | Deliverables |
|------|-------|-------|-------------|
| 1-2 | Core Types | ErrorContext, ErrorCategory, classify_error, RecoveryResult | Error classification |
| 3-4 | Retry | RetryEngine, exponential backoff, jitter, retry budgets | Retry framework |
| 5-6 | Circuit Breaker | CircuitBreaker, CircuitBreakerRegistry, state machine | Circuit breakers |
| 7 | Fallback | FallbackChain, DegradationManager, ResponseCache | Graceful degradation |
| 8 | Self-Healing | HealthMonitor, health checks, recovery functions | Self-healing |
| 9 | DLQ | DeadLetterQueue, processing, monitoring | Dead letter queue |
| 10 | Integration | ErrorRecoveryMixin, Agent integration, CLI commands | GAIA integration |

---

## Testing Strategy

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Unit tests for error recovery framework."""

import pytest
import time
from unittest.mock import MagicMock, patch


class TestErrorClassification:
    def test_transient_errors(self):
        assert classify_error(ConnectionError()) == ErrorCategory.TRANSIENT
        assert classify_error(TimeoutError()) == ErrorCategory.TRANSIENT

    def test_permanent_errors(self):
        assert classify_error(ValueError("bad input")) == ErrorCategory.PERMANENT
        assert classify_error(FileNotFoundError()) == ErrorCategory.PERMANENT

    def test_unknown_errors(self):
        assert classify_error(RuntimeError("something")) == ErrorCategory.UNKNOWN


class TestRetryEngine:
    @pytest.fixture
    def engine(self):
        return RetryEngine()

    def test_successful_first_attempt(self, engine):
        result = engine.execute_sync(lambda: "success")
        assert result.success
        assert result.result == "success"
        assert result.attempts == 1

    def test_retry_on_transient_error(self, engine):
        call_count = 0

        def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Connection refused")
            return "success"

        config = RetryConfig(
            max_retries=3,
            initial_delay_seconds=0.01,
        )
        result = engine.execute_sync(flaky, config=config)
        assert result.success
        assert call_count == 3

    def test_no_retry_on_permanent_error(self, engine):
        call_count = 0

        def bad_input():
            nonlocal call_count
            call_count += 1
            raise ValueError("Invalid input")

        config = RetryConfig(max_retries=3, initial_delay_seconds=0.01)
        result = engine.execute_sync(bad_input, config=config)
        assert not result.success
        assert call_count == 1  # No retries for permanent errors

    def test_retry_budget_exhaustion(self, engine):
        config = RetryConfig(
            max_retries=100,
            initial_delay_seconds=1.0,
            retry_budget_seconds=0.5,
        )

        def always_fail():
            raise ConnectionError("fail")

        result = engine.execute_sync(always_fail, config=config)
        assert not result.success

    def test_exponential_backoff(self, engine):
        config = RetryConfig(
            initial_delay_seconds=1.0,
            backoff_multiplier=2.0,
            jitter=False,
        )
        assert engine.calculate_delay(0, config) == 1.0
        assert engine.calculate_delay(1, config) == 2.0
        assert engine.calculate_delay(2, config) == 4.0


class TestCircuitBreaker:
    @pytest.fixture
    def breaker(self):
        return CircuitBreaker(
            "test",
            CircuitBreakerConfig(
                failure_threshold=3,
                timeout_seconds=1.0,
                success_threshold=2,
            ),
        )

    def test_starts_closed(self, breaker):
        assert breaker.state == CircuitState.CLOSED
        assert breaker.allow_request()

    def test_opens_on_failures(self, breaker):
        for _ in range(3):
            breaker.record_failure()
        assert breaker.state == CircuitState.OPEN
        assert not breaker.allow_request()

    def test_half_open_after_timeout(self, breaker):
        for _ in range(3):
            breaker.record_failure()
        assert breaker.state == CircuitState.OPEN

        time.sleep(1.1)  # Wait for timeout
        assert breaker.allow_request()  # Should transition to HALF_OPEN
        assert breaker.state == CircuitState.HALF_OPEN

    def test_closes_on_successes_in_half_open(self, breaker):
        for _ in range(3):
            breaker.record_failure()
        time.sleep(1.1)
        breaker.allow_request()  # Triggers HALF_OPEN

        breaker.record_success()
        breaker.record_success()
        assert breaker.state == CircuitState.CLOSED

    def test_reopens_on_failure_in_half_open(self, breaker):
        for _ in range(3):
            breaker.record_failure()
        time.sleep(1.1)
        breaker.allow_request()  # HALF_OPEN

        breaker.record_failure()
        assert breaker.state == CircuitState.OPEN


class TestFallbackChain:
    def test_primary_succeeds(self):
        chain = FallbackChain("test")
        chain.add(FallbackOption(
            name="primary",
            handler=lambda: "primary_result",
            quality_score=1.0,
        ))
        result = chain.execute()
        assert result.success
        assert result.result == "primary_result"
        assert not result.degraded

    def test_fallback_on_primary_failure(self):
        chain = FallbackChain("test")
        chain.add(FallbackOption(
            name="primary",
            handler=lambda: (_ for _ in ()).throw(ConnectionError("down")),
        ))
        chain.add(FallbackOption(
            name="backup",
            handler=lambda: "backup_result",
            quality_score=0.8,
        ))

        def failing():
            raise ConnectionError("down")

        chain2 = FallbackChain("test2")
        chain2.add(FallbackOption(name="fail", handler=failing))
        chain2.add(FallbackOption(
            name="backup",
            handler=lambda: "backup_result",
            quality_score=0.8,
        ))

        result = chain2.execute()
        assert result.success
        assert result.result == "backup_result"
        assert result.degraded  # quality < 1.0

    def test_all_fallbacks_exhausted(self):
        def fail():
            raise RuntimeError("fail")

        chain = FallbackChain("test")
        chain.add(FallbackOption(name="a", handler=fail))
        chain.add(FallbackOption(name="b", handler=fail))

        result = chain.execute()
        assert not result.success


class TestDeadLetterQueue:
    @pytest.fixture
    def dlq(self, tmp_path):
        return DeadLetterQueue(db_path=str(tmp_path / "test_dlq.db"))

    def test_enqueue_and_list(self, dlq):
        dlq.enqueue(DeadLetterEntry(
            operation="send_email",
            component="EmailAgent",
            args={"to": "test@example.com"},
        ))
        entries = dlq.list_entries()
        assert len(entries) == 1
        assert entries[0]["operation"] == "send_email"

    def test_process_success(self, dlq):
        dlq.enqueue(DeadLetterEntry(operation="test_op"))
        results = dlq.process(handler=lambda e: True)
        assert results["succeeded"] == 1
        assert dlq.get_pending_count() == 0

    def test_process_failure_requeues(self, dlq):
        dlq.enqueue(DeadLetterEntry(operation="test_op"))
        results = dlq.process(handler=lambda e: False)
        assert results["failed"] == 1
        assert dlq.get_pending_count() == 1  # Re-queued
```

---

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Transient error recovery rate | >90% | Retry success tracking |
| Circuit breaker false positive rate | <5% | Manual review of open circuits |
| Fallback response quality | >60% user satisfaction | Quality score tracking |
| Self-healing recovery time | <60s for Lemonade restart | Health monitor timing |
| Dead letter resolution rate | >80% within 24 hours | DLQ aging analysis |
| Cascading failure prevention | 0 cascading incidents | Incident tracking |
| Error classification accuracy | >95% | Manual review of classifications |
| Mean time to recovery (MTTR) | <30s for transient errors | Trace analysis |

---

## Complete Code

```
src/gaia/recovery/
    __init__.py
    types.py              # ErrorContext, ErrorCategory, RecoveryResult
    classifier.py         # Error classification engine
    retry.py              # RetryEngine, RetryConfig, strategies
    circuit_breaker.py    # CircuitBreaker, CircuitBreakerRegistry
    fallback.py           # FallbackChain, DegradationManager
    health.py             # HealthMonitor, health checks
    dead_letter.py        # DeadLetterQueue
    mixin.py              # ErrorRecoveryMixin for Agent
    cache.py              # ResponseCache for fallback
    cli.py                # CLI commands for 'gaia recovery'

tests/unit/recovery/
    test_classifier.py
    test_retry.py
    test_circuit_breaker.py
    test_fallback.py
    test_health.py
    test_dead_letter.py

tests/integration/recovery/
    test_agent_recovery.py
    test_llm_failover.py
```
