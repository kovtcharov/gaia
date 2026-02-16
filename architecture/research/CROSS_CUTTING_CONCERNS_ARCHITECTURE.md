# Cross-Cutting Concerns Architecture for GAIA

**Date**: February 7, 2026
**Version**: 1.0
**Status**: Specification
**Priority**: HIGH
**Estimated Effort**: 14-16 weeks (3 engineers)
**Target**: Unified infrastructure patterns shared across all GAIA architectures

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Storage Abstraction Layer](#1-storage-abstraction-layer)
3. [Dependency Injection Framework](#2-dependency-injection-framework)
4. [Async/Sync Bridging](#3-asyncsync-bridging)
5. [Configuration Management](#4-configuration-management)
6. [Logging and Error Formatting](#5-logging-and-error-formatting)
7. [Resource Management](#6-resource-management)
8. [Plugin Architecture](#7-plugin-architecture)
9. [Database Migration Framework](#8-database-migration-framework)
10. [Integration Map](#integration-map)
11. [Implementation Plan](#implementation-plan)
12. [Testing Strategy](#testing-strategy)

---

## Executive Summary

### The Problem

GAIA's architecture documents -- Persistent Memory, Observability, Security, Workflow Orchestration, and others -- each introduce their own SQLite schemas, configuration formats, logging approaches, and resource management patterns. This duplication creates:

- **7+ independent SQLite schemas** with no migration story and no abstraction over the storage engine
- **Direct instantiation everywhere** -- agents `__init__` methods create their own `LemonadeClient`, `ChatSDK`, and database connections, making unit testing require heavy mocking
- **Mixed async/sync code** -- `ChatSDK.send()` is async, `DatabaseMixin.query()` is sync, and bridging between them uses ad-hoc `asyncio.run()` calls
- **No unified configuration** -- each module reads its own env vars, has its own defaults, and defines its own dataclass (`ChatConfig`, `LemonadeClient` defaults, etc.)
- **Inconsistent logging** -- 51 files call `logging.getLogger()` independently, with no correlation IDs and no structured output
- **No resource pooling** -- every agent opens its own SQLite connection, HTTP session, and thread pool
- **No plugin interface** -- third-party tool integration requires forking the codebase
- **No migration framework** -- schema changes require manual `table_exists()` checks scattered across agent code

### The Solution

Eight cross-cutting infrastructure systems that provide shared foundations for all GAIA architectures:

| System | What It Solves | Lines of Code |
|--------|---------------|---------------|
| Storage Abstraction | SQLite lock-in, no backend swapping | ~400 |
| Dependency Injection | Hard-to-test singletons, direct instantiation | ~300 |
| Async/Sync Bridging | Mixed paradigms, threading bugs | ~200 |
| Configuration Management | Scattered env vars, no validation | ~350 |
| Logging & Error Formatting | No correlation IDs, inconsistent formats | ~300 |
| Resource Management | No pooling, leaked connections | ~350 |
| Plugin Architecture | No extensibility for third parties | ~400 |
| Database Migration | No schema versioning, manual checks | ~350 |

### Design Principles

1. **Zero mandatory dependencies** -- every system works with Python's stdlib; optional packages enhance capability
2. **Backward compatible** -- existing `DatabaseMixin`, `GaiaLogger`, and `create_client()` patterns continue to work
3. **Opt-in adoption** -- teams adopt one system at a time; no big-bang migration required
4. **Testability first** -- every component is designed for in-memory testing without external services

---

## 1. Storage Abstraction Layer

### Problem Analysis

The current `DatabaseMixin` in `src/gaia/database/mixin.py` hard-codes `sqlite3` throughout:

```python
# Current: src/gaia/database/mixin.py (lines 7-8, 42, 63)
import sqlite3
from contextlib import contextmanager

class DatabaseMixin:
    _db: Optional[sqlite3.Connection] = None

    def init_db(self, path: str = ":memory:") -> None:
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
```

Every architecture document that stores data (Persistent Memory, Observability, Security, Workflow Orchestration) creates its own SQLite tables with no shared abstraction. This means:

- Cannot swap to PostgreSQL for production multi-user deployments
- Cannot use MongoDB for document-heavy workloads (RAG metadata)
- No connection pooling (each mixin instance opens a raw connection)
- No query parameterization validation across backends

### Solution: Repository Pattern

The Repository pattern decouples data access from storage implementation. Each domain defines a repository interface; backends implement it.

```
src/gaia/storage/
    __init__.py
    repository.py       # Abstract Repository interface
    backends/
        __init__.py
        sqlite.py       # SQLite backend (default)
        postgres.py     # PostgreSQL backend (optional)
        memory.py       # In-memory backend (testing)
    pool.py             # Connection pool manager
    types.py            # Shared types (QueryResult, etc.)
```

### Repository Interface

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Abstract repository interface for GAIA storage."""

from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, Sequence, Union


@dataclass
class QueryResult:
    """Result of a database query."""

    rows: List[Dict[str, Any]]
    count: int
    columns: List[str] = field(default_factory=list)

    @property
    def first(self) -> Optional[Dict[str, Any]]:
        """Return first row or None."""
        return self.rows[0] if self.rows else None

    @property
    def scalar(self) -> Optional[Any]:
        """Return first column of first row, or None."""
        if self.rows and self.columns:
            return self.rows[0].get(self.columns[0])
        return None

    def __iter__(self):
        return iter(self.rows)

    def __len__(self):
        return self.count


@dataclass
class TableSchema:
    """Description of a database table."""

    name: str
    columns: List[Dict[str, Any]]
    primary_key: Optional[str] = None
    indexes: List[str] = field(default_factory=list)


class Repository(ABC):
    """
    Abstract storage repository for GAIA.

    All data access goes through this interface. Backends implement
    the actual storage logic (SQLite, PostgreSQL, in-memory, etc.).

    Example:
        repo = SQLiteRepository("data/myagent.db")
        repo.execute_ddl('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL
            )
        ''')
        row_id = repo.insert("users", {"name": "Alice"})
        result = repo.query("SELECT * FROM users WHERE id = :id", {"id": row_id})
        print(result.first)  # {"id": 1, "name": "Alice"}
    """

    @abstractmethod
    def connect(self) -> None:
        """Establish connection to the storage backend."""
        ...

    @abstractmethod
    def disconnect(self) -> None:
        """Close connection to the storage backend."""
        ...

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """True if the repository has an active connection."""
        ...

    @abstractmethod
    def query(
        self,
        sql: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> QueryResult:
        """
        Execute a SELECT query and return results.

        Args:
            sql: SQL query with :param_name placeholders.
            params: Dictionary of parameter values.

        Returns:
            QueryResult containing rows, count, and column names.
        """
        ...

    @abstractmethod
    def insert(self, table: str, data: Dict[str, Any]) -> int:
        """
        Insert a row and return its ID.

        Args:
            table: Table name.
            data: Column-value dictionary.

        Returns:
            The inserted row's ID.
        """
        ...

    @abstractmethod
    def insert_many(self, table: str, rows: Sequence[Dict[str, Any]]) -> int:
        """
        Insert multiple rows efficiently.

        Args:
            table: Table name.
            rows: Sequence of column-value dictionaries.

        Returns:
            Number of rows inserted.
        """
        ...

    @abstractmethod
    def update(
        self,
        table: str,
        data: Dict[str, Any],
        where: str,
        params: Dict[str, Any],
    ) -> int:
        """
        Update rows matching condition.

        Args:
            table: Table name.
            data: Column-value dictionary of new values.
            where: WHERE clause with :param placeholders.
            params: Parameters for WHERE clause.

        Returns:
            Number of rows affected.
        """
        ...

    @abstractmethod
    def delete(self, table: str, where: str, params: Dict[str, Any]) -> int:
        """
        Delete rows matching condition.

        Args:
            table: Table name.
            where: WHERE clause with :param placeholders.
            params: Parameters for WHERE clause.

        Returns:
            Number of rows deleted.
        """
        ...

    @abstractmethod
    def execute_ddl(self, sql: str) -> None:
        """
        Execute DDL statements (CREATE TABLE, ALTER TABLE, etc.).

        Args:
            sql: DDL SQL statement(s).
        """
        ...

    @abstractmethod
    def table_exists(self, name: str) -> bool:
        """Check if a table exists."""
        ...

    @abstractmethod
    def get_table_schema(self, name: str) -> Optional[TableSchema]:
        """Get schema information for a table."""
        ...

    @abstractmethod
    @contextmanager
    def transaction(self) -> Generator[None, None, None]:
        """
        Execute operations atomically.

        Auto-commits on success, rolls back on exception.
        """
        ...
```

### SQLite Backend

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""SQLite storage backend for GAIA."""

import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Sequence

from gaia.storage.repository import QueryResult, Repository, TableSchema

logger = logging.getLogger(__name__)


class SQLiteRepository(Repository):
    """
    SQLite implementation of the Repository interface.

    This is the default backend for GAIA, using Python's built-in
    sqlite3 module with zero external dependencies.

    Args:
        path: Database file path, or ":memory:" for in-memory.
        wal_mode: Enable WAL mode for concurrent readers (default True).
        busy_timeout_ms: Milliseconds to wait on locked database (default 5000).
    """

    def __init__(
        self,
        path: str = ":memory:",
        wal_mode: bool = True,
        busy_timeout_ms: int = 5000,
    ):
        self._path = path
        self._wal_mode = wal_mode
        self._busy_timeout_ms = busy_timeout_ms
        self._conn: Optional[sqlite3.Connection] = None
        self._in_tx: bool = False

    def connect(self) -> None:
        """Establish SQLite connection with recommended pragmas."""
        if self._conn is not None:
            return

        if self._path != ":memory:":
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(
            self._path,
            check_same_thread=False,
            timeout=self._busy_timeout_ms / 1000.0,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute(f"PRAGMA busy_timeout = {self._busy_timeout_ms}")

        if self._wal_mode and self._path != ":memory:":
            self._conn.execute("PRAGMA journal_mode = WAL")

        logger.info("SQLite connected: %s", self._path)

    def disconnect(self) -> None:
        """Close SQLite connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
            self._in_tx = False

    @property
    def is_connected(self) -> bool:
        return self._conn is not None

    def _require_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError(
                "Repository not connected. Call connect() first."
            )
        return self._conn

    def query(
        self,
        sql: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> QueryResult:
        conn = self._require_conn()
        cursor = conn.execute(sql, params or {})
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        rows = [dict(row) for row in cursor.fetchall()]
        return QueryResult(rows=rows, count=len(rows), columns=columns)

    def insert(self, table: str, data: Dict[str, Any]) -> int:
        conn = self._require_conn()
        cols = ", ".join(data.keys())
        placeholders = ", ".join(f":{k}" for k in data.keys())
        sql = f"INSERT INTO {table} ({cols}) VALUES ({placeholders})"
        cursor = conn.execute(sql, data)
        if not self._in_tx:
            conn.commit()
        return cursor.lastrowid

    def insert_many(self, table: str, rows: Sequence[Dict[str, Any]]) -> int:
        if not rows:
            return 0
        conn = self._require_conn()
        cols = ", ".join(rows[0].keys())
        placeholders = ", ".join(f":{k}" for k in rows[0].keys())
        sql = f"INSERT INTO {table} ({cols}) VALUES ({placeholders})"
        cursor = conn.executemany(sql, rows)
        if not self._in_tx:
            conn.commit()
        return cursor.rowcount

    def update(
        self,
        table: str,
        data: Dict[str, Any],
        where: str,
        params: Dict[str, Any],
    ) -> int:
        conn = self._require_conn()
        set_clause = ", ".join(f"{k} = :__set_{k}" for k in data.keys())
        merged = {f"__set_{k}": v for k, v in data.items()}
        merged.update(params)
        sql = f"UPDATE {table} SET {set_clause} WHERE {where}"
        cursor = conn.execute(sql, merged)
        if not self._in_tx:
            conn.commit()
        return cursor.rowcount

    def delete(self, table: str, where: str, params: Dict[str, Any]) -> int:
        conn = self._require_conn()
        sql = f"DELETE FROM {table} WHERE {where}"
        cursor = conn.execute(sql, params)
        if not self._in_tx:
            conn.commit()
        return cursor.rowcount

    def execute_ddl(self, sql: str) -> None:
        conn = self._require_conn()
        if self._in_tx:
            raise RuntimeError(
                "execute_ddl() cannot be called inside a transaction. "
                "DDL statements auto-commit in SQLite."
            )
        conn.executescript(sql)

    def table_exists(self, name: str) -> bool:
        result = self.query(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=:name",
            {"name": name},
        )
        return result.count > 0

    def get_table_schema(self, name: str) -> Optional[TableSchema]:
        if not self.table_exists(name):
            return None
        result = self.query(f"PRAGMA table_info({name})")
        columns = [
            {
                "name": row["name"],
                "type": row["type"],
                "nullable": not row["notnull"],
                "primary_key": bool(row["pk"]),
                "default": row["dflt_value"],
            }
            for row in result
        ]
        pk_cols = [c["name"] for c in columns if c["primary_key"]]
        return TableSchema(
            name=name,
            columns=columns,
            primary_key=pk_cols[0] if pk_cols else None,
        )

    @contextmanager
    def transaction(self) -> Generator[None, None, None]:
        conn = self._require_conn()
        self._in_tx = True
        try:
            yield
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._in_tx = False
```

### In-Memory Backend (Testing)

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""In-memory storage backend for testing."""

from gaia.storage.backends.sqlite import SQLiteRepository


class MemoryRepository(SQLiteRepository):
    """
    In-memory SQLite repository for testing.

    Convenience wrapper that always uses ":memory:" and auto-connects.

    Example:
        repo = MemoryRepository()
        repo.execute_ddl("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)")
        repo.insert("t", {"v": "test"})
        assert repo.query("SELECT * FROM t").count == 1
    """

    def __init__(self):
        super().__init__(path=":memory:", wal_mode=False)
        self.connect()
```

### Migration from DatabaseMixin

The existing `DatabaseMixin` remains as a backward-compatible wrapper over the new Repository:

```python
# Updated DatabaseMixin (backward compatible)
class DatabaseMixin:
    """
    Mixin providing database access for GAIA agents.

    Now backed by the Repository abstraction. Existing code continues
    to work unchanged; new code should prefer Repository directly.
    """

    _repo: Optional[Repository] = None

    def init_db(self, path: str = ":memory:", backend: str = "sqlite") -> None:
        if self._repo and self._repo.is_connected:
            self._repo.disconnect()

        if backend == "sqlite":
            from gaia.storage.backends.sqlite import SQLiteRepository
            self._repo = SQLiteRepository(path)
        elif backend == "memory":
            from gaia.storage.backends.memory import MemoryRepository
            self._repo = MemoryRepository()
        else:
            raise ValueError(f"Unknown backend: {backend}")

        self._repo.connect()

    def close_db(self) -> None:
        if self._repo:
            self._repo.disconnect()
            self._repo = None

    @property
    def db_ready(self) -> bool:
        return self._repo is not None and self._repo.is_connected

    def query(self, sql, params=None, one=False):
        result = self._repo.query(sql, params)
        if one:
            return result.first
        return result.rows

    def insert(self, table, data):
        return self._repo.insert(table, data)

    def update(self, table, data, where, params):
        return self._repo.update(table, data, where, params)

    def delete(self, table, where, params):
        return self._repo.delete(table, where, params)

    def execute(self, sql):
        return self._repo.execute_ddl(sql)

    def table_exists(self, name):
        return self._repo.table_exists(name)

    @contextmanager
    def transaction(self):
        with self._repo.transaction():
            yield
```

---

## 2. Dependency Injection Framework

### Problem Analysis

GAIA agents directly instantiate their dependencies in `__init__`:

```python
# Current: src/gaia/agents/base/agent.py (lines 64-84)
class Agent(abc.ABC):
    def __init__(self, use_claude=False, base_url=None, model_id=None, ...):
        # Direct instantiation -- hard to test, hard to swap
        self.chat_sdk = ChatSDK(ChatConfig(
            model=model_id or DEFAULT_MODEL_NAME,
            base_url=base_url,
            use_claude=use_claude,
        ))
```

This creates tight coupling: testing an agent requires either a running Lemonade server or heavy mocking of internal implementation details. The `skip_lemonade` parameter was added as a workaround, but it is a code smell indicating that the real solution is dependency injection.

### Solution: Lightweight DI Container

GAIA does not need a heavyweight DI framework like those in Java. A Python-native approach using a service registry with type-based resolution is sufficient.

```
src/gaia/di/
    __init__.py
    container.py    # Service container
    providers.py    # Provider protocols
    scope.py        # Scoping (singleton, transient, request)
```

### Service Container

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Lightweight dependency injection container for GAIA."""

import inspect
import logging
from contextlib import contextmanager
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    Optional,
    Type,
    TypeVar,
    Union,
    get_type_hints,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


class Scope(Enum):
    """Service lifetime scope."""

    SINGLETON = auto()   # One instance for the container lifetime
    TRANSIENT = auto()   # New instance on every resolve
    REQUEST = auto()     # One instance per request scope


class ServiceDescriptor:
    """Describes how to create and manage a service."""

    def __init__(
        self,
        service_type: Type,
        factory: Callable[..., Any],
        scope: Scope = Scope.SINGLETON,
    ):
        self.service_type = service_type
        self.factory = factory
        self.scope = scope
        self.instance: Optional[Any] = None


class Container:
    """
    Lightweight dependency injection container.

    Register services with factories, then resolve them by type.
    The container handles lifecycle management and dependency chains.

    Example:
        container = Container()

        # Register services
        container.register(Repository, lambda: SQLiteRepository(":memory:"))
        container.register(LLMClient, lambda: create_client(provider="lemonade"))

        # Register with dependencies auto-resolved
        container.register_class(ChatSDK)

        # Resolve
        repo = container.resolve(Repository)
        chat = container.resolve(ChatSDK)

        # Override for testing
        test_container = Container()
        test_container.register(Repository, lambda: MemoryRepository())
        test_container.register(LLMClient, lambda: MockLLMClient())
    """

    def __init__(self, parent: Optional["Container"] = None):
        self._descriptors: Dict[Type, ServiceDescriptor] = {}
        self._parent = parent
        self._request_instances: Dict[Type, Any] = {}

    def register(
        self,
        service_type: Type[T],
        factory: Callable[..., T],
        scope: Scope = Scope.SINGLETON,
    ) -> "Container":
        """
        Register a service with an explicit factory.

        Args:
            service_type: The type (usually an ABC or Protocol) to register.
            factory: Callable that creates the service instance.
            scope: Lifetime scope for the service.

        Returns:
            Self for chaining.
        """
        self._descriptors[service_type] = ServiceDescriptor(
            service_type=service_type,
            factory=factory,
            scope=scope,
        )
        return self

    def register_instance(
        self,
        service_type: Type[T],
        instance: T,
    ) -> "Container":
        """
        Register a pre-created instance (always singleton).

        Args:
            service_type: The type to register.
            instance: The existing instance to use.

        Returns:
            Self for chaining.
        """
        desc = ServiceDescriptor(
            service_type=service_type,
            factory=lambda: instance,
            scope=Scope.SINGLETON,
        )
        desc.instance = instance
        self._descriptors[service_type] = desc
        return self

    def register_class(
        self,
        implementation: Type[T],
        service_type: Optional[Type] = None,
        scope: Scope = Scope.SINGLETON,
    ) -> "Container":
        """
        Register a class, auto-resolving constructor dependencies.

        The container inspects __init__ type hints and resolves
        each parameter from the container.

        Args:
            implementation: The concrete class to instantiate.
            service_type: The abstract type to register as (default: same as implementation).
            scope: Lifetime scope.

        Returns:
            Self for chaining.
        """
        if service_type is None:
            service_type = implementation

        def factory():
            return self._auto_create(implementation)

        self._descriptors[service_type] = ServiceDescriptor(
            service_type=service_type,
            factory=factory,
            scope=scope,
        )
        return self

    def resolve(self, service_type: Type[T]) -> T:
        """
        Resolve a service by type.

        Checks this container first, then parent container.

        Args:
            service_type: The type to resolve.

        Returns:
            The service instance.

        Raises:
            KeyError: If service type is not registered.
        """
        desc = self._descriptors.get(service_type)

        if desc is None:
            if self._parent is not None:
                return self._parent.resolve(service_type)
            raise KeyError(
                f"Service not registered: {service_type.__name__}. "
                f"Available: {[t.__name__ for t in self._descriptors]}"
            )

        if desc.scope == Scope.SINGLETON:
            if desc.instance is None:
                desc.instance = desc.factory()
                logger.debug("Created singleton: %s", service_type.__name__)
            return desc.instance

        elif desc.scope == Scope.REQUEST:
            if service_type in self._request_instances:
                return self._request_instances[service_type]
            instance = desc.factory()
            self._request_instances[service_type] = instance
            return instance

        else:  # TRANSIENT
            return desc.factory()

    def has(self, service_type: Type) -> bool:
        """Check if a service type is registered."""
        if service_type in self._descriptors:
            return True
        if self._parent is not None:
            return self._parent.has(service_type)
        return False

    @contextmanager
    def request_scope(self):
        """
        Create a request scope. Request-scoped services are shared
        within this context and disposed afterward.

        Example:
            with container.request_scope():
                svc1 = container.resolve(MyService)
                svc2 = container.resolve(MyService)
                assert svc1 is svc2  # Same instance within scope
        """
        self._request_instances.clear()
        try:
            yield self
        finally:
            # Dispose request-scoped instances that have cleanup methods
            for instance in self._request_instances.values():
                if hasattr(instance, "close"):
                    try:
                        instance.close()
                    except Exception as exc:
                        logger.warning("Error closing %s: %s", type(instance).__name__, exc)
                elif hasattr(instance, "disconnect"):
                    try:
                        instance.disconnect()
                    except Exception as exc:
                        logger.warning("Error disconnecting %s: %s", type(instance).__name__, exc)
            self._request_instances.clear()

    def create_child(self) -> "Container":
        """Create a child container that inherits this container's registrations."""
        return Container(parent=self)

    def dispose(self) -> None:
        """Dispose all singleton instances that have cleanup methods."""
        for desc in self._descriptors.values():
            if desc.instance is not None and desc.scope == Scope.SINGLETON:
                if hasattr(desc.instance, "close"):
                    try:
                        desc.instance.close()
                    except Exception as exc:
                        logger.warning(
                            "Error closing %s: %s",
                            desc.service_type.__name__,
                            exc,
                        )
                elif hasattr(desc.instance, "disconnect"):
                    try:
                        desc.instance.disconnect()
                    except Exception as exc:
                        logger.warning(
                            "Error disconnecting %s: %s",
                            desc.service_type.__name__,
                            exc,
                        )
                desc.instance = None

    def _auto_create(self, cls: Type) -> Any:
        """Create an instance by resolving constructor type hints."""
        try:
            hints = get_type_hints(cls.__init__)
        except Exception:
            hints = {}

        hints.pop("return", None)

        sig = inspect.signature(cls.__init__)
        kwargs = {}

        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue

            param_type = hints.get(param_name)

            if param_type is not None and self.has(param_type):
                kwargs[param_name] = self.resolve(param_type)
            elif param.default is not inspect.Parameter.empty:
                continue  # Use the default
            else:
                logger.debug(
                    "Cannot resolve param '%s' of type %s for %s",
                    param_name,
                    param_type,
                    cls.__name__,
                )

        return cls(**kwargs)
```

### Agent Factory with DI

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Agent factory using dependency injection."""

from typing import Type, TypeVar

from gaia.di.container import Container, Scope
from gaia.llm.base_client import LLMClient
from gaia.storage.repository import Repository

T = TypeVar("T")


def create_default_container(**overrides) -> Container:
    """
    Create a container with default GAIA service registrations.

    Args:
        **overrides: Keyword arguments to override default configuration.
            - provider: LLM provider name ("lemonade", "claude", "openai")
            - db_path: Database path (default ":memory:")
            - base_url: LLM server base URL

    Returns:
        Configured Container instance.
    """
    from gaia.llm.factory import create_client
    from gaia.storage.backends.sqlite import SQLiteRepository

    container = Container()

    # Storage
    db_path = overrides.get("db_path", ":memory:")
    container.register(
        Repository,
        lambda: _connect(SQLiteRepository(db_path)),
        scope=Scope.SINGLETON,
    )

    # LLM Client
    provider = overrides.get("provider", "lemonade")
    base_url = overrides.get("base_url")
    llm_kwargs = {}
    if base_url:
        llm_kwargs["base_url"] = base_url
    container.register(
        LLMClient,
        lambda: create_client(provider=provider, **llm_kwargs),
        scope=Scope.SINGLETON,
    )

    return container


def create_test_container() -> Container:
    """
    Create a container with mock/in-memory services for testing.

    Returns:
        Container configured for unit testing (no external services).
    """
    from gaia.storage.backends.memory import MemoryRepository

    container = Container()

    container.register(
        Repository,
        MemoryRepository,
        scope=Scope.SINGLETON,
    )

    # LLM client is not registered -- agents under test should
    # either use skip_lemonade=True or register a mock explicitly.

    return container


def _connect(repo: Repository) -> Repository:
    """Helper to connect a repository before returning it."""
    repo.connect()
    return repo
```

### Usage in Agents

```python
# Before (current pattern):
class ChatAgent(Agent, DatabaseMixin):
    def __init__(self, db_path="data/chat.db", **kwargs):
        super().__init__(**kwargs)         # Creates its own LLM client
        self.init_db(db_path)             # Creates its own SQLite connection

# After (with DI, fully backward compatible):
class ChatAgent(Agent, DatabaseMixin):
    def __init__(self, db_path="data/chat.db", container=None, **kwargs):
        super().__init__(**kwargs)

        if container and container.has(Repository):
            # Use injected repository
            self._repo = container.resolve(Repository)
        else:
            # Fallback to existing behavior
            self.init_db(db_path)
```

---

## 3. Async/Sync Bridging

### Problem Analysis

GAIA has a fundamental paradigm split:

| Component | Pattern | Example |
|-----------|---------|---------|
| `ChatSDK.send()` | `async def` | `response = await chat.send("hello")` |
| `DatabaseMixin.query()` | Synchronous | `rows = self.query("SELECT ...")` |
| `LemonadeClient.chat_completion()` | Synchronous (uses `requests`) | `response = client.chat_completion(...)` |
| `Agent.process_query()` | Synchronous | `result = agent.process_query("hello")` |
| MCP bridge | `async def` (aiohttp) | WebSocket server |
| Audio pipeline | `threading.Thread` | `whisper_asr.py`, `kokoro_tts.py` |

When sync code needs to call async code (or vice versa), developers reach for ad-hoc solutions:

```python
# Pattern 1: asyncio.run() -- blocks, cannot nest
result = asyncio.run(chat.send("hello"))

# Pattern 2: threading -- loses context
thread = Thread(target=lambda: asyncio.run(coro()))
thread.start()

# Pattern 3: Run loop in thread -- complex, error-prone
loop = asyncio.new_event_loop()
threading.Thread(target=loop.run_forever, daemon=True).start()
future = asyncio.run_coroutine_threadsafe(coro(), loop)
result = future.result(timeout=30)
```

### Solution: Bridging Utilities

```
src/gaia/async_bridge/
    __init__.py
    bridge.py       # Sync-to-async and async-to-sync utilities
    context.py      # Context propagation (correlation IDs across boundaries)
```

### Bridge Implementation

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Async/sync bridging utilities for GAIA."""

import asyncio
import contextvars
import functools
import logging
import threading
from concurrent.futures import Future
from typing import Any, Awaitable, Callable, Coroutine, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Module-level background loop for sync->async bridging
_bridge_loop: asyncio.AbstractEventLoop = None
_bridge_thread: threading.Thread = None
_bridge_lock = threading.Lock()


def _get_bridge_loop() -> asyncio.AbstractEventLoop:
    """
    Get or create a background event loop for sync-to-async bridging.

    This loop runs in a dedicated daemon thread and is shared across
    all callers. It is created on first use and lives for the process
    lifetime.
    """
    global _bridge_loop, _bridge_thread

    if _bridge_loop is not None and _bridge_loop.is_running():
        return _bridge_loop

    with _bridge_lock:
        # Double-check after acquiring lock
        if _bridge_loop is not None and _bridge_loop.is_running():
            return _bridge_loop

        _bridge_loop = asyncio.new_event_loop()
        _bridge_thread = threading.Thread(
            target=_bridge_loop.run_forever,
            name="gaia-async-bridge",
            daemon=True,
        )
        _bridge_thread.start()
        logger.debug("Started async bridge loop in background thread")
        return _bridge_loop


def run_sync(coro: Coroutine[Any, Any, T], timeout: float = 60.0) -> T:
    """
    Run an async coroutine from synchronous code.

    This is the primary entry point for calling async functions from
    sync contexts (e.g., calling ChatSDK.send() from Agent.process_query()).

    Unlike asyncio.run(), this works even when an event loop is already
    running (e.g., inside Jupyter notebooks or nested frameworks).

    Args:
        coro: The coroutine to execute.
        timeout: Maximum seconds to wait (default 60).

    Returns:
        The coroutine's return value.

    Raises:
        TimeoutError: If the coroutine does not complete within timeout.
        Exception: Any exception raised by the coroutine.

    Example:
        # From synchronous agent code:
        async def fetch_data():
            return await http_client.get("/api/data")

        data = run_sync(fetch_data())
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is None:
        # No event loop running -- use asyncio.run() directly
        return asyncio.run(coro)

    # Event loop already running -- use the bridge loop
    bridge_loop = _get_bridge_loop()

    # Copy context variables to the bridge thread
    ctx = contextvars.copy_context()
    future = asyncio.run_coroutine_threadsafe(
        _run_with_context(ctx, coro),
        bridge_loop,
    )

    try:
        return future.result(timeout=timeout)
    except TimeoutError:
        future.cancel()
        raise TimeoutError(
            f"Async operation did not complete within {timeout}s"
        )


async def _run_with_context(
    ctx: contextvars.Context,
    coro: Coroutine[Any, Any, T],
) -> T:
    """Run a coroutine while restoring context variables."""
    # contextvars are thread-local in CPython; we copy them manually
    for var, value in ctx.items():
        var.set(value)
    return await coro


def run_async(func: Callable[..., T], *args: Any, **kwargs: Any) -> Awaitable[T]:
    """
    Run a synchronous (blocking) function from async code without
    blocking the event loop.

    Uses the default executor (thread pool) to run the function.

    Args:
        func: Synchronous callable.
        *args: Positional arguments to pass to func.
        **kwargs: Keyword arguments to pass to func.

    Returns:
        Awaitable that resolves to the function's return value.

    Example:
        # From async handler:
        def heavy_computation(data):
            return process(data)  # CPU-bound

        result = await run_async(heavy_computation, my_data)
    """
    loop = asyncio.get_running_loop()

    # functools.partial preserves kwargs
    if kwargs:
        func_with_args = functools.partial(func, *args, **kwargs)
        return loop.run_in_executor(None, func_with_args)
    else:
        return loop.run_in_executor(None, func, *args)


def sync_compatible(async_func: Callable[..., Coroutine]) -> Callable:
    """
    Decorator that makes an async function callable from both sync and async contexts.

    When called from async code, returns the coroutine directly.
    When called from sync code, runs the coroutine with run_sync().

    Example:
        @sync_compatible
        async def fetch_data(url: str) -> dict:
            async with aiohttp.ClientSession() as session:
                resp = await session.get(url)
                return await resp.json()

        # Works from sync code:
        data = fetch_data("https://api.example.com/data")

        # Works from async code:
        data = await fetch_data("https://api.example.com/data")
    """

    @functools.wraps(async_func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        coro = async_func(*args, **kwargs)

        try:
            asyncio.get_running_loop()
            # We are in an async context -- return the coroutine
            return coro
        except RuntimeError:
            # We are in a sync context -- run it
            return run_sync(coro)

    return wrapper
```

### Context Propagation

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Context propagation across async/sync boundaries."""

import contextvars
import uuid
from typing import Optional

# Context variables that propagate across async/sync boundaries
correlation_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "correlation_id", default=""
)
agent_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "agent_id", default=""
)
request_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default=""
)


def new_correlation_id() -> str:
    """Generate and set a new correlation ID."""
    cid = str(uuid.uuid4())[:12]
    correlation_id.set(cid)
    return cid


def get_correlation_id() -> str:
    """Get the current correlation ID, generating one if needed."""
    cid = correlation_id.get()
    if not cid:
        cid = new_correlation_id()
    return cid


def set_agent_context(aid: str) -> None:
    """Set the current agent ID for context propagation."""
    agent_id.set(aid)


def get_context_dict() -> dict:
    """Get all context variables as a dictionary for logging/tracing."""
    return {
        "correlation_id": correlation_id.get(""),
        "agent_id": agent_id.get(""),
        "request_id": request_id.get(""),
    }
```

### When to Use Each Pattern

| Scenario | Pattern | Function |
|----------|---------|----------|
| Sync agent calls async SDK | Sync-to-async | `run_sync(chat.send("hello"))` |
| Async handler calls sync DB | Async-to-sync | `await run_async(repo.query, sql)` |
| Function used both ways | Dual-compatible | `@sync_compatible` decorator |
| CPU-bound in async context | Offload to thread | `await run_async(heavy_fn, data)` |
| Context across boundaries | ContextVars | `get_correlation_id()` |

### Decision Matrix

```
Is the caller async?
  YES -> Is the callee async?
    YES -> Just await it: result = await callee()
    NO  -> Offload to thread: result = await run_async(callee, args)
  NO  -> Is the callee async?
    YES -> Bridge: result = run_sync(callee())
    NO  -> Just call it: result = callee(args)
```

---

## 4. Configuration Management

### Problem Analysis

Configuration in GAIA is scattered across multiple mechanisms:

| Component | Config Source | Format |
|-----------|-------------|--------|
| LemonadeClient | `LEMONADE_BASE_URL` env var + hardcoded defaults | N/A |
| ChatSDK | `ChatConfig` dataclass | Python |
| Agent | Constructor parameters (15+ kwargs) | Python |
| MCP Bridge | CLI arguments + env vars | N/A |
| RAG | Hardcoded chunk sizes and thresholds | N/A |
| Audio | Hardcoded model paths | N/A |

There is no unified config file, no schema validation, and no environment-specific overrides.

### Solution: Unified Configuration System

```
src/gaia/config/
    __init__.py
    schema.py       # Pydantic-style config models (stdlib only)
    loader.py       # Config file loading and merging
    env.py          # Environment variable integration
    secrets.py      # Integration point with Security architecture
```

### Configuration Schema

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Configuration schema for GAIA."""

import json
import logging
import os
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Default config locations, checked in order
CONFIG_SEARCH_PATHS = [
    Path.home() / ".gaia" / "config.json",           # User config
    Path.cwd() / ".gaia" / "config.json",             # Project config
    Path.cwd() / "gaia.config.json",                  # Project root config
]

ENV_PREFIX = "GAIA_"


@dataclass
class LLMConfig:
    """LLM provider configuration."""

    provider: str = "lemonade"
    model: str = "Qwen3-0.6B-GGUF"
    base_url: str = "http://localhost:8000/api/v1"
    temperature: Optional[float] = None
    max_tokens: int = 4096
    context_size: int = 32768
    request_timeout: int = 900
    model_load_timeout: int = 12000

    # Cloud provider settings (used when provider != "lemonade")
    api_key: Optional[str] = None  # Loaded from secrets, never from file
    claude_model: str = "claude-sonnet-4-20250514"
    openai_model: str = "gpt-4o"


@dataclass
class StorageConfig:
    """Storage configuration."""

    backend: str = "sqlite"
    path: str = "~/.gaia/data/gaia.db"
    wal_mode: bool = True
    busy_timeout_ms: int = 5000
    pool_size: int = 5

    # PostgreSQL (when backend == "postgres")
    pg_host: str = "localhost"
    pg_port: int = 5432
    pg_database: str = "gaia"
    pg_user: str = "gaia"
    pg_password: Optional[str] = None  # Loaded from secrets


@dataclass
class AgentConfig:
    """Agent behavior configuration."""

    max_steps: int = 20
    max_plan_iterations: int = 3
    max_consecutive_repeats: int = 4
    streaming: bool = False
    show_stats: bool = False
    debug: bool = False
    output_dir: Optional[str] = None


@dataclass
class AudioConfig:
    """Audio pipeline configuration."""

    asr_model: str = "whisper-tiny"
    tts_model: str = "kokoro"
    sample_rate: int = 16000
    vad_enabled: bool = True


@dataclass
class RAGConfig:
    """RAG pipeline configuration."""

    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k: int = 5
    similarity_threshold: float = 0.7
    embedding_model: str = "all-MiniLM-L6-v2"


@dataclass
class LoggingConfig:
    """Logging configuration."""

    level: str = "INFO"
    format: str = "structured"  # "structured" or "plain"
    log_file: str = "gaia.log"
    enable_correlation_ids: bool = True
    module_levels: Dict[str, str] = field(default_factory=lambda: {
        "gaia.agents": "INFO",
        "gaia.llm": "INFO",
        "gaia.rag": "INFO",
    })


@dataclass
class PluginConfig:
    """Plugin system configuration."""

    enabled: bool = True
    search_paths: List[str] = field(default_factory=lambda: [
        "~/.gaia/plugins",
    ])
    allowed_plugins: List[str] = field(default_factory=list)  # Empty = all allowed
    sandbox_enabled: bool = True


@dataclass
class GaiaConfig:
    """
    Root configuration for GAIA.

    Aggregates all subsystem configurations into a single tree.
    Can be loaded from JSON files, environment variables, or
    constructed programmatically.

    Example:
        # Load from default locations
        config = GaiaConfig.load()

        # Load from specific file
        config = GaiaConfig.load("/path/to/config.json")

        # Override specific values
        config = GaiaConfig.load()
        config.llm.provider = "claude"
        config.llm.model = "claude-sonnet-4-20250514"
    """

    llm: LLMConfig = field(default_factory=LLMConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    plugins: PluginConfig = field(default_factory=PluginConfig)

    # Environment name for environment-specific overrides
    environment: str = "development"

    @classmethod
    def load(cls, path: Optional[str] = None) -> "GaiaConfig":
        """
        Load configuration from file with environment variable overrides.

        Loading order (later values override earlier):
        1. Built-in defaults (dataclass defaults above)
        2. Config file (first found in search paths, or explicit path)
        3. Environment-specific file (e.g., config.production.json)
        4. Environment variables (GAIA_LLM_PROVIDER, etc.)

        Args:
            path: Explicit config file path. If None, searches default locations.

        Returns:
            Merged GaiaConfig instance.
        """
        config = cls()

        # Step 1: Load base config file
        config_data = _load_config_file(path)
        if config_data:
            _apply_dict_to_config(config, config_data)

        # Step 2: Load environment-specific overlay
        env = os.getenv(f"{ENV_PREFIX}ENVIRONMENT", config.environment)
        config.environment = env
        env_data = _load_env_config_file(path, env)
        if env_data:
            _apply_dict_to_config(config, env_data)

        # Step 3: Apply environment variable overrides
        _apply_env_vars(config)

        logger.info(
            "Configuration loaded (env=%s, provider=%s, model=%s)",
            config.environment,
            config.llm.provider,
            config.llm.model,
        )
        return config

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to dictionary (secrets redacted)."""
        d = asdict(self)
        # Redact sensitive fields
        if d.get("llm", {}).get("api_key"):
            d["llm"]["api_key"] = "***REDACTED***"
        if d.get("storage", {}).get("pg_password"):
            d["storage"]["pg_password"] = "***REDACTED***"
        return d

    def save(self, path: str) -> None:
        """Save configuration to a JSON file (secrets redacted)."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def _load_config_file(path: Optional[str]) -> Optional[Dict]:
    """Load a config file from explicit path or search paths."""
    if path:
        p = Path(path)
        if p.exists():
            with open(p) as f:
                return json.load(f)
        logger.warning("Config file not found: %s", path)
        return None

    for search_path in CONFIG_SEARCH_PATHS:
        expanded = Path(str(search_path).replace("~", str(Path.home())))
        if expanded.exists():
            logger.debug("Loading config from: %s", expanded)
            with open(expanded) as f:
                return json.load(f)

    return None


def _load_env_config_file(base_path: Optional[str], env: str) -> Optional[Dict]:
    """Load environment-specific config overlay (e.g., config.production.json)."""
    if not base_path:
        for search_path in CONFIG_SEARCH_PATHS:
            expanded = Path(str(search_path).replace("~", str(Path.home())))
            base_path = str(expanded)
            env_path = Path(base_path.replace(".json", f".{env}.json"))
            if env_path.exists():
                with open(env_path) as f:
                    return json.load(f)
        return None

    env_path = Path(base_path.replace(".json", f".{env}.json"))
    if env_path.exists():
        with open(env_path) as f:
            return json.load(f)
    return None


def _apply_dict_to_config(config: GaiaConfig, data: Dict) -> None:
    """Apply a dictionary of values to a config object, handling nested dataclasses."""
    for key, value in data.items():
        if hasattr(config, key):
            attr = getattr(config, key)
            if isinstance(value, dict) and hasattr(attr, "__dataclass_fields__"):
                # Nested dataclass -- recurse
                for sub_key, sub_value in value.items():
                    if hasattr(attr, sub_key):
                        setattr(attr, sub_key, sub_value)
            else:
                setattr(config, key, value)


def _apply_env_vars(config: GaiaConfig) -> None:
    """
    Apply environment variable overrides.

    Pattern: GAIA_{SECTION}_{FIELD} maps to config.{section}.{field}.
    Example: GAIA_LLM_PROVIDER=claude -> config.llm.provider = "claude"
    """
    # Also support the legacy LEMONADE_BASE_URL env var
    legacy_url = os.getenv("LEMONADE_BASE_URL")
    if legacy_url:
        config.llm.base_url = legacy_url

    for env_key, env_value in os.environ.items():
        if not env_key.startswith(ENV_PREFIX):
            continue

        parts = env_key[len(ENV_PREFIX):].lower().split("_", 1)
        if len(parts) != 2:
            continue

        section, field_name = parts
        section_obj = getattr(config, section, None)
        if section_obj is None or not hasattr(section_obj, field_name):
            continue

        current = getattr(section_obj, field_name)
        converted = _convert_env_value(env_value, type(current))
        setattr(section_obj, field_name, converted)


def _convert_env_value(value: str, target_type: type) -> Any:
    """Convert an environment variable string to the target type."""
    if target_type == bool:
        return value.lower() in ("true", "1", "yes")
    elif target_type == int:
        return int(value)
    elif target_type == float:
        return float(value)
    elif target_type == list:
        return [v.strip() for v in value.split(",")]
    return value
```

### Example Config File

```json
{
  "environment": "development",
  "llm": {
    "provider": "lemonade",
    "model": "Qwen3-Coder-30B-A3B-Instruct-GGUF",
    "base_url": "http://localhost:8000/api/v1",
    "temperature": 0.7,
    "context_size": 32768
  },
  "storage": {
    "backend": "sqlite",
    "path": "~/.gaia/data/gaia.db"
  },
  "agent": {
    "max_steps": 20,
    "streaming": true,
    "show_stats": false
  },
  "logging": {
    "level": "INFO",
    "format": "structured",
    "enable_correlation_ids": true
  },
  "plugins": {
    "enabled": true,
    "search_paths": ["~/.gaia/plugins"]
  }
}
```

### Production Override (config.production.json)

```json
{
  "llm": {
    "model": "Qwen3-Coder-30B-A3B-Instruct-GGUF",
    "context_size": 65536,
    "request_timeout": 1800
  },
  "storage": {
    "backend": "postgres",
    "pg_host": "db.internal.example.com",
    "pg_database": "gaia_prod",
    "pool_size": 20
  },
  "logging": {
    "level": "WARNING",
    "format": "structured"
  },
  "plugins": {
    "sandbox_enabled": true
  }
}
```

---

## 5. Logging and Error Formatting

### Problem Analysis

GAIA's current logging has several issues:

1. **No correlation IDs** -- when a user query triggers ChatAgent -> RAG -> LLM, the log lines from each component have no way to be linked together
2. **Duplicate `basicConfig` calls** -- 51 files call `logging.getLogger()`, and several call `logging.basicConfig()` (which only works for the first call); this causes inconsistent handler configuration
3. **Two separate logging systems** -- `GaiaLogger` in `src/gaia/logger.py` (global singleton with color formatting) and per-module `logging.basicConfig()` calls that conflict with it
4. **No structured output** -- log messages are free-form strings, not parseable JSON for log aggregation tools
5. **Error formatting in one place only** -- `src/gaia/agents/base/errors.py` provides `format_execution_trace()` for agent errors, but other modules format errors ad-hoc

### Solution: Structured Logging with Correlation IDs

```
src/gaia/logging/
    __init__.py
    structured.py   # Structured log formatter
    context.py      # Correlation ID injection (uses async_bridge.context)
    filters.py      # Log filtering (suppress noisy libraries)
    setup.py        # One-time logging setup
```

### Structured Logger

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Structured logging for GAIA."""

import json
import logging
import sys
import time
from typing import Any, Dict, Optional

from gaia.async_bridge.context import get_correlation_id, get_context_dict


class StructuredFormatter(logging.Formatter):
    """
    JSON-structured log formatter with correlation ID injection.

    Each log line is a valid JSON object containing:
    - timestamp (ISO 8601)
    - level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    - logger (module name)
    - message (the log message)
    - correlation_id (from context propagation)
    - agent_id (if set)
    - Extra fields from the LogRecord

    Example output:
        {"ts":"2026-02-07T10:30:45.123","level":"INFO","logger":"gaia.agents.chat",
         "msg":"Processing query","cid":"a1b2c3d4e5f6","agent":"chat-001"}
    """

    def format(self, record: logging.LogRecord) -> str:
        ctx = get_context_dict()

        entry: Dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S")
                  + f".{int(record.msecs):03d}",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }

        # Add correlation context
        if ctx.get("correlation_id"):
            entry["cid"] = ctx["correlation_id"]
        if ctx.get("agent_id"):
            entry["agent"] = ctx["agent_id"]
        if ctx.get("request_id"):
            entry["rid"] = ctx["request_id"]

        # Add source location for DEBUG and ERROR
        if record.levelno >= logging.ERROR or record.levelno <= logging.DEBUG:
            entry["file"] = f"{record.filename}:{record.lineno}"
            entry["func"] = record.funcName

        # Add exception info
        if record.exc_info and record.exc_info[1]:
            entry["error"] = {
                "type": type(record.exc_info[1]).__name__,
                "message": str(record.exc_info[1]),
            }
            entry["traceback"] = self.formatException(record.exc_info)

        # Add extra fields (set via logger.info("msg", extra={...}))
        standard_attrs = {
            "name", "msg", "args", "created", "filename", "funcName",
            "levelname", "levelno", "lineno", "module", "msecs", "pathname",
            "process", "processName", "relativeCreated", "stack_info",
            "thread", "threadName", "exc_info", "exc_text", "message",
        }
        for key, value in record.__dict__.items():
            if key not in standard_attrs and not key.startswith("_"):
                entry[key] = value

        return json.dumps(entry, default=str, ensure_ascii=False)


class ColorFormatter(logging.Formatter):
    """
    Human-readable colored formatter for console output.

    Injects correlation IDs into the standard format for development use.
    Falls back to plain text if the terminal does not support colors.
    """

    COLORS = {
        "DEBUG": "\033[37m",      # White
        "INFO": "\033[37m",       # White
        "WARNING": "\033[33m",    # Yellow
        "ERROR": "\033[31m",      # Red
        "CRITICAL": "\033[41m",   # Red background
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        ctx = get_context_dict()
        color = self.COLORS.get(record.levelname, "")
        reset = self.RESET

        cid = ctx.get("correlation_id", "")
        cid_str = f" [{cid}]" if cid else ""

        timestamp = self.formatTime(record, "%H:%M:%S")

        msg = record.getMessage()
        formatted = (
            f"{timestamp} {color}{record.levelname:8s}{reset}"
            f"{cid_str} {record.name}.{record.funcName} | {msg}"
        )

        if record.exc_info and record.exc_info[1]:
            formatted += "\n" + self.formatException(record.exc_info)

        return formatted


def setup_logging(
    level: str = "INFO",
    format_type: str = "color",
    log_file: Optional[str] = None,
    module_levels: Optional[Dict[str, str]] = None,
) -> None:
    """
    One-time logging setup for GAIA.

    Should be called once at application startup (e.g., in CLI entry point).
    Configures the root logger and applies module-level overrides.

    Args:
        level: Default log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        format_type: "structured" for JSON, "color" for human-readable.
        log_file: Optional file path for log output (always structured JSON).
        module_levels: Dict of module name -> level overrides.

    Example:
        # In CLI entry point:
        setup_logging(
            level="INFO",
            format_type="color",
            log_file="gaia.log",
            module_levels={"gaia.llm": "WARNING", "aiohttp": "ERROR"},
        )
    """
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers to prevent duplicates
    root.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    if format_type == "structured":
        console_handler.setFormatter(StructuredFormatter())
    else:
        console_handler.setFormatter(ColorFormatter())
    root.addHandler(console_handler)

    # File handler (always structured JSON for machine parsing)
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(StructuredFormatter())
        root.addHandler(file_handler)

    # Module-level overrides
    if module_levels:
        for module_name, mod_level in module_levels.items():
            logging.getLogger(module_name).setLevel(
                getattr(logging, mod_level.upper(), logging.INFO)
            )

    # Suppress noisy third-party loggers
    for noisy in ("httpx", "httpcore", "urllib3", "aiohttp.access", "datasets"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
```

### Enhanced Error Formatting

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Enhanced error formatting with context for GAIA."""

import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from gaia.async_bridge.context import get_context_dict


@dataclass
class ErrorContext:
    """Rich error context for debugging."""

    exception: Exception
    correlation_id: str = ""
    agent_id: str = ""
    query: Optional[str] = None
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    plan_step: Optional[int] = None
    total_steps: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        ctx = get_context_dict()
        if not self.correlation_id:
            self.correlation_id = ctx.get("correlation_id", "")
        if not self.agent_id:
            self.agent_id = ctx.get("agent_id", "")


def format_error(
    error: ErrorContext,
    include_traceback: bool = True,
    include_context: bool = True,
    max_tb_frames: int = 10,
) -> str:
    """
    Format an error with full context for display or logging.

    Produces a structured error report that includes:
    - Error type and message
    - Correlation ID for cross-referencing logs
    - Execution context (query, tool, plan step)
    - Filtered traceback (user code highlighted)

    Args:
        error: ErrorContext with all available context.
        include_traceback: Whether to include the traceback.
        include_context: Whether to include execution context.
        max_tb_frames: Maximum number of traceback frames to show.

    Returns:
        Formatted error string.
    """
    sep = "-" * 63
    lines: List[str] = []

    # Header
    lines.append(sep)
    exc = error.exception
    lines.append(f"ERROR: {type(exc).__name__}: {exc}")
    lines.append(sep)

    # Context section
    if include_context:
        lines.append("")
        if error.correlation_id:
            lines.append(f"  Correlation ID: {error.correlation_id}")
        if error.agent_id:
            lines.append(f"  Agent: {error.agent_id}")
        if error.query:
            display = error.query[:80] + "..." if len(error.query) > 80 else error.query
            lines.append(f"  Query: \"{display}\"")
        if error.plan_step is not None and error.total_steps is not None:
            lines.append(f"  Plan Step: {error.plan_step}/{error.total_steps}")
        if error.tool_name:
            lines.append(f"  Tool: {error.tool_name}")
        if error.tool_args:
            import json
            args_str = json.dumps(error.tool_args, default=str)
            if len(args_str) > 100:
                args_str = args_str[:97] + "..."
            lines.append(f"  Args: {args_str}")
        for key, value in error.extra.items():
            lines.append(f"  {key}: {value}")

    # Traceback section
    if include_traceback and exc.__traceback__:
        lines.append("")
        lines.append("Traceback (most recent call last):")
        tb_lines = traceback.format_tb(exc.__traceback__)
        # Show last N frames
        for tb_line in tb_lines[-max_tb_frames:]:
            lines.append(tb_line.rstrip())

    lines.append(sep)
    return "\n".join(lines)


def format_error_json(error: ErrorContext) -> Dict[str, Any]:
    """
    Format an error as a JSON-serializable dictionary.

    Useful for structured logging and API error responses.

    Args:
        error: ErrorContext with all available context.

    Returns:
        Dictionary suitable for JSON serialization.
    """
    result: Dict[str, Any] = {
        "error_type": type(error.exception).__name__,
        "error_message": str(error.exception),
    }

    if error.correlation_id:
        result["correlation_id"] = error.correlation_id
    if error.agent_id:
        result["agent_id"] = error.agent_id
    if error.query:
        result["query"] = error.query
    if error.tool_name:
        result["tool_name"] = error.tool_name
    if error.tool_args:
        result["tool_args"] = error.tool_args
    if error.plan_step is not None:
        result["plan_step"] = error.plan_step
        result["total_steps"] = error.total_steps

    if error.exception.__traceback__:
        result["traceback"] = traceback.format_tb(error.exception.__traceback__)

    if error.extra:
        result["extra"] = error.extra

    return result
```

### Backward Compatibility with GaiaLogger

The existing `GaiaLogger` in `src/gaia/logger.py` continues to work. The new system integrates with it:

```python
# In src/gaia/logger.py -- add at the bottom:

def setup_structured_logging(config: Optional["LoggingConfig"] = None) -> None:
    """
    Upgrade to structured logging if configured.

    Called during application startup. If config specifies structured format,
    replaces the existing handlers with the new StructuredFormatter.
    Otherwise, keeps the existing GaiaLogger behavior.
    """
    if config is None:
        return  # Keep existing behavior

    from gaia.logging.structured import setup_logging
    setup_logging(
        level=config.level,
        format_type=config.format,
        log_file=config.log_file,
        module_levels=config.module_levels,
    )
```

### Integration with Observability Architecture

The structured logging system feeds directly into the Observability Architecture's tracing:

```python
# Example: Agent logs with correlation ID automatically included
import logging
from gaia.async_bridge.context import new_correlation_id

logger = logging.getLogger(__name__)

class ChatAgent(Agent):
    def process_query(self, query: str) -> str:
        # Generate correlation ID for this request
        cid = new_correlation_id()

        # All subsequent log calls include the correlation ID
        logger.info("Processing query", extra={"query_length": len(query)})

        # Structured JSON output:
        # {"ts":"2026-02-07T10:30:45.123","level":"INFO",
        #  "logger":"gaia.agents.chat","msg":"Processing query",
        #  "cid":"a1b2c3d4e5f6","query_length":42}

        try:
            result = self._execute(query)
            logger.info("Query completed", extra={"result_length": len(result)})
            return result
        except Exception as exc:
            from gaia.logging.errors import ErrorContext, format_error
            ctx = ErrorContext(exception=exc, query=query)
            logger.error(format_error(ctx))
            raise
```

---

## 6. Resource Management

### Problem Analysis

GAIA has no resource pooling or lifecycle management:

1. **SQLite connections** -- each `DatabaseMixin` instance opens its own `sqlite3.connect()` with no pooling and no maximum connection limits
2. **HTTP clients** -- `LemonadeClient` creates a new `requests.Session` per instance; agents each create their own client
3. **Thread pools** -- audio recording, MCP bridge, and progress indicators each spawn raw `threading.Thread` instances with no shared pool or limits
4. **File handles** -- RAG document processing opens files directly with no handle limits
5. **No cleanup guarantees** -- if an agent crashes mid-operation, database connections and file handles may leak

### Solution: Resource Pool Manager

```
src/gaia/resources/
    __init__.py
    pool.py         # Generic resource pool
    http_pool.py    # HTTP session pool
    db_pool.py      # Database connection pool
    executor.py     # Thread pool executor management
    manager.py      # Unified resource lifecycle manager
```

### Generic Resource Pool

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Generic resource pool for GAIA."""

import logging
import queue
import threading
import time
from contextlib import contextmanager
from typing import Callable, Generic, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ResourcePool(Generic[T]):
    """
    Thread-safe generic resource pool.

    Manages a pool of reusable resources (connections, sessions, etc.)
    with configurable limits, idle timeout, and health checks.

    Args:
        factory: Callable that creates a new resource.
        dispose: Callable that cleans up a resource.
        validate: Optional callable to check if a resource is healthy.
        min_size: Minimum pool size (pre-created on init).
        max_size: Maximum pool size (hard limit).
        idle_timeout: Seconds before idle resources are disposed.
        acquire_timeout: Seconds to wait when pool is exhausted.

    Example:
        import sqlite3

        pool = ResourcePool(
            factory=lambda: sqlite3.connect("app.db"),
            dispose=lambda conn: conn.close(),
            validate=lambda conn: conn.execute("SELECT 1"),
            min_size=2,
            max_size=10,
        )

        with pool.acquire() as conn:
            conn.execute("SELECT * FROM users")
    """

    def __init__(
        self,
        factory: Callable[[], T],
        dispose: Callable[[T], None],
        validate: Optional[Callable[[T], bool]] = None,
        min_size: int = 1,
        max_size: int = 10,
        idle_timeout: float = 300.0,
        acquire_timeout: float = 30.0,
    ):
        self._factory = factory
        self._dispose = dispose
        self._validate = validate
        self._min_size = min_size
        self._max_size = max_size
        self._idle_timeout = idle_timeout
        self._acquire_timeout = acquire_timeout

        self._pool: queue.Queue = queue.Queue(maxsize=max_size)
        self._size = 0
        self._lock = threading.Lock()
        self._closed = False

        # Pre-create minimum resources
        for _ in range(min_size):
            self._add_resource()

        # Start idle reaper thread
        self._reaper_thread = threading.Thread(
            target=self._reap_idle,
            name="resource-pool-reaper",
            daemon=True,
        )
        self._reaper_thread.start()

    def _add_resource(self) -> None:
        """Create and add a new resource to the pool."""
        with self._lock:
            if self._size >= self._max_size:
                return
            self._size += 1

        try:
            resource = self._factory()
            self._pool.put(_PoolEntry(resource, time.time()))
        except Exception:
            with self._lock:
                self._size -= 1
            raise

    @contextmanager
    def acquire(self):
        """
        Acquire a resource from the pool.

        Returns the resource to the pool on context exit.
        If the resource fails validation, it is disposed and a new one is created.

        Yields:
            A pooled resource.

        Raises:
            RuntimeError: If pool is closed.
            TimeoutError: If no resource available within acquire_timeout.
        """
        if self._closed:
            raise RuntimeError("Resource pool is closed")

        resource = self._get_resource()
        try:
            yield resource
        except Exception:
            # On error, dispose the resource instead of returning it
            self._dispose_resource(resource)
            raise
        else:
            # Return healthy resource to pool
            self._return_resource(resource)

    def _get_resource(self) -> T:
        """Get a validated resource from the pool or create a new one."""
        deadline = time.time() + self._acquire_timeout

        while True:
            try:
                entry = self._pool.get(timeout=0.1)
                resource = entry.resource

                # Validate the resource
                if self._validate:
                    try:
                        if not self._validate(resource):
                            self._dispose_resource(resource)
                            continue
                    except Exception:
                        self._dispose_resource(resource)
                        continue

                return resource

            except queue.Empty:
                # Pool empty -- try to create a new resource
                with self._lock:
                    if self._size < self._max_size:
                        self._size += 1
                        try:
                            return self._factory()
                        except Exception:
                            self._size -= 1
                            raise

                # At max capacity -- wait for a resource to be returned
                if time.time() > deadline:
                    raise TimeoutError(
                        f"Could not acquire resource within {self._acquire_timeout}s. "
                        f"Pool size: {self._size}/{self._max_size}"
                    )

    def _return_resource(self, resource: T) -> None:
        """Return a resource to the pool."""
        if self._closed:
            self._dispose_resource(resource)
            return
        try:
            self._pool.put_nowait(_PoolEntry(resource, time.time()))
        except queue.Full:
            self._dispose_resource(resource)

    def _dispose_resource(self, resource: T) -> None:
        """Dispose a resource and decrement the pool size."""
        try:
            self._dispose(resource)
        except Exception as exc:
            logger.warning("Error disposing resource: %s", exc)
        finally:
            with self._lock:
                self._size -= 1

    def _reap_idle(self) -> None:
        """Background thread that disposes idle resources."""
        while not self._closed:
            time.sleep(min(self._idle_timeout / 2, 60.0))

            if self._closed:
                break

            now = time.time()
            reaped = 0

            # Check for idle resources beyond the minimum pool size
            while self._pool.qsize() > self._min_size:
                try:
                    entry = self._pool.get_nowait()
                    if now - entry.last_used > self._idle_timeout:
                        self._dispose_resource(entry.resource)
                        reaped += 1
                    else:
                        # Not idle -- put it back
                        self._pool.put_nowait(entry)
                        break
                except queue.Empty:
                    break

            if reaped > 0:
                logger.debug("Reaped %d idle resources", reaped)

    @property
    def size(self) -> int:
        """Current number of resources (in-use + available)."""
        return self._size

    @property
    def available(self) -> int:
        """Number of resources currently available in the pool."""
        return self._pool.qsize()

    def close(self) -> None:
        """Close the pool and dispose all resources."""
        self._closed = True

        while True:
            try:
                entry = self._pool.get_nowait()
                self._dispose_resource(entry.resource)
            except queue.Empty:
                break

        logger.info("Resource pool closed")


class _PoolEntry:
    """Internal wrapper to track resource idle time."""

    __slots__ = ("resource", "last_used")

    def __init__(self, resource, last_used: float):
        self.resource = resource
        self.last_used = last_used
```

### Database Connection Pool

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Database connection pool for GAIA."""

import sqlite3
from typing import Optional

from gaia.resources.pool import ResourcePool


class SQLitePool:
    """
    Connection pool for SQLite databases.

    Wraps ResourcePool with SQLite-specific factory, validation,
    and disposal logic.

    Args:
        path: Database file path.
        min_size: Minimum connections to maintain.
        max_size: Maximum connections allowed.
        wal_mode: Enable WAL mode for concurrent readers.

    Example:
        pool = SQLitePool("data/app.db", min_size=2, max_size=10)

        with pool.connection() as conn:
            cursor = conn.execute("SELECT * FROM users")
            rows = cursor.fetchall()

        pool.close()
    """

    def __init__(
        self,
        path: str,
        min_size: int = 2,
        max_size: int = 10,
        wal_mode: bool = True,
    ):
        self._path = path
        self._wal_mode = wal_mode

        self._pool = ResourcePool(
            factory=self._create_connection,
            dispose=self._close_connection,
            validate=self._validate_connection,
            min_size=min_size,
            max_size=max_size,
            idle_timeout=300.0,
        )

    def _create_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 5000")
        if self._wal_mode:
            conn.execute("PRAGMA journal_mode = WAL")
        return conn

    def _close_connection(self, conn: sqlite3.Connection) -> None:
        conn.close()

    def _validate_connection(self, conn: sqlite3.Connection) -> bool:
        try:
            conn.execute("SELECT 1")
            return True
        except Exception:
            return False

    def connection(self):
        """Acquire a connection from the pool (context manager)."""
        return self._pool.acquire()

    @property
    def size(self) -> int:
        return self._pool.size

    @property
    def available(self) -> int:
        return self._pool.available

    def close(self) -> None:
        self._pool.close()
```

### Thread Pool Executor Manager

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Shared thread pool executor for GAIA."""

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

logger = logging.getLogger(__name__)

# Module-level shared executor
_executor: Optional[ThreadPoolExecutor] = None
_max_workers: int = min(32, (os.cpu_count() or 4) + 4)


def get_executor(max_workers: Optional[int] = None) -> ThreadPoolExecutor:
    """
    Get the shared thread pool executor.

    GAIA uses a single shared executor to prevent thread explosion.
    Individual components should use this instead of creating their
    own ThreadPoolExecutor or raw Thread instances.

    Args:
        max_workers: Override the default max workers (only applies on first call).

    Returns:
        Shared ThreadPoolExecutor instance.
    """
    global _executor, _max_workers

    if _executor is None or _executor._shutdown:
        if max_workers:
            _max_workers = max_workers
        _executor = ThreadPoolExecutor(
            max_workers=_max_workers,
            thread_name_prefix="gaia-worker",
        )
        logger.debug("Created thread pool executor (max_workers=%d)", _max_workers)

    return _executor


def shutdown_executor(wait: bool = True) -> None:
    """
    Shutdown the shared executor.

    Called during application shutdown to ensure clean cleanup.

    Args:
        wait: If True, wait for pending tasks to complete.
    """
    global _executor
    if _executor is not None:
        _executor.shutdown(wait=wait)
        _executor = None
        logger.debug("Thread pool executor shut down")
```

### Unified Resource Manager

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Unified resource lifecycle manager for GAIA."""

import atexit
import logging
from typing import List

logger = logging.getLogger(__name__)


class ResourceManager:
    """
    Manages the lifecycle of all GAIA resources.

    Provides centralized startup and shutdown for connection pools,
    thread executors, and other long-lived resources.

    Example:
        manager = ResourceManager()
        manager.register_shutdown(db_pool.close)
        manager.register_shutdown(http_pool.close)

        # At application exit, all resources are cleaned up:
        manager.shutdown()
    """

    def __init__(self):
        self._shutdown_hooks: List = []
        self._started = False

        # Register atexit handler for safety
        atexit.register(self.shutdown)

    def register_shutdown(self, hook) -> None:
        """
        Register a cleanup function to be called on shutdown.

        Functions are called in reverse registration order (LIFO).

        Args:
            hook: Callable with no arguments.
        """
        self._shutdown_hooks.append(hook)

    def shutdown(self) -> None:
        """Execute all registered shutdown hooks."""
        if not self._shutdown_hooks:
            return

        logger.info("Shutting down resources (%d hooks)", len(self._shutdown_hooks))

        # Execute in reverse order (LIFO)
        for hook in reversed(self._shutdown_hooks):
            try:
                hook()
            except Exception as exc:
                logger.warning("Error during shutdown: %s", exc)

        self._shutdown_hooks.clear()


# Global resource manager instance
_manager = ResourceManager()


def get_resource_manager() -> ResourceManager:
    """Get the global resource manager."""
    return _manager
```

---

## 7. Plugin Architecture

### Problem Analysis

GAIA currently has no way for third parties to extend its capabilities without modifying the core codebase:

- **Tools** must be registered inside agent classes using the `@tool` decorator in `src/gaia/agents/base/tools.py` -- there is no external registration mechanism
- **MCP servers** can be added externally, but there is no discovery or loading mechanism beyond the MCP bridge's hard-coded configuration
- **Agents** cannot be loaded dynamically -- the routing agent has a hard-coded list of available agents
- **Storage backends, LLM providers, and formatters** are all registered statically in their respective factory modules

### Solution: Plugin Interface with Discovery, Loading, and Sandboxing

```
src/gaia/plugins/
    __init__.py
    interface.py    # Plugin protocol definition
    discovery.py    # Plugin discovery from paths and entry points
    loader.py       # Plugin loading with sandboxing
    registry.py     # Plugin registry (manages loaded plugins)
    hooks.py        # Lifecycle hook definitions
```

### Plugin Interface

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Plugin interface for GAIA extensions."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Type


class PluginType(Enum):
    """Types of plugins that GAIA supports."""

    TOOL = auto()        # Adds new @tool functions to agents
    AGENT = auto()       # Adds new agent implementations
    STORAGE = auto()     # Adds storage backends (Repository implementations)
    LLM_PROVIDER = auto()  # Adds LLM providers
    FORMATTER = auto()   # Adds output formatters
    HOOK = auto()        # Adds lifecycle hooks (on_query, on_tool_call, etc.)


@dataclass
class PluginMetadata:
    """
    Metadata describing a plugin.

    Loaded from the plugin's manifest file or class attributes.
    """

    name: str
    version: str
    description: str
    author: str = ""
    plugin_type: PluginType = PluginType.TOOL
    requires_gaia: str = ">=0.15.0"  # Minimum GAIA version
    dependencies: List[str] = field(default_factory=list)
    permissions: List[str] = field(default_factory=list)  # e.g., ["filesystem", "network"]


class GaiaPlugin(ABC):
    """
    Base class for all GAIA plugins.

    Plugins must implement activate() and deactivate(). They receive
    a PluginContext that provides safe access to GAIA internals.

    Example:
        class WeatherPlugin(GaiaPlugin):
            metadata = PluginMetadata(
                name="weather",
                version="1.0.0",
                description="Adds weather lookup tools",
                plugin_type=PluginType.TOOL,
                permissions=["network"],
            )

            def activate(self, context: "PluginContext") -> None:
                @context.register_tool
                def get_weather(city: str) -> dict:
                    '''Get current weather for a city.'''
                    import requests
                    resp = requests.get(f"https://wttr.in/{city}?format=j1")
                    return resp.json()

            def deactivate(self, context: "PluginContext") -> None:
                context.unregister_tool("get_weather")
    """

    metadata: PluginMetadata

    @abstractmethod
    def activate(self, context: "PluginContext") -> None:
        """
        Activate the plugin.

        Called when the plugin is loaded. Use the context to register
        tools, hooks, or other extensions.

        Args:
            context: Safe interface to GAIA internals.
        """
        ...

    @abstractmethod
    def deactivate(self, context: "PluginContext") -> None:
        """
        Deactivate the plugin.

        Called when the plugin is unloaded. Clean up any resources
        and unregister any extensions.

        Args:
            context: Same context provided during activation.
        """
        ...

    def on_error(self, error: Exception) -> None:
        """
        Called when an error occurs during plugin execution.

        Override to add custom error handling (e.g., logging, retry).

        Args:
            error: The exception that occurred.
        """
        pass


class PluginContext:
    """
    Safe interface provided to plugins for interacting with GAIA.

    Limits what plugins can access to prevent security issues and
    maintain API stability.
    """

    def __init__(self, plugin_name: str, tool_registry: Dict, config: Any):
        self._plugin_name = plugin_name
        self._tool_registry = tool_registry
        self._config = config
        self._registered_tools: List[str] = []
        self._registered_hooks: List[str] = []

    def register_tool(self, func: Callable) -> Callable:
        """
        Register a function as a GAIA tool.

        Works like the @tool decorator but with plugin namespacing.
        Tools are prefixed with the plugin name to avoid conflicts.

        Args:
            func: The tool function to register.

        Returns:
            The original function, unchanged.
        """
        from gaia.agents.base.tools import tool as tool_decorator

        # Namespace the tool to prevent conflicts
        qualified_name = f"{self._plugin_name}__{func.__name__}"
        func.__name__ = qualified_name
        tool_decorator(func)
        self._registered_tools.append(qualified_name)
        return func

    def unregister_tool(self, name: str) -> None:
        """Remove a previously registered tool."""
        qualified = f"{self._plugin_name}__{name}"
        self._tool_registry.pop(qualified, None)
        if qualified in self._registered_tools:
            self._registered_tools.remove(qualified)

    def get_config(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.

        Plugins can only read configuration, not modify it.

        Args:
            key: Dotted config key (e.g., "llm.model").
            default: Default value if key not found.

        Returns:
            Configuration value.
        """
        parts = key.split(".")
        obj = self._config
        for part in parts:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                return default
        return obj

    @property
    def plugin_name(self) -> str:
        """The name of the plugin using this context."""
        return self._plugin_name
```

### Plugin Discovery and Loading

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Plugin discovery and loading for GAIA."""

import importlib
import importlib.metadata
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Type

from gaia.plugins.interface import GaiaPlugin, PluginMetadata, PluginContext

logger = logging.getLogger(__name__)

# Entry point group name for pip-installed plugins
ENTRY_POINT_GROUP = "gaia.plugins"


class PluginRegistry:
    """
    Discovers, loads, and manages GAIA plugins.

    Plugins can be discovered from:
    1. File system paths (~/.gaia/plugins/)
    2. Python entry points (pip-installed packages)
    3. Explicit registration

    Example:
        registry = PluginRegistry(
            search_paths=["~/.gaia/plugins"],
            allowed=["weather", "slack"],
        )
        registry.discover()
        registry.activate_all(tool_registry=_TOOL_REGISTRY, config=gaia_config)

        # Later:
        registry.deactivate_all()
    """

    def __init__(
        self,
        search_paths: Optional[List[str]] = None,
        allowed: Optional[List[str]] = None,
        sandbox_enabled: bool = True,
    ):
        self._search_paths = [
            Path(p.replace("~", str(Path.home())))
            for p in (search_paths or [])
        ]
        self._allowed = set(allowed) if allowed else None  # None = all allowed
        self._sandbox_enabled = sandbox_enabled
        self._plugins: Dict[str, GaiaPlugin] = {}
        self._contexts: Dict[str, PluginContext] = {}

    def discover(self) -> List[PluginMetadata]:
        """
        Discover available plugins from all sources.

        Returns:
            List of PluginMetadata for discovered plugins.
        """
        discovered = []

        # 1. Search file system paths
        for search_path in self._search_paths:
            if not search_path.exists():
                continue
            for plugin_dir in search_path.iterdir():
                if not plugin_dir.is_dir():
                    continue
                manifest_path = plugin_dir / "plugin.json"
                if manifest_path.exists():
                    meta = self._load_manifest(manifest_path)
                    if meta and self._is_allowed(meta.name):
                        discovered.append(meta)
                        logger.info("Discovered plugin: %s v%s", meta.name, meta.version)

        # 2. Search Python entry points
        try:
            eps = importlib.metadata.entry_points()
            # Python 3.12+ returns a SelectableGroups
            if hasattr(eps, "select"):
                plugin_eps = eps.select(group=ENTRY_POINT_GROUP)
            else:
                plugin_eps = eps.get(ENTRY_POINT_GROUP, [])

            for ep in plugin_eps:
                if self._is_allowed(ep.name):
                    discovered.append(PluginMetadata(
                        name=ep.name,
                        version="0.0.0",  # Will be updated on load
                        description=f"Entry point plugin: {ep.value}",
                    ))
                    logger.info("Discovered entry point plugin: %s", ep.name)
        except Exception as exc:
            logger.debug("Error scanning entry points: %s", exc)

        return discovered

    def load_plugin(
        self,
        name: str,
        tool_registry: Dict,
        config: object,
    ) -> Optional[GaiaPlugin]:
        """
        Load and activate a single plugin by name.

        Args:
            name: Plugin name.
            tool_registry: The GAIA tool registry dict.
            config: The GaiaConfig instance.

        Returns:
            The loaded GaiaPlugin instance, or None if loading failed.
        """
        if name in self._plugins:
            logger.warning("Plugin already loaded: %s", name)
            return self._plugins[name]

        plugin_class = self._find_plugin_class(name)
        if plugin_class is None:
            logger.error("Plugin not found: %s", name)
            return None

        try:
            plugin = plugin_class()
            context = PluginContext(
                plugin_name=name,
                tool_registry=tool_registry,
                config=config,
            )

            plugin.activate(context)

            self._plugins[name] = plugin
            self._contexts[name] = context
            logger.info("Loaded plugin: %s", name)
            return plugin

        except Exception as exc:
            logger.error("Failed to load plugin %s: %s", name, exc)
            return None

    def unload_plugin(self, name: str) -> None:
        """Deactivate and remove a plugin."""
        plugin = self._plugins.get(name)
        context = self._contexts.get(name)

        if plugin and context:
            try:
                plugin.deactivate(context)
            except Exception as exc:
                logger.warning("Error deactivating plugin %s: %s", name, exc)

        self._plugins.pop(name, None)
        self._contexts.pop(name, None)

    def activate_all(self, tool_registry: Dict, config: object) -> int:
        """
        Discover and load all available plugins.

        Args:
            tool_registry: The GAIA tool registry dict.
            config: The GaiaConfig instance.

        Returns:
            Number of plugins successfully loaded.
        """
        discovered = self.discover()
        loaded = 0
        for meta in discovered:
            if self.load_plugin(meta.name, tool_registry, config):
                loaded += 1
        return loaded

    def deactivate_all(self) -> None:
        """Deactivate all loaded plugins."""
        for name in list(self._plugins.keys()):
            self.unload_plugin(name)

    @property
    def loaded_plugins(self) -> List[str]:
        """Names of currently loaded plugins."""
        return list(self._plugins.keys())

    def _is_allowed(self, name: str) -> bool:
        """Check if a plugin is in the allowed list."""
        if self._allowed is None:
            return True  # All allowed
        return name in self._allowed

    def _load_manifest(self, path: Path) -> Optional[PluginMetadata]:
        """Load plugin metadata from a plugin.json manifest."""
        try:
            with open(path) as f:
                data = json.load(f)
            return PluginMetadata(**data)
        except Exception as exc:
            logger.warning("Invalid plugin manifest %s: %s", path, exc)
            return None

    def _find_plugin_class(self, name: str) -> Optional[Type[GaiaPlugin]]:
        """Find a plugin class by name from all sources."""
        # Check file system plugins
        for search_path in self._search_paths:
            plugin_dir = search_path / name
            init_file = plugin_dir / "__init__.py"
            if init_file.exists():
                try:
                    # Add plugin directory to sys.path temporarily
                    parent = str(search_path)
                    if parent not in sys.path:
                        sys.path.insert(0, parent)
                    module = importlib.import_module(name)
                    # Look for a class that inherits from GaiaPlugin
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if (
                            isinstance(attr, type)
                            and issubclass(attr, GaiaPlugin)
                            and attr is not GaiaPlugin
                        ):
                            return attr
                except Exception as exc:
                    logger.warning("Error loading plugin module %s: %s", name, exc)

        # Check entry points
        try:
            eps = importlib.metadata.entry_points()
            if hasattr(eps, "select"):
                plugin_eps = eps.select(group=ENTRY_POINT_GROUP, name=name)
            else:
                plugin_eps = [
                    ep for ep in eps.get(ENTRY_POINT_GROUP, [])
                    if ep.name == name
                ]
            for ep in plugin_eps:
                return ep.load()
        except Exception:
            pass

        return None
```

### Creating a Plugin (Example)

```
~/.gaia/plugins/weather/
    __init__.py
    plugin.json
```

**plugin.json:**

```json
{
    "name": "weather",
    "version": "1.0.0",
    "description": "Adds weather lookup tools to GAIA agents",
    "author": "GAIA Community",
    "plugin_type": "TOOL",
    "requires_gaia": ">=0.15.0",
    "permissions": ["network"]
}
```

**__init__.py:**

```python
# Third-party GAIA plugin example

from gaia.plugins.interface import GaiaPlugin, PluginMetadata, PluginType, PluginContext


class WeatherPlugin(GaiaPlugin):
    """Adds weather lookup capability to GAIA agents."""

    metadata = PluginMetadata(
        name="weather",
        version="1.0.0",
        description="Weather lookup tools",
        author="GAIA Community",
        plugin_type=PluginType.TOOL,
        permissions=["network"],
    )

    def activate(self, context: PluginContext) -> None:
        @context.register_tool
        def get_weather(city: str) -> dict:
            """Get current weather for a city. Returns temperature and conditions."""
            import requests
            try:
                resp = requests.get(
                    f"https://wttr.in/{city}?format=j1",
                    timeout=10,
                )
                resp.raise_for_status()
                data = resp.json()
                current = data["current_condition"][0]
                return {
                    "city": city,
                    "temperature_c": current["temp_C"],
                    "temperature_f": current["temp_F"],
                    "condition": current["weatherDesc"][0]["value"],
                    "humidity": current["humidity"],
                }
            except Exception as exc:
                return {"error": str(exc)}

    def deactivate(self, context: PluginContext) -> None:
        context.unregister_tool("get_weather")
```

### Installing a Plugin via pip

Third-party plugins can also be distributed as Python packages:

```toml
# In the plugin's pyproject.toml:
[project.entry-points."gaia.plugins"]
weather = "gaia_weather_plugin:WeatherPlugin"
```

After `pip install gaia-weather-plugin`, the plugin is automatically discovered.

---

## 8. Database Migration Framework

### Problem Analysis

GAIA has 10+ SQLite schemas scattered across architecture documents and agent implementations:

| Component | Schema Location | Tables |
|-----------|----------------|--------|
| DatabaseMixin | Agent `__init__` methods | Varies per agent |
| Persistent Memory | Architecture doc | sessions, memories, knowledge, facts |
| Observability | Architecture doc | traces, spans, metrics, events |
| Security | Architecture doc | secrets, permissions, audit_log |
| Workflow Orchestration | Architecture doc | workflows, steps, executions |
| EMR Agent | `agents/emr/agent.py` | patients, encounters, forms |
| Chat Session | `agents/chat/session.py` | sessions, messages |
| Eval Framework | `eval/eval.py` | experiments, results |
| Code Orchestration | `orchestration/` | templates, checklists |
| RAG | `rag/sdk.py` | documents, chunks, embeddings |

None of these have migration support. Schema changes require:

1. Manually checking if a table exists with `table_exists()`
2. Adding `ALTER TABLE` statements that run on every startup
3. No rollback capability -- failed migrations leave the database in an inconsistent state
4. No version tracking -- no way to know which migrations have been applied

### Solution: Migration Framework

```
src/gaia/migrations/
    __init__.py
    engine.py       # Migration execution engine
    migration.py    # Migration base class
    registry.py     # Migration registry and ordering
    cli.py          # CLI commands (gaia migrate)
    versions/       # Migration files
        __init__.py
        v001_initial_schema.py
        v002_add_sessions.py
        ...
```

### Migration Base Class

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Database migration framework for GAIA."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from gaia.storage.repository import Repository

logger = logging.getLogger(__name__)


@dataclass
class MigrationInfo:
    """Metadata about a migration."""

    version: int
    name: str
    description: str
    reversible: bool = True


class Migration(ABC):
    """
    Base class for database migrations.

    Each migration has an up() method (apply) and an optional down()
    method (rollback). Migrations are versioned and executed in order.

    Example:
        class V001InitialSchema(Migration):
            info = MigrationInfo(
                version=1,
                name="initial_schema",
                description="Create core tables",
            )

            def up(self, repo: Repository) -> None:
                repo.execute_ddl('''
                    CREATE TABLE users (
                        id INTEGER PRIMARY KEY,
                        name TEXT NOT NULL,
                        email TEXT UNIQUE,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    );
                    CREATE TABLE sessions (
                        id INTEGER PRIMARY KEY,
                        user_id INTEGER REFERENCES users(id),
                        started_at TEXT DEFAULT CURRENT_TIMESTAMP
                    );
                ''')

            def down(self, repo: Repository) -> None:
                repo.execute_ddl('''
                    DROP TABLE IF EXISTS sessions;
                    DROP TABLE IF EXISTS users;
                ''')
    """

    info: MigrationInfo

    @abstractmethod
    def up(self, repo: Repository) -> None:
        """Apply the migration."""
        ...

    def down(self, repo: Repository) -> None:
        """
        Rollback the migration.

        Optional -- raise NotImplementedError if migration is not reversible.
        """
        raise NotImplementedError(
            f"Migration {self.info.name} (v{self.info.version}) "
            f"does not support rollback."
        )
```

### Migration Engine

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Migration execution engine for GAIA."""

import importlib
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Type

from gaia.migrations.migration import Migration, MigrationInfo
from gaia.storage.repository import Repository

logger = logging.getLogger(__name__)

# Schema version tracking table
MIGRATION_TABLE = "_gaia_migrations"
MIGRATION_TABLE_DDL = f"""
    CREATE TABLE IF NOT EXISTS {MIGRATION_TABLE} (
        version INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        description TEXT,
        applied_at TEXT DEFAULT CURRENT_TIMESTAMP,
        duration_ms INTEGER,
        checksum TEXT
    );
"""


class MigrationEngine:
    """
    Executes database migrations in order.

    Tracks applied migrations in a _gaia_migrations table.
    Supports forward migration (up) and rollback (down).

    Args:
        repo: Repository instance to migrate.
        migrations_path: Path to the migrations/ directory.

    Example:
        repo = SQLiteRepository("data/app.db")
        repo.connect()

        engine = MigrationEngine(repo)
        engine.register(V001InitialSchema)
        engine.register(V002AddSessions)

        # Apply all pending migrations
        applied = engine.migrate()
        print(f"Applied {applied} migrations")

        # Check current version
        print(f"Database at version {engine.current_version()}")

        # Rollback last migration
        engine.rollback()
    """

    def __init__(
        self,
        repo: Repository,
        migrations_path: Optional[str] = None,
    ):
        self._repo = repo
        self._migrations: Dict[int, Type[Migration]] = {}

        # Ensure migration tracking table exists
        self._ensure_migration_table()

        # Auto-discover migrations from path
        if migrations_path:
            self._discover_migrations(Path(migrations_path))

    def register(self, migration_class: Type[Migration]) -> None:
        """
        Register a migration class.

        Args:
            migration_class: A Migration subclass with info attribute.

        Raises:
            ValueError: If version is already registered.
        """
        info = migration_class.info
        if info.version in self._migrations:
            existing = self._migrations[info.version]
            raise ValueError(
                f"Migration version {info.version} already registered "
                f"({existing.info.name}). Cannot register {info.name}."
            )
        self._migrations[info.version] = migration_class
        logger.debug("Registered migration: v%03d_%s", info.version, info.name)

    def current_version(self) -> int:
        """Get the current database schema version."""
        result = self._repo.query(
            f"SELECT MAX(version) as v FROM {MIGRATION_TABLE}"
        )
        if result.first and result.first.get("v") is not None:
            return result.first["v"]
        return 0

    def pending_migrations(self) -> List[MigrationInfo]:
        """Get list of migrations that have not been applied yet."""
        current = self.current_version()
        pending = []
        for version in sorted(self._migrations.keys()):
            if version > current:
                pending.append(self._migrations[version].info)
        return pending

    def applied_migrations(self) -> List[Dict]:
        """Get list of migrations that have been applied."""
        result = self._repo.query(
            f"SELECT * FROM {MIGRATION_TABLE} ORDER BY version"
        )
        return result.rows

    def migrate(self, target_version: Optional[int] = None) -> int:
        """
        Apply pending migrations up to target_version.

        Args:
            target_version: Version to migrate to. None = latest.

        Returns:
            Number of migrations applied.

        Raises:
            RuntimeError: If a migration fails (database is rolled back
                          to the state before the failed migration).
        """
        current = self.current_version()

        if target_version is None:
            target_version = max(self._migrations.keys()) if self._migrations else 0

        if target_version <= current:
            logger.info("Database already at version %d", current)
            return 0

        applied_count = 0
        for version in sorted(self._migrations.keys()):
            if version <= current or version > target_version:
                continue

            migration_class = self._migrations[version]
            migration = migration_class()
            info = migration.info

            logger.info(
                "Applying migration v%03d: %s - %s",
                info.version,
                info.name,
                info.description,
            )

            start_time = time.time()
            try:
                with self._repo.transaction():
                    migration.up(self._repo)
                    duration_ms = int((time.time() - start_time) * 1000)
                    self._repo.insert(MIGRATION_TABLE, {
                        "version": info.version,
                        "name": info.name,
                        "description": info.description,
                        "duration_ms": duration_ms,
                    })
                applied_count += 1
                logger.info(
                    "Applied migration v%03d in %dms",
                    info.version,
                    duration_ms,
                )
            except Exception as exc:
                logger.error(
                    "Migration v%03d failed: %s. Database rolled back.",
                    info.version,
                    exc,
                )
                raise RuntimeError(
                    f"Migration v{info.version:03d}_{info.name} failed: {exc}"
                ) from exc

        logger.info(
            "Migration complete: %d applied, now at version %d",
            applied_count,
            self.current_version(),
        )
        return applied_count

    def rollback(self, steps: int = 1) -> int:
        """
        Rollback the last N migrations.

        Args:
            steps: Number of migrations to rollback.

        Returns:
            Number of migrations rolled back.

        Raises:
            RuntimeError: If a rollback fails or migration is not reversible.
        """
        current = self.current_version()
        rolled_back = 0

        for _ in range(steps):
            if current <= 0:
                break

            migration_class = self._migrations.get(current)
            if migration_class is None:
                raise RuntimeError(
                    f"Migration v{current:03d} not found in registry. "
                    f"Cannot rollback."
                )

            migration = migration_class()
            info = migration.info

            if not info.reversible:
                raise RuntimeError(
                    f"Migration v{info.version:03d}_{info.name} is not reversible."
                )

            logger.info("Rolling back migration v%03d: %s", info.version, info.name)

            try:
                with self._repo.transaction():
                    migration.down(self._repo)
                    self._repo.delete(
                        MIGRATION_TABLE,
                        "version = :version",
                        {"version": info.version},
                    )
                rolled_back += 1
                current -= 1
                logger.info("Rolled back migration v%03d", info.version)
            except NotImplementedError:
                raise RuntimeError(
                    f"Migration v{info.version:03d}_{info.name} "
                    f"does not support rollback."
                )
            except Exception as exc:
                logger.error("Rollback of v%03d failed: %s", info.version, exc)
                raise

        return rolled_back

    def _ensure_migration_table(self) -> None:
        """Create the migration tracking table if it does not exist."""
        if not self._repo.table_exists(MIGRATION_TABLE):
            self._repo.execute_ddl(MIGRATION_TABLE_DDL)

    def _discover_migrations(self, path: Path) -> None:
        """Auto-discover migration classes from a directory."""
        if not path.exists():
            return

        for file in sorted(path.glob("v*.py")):
            if file.name.startswith("__"):
                continue
            module_name = file.stem
            try:
                # Import the module
                import sys
                parent = str(path.parent)
                if parent not in sys.path:
                    sys.path.insert(0, parent)
                module = importlib.import_module(
                    f"{path.name}.{module_name}"
                )
                # Find Migration subclasses
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (
                        isinstance(attr, type)
                        and issubclass(attr, Migration)
                        and attr is not Migration
                        and hasattr(attr, "info")
                    ):
                        self.register(attr)
            except Exception as exc:
                logger.warning("Error loading migration %s: %s", file.name, exc)
```

### Example Migration Files

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""v001: Initial schema for GAIA core tables."""

from gaia.migrations.migration import Migration, MigrationInfo
from gaia.storage.repository import Repository


class V001InitialSchema(Migration):
    info = MigrationInfo(
        version=1,
        name="initial_schema",
        description="Create core agent and session tables",
        reversible=True,
    )

    def up(self, repo: Repository) -> None:
        repo.execute_ddl("""
            CREATE TABLE agent_sessions (
                id TEXT PRIMARY KEY,
                agent_type TEXT NOT NULL,
                started_at TEXT DEFAULT CURRENT_TIMESTAMP,
                ended_at TEXT,
                query_count INTEGER DEFAULT 0,
                status TEXT DEFAULT 'active'
            );

            CREATE TABLE conversation_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES agent_sessions(id),
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system', 'tool')),
                content TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                token_count INTEGER
            );

            CREATE INDEX idx_messages_session
                ON conversation_messages(session_id, created_at);

            CREATE TABLE tool_executions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES agent_sessions(id),
                tool_name TEXT NOT NULL,
                arguments TEXT,
                result TEXT,
                duration_ms INTEGER,
                success INTEGER DEFAULT 1,
                executed_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX idx_tool_exec_session
                ON tool_executions(session_id, executed_at);
        """)

    def down(self, repo: Repository) -> None:
        repo.execute_ddl("""
            DROP TABLE IF EXISTS tool_executions;
            DROP TABLE IF EXISTS conversation_messages;
            DROP TABLE IF EXISTS agent_sessions;
        """)
```

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""v002: Add persistent memory tables."""

from gaia.migrations.migration import Migration, MigrationInfo
from gaia.storage.repository import Repository


class V002PersistentMemory(Migration):
    info = MigrationInfo(
        version=2,
        name="persistent_memory",
        description="Add episodic and semantic memory tables",
        reversible=True,
    )

    def up(self, repo: Repository) -> None:
        repo.execute_ddl("""
            CREATE TABLE episodic_memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT REFERENCES agent_sessions(id),
                summary TEXT NOT NULL,
                importance REAL DEFAULT 0.5,
                embedding BLOB,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                expires_at TEXT
            );

            CREATE TABLE semantic_knowledge (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT NOT NULL,
                fact TEXT NOT NULL,
                confidence REAL DEFAULT 1.0,
                source_session TEXT,
                learned_at TEXT DEFAULT CURRENT_TIMESTAMP,
                last_accessed TEXT,
                access_count INTEGER DEFAULT 0
            );

            CREATE INDEX idx_knowledge_category
                ON semantic_knowledge(category, confidence DESC);
        """)

    def down(self, repo: Repository) -> None:
        repo.execute_ddl("""
            DROP TABLE IF EXISTS semantic_knowledge;
            DROP TABLE IF EXISTS episodic_memories;
        """)
```

### Auto-Migration on Startup

```python
# In application startup (e.g., cli.py or agent __init__):

def ensure_database_migrated(repo: Repository) -> None:
    """
    Run pending migrations on application startup.

    Called once during GAIA initialization to ensure the database
    schema is up to date.
    """
    from gaia.migrations.engine import MigrationEngine

    engine = MigrationEngine(
        repo=repo,
        migrations_path="gaia/migrations/versions",
    )

    pending = engine.pending_migrations()
    if pending:
        logger.info(
            "%d pending migrations: %s",
            len(pending),
            ", ".join(f"v{m.version:03d}" for m in pending),
        )
        engine.migrate()
    else:
        logger.debug("Database schema up to date (v%d)", engine.current_version())
```

---

## Integration Map

This diagram shows how the eight cross-cutting systems integrate with each other and with existing GAIA components.

```
                    +-------------------+
                    | GaiaConfig (4)    |
                    | (loads from file  |
                    |  + env vars)      |
                    +--------+----------+
                             |
              +--------------+------------------+
              |              |                  |
     +--------v------+ +----v------+  +--------v---------+
     | Container (2)  | | Logging(5)| | PluginRegistry (7)|
     | (DI services)  | | (setup)   | | (discovery/load)  |
     +---+----+-------+ +-----+----+ +-------+----------+
         |    |                |              |
    +----v-+  +---v----+  +---v------+  +----v---------+
    |Repos.| |LLMClient| |Structured|  |PluginContext  |
    |(1)   | |(factory)| |Formatter | |(registers tools|
    +--+---+ +---+-----+ +---+------+  | into registry)|
       |         |            |         +----+----------+
       |    +----v---------+  |              |
       |    |AsyncBridge(3)| correlation_id  |
       |    |(run_sync,    +--+              |
       |    | run_async)   |                 |
       |    +----+---------+                 |
       |         |                           |
  +----v---------v---------+-----------------v---+
  |           Agent (base/agent.py)               |
  |  - uses Repository for storage                |
  |  - uses LLMClient for inference               |
  |  - uses AsyncBridge for ChatSDK calls         |
  |  - uses structured logging with correlation   |
  |  - tools from core + plugins                  |
  +---+-------------------------------------------+
      |
  +---v------------------+
  | MigrationEngine (8)  |
  | (runs on startup,    |
  |  uses Repository)    |
  +---+------------------+
      |
  +---v------------------+
  | ResourceManager (6)  |
  | (pools, executors,   |
  |  cleanup on exit)    |
  +----------------------+
```

### Cross-Reference: How Each System Uses Others

| System | Uses | Used By |
|--------|------|---------|
| 1. Storage Abstraction | - | 2 (DI registers repos), 6 (pool wraps repos), 8 (migrations target repos) |
| 2. Dependency Injection | 1 (registers repos), 4 (reads config) | All agents, tests |
| 3. Async/Sync Bridge | 5 (propagates correlation IDs) | Agents, ChatSDK, MCP bridge |
| 4. Configuration | - | 2 (configures DI), 5 (logging config), 6 (pool sizes), 7 (plugin paths) |
| 5. Logging | 3 (reads correlation context) | All modules |
| 6. Resource Management | 1 (pools repos), 4 (reads pool sizes) | Agents, API server |
| 7. Plugin Architecture | 2 (uses DI for context), 4 (reads config) | Third-party extensions |
| 8. Migration Framework | 1 (executes against repos) | Startup, CLI |

---

## Implementation Plan

### Phase 1: Foundations (Weeks 1-4)

| Week | System | Deliverable | Risk |
|------|--------|-------------|------|
| 1 | Storage Abstraction (1) | Repository interface, SQLite backend, MemoryRepository | Low |
| 2 | Storage Abstraction (1) | Updated DatabaseMixin wrapper, backward compatibility tests | Low |
| 3 | Configuration (4) | GaiaConfig schema, loader, env var integration | Low |
| 4 | Logging (5) | StructuredFormatter, ColorFormatter, setup_logging(), ErrorContext | Low |

**Exit criteria**: All existing tests pass. No behavioral changes. New systems usable but not mandatory.

### Phase 2: Infrastructure (Weeks 5-8)

| Week | System | Deliverable | Risk |
|------|--------|-------------|------|
| 5 | Dependency Injection (2) | Container, Scope, register/resolve | Medium |
| 6 | Dependency Injection (2) | Agent factory, create_test_container | Medium |
| 7 | Async/Sync Bridge (3) | run_sync, run_async, sync_compatible, context propagation | Medium |
| 8 | Resource Management (6) | ResourcePool, SQLitePool, shared ThreadPoolExecutor | Medium |

**Exit criteria**: DI container optional in Agent constructors. Bridge utilities replace ad-hoc asyncio patterns.

### Phase 3: Extensions (Weeks 9-12)

| Week | System | Deliverable | Risk |
|------|--------|-------------|------|
| 9 | Plugin Architecture (7) | GaiaPlugin interface, PluginContext, PluginRegistry | Medium |
| 10 | Plugin Architecture (7) | File-system discovery, entry point discovery, example plugin | Medium |
| 11 | Migration Framework (8) | Migration base class, MigrationEngine, version tracking | Low |
| 12 | Migration Framework (8) | Initial migration files for core schemas, auto-migration on startup | Low |

**Exit criteria**: Weather plugin example works end-to-end. Migrations track schema version.

### Phase 4: Integration and Hardening (Weeks 13-16)

| Week | System | Deliverable | Risk |
|------|--------|-------------|------|
| 13 | Integration | Wire all systems through GaiaConfig and Container | High |
| 14 | Integration | Update existing agents to use new systems (opt-in) | High |
| 15 | Testing | Integration tests, performance benchmarks, edge cases | Medium |
| 16 | Documentation | Update docs/sdk/, migration guides, plugin authoring guide | Low |

**Exit criteria**: All 8 systems work together. Existing agents unchanged. New agents can use new systems.

---

## Testing Strategy

### Unit Tests

Each system has its own test file:

```
tests/unit/
    test_storage_repository.py
    test_storage_sqlite.py
    test_storage_memory.py
    test_di_container.py
    test_async_bridge.py
    test_config_schema.py
    test_config_loader.py
    test_logging_structured.py
    test_logging_errors.py
    test_resource_pool.py
    test_resource_db_pool.py
    test_plugin_interface.py
    test_plugin_registry.py
    test_migration_engine.py
```

### Example Tests

```python
# test_storage_repository.py
import pytest
from gaia.storage.backends.memory import MemoryRepository


class TestMemoryRepository:
    def test_insert_and_query(self):
        repo = MemoryRepository()
        repo.execute_ddl("CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT)")
        row_id = repo.insert("t", {"name": "Alice"})
        result = repo.query("SELECT * FROM t WHERE id = :id", {"id": row_id})
        assert result.count == 1
        assert result.first["name"] == "Alice"

    def test_transaction_rollback(self):
        repo = MemoryRepository()
        repo.execute_ddl("CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT)")
        repo.insert("t", {"val": "original"})

        with pytest.raises(ValueError):
            with repo.transaction():
                repo.insert("t", {"val": "should_rollback"})
                raise ValueError("rollback trigger")

        result = repo.query("SELECT * FROM t")
        assert result.count == 1
        assert result.first["val"] == "original"

    def test_table_exists(self):
        repo = MemoryRepository()
        assert not repo.table_exists("missing")
        repo.execute_ddl("CREATE TABLE present (id INTEGER PRIMARY KEY)")
        assert repo.table_exists("present")


# test_di_container.py
import pytest
from gaia.di.container import Container, Scope
from gaia.storage.repository import Repository
from gaia.storage.backends.memory import MemoryRepository


class TestContainer:
    def test_singleton_scope(self):
        container = Container()
        container.register(Repository, MemoryRepository, Scope.SINGLETON)
        a = container.resolve(Repository)
        b = container.resolve(Repository)
        assert a is b

    def test_transient_scope(self):
        container = Container()
        container.register(Repository, MemoryRepository, Scope.TRANSIENT)
        a = container.resolve(Repository)
        b = container.resolve(Repository)
        assert a is not b

    def test_child_container_override(self):
        parent = Container()
        parent.register_instance(str, "parent_value")
        child = parent.create_child()
        child.register_instance(str, "child_value")
        assert parent.resolve(str) == "parent_value"
        assert child.resolve(str) == "child_value"

    def test_missing_service_raises(self):
        container = Container()
        with pytest.raises(KeyError, match="Service not registered"):
            container.resolve(Repository)


# test_async_bridge.py
import asyncio
import pytest
from gaia.async_bridge.bridge import run_sync, run_async, sync_compatible


class TestAsyncBridge:
    def test_run_sync_from_sync(self):
        async def coro():
            return 42
        assert run_sync(coro()) == 42

    def test_run_sync_timeout(self):
        async def slow():
            await asyncio.sleep(10)
        with pytest.raises(TimeoutError):
            run_sync(slow(), timeout=0.1)

    @pytest.mark.asyncio
    async def test_run_async_from_async(self):
        def sync_fn(x):
            return x * 2
        result = await run_async(sync_fn, 21)
        assert result == 42

    def test_sync_compatible_from_sync(self):
        @sync_compatible
        async def fetch():
            return "data"
        assert fetch() == "data"


# test_migration_engine.py
import pytest
from gaia.storage.backends.memory import MemoryRepository
from gaia.migrations.engine import MigrationEngine
from gaia.migrations.migration import Migration, MigrationInfo


class V001Test(Migration):
    info = MigrationInfo(version=1, name="test_v1", description="Test migration 1")

    def up(self, repo):
        repo.execute_ddl("CREATE TABLE test_table (id INTEGER PRIMARY KEY, val TEXT)")

    def down(self, repo):
        repo.execute_ddl("DROP TABLE IF EXISTS test_table")


class V002Test(Migration):
    info = MigrationInfo(version=2, name="test_v2", description="Test migration 2")

    def up(self, repo):
        repo.execute_ddl("CREATE TABLE test_table_2 (id INTEGER PRIMARY KEY)")

    def down(self, repo):
        repo.execute_ddl("DROP TABLE IF EXISTS test_table_2")


class TestMigrationEngine:
    def test_migrate_applies_pending(self):
        repo = MemoryRepository()
        engine = MigrationEngine(repo)
        engine.register(V001Test)
        engine.register(V002Test)

        assert engine.current_version() == 0
        applied = engine.migrate()
        assert applied == 2
        assert engine.current_version() == 2
        assert repo.table_exists("test_table")
        assert repo.table_exists("test_table_2")

    def test_migrate_is_idempotent(self):
        repo = MemoryRepository()
        engine = MigrationEngine(repo)
        engine.register(V001Test)

        engine.migrate()
        applied = engine.migrate()
        assert applied == 0

    def test_rollback(self):
        repo = MemoryRepository()
        engine = MigrationEngine(repo)
        engine.register(V001Test)
        engine.register(V002Test)

        engine.migrate()
        assert engine.current_version() == 2

        engine.rollback(steps=1)
        assert engine.current_version() == 1
        assert repo.table_exists("test_table")
        assert not repo.table_exists("test_table_2")

    def test_pending_migrations(self):
        repo = MemoryRepository()
        engine = MigrationEngine(repo)
        engine.register(V001Test)
        engine.register(V002Test)

        pending = engine.pending_migrations()
        assert len(pending) == 2
        assert pending[0].version == 1
        assert pending[1].version == 2

        engine.migrate(target_version=1)
        pending = engine.pending_migrations()
        assert len(pending) == 1
        assert pending[0].version == 2
```

### Integration Tests

```python
# tests/integration/test_cross_cutting_integration.py

import pytest
from gaia.config.schema import GaiaConfig
from gaia.di.container import Container, Scope
from gaia.storage.repository import Repository
from gaia.storage.backends.memory import MemoryRepository
from gaia.migrations.engine import MigrationEngine


class TestCrossCuttingIntegration:
    """Tests that verify the 8 cross-cutting systems work together."""

    def test_config_drives_container_registration(self):
        """Config -> DI -> Repository chain."""
        config = GaiaConfig()
        config.storage.backend = "sqlite"
        config.storage.path = ":memory:"

        container = Container()
        container.register_instance(GaiaConfig, config)
        container.register(Repository, MemoryRepository, Scope.SINGLETON)

        repo = container.resolve(Repository)
        assert repo.is_connected

    def test_migration_through_di_container(self):
        """DI -> Repository -> Migration chain."""
        container = Container()
        container.register(Repository, MemoryRepository, Scope.SINGLETON)

        repo = container.resolve(Repository)
        engine = MigrationEngine(repo)
        engine.register(V001Test)
        engine.migrate()

        assert repo.table_exists("test_table")
        assert engine.current_version() == 1
```

---

## Appendix A: File Listing

Complete set of new files introduced by this architecture:

```
src/gaia/
    storage/
        __init__.py
        repository.py           # Repository ABC, QueryResult, TableSchema
        pool.py                 # Connection pool (re-exported from resources)
        backends/
            __init__.py
            sqlite.py           # SQLiteRepository
            postgres.py         # PostgresRepository (optional)
            memory.py           # MemoryRepository (testing)

    di/
        __init__.py
        container.py            # Container, Scope, ServiceDescriptor
        providers.py            # Default provider registrations

    async_bridge/
        __init__.py
        bridge.py               # run_sync, run_async, sync_compatible
        context.py              # correlation_id, agent_id, request_id

    config/
        __init__.py
        schema.py               # GaiaConfig, LLMConfig, StorageConfig, etc.
        loader.py               # File loading and merging
        env.py                  # Environment variable integration

    logging/
        __init__.py
        structured.py           # StructuredFormatter, ColorFormatter, setup_logging
        errors.py               # ErrorContext, format_error, format_error_json

    resources/
        __init__.py
        pool.py                 # ResourcePool generic pool
        db_pool.py              # SQLitePool
        executor.py             # Shared ThreadPoolExecutor
        manager.py              # ResourceManager lifecycle

    plugins/
        __init__.py
        interface.py            # GaiaPlugin, PluginMetadata, PluginContext
        discovery.py            # File system and entry point discovery
        loader.py               # Plugin loading
        registry.py             # PluginRegistry

    migrations/
        __init__.py
        migration.py            # Migration ABC, MigrationInfo
        engine.py               # MigrationEngine
        versions/
            __init__.py
            v001_initial_schema.py
            v002_persistent_memory.py
```

**Total new files**: 31
**Estimated total lines of code**: ~2,800
**External dependencies**: Zero (all stdlib)
**Optional dependencies**: psycopg2 (PostgreSQL backend), pydantic (config validation)

---

## Appendix B: Glossary

| Term | Definition |
|------|-----------|
| **Repository** | Abstract interface for data storage operations (CRUD + DDL) |
| **Backend** | Concrete implementation of Repository (SQLite, PostgreSQL, etc.) |
| **Container** | Dependency injection service registry with lifecycle management |
| **Scope** | Service lifetime: SINGLETON (one per container), TRANSIENT (new each time), REQUEST (one per request) |
| **Bridge** | Utility for crossing async/sync boundaries without losing context |
| **Correlation ID** | Unique identifier linking all log entries and traces for a single user request |
| **Migration** | Versioned database schema change with up (apply) and down (rollback) methods |
| **Plugin** | Third-party extension that adds tools, agents, or other capabilities to GAIA |
| **Resource Pool** | Thread-safe collection of reusable resources with idle timeout and validation |
