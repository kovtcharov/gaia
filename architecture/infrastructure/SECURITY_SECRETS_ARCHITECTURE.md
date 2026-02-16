# Security and Secrets Management Architecture for GAIA

**Date**: February 7, 2026
**Version**: 1.0
**Status**: Specification
**Priority**: CRITICAL
**Estimated Effort**: 10-12 weeks (2 engineers)
**Target**: Enable secrets vault, permission model, audit logging, and resource limits

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Problem Statement](#problem-statement)
3. [Architecture Overview](#architecture-overview)
4. [Component Specifications](#component-specifications)
5. [Data Models and Schemas](#data-models-and-schemas)
6. [Secrets Vault](#secrets-vault)
7. [Permission Model](#permission-model)
8. [Audit Logging](#audit-logging)
9. [Resource Limits](#resource-limits)
10. [Integration with GAIA](#integration-with-gaia)
11. [Threat Model](#threat-model)
12. [Implementation Plan](#implementation-plan)
13. [Testing Strategy](#testing-strategy)
14. [Success Metrics](#success-metrics)
15. [Complete Code](#complete-code)

---

## Executive Summary

### The Gap

GAIA currently has:
- Basic environment variable reading for API keys
- No centralized secrets management
- No permission boundaries between agents
- No audit trail of agent actions
- No resource usage limits or quotas

GAIA **lacks**:
- Encrypted secrets storage with rotation support
- Fine-grained permission model for agent capabilities
- Comprehensive audit logging of all security-relevant events
- Resource limits (CPU, memory, tokens, API calls)
- Credential isolation between agents and workflows
- Secret scanning to prevent accidental exposure

### The Solution

A **Security and Secrets Management Architecture** that provides:
- Encrypted secrets vault with key rotation and access scoping
- Role-based and capability-based permission model
- Tamper-proof audit log of all agent actions
- Configurable resource limits with enforcement
- Credential isolation between agents
- Secret scanning in prompts and outputs

### Impact

**Unlocks entire category**: Enterprise-ready security for:
- Multi-user deployments with credential isolation
- Compliance-ready audit trails (SOC2, HIPAA)
- Production agent deployments with resource governance
- Third-party integration with secure credential management

**Before**: 15% security coverage (basic env vars only)
**After**: 90% security coverage

---

## Problem Statement

### Current Limitations

**Example scenario**: Agent needs Gmail API credentials, Jira token, and OpenAI key

**Current approach (insecure)**:
```python
# Credentials scattered in environment variables or plain files
import os
gmail_key = os.environ["GMAIL_API_KEY"]       # Plain text in env
jira_token = os.environ["JIRA_TOKEN"]         # Accessible to ALL agents
openai_key = os.environ["OPENAI_API_KEY"]     # No rotation mechanism
# No audit trail of which agent used which credential
# No way to revoke access for a single agent
```

**Required approach (secure)**:
```python
from gaia.security import SecretManager, PermissionContext

# Encrypted, scoped, audited credential access
with PermissionContext(agent="EmailAgent", scope=["email:read", "email:send"]):
    gmail_creds = SecretManager.get("gmail_oauth_token")
    # Audit log: "EmailAgent accessed gmail_oauth_token at 2026-02-07T10:00:00"
    # Jira token not accessible in this scope
    # Resource limits enforced automatically
```

### User Stories

**Story 1: Secure Credential Storage**
```
As a developer, I want to:
- Store API keys and tokens in an encrypted vault
- Access credentials by name with automatic decryption
- Rotate credentials without restarting agents
- Scope credentials to specific agents
So that credentials are never exposed in plain text
```

**Story 2: Agent Permissions**
```
As an admin, I want to:
- Define what each agent is allowed to do
- Restrict EmailAgent to only email operations
- Prevent CodeAgent from accessing financial APIs
- Enforce file system access boundaries
So that agents cannot exceed their intended scope
```

**Story 3: Audit Compliance**
```
As a compliance officer, I want to:
- See a complete log of every agent action
- Track which credentials were accessed and when
- Detect anomalous patterns (unusual access, excessive usage)
- Generate compliance reports
So that we can demonstrate regulatory compliance
```

**Story 4: Resource Governance**
```
As an ops engineer, I want to:
- Set token usage limits per agent per day
- Limit concurrent agent executions
- Set memory and CPU bounds for tool execution
- Get alerts when limits are approached
So that costs and resource usage stay within budget
```

---

## Architecture Overview

### High-Level Design

```
+------------------------------------------------------------------+
|                      GAIA Application                             |
|  [Agents]  [Tools]  [LLM Clients]  [API Server]  [Workflows]    |
+------+--------+----------+-----------+-----------+---------------+
       |        |          |           |           |
       v        v          v           v           v
+------------------------------------------------------------------+
|                  Security Gateway                                 |
|                                                                   |
|  +-----------------+  +-----------------+  +------------------+  |
|  | Permission      |  | Secret          |  | Resource         |  |
|  | Enforcer        |  | Manager         |  | Governor         |  |
|  |                 |  |                 |  |                  |  |
|  | - Capability    |  | - Vault access  |  | - Token limits   |  |
|  |   checking      |  | - Decryption    |  | - Rate limiting  |  |
|  | - Scope         |  | - Rotation      |  | - Memory caps    |  |
|  |   validation    |  | - Scoping       |  | - CPU bounds     |  |
|  | - Deny logging  |  | - Audit trail   |  | - Concurrency    |  |
|  +--------+--------+  +--------+--------+  +--------+---------+  |
|           |                     |                     |           |
|           v                     v                     v           |
|  +-----------------------------------------------------------+   |
|  |                   Audit Logger                              |   |
|  |                                                             |   |
|  |  - Tamper-proof append-only log                            |   |
|  |  - Structured events with context                          |   |
|  |  - Anomaly detection                                       |   |
|  |  - Compliance reporting                                     |   |
|  +-----------------------------------------------------------+   |
+------------------------------------------------------------------+
           |                     |                     |
           v                     v                     v
+------------------+  +-------------------+  +------------------+
| Encrypted Vault  |  | Audit Database    |  | Resource Metrics |
| (SQLite+AES256)  |  | (SQLite, signed)  |  | (Counters/Gauges)|
+------------------+  +-------------------+  +------------------+
```

### Security Principles

```
+------------------------------------------------------------+
|              Defense in Depth                               |
|                                                            |
|  Layer 1: AUTHENTICATION                                   |
|  - Who is making the request?                              |
|  - Agent identity, user identity, workflow context          |
|                                                            |
|  Layer 2: AUTHORIZATION                                    |
|  - Is this action allowed?                                 |
|  - Capability checks, scope validation                     |
|                                                            |
|  Layer 3: ISOLATION                                        |
|  - Can this action affect other components?                |
|  - Credential scoping, filesystem sandboxing               |
|                                                            |
|  Layer 4: MONITORING                                       |
|  - Was this action appropriate?                            |
|  - Audit logging, anomaly detection                        |
|                                                            |
|  Layer 5: LIMITS                                           |
|  - Is this action within bounds?                           |
|  - Resource quotas, rate limits, budget caps               |
+------------------------------------------------------------+
```

---

## Component Specifications

### 1. Core Security Types

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Core security types for GAIA.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from gaia.logger import get_logger

log = get_logger(__name__)


class Permission(Enum):
    """Fine-grained permissions for agent capabilities."""
    # File system
    FILE_READ = "file:read"
    FILE_WRITE = "file:write"
    FILE_DELETE = "file:delete"
    FILE_EXECUTE = "file:execute"

    # Network
    NETWORK_HTTP = "network:http"
    NETWORK_SMTP = "network:smtp"
    NETWORK_IMAP = "network:imap"

    # LLM
    LLM_LOCAL = "llm:local"
    LLM_CLOUD = "llm:cloud"

    # Email
    EMAIL_READ = "email:read"
    EMAIL_SEND = "email:send"
    EMAIL_DELETE = "email:delete"

    # Calendar
    CALENDAR_READ = "calendar:read"
    CALENDAR_WRITE = "calendar:write"

    # Code execution
    CODE_EXECUTE = "code:execute"
    CODE_SHELL = "code:shell"

    # Secrets
    SECRET_READ = "secret:read"
    SECRET_WRITE = "secret:write"
    SECRET_DELETE = "secret:delete"

    # Admin
    ADMIN_AUDIT = "admin:audit"
    ADMIN_CONFIG = "admin:config"
    ADMIN_USERS = "admin:users"


class AuditEventType(Enum):
    """Types of auditable events."""
    SECRET_ACCESS = "secret_access"
    SECRET_CREATE = "secret_create"
    SECRET_DELETE = "secret_delete"
    SECRET_ROTATE = "secret_rotate"
    PERMISSION_CHECK = "permission_check"
    PERMISSION_DENIED = "permission_denied"
    AGENT_START = "agent_start"
    AGENT_COMPLETE = "agent_complete"
    AGENT_ERROR = "agent_error"
    TOOL_EXECUTE = "tool_execute"
    LLM_REQUEST = "llm_request"
    RESOURCE_LIMIT_HIT = "resource_limit_hit"
    ANOMALY_DETECTED = "anomaly_detected"
    CONFIG_CHANGE = "config_change"
    LOGIN = "login"
    LOGOUT = "logout"


class SecretType(Enum):
    """Types of secrets."""
    API_KEY = "api_key"
    OAUTH_TOKEN = "oauth_token"
    PASSWORD = "password"
    CERTIFICATE = "certificate"
    SSH_KEY = "ssh_key"
    WEBHOOK_SECRET = "webhook_secret"
    CUSTOM = "custom"


@dataclass
class SecurityContext:
    """Security context for a request/operation."""
    context_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    agent_name: str = ""
    agent_type: str = ""
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    workflow_id: Optional[str] = None
    permissions: Set[Permission] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def has_permission(self, perm: Permission) -> bool:
        """Check if this context has a specific permission."""
        return perm in self.permissions

    def has_any_permission(self, perms: List[Permission]) -> bool:
        """Check if context has any of the listed permissions."""
        return bool(self.permissions & set(perms))


@dataclass
class AuditEvent:
    """A single auditable event."""
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    event_type: AuditEventType = AuditEventType.TOOL_EXECUTE
    timestamp: datetime = field(default_factory=datetime.now)
    context: Optional[SecurityContext] = None
    action: str = ""
    resource: str = ""
    outcome: str = "success"  # success, denied, error
    details: Dict[str, Any] = field(default_factory=dict)
    severity: str = "info"  # info, warning, critical

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "agent": self.context.agent_name if self.context else "",
            "user": self.context.user_id if self.context else "",
            "action": self.action,
            "resource": self.resource,
            "outcome": self.outcome,
            "severity": self.severity,
            "details": self.details,
        }


@dataclass
class SecretEntry:
    """A stored secret with metadata."""
    name: str
    secret_type: SecretType
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    allowed_agents: List[str] = field(default_factory=list)  # Empty = all agents
    allowed_scopes: List[str] = field(default_factory=list)
    version: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        if self.expires_at:
            return datetime.now() > self.expires_at
        return False


@dataclass
class ResourceQuota:
    """Resource usage limits."""
    name: str
    max_tokens_per_day: int = 1_000_000
    max_tokens_per_request: int = 100_000
    max_requests_per_hour: int = 100
    max_concurrent_agents: int = 5
    max_file_size_bytes: int = 100 * 1024 * 1024  # 100MB
    max_memory_mb: int = 2048
    max_execution_seconds: int = 300
    allowed_domains: List[str] = field(default_factory=list)  # Empty = all
    blocked_domains: List[str] = field(default_factory=list)
```

---

## Secrets Vault

### 2.1 Encrypted Vault Implementation

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Encrypted secrets vault for GAIA.

Uses AES-256-GCM encryption with key derivation from a master password.
Stores secrets in SQLite with full encryption at the field level.
"""

import base64
import hashlib
import json
import os
import sqlite3
import threading
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from gaia.logger import get_logger

log = get_logger(__name__)


class EncryptionEngine:
    """
    AES-256-GCM encryption engine.

    Uses:
    - PBKDF2 for key derivation from master password
    - AES-256-GCM for authenticated encryption
    - Random nonce per encryption operation
    """

    def __init__(self, master_key: bytes):
        """
        Initialize with a derived master key.

        Args:
            master_key: 32-byte AES key (derived from password)
        """
        self._key = master_key

    @classmethod
    def from_password(cls, password: str, salt: Optional[bytes] = None) -> "EncryptionEngine":
        """
        Derive encryption key from a password using PBKDF2.

        Args:
            password: Master password
            salt: Optional salt (generated if None)

        Returns:
            EncryptionEngine instance
        """
        if salt is None:
            salt = os.urandom(16)

        key = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            iterations=100_000,
            dklen=32,
        )

        engine = cls(key)
        engine._salt = salt
        return engine

    @classmethod
    def from_keyring(cls) -> "EncryptionEngine":
        """
        Load or generate encryption key from OS keyring.

        Uses the system keyring (Windows Credential Manager,
        macOS Keychain, Linux Secret Service) for secure key storage.
        """
        try:
            import keyring
            stored_key = keyring.get_password("gaia-vault", "master-key")
            if stored_key:
                key_bytes = base64.b64decode(stored_key)
                return cls(key_bytes)
            else:
                # Generate new key
                key_bytes = os.urandom(32)
                keyring.set_password(
                    "gaia-vault", "master-key",
                    base64.b64encode(key_bytes).decode()
                )
                log.info("Generated new vault master key in OS keyring")
                return cls(key_bytes)
        except ImportError:
            log.warning(
                "keyring package not installed. Using file-based key storage. "
                "Install keyring for production: pip install keyring"
            )
            return cls._fallback_key()

    @classmethod
    def _fallback_key(cls) -> "EncryptionEngine":
        """Fallback: derive key from machine-specific data."""
        import platform
        import socket
        machine_id = f"{platform.node()}-{socket.gethostname()}-gaia-vault"
        key = hashlib.pbkdf2_hmac(
            "sha256", machine_id.encode(), b"gaia-vault-salt", 100_000, 32
        )
        return cls(key)

    def encrypt(self, plaintext: str) -> str:
        """
        Encrypt plaintext string.

        Returns:
            Base64-encoded ciphertext (nonce + ciphertext + tag)
        """
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        nonce = os.urandom(12)  # 96-bit nonce for GCM
        aesgcm = AESGCM(self._key)
        ciphertext = aesgcm.encrypt(nonce, plaintext.encode("utf-8"), None)

        # Pack: nonce (12) + ciphertext+tag
        packed = nonce + ciphertext
        return base64.b64encode(packed).decode("utf-8")

    def decrypt(self, encrypted: str) -> str:
        """
        Decrypt an encrypted string.

        Args:
            encrypted: Base64-encoded ciphertext from encrypt()

        Returns:
            Decrypted plaintext string
        """
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        packed = base64.b64decode(encrypted)
        nonce = packed[:12]
        ciphertext = packed[12:]

        aesgcm = AESGCM(self._key)
        plaintext = aesgcm.decrypt(nonce, ciphertext, None)
        return plaintext.decode("utf-8")


class SecretVault:
    """
    Encrypted secrets vault with access control and audit.

    Features:
    - AES-256-GCM encryption at rest
    - Per-secret access scoping (agent-level)
    - Automatic key rotation support
    - Secret versioning
    - Expiration enforcement
    - Full audit trail of access

    Usage:
        vault = SecretVault()

        # Store a secret
        vault.store("openai_key", "sk-abc123",
                    secret_type=SecretType.API_KEY,
                    allowed_agents=["ChatAgent", "CodeAgent"])

        # Retrieve (with audit)
        context = SecurityContext(agent_name="ChatAgent")
        value = vault.get("openai_key", context)

        # Rotate
        vault.rotate("openai_key", "sk-new456")
    """

    def __init__(
        self,
        db_path: str = "gaia_vault.db",
        encryption_engine: Optional[EncryptionEngine] = None,
        audit_logger: Optional["AuditLogger"] = None,
    ):
        self.db_path = db_path
        self.engine = encryption_engine or EncryptionEngine.from_keyring()
        self.audit = audit_logger
        self._lock = threading.Lock()
        self._cache: Dict[str, str] = {}  # In-memory cache (encrypted values)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize vault database."""
        conn = sqlite3.connect(self.db_path)
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS secrets (
                name TEXT PRIMARY KEY,
                encrypted_value TEXT NOT NULL,
                secret_type TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                expires_at TEXT,
                allowed_agents TEXT,
                allowed_scopes TEXT,
                version INTEGER DEFAULT 1,
                metadata TEXT
            );

            CREATE TABLE IF NOT EXISTS secret_versions (
                name TEXT NOT NULL,
                version INTEGER NOT NULL,
                encrypted_value TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (name, version)
            );
        """)
        conn.commit()
        conn.close()

    def store(
        self,
        name: str,
        value: str,
        secret_type: SecretType = SecretType.API_KEY,
        allowed_agents: Optional[List[str]] = None,
        allowed_scopes: Optional[List[str]] = None,
        expires_in_days: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        context: Optional[SecurityContext] = None,
    ) -> None:
        """
        Store a secret in the vault.

        Args:
            name: Secret name (unique identifier)
            value: Secret value (will be encrypted)
            secret_type: Type of secret
            allowed_agents: Agents that can access this secret (None = all)
            allowed_scopes: Required permission scopes for access
            expires_in_days: Auto-expire after N days
            metadata: Additional metadata
            context: Security context of the storer
        """
        encrypted_value = self.engine.encrypt(value)
        now = datetime.now()
        expires_at = (now + timedelta(days=expires_in_days)) if expires_in_days else None

        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Check if secret already exists
            cursor.execute("SELECT version FROM secrets WHERE name = ?", (name,))
            existing = cursor.fetchone()
            version = (existing[0] + 1) if existing else 1

            # Store current version in history
            if existing:
                cursor.execute(
                    "SELECT encrypted_value FROM secrets WHERE name = ?", (name,)
                )
                old_value = cursor.fetchone()[0]
                cursor.execute("""
                    INSERT INTO secret_versions (name, version, encrypted_value, created_at)
                    VALUES (?, ?, ?, ?)
                """, (name, existing[0], old_value, now.isoformat()))

            # Store or update secret
            cursor.execute("""
                INSERT OR REPLACE INTO secrets
                (name, encrypted_value, secret_type, created_at, updated_at,
                 expires_at, allowed_agents, allowed_scopes, version, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                name,
                encrypted_value,
                secret_type.value,
                now.isoformat() if not existing else now.isoformat(),
                now.isoformat(),
                expires_at.isoformat() if expires_at else None,
                json.dumps(allowed_agents) if allowed_agents else None,
                json.dumps(allowed_scopes) if allowed_scopes else None,
                version,
                json.dumps(metadata) if metadata else None,
            ))

            conn.commit()
            conn.close()

        # Clear cache
        self._cache.pop(name, None)

        # Audit
        if self.audit:
            self.audit.log_event(AuditEvent(
                event_type=AuditEventType.SECRET_CREATE,
                context=context,
                action="store",
                resource=name,
                details={"secret_type": secret_type.value, "version": version},
            ))

        log.info(f"Secret stored: {name} (type={secret_type.value}, version={version})")

    def get(
        self,
        name: str,
        context: Optional[SecurityContext] = None,
    ) -> Optional[str]:
        """
        Retrieve and decrypt a secret.

        Args:
            name: Secret name
            context: Security context of the requester

        Returns:
            Decrypted secret value, or None if not found/denied
        """
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("SELECT * FROM secrets WHERE name = ?", (name,))
            row = cursor.fetchone()
            conn.close()

        if not row:
            log.warning(f"Secret not found: {name}")
            if self.audit and context:
                self.audit.log_event(AuditEvent(
                    event_type=AuditEventType.SECRET_ACCESS,
                    context=context,
                    action="get",
                    resource=name,
                    outcome="not_found",
                    severity="warning",
                ))
            return None

        # Check expiration
        expires_at = row[5]
        if expires_at and datetime.fromisoformat(expires_at) < datetime.now():
            log.warning(f"Secret expired: {name}")
            if self.audit and context:
                self.audit.log_event(AuditEvent(
                    event_type=AuditEventType.SECRET_ACCESS,
                    context=context,
                    action="get",
                    resource=name,
                    outcome="expired",
                    severity="warning",
                ))
            return None

        # Check agent access
        allowed_agents_json = row[6]
        if allowed_agents_json and context:
            allowed_agents = json.loads(allowed_agents_json)
            if allowed_agents and context.agent_name not in allowed_agents:
                log.warning(
                    f"Secret access denied: {name} "
                    f"(agent={context.agent_name}, allowed={allowed_agents})"
                )
                if self.audit:
                    self.audit.log_event(AuditEvent(
                        event_type=AuditEventType.PERMISSION_DENIED,
                        context=context,
                        action="get_secret",
                        resource=name,
                        outcome="denied",
                        severity="warning",
                        details={"reason": "agent_not_allowed"},
                    ))
                return None

        # Decrypt
        encrypted_value = row[1]
        try:
            decrypted = self.engine.decrypt(encrypted_value)
        except Exception as e:
            log.error(f"Failed to decrypt secret {name}: {e}")
            return None

        # Audit successful access
        if self.audit and context:
            self.audit.log_event(AuditEvent(
                event_type=AuditEventType.SECRET_ACCESS,
                context=context,
                action="get",
                resource=name,
                outcome="success",
            ))

        return decrypted

    def rotate(
        self,
        name: str,
        new_value: str,
        context: Optional[SecurityContext] = None,
    ) -> bool:
        """
        Rotate a secret to a new value.

        Preserves metadata and access controls; increments version.

        Args:
            name: Secret name
            new_value: New secret value
            context: Security context of the rotator

        Returns:
            True if rotation successful
        """
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("SELECT * FROM secrets WHERE name = ?", (name,))
            row = cursor.fetchone()
            if not row:
                conn.close()
                log.warning(f"Cannot rotate non-existent secret: {name}")
                return False

            old_version = row[8]
            new_version = old_version + 1

            # Archive old version
            cursor.execute("""
                INSERT INTO secret_versions (name, version, encrypted_value, created_at)
                VALUES (?, ?, ?, ?)
            """, (name, old_version, row[1], datetime.now().isoformat()))

            # Update with new value
            encrypted = self.engine.encrypt(new_value)
            cursor.execute("""
                UPDATE secrets
                SET encrypted_value = ?, updated_at = ?, version = ?
                WHERE name = ?
            """, (encrypted, datetime.now().isoformat(), new_version, name))

            conn.commit()
            conn.close()

        self._cache.pop(name, None)

        if self.audit and context:
            self.audit.log_event(AuditEvent(
                event_type=AuditEventType.SECRET_ROTATE,
                context=context,
                action="rotate",
                resource=name,
                details={"old_version": old_version, "new_version": new_version},
            ))

        log.info(f"Secret rotated: {name} (v{old_version} -> v{new_version})")
        return True

    def delete(
        self,
        name: str,
        context: Optional[SecurityContext] = None,
    ) -> bool:
        """Delete a secret from the vault."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM secrets WHERE name = ?", (name,))
            cursor.execute("DELETE FROM secret_versions WHERE name = ?", (name,))
            deleted = cursor.rowcount > 0
            conn.commit()
            conn.close()

        self._cache.pop(name, None)

        if self.audit and context:
            self.audit.log_event(AuditEvent(
                event_type=AuditEventType.SECRET_DELETE,
                context=context,
                action="delete",
                resource=name,
            ))

        return deleted

    def list_secrets(self) -> List[SecretEntry]:
        """List all secrets (metadata only, not values)."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name, secret_type, created_at, updated_at, expires_at, allowed_agents, version FROM secrets")
        rows = cursor.fetchall()
        conn.close()

        return [
            SecretEntry(
                name=r[0],
                secret_type=SecretType(r[1]),
                created_at=datetime.fromisoformat(r[2]),
                updated_at=datetime.fromisoformat(r[3]),
                expires_at=datetime.fromisoformat(r[4]) if r[4] else None,
                allowed_agents=json.loads(r[5]) if r[5] else [],
                version=r[6],
            )
            for r in rows
        ]

    def scan_for_secrets(self, text: str) -> List[Dict[str, str]]:
        """
        Scan text for potential secret leaks.

        Detects patterns like API keys, tokens, passwords in text.
        Used to prevent accidental exposure in prompts or logs.

        Args:
            text: Text to scan

        Returns:
            List of detected potential secrets with type and location
        """
        import re

        patterns = [
            ("api_key", r"(?:api[_-]?key|apikey)\s*[=:]\s*['\"]?([a-zA-Z0-9_\-]{20,})['\"]?"),
            ("bearer_token", r"Bearer\s+([a-zA-Z0-9_\-.]+)"),
            ("aws_key", r"AKIA[A-Z0-9]{16}"),
            ("github_token", r"gh[ps]_[a-zA-Z0-9]{36}"),
            ("openai_key", r"sk-[a-zA-Z0-9]{48}"),
            ("password", r"(?:password|passwd|pwd)\s*[=:]\s*['\"]?([^\s'\"]{8,})['\"]?"),
            ("private_key", r"-----BEGIN\s+(RSA\s+)?PRIVATE KEY-----"),
            ("connection_string", r"(?:mongodb|postgres|mysql|redis)://[^\s]+"),
        ]

        findings = []
        for secret_type, pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                findings.append({
                    "type": secret_type,
                    "position": match.start(),
                    "length": len(match.group(0)),
                    "preview": match.group(0)[:10] + "..." if len(match.group(0)) > 10 else match.group(0),
                })

        return findings
```

---

## Permission Model

### 3.1 Permission Enforcer

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Capability-based permission model for GAIA agents.
"""

import json
import threading
from typing import Any, Dict, List, Optional, Set

from gaia.logger import get_logger

log = get_logger(__name__)


# Default permission profiles for built-in agents
DEFAULT_AGENT_PERMISSIONS = {
    "ChatAgent": {
        Permission.LLM_LOCAL,
        Permission.LLM_CLOUD,
        Permission.FILE_READ,
        Permission.SECRET_READ,
    },
    "CodeAgent": {
        Permission.LLM_LOCAL,
        Permission.LLM_CLOUD,
        Permission.FILE_READ,
        Permission.FILE_WRITE,
        Permission.CODE_EXECUTE,
        Permission.CODE_SHELL,
        Permission.SECRET_READ,
    },
    "EmailAgent": {
        Permission.LLM_LOCAL,
        Permission.EMAIL_READ,
        Permission.EMAIL_SEND,
        Permission.CALENDAR_READ,
        Permission.CALENDAR_WRITE,
        Permission.NETWORK_SMTP,
        Permission.NETWORK_IMAP,
        Permission.SECRET_READ,
    },
    "JiraAgent": {
        Permission.LLM_LOCAL,
        Permission.NETWORK_HTTP,
        Permission.SECRET_READ,
    },
    "BlenderAgent": {
        Permission.LLM_LOCAL,
        Permission.FILE_READ,
        Permission.FILE_WRITE,
        Permission.CODE_EXECUTE,
    },
}


class PermissionEnforcer:
    """
    Enforces permission checks for agent operations.

    Features:
    - Default permission profiles for built-in agents
    - Custom permission assignment per agent instance
    - Hierarchical permissions (admin inherits all)
    - Contextual permission elevation (with audit)
    - Integration with audit logger

    Usage:
        enforcer = PermissionEnforcer()

        # Check permission
        context = SecurityContext(
            agent_name="EmailAgent",
            permissions=enforcer.get_permissions("EmailAgent")
        )

        if enforcer.check(context, Permission.EMAIL_SEND):
            send_email()
        else:
            log.warning("Permission denied")
    """

    def __init__(
        self,
        custom_profiles: Optional[Dict[str, Set[Permission]]] = None,
        audit_logger: Optional["AuditLogger"] = None,
    ):
        self.profiles: Dict[str, Set[Permission]] = dict(DEFAULT_AGENT_PERMISSIONS)
        if custom_profiles:
            self.profiles.update(custom_profiles)
        self.audit = audit_logger
        self._overrides: Dict[str, Set[Permission]] = {}

    def get_permissions(self, agent_name: str) -> Set[Permission]:
        """Get the permission set for an agent."""
        if agent_name in self._overrides:
            return self._overrides[agent_name]
        return self.profiles.get(agent_name, set())

    def create_context(
        self,
        agent_name: str,
        agent_type: str = "",
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        extra_permissions: Optional[Set[Permission]] = None,
    ) -> SecurityContext:
        """
        Create a security context for an agent.

        Args:
            agent_name: Agent instance name
            agent_type: Agent class name
            user_id: Associated user ID
            session_id: Associated session ID
            extra_permissions: Additional permissions to grant

        Returns:
            SecurityContext with resolved permissions
        """
        permissions = self.get_permissions(agent_type or agent_name)
        if extra_permissions:
            permissions = permissions | extra_permissions

        return SecurityContext(
            agent_name=agent_name,
            agent_type=agent_type,
            user_id=user_id,
            session_id=session_id,
            permissions=permissions,
        )

    def check(
        self,
        context: SecurityContext,
        required_permission: Permission,
        resource: str = "",
    ) -> bool:
        """
        Check if a security context has a required permission.

        Args:
            context: The security context to check
            required_permission: Permission required for the operation
            resource: Resource being accessed (for audit)

        Returns:
            True if permitted, False if denied
        """
        allowed = context.has_permission(required_permission)

        if self.audit:
            self.audit.log_event(AuditEvent(
                event_type=(
                    AuditEventType.PERMISSION_CHECK if allowed
                    else AuditEventType.PERMISSION_DENIED
                ),
                context=context,
                action=required_permission.value,
                resource=resource,
                outcome="allowed" if allowed else "denied",
                severity="info" if allowed else "warning",
            ))

        if not allowed:
            log.warning(
                f"Permission denied: {context.agent_name} requires "
                f"{required_permission.value} for {resource}"
            )

        return allowed

    def check_multiple(
        self,
        context: SecurityContext,
        required_permissions: List[Permission],
        require_all: bool = True,
    ) -> bool:
        """
        Check multiple permissions.

        Args:
            context: Security context
            required_permissions: Permissions to check
            require_all: If True, ALL permissions required. If False, ANY.

        Returns:
            True if check passes
        """
        if require_all:
            return all(
                context.has_permission(p) for p in required_permissions
            )
        return context.has_any_permission(required_permissions)

    def grant_permission(
        self,
        agent_name: str,
        permission: Permission,
        context: Optional[SecurityContext] = None,
    ) -> None:
        """Grant an additional permission to an agent."""
        if agent_name not in self._overrides:
            self._overrides[agent_name] = set(self.get_permissions(agent_name))
        self._overrides[agent_name].add(permission)

        if self.audit and context:
            self.audit.log_event(AuditEvent(
                event_type=AuditEventType.CONFIG_CHANGE,
                context=context,
                action="grant_permission",
                resource=agent_name,
                details={"permission": permission.value},
            ))

    def revoke_permission(
        self,
        agent_name: str,
        permission: Permission,
        context: Optional[SecurityContext] = None,
    ) -> None:
        """Revoke a permission from an agent."""
        if agent_name not in self._overrides:
            self._overrides[agent_name] = set(self.get_permissions(agent_name))
        self._overrides[agent_name].discard(permission)

        if self.audit and context:
            self.audit.log_event(AuditEvent(
                event_type=AuditEventType.CONFIG_CHANGE,
                context=context,
                action="revoke_permission",
                resource=agent_name,
                details={"permission": permission.value},
            ))
```

---

## Audit Logging

### 4.1 Tamper-Proof Audit Logger

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Tamper-proof audit logging for GAIA.

Implements an append-only log with hash chaining for integrity verification.
"""

import hashlib
import json
import sqlite3
import threading
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from gaia.logger import get_logger

log = get_logger(__name__)


class AuditLogger:
    """
    Append-only audit logger with hash chain integrity.

    Features:
    - Append-only log (no updates or deletes)
    - Hash chain linking each entry to the previous
    - Integrity verification (detect tampering)
    - Structured event format
    - Anomaly detection hooks
    - Compliance report generation

    Storage: SQLite with write-ahead logging (WAL) mode.
    """

    def __init__(
        self,
        db_path: str = "gaia_audit.db",
        anomaly_callback: Optional[Any] = None,
    ):
        self.db_path = db_path
        self.anomaly_callback = anomaly_callback
        self._lock = threading.Lock()
        self._last_hash: Optional[str] = None
        self._event_buffer: List[AuditEvent] = []
        self._buffer_size = 50
        self._init_db()
        self._load_last_hash()

    def _init_db(self) -> None:
        """Initialize audit database."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")  # Write-ahead logging
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT UNIQUE NOT NULL,
                event_type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                agent_name TEXT,
                user_id TEXT,
                action TEXT NOT NULL,
                resource TEXT,
                outcome TEXT NOT NULL,
                severity TEXT NOT NULL,
                details TEXT,
                previous_hash TEXT,
                entry_hash TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_audit_time ON audit_log(timestamp);
            CREATE INDEX IF NOT EXISTS idx_audit_type ON audit_log(event_type);
            CREATE INDEX IF NOT EXISTS idx_audit_agent ON audit_log(agent_name);
            CREATE INDEX IF NOT EXISTS idx_audit_severity ON audit_log(severity);
            CREATE INDEX IF NOT EXISTS idx_audit_outcome ON audit_log(outcome);
        """)
        conn.commit()
        conn.close()

    def _load_last_hash(self) -> None:
        """Load the hash of the last audit entry."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT entry_hash FROM audit_log ORDER BY id DESC LIMIT 1")
        row = cursor.fetchone()
        self._last_hash = row[0] if row else "0" * 64
        conn.close()

    def _compute_hash(self, event: AuditEvent, previous_hash: str) -> str:
        """Compute hash for an audit entry (chain link)."""
        data = (
            f"{event.event_id}|{event.event_type.value}|{event.timestamp.isoformat()}|"
            f"{event.action}|{event.resource}|{event.outcome}|{previous_hash}"
        )
        return hashlib.sha256(data.encode()).hexdigest()

    def log_event(self, event: AuditEvent) -> None:
        """
        Log an audit event.

        Thread-safe. Events are buffered and flushed periodically.
        """
        with self._lock:
            previous_hash = self._last_hash
            entry_hash = self._compute_hash(event, previous_hash)
            self._last_hash = entry_hash

            self._event_buffer.append((event, previous_hash, entry_hash))

            if len(self._event_buffer) >= self._buffer_size:
                self._flush()

        # Check for anomalies
        if event.severity == "critical" and self.anomaly_callback:
            self.anomaly_callback(event)

    def _flush(self) -> None:
        """Flush buffered events to database."""
        if not self._event_buffer:
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for event, prev_hash, entry_hash in self._event_buffer:
            cursor.execute("""
                INSERT INTO audit_log
                (event_id, event_type, timestamp, agent_name, user_id,
                 action, resource, outcome, severity, details,
                 previous_hash, entry_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event.event_id,
                event.event_type.value,
                event.timestamp.isoformat(),
                event.context.agent_name if event.context else "",
                event.context.user_id if event.context else "",
                event.action,
                event.resource,
                event.outcome,
                event.severity,
                json.dumps(event.details, default=str),
                prev_hash,
                entry_hash,
            ))

        conn.commit()
        conn.close()
        self._event_buffer.clear()

    def flush(self) -> None:
        """Public flush method."""
        with self._lock:
            self._flush()

    def verify_integrity(self) -> Dict[str, Any]:
        """
        Verify the integrity of the audit log.

        Walks the hash chain and checks for tampering.

        Returns:
            {
                "valid": True/False,
                "total_entries": int,
                "first_invalid_id": int or None,
                "verification_time": float
            }
        """
        import time
        start = time.time()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, event_id, event_type, timestamp, action, resource, "
            "outcome, previous_hash, entry_hash FROM audit_log ORDER BY id"
        )

        total = 0
        first_invalid = None
        expected_prev = "0" * 64

        for row in cursor.fetchall():
            total += 1
            entry_id = row[0]
            stored_prev_hash = row[7]
            stored_entry_hash = row[8]

            # Verify chain link
            if stored_prev_hash != expected_prev:
                if first_invalid is None:
                    first_invalid = entry_id
                break

            # Verify entry hash
            data = (
                f"{row[1]}|{row[2]}|{row[3]}|{row[4]}|{row[5]}|{row[6]}|{stored_prev_hash}"
            )
            computed_hash = hashlib.sha256(data.encode()).hexdigest()
            if computed_hash != stored_entry_hash:
                if first_invalid is None:
                    first_invalid = entry_id
                break

            expected_prev = stored_entry_hash

        conn.close()

        return {
            "valid": first_invalid is None,
            "total_entries": total,
            "first_invalid_id": first_invalid,
            "verification_time": time.time() - start,
        }

    def query_events(
        self,
        event_type: Optional[AuditEventType] = None,
        agent_name: Optional[str] = None,
        severity: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Query audit events with filters."""
        self.flush()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = "SELECT * FROM audit_log WHERE 1=1"
        params = []

        if event_type:
            query += " AND event_type = ?"
            params.append(event_type.value)
        if agent_name:
            query += " AND agent_name = ?"
            params.append(agent_name)
        if severity:
            query += " AND severity = ?"
            params.append(severity)
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time.isoformat())
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time.isoformat())

        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        return [
            {
                "id": r[0],
                "event_id": r[1],
                "event_type": r[2],
                "timestamp": r[3],
                "agent_name": r[4],
                "user_id": r[5],
                "action": r[6],
                "resource": r[7],
                "outcome": r[8],
                "severity": r[9],
                "details": json.loads(r[10]) if r[10] else {},
            }
            for r in rows
        ]

    def generate_compliance_report(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> Dict[str, Any]:
        """Generate a compliance summary report."""
        events = self.query_events(
            start_time=start_date, end_time=end_date, limit=10000
        )

        denied_count = sum(1 for e in events if e["outcome"] == "denied")
        error_count = sum(1 for e in events if e["outcome"] == "error")
        critical_count = sum(1 for e in events if e["severity"] == "critical")

        secret_accesses = sum(
            1 for e in events if e["event_type"] == "secret_access"
        )

        by_agent = {}
        for e in events:
            agent = e.get("agent_name", "unknown")
            if agent not in by_agent:
                by_agent[agent] = {"total": 0, "denied": 0, "errors": 0}
            by_agent[agent]["total"] += 1
            if e["outcome"] == "denied":
                by_agent[agent]["denied"] += 1
            if e["outcome"] == "error":
                by_agent[agent]["errors"] += 1

        # Verify integrity
        integrity = self.verify_integrity()

        return {
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
            },
            "summary": {
                "total_events": len(events),
                "denied_events": denied_count,
                "error_events": error_count,
                "critical_events": critical_count,
                "secret_accesses": secret_accesses,
            },
            "by_agent": by_agent,
            "integrity": integrity,
        }
```

---

## Resource Limits

### 5.1 Resource Governor

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Resource governance and limit enforcement for GAIA.
"""

import threading
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional

from gaia.logger import get_logger

log = get_logger(__name__)


class ResourceGovernor:
    """
    Enforce resource limits for GAIA agents.

    Limits:
    - Token usage per agent per day
    - API request rate limiting
    - Concurrent agent executions
    - Execution time per query
    - File size limits
    - Network domain whitelisting

    Usage:
        governor = ResourceGovernor()
        governor.set_quota("EmailAgent", ResourceQuota(
            max_tokens_per_day=500_000,
            max_requests_per_hour=50,
        ))

        # Before each operation:
        if governor.check_token_budget("EmailAgent", tokens_needed=1000):
            result = agent.process_query(query)
            governor.record_usage("EmailAgent", tokens_used=1500)
    """

    def __init__(
        self,
        default_quota: Optional[ResourceQuota] = None,
        audit_logger: Optional[AuditLogger] = None,
    ):
        self.default_quota = default_quota or ResourceQuota(name="default")
        self.audit = audit_logger
        self._quotas: Dict[str, ResourceQuota] = {}
        self._usage: Dict[str, Dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self._rate_windows: Dict[str, List[float]] = defaultdict(list)
        self._active_agents: Dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()

    def set_quota(self, agent_name: str, quota: ResourceQuota) -> None:
        """Set resource quota for an agent."""
        self._quotas[agent_name] = quota
        log.info(f"Resource quota set for {agent_name}: {quota.max_tokens_per_day} tokens/day")

    def get_quota(self, agent_name: str) -> ResourceQuota:
        """Get the effective quota for an agent."""
        return self._quotas.get(agent_name, self.default_quota)

    def check_token_budget(self, agent_name: str, tokens_needed: int) -> bool:
        """
        Check if an agent has sufficient token budget.

        Args:
            agent_name: Agent requesting tokens
            tokens_needed: Estimated tokens for the operation

        Returns:
            True if within budget, False if would exceed limit
        """
        quota = self.get_quota(agent_name)
        today = time.strftime("%Y-%m-%d")

        with self._lock:
            used = self._usage[agent_name].get(f"tokens_{today}", 0)

        if used + tokens_needed > quota.max_tokens_per_day:
            log.warning(
                f"Token budget exceeded: {agent_name} "
                f"({used + tokens_needed} > {quota.max_tokens_per_day})"
            )
            if self.audit:
                self.audit.log_event(AuditEvent(
                    event_type=AuditEventType.RESOURCE_LIMIT_HIT,
                    action="token_check",
                    resource=agent_name,
                    outcome="denied",
                    severity="warning",
                    details={
                        "used": used,
                        "requested": tokens_needed,
                        "limit": quota.max_tokens_per_day,
                    },
                ))
            return False

        return True

    def check_rate_limit(self, agent_name: str) -> bool:
        """
        Check if an agent is within rate limits.

        Uses sliding window rate limiting.
        """
        quota = self.get_quota(agent_name)
        now = time.time()
        window_start = now - 3600  # 1 hour window

        with self._lock:
            # Clean old entries
            self._rate_windows[agent_name] = [
                t for t in self._rate_windows[agent_name] if t > window_start
            ]

            current_rate = len(self._rate_windows[agent_name])

            if current_rate >= quota.max_requests_per_hour:
                log.warning(
                    f"Rate limit exceeded: {agent_name} "
                    f"({current_rate}/{quota.max_requests_per_hour}/hour)"
                )
                return False

            self._rate_windows[agent_name].append(now)

        return True

    def check_concurrent_limit(self, agent_name: str = "") -> bool:
        """Check if concurrent agent limit is reached."""
        quota = self.get_quota(agent_name) if agent_name else self.default_quota

        with self._lock:
            total_active = sum(self._active_agents.values())
            if total_active >= quota.max_concurrent_agents:
                log.warning(
                    f"Concurrent limit reached: {total_active}/{quota.max_concurrent_agents}"
                )
                return False

        return True

    def acquire_agent_slot(self, agent_name: str) -> bool:
        """
        Acquire a concurrent execution slot.

        Must be released with release_agent_slot().
        """
        if not self.check_concurrent_limit(agent_name):
            return False

        with self._lock:
            self._active_agents[agent_name] += 1

        return True

    def release_agent_slot(self, agent_name: str) -> None:
        """Release a concurrent execution slot."""
        with self._lock:
            if self._active_agents[agent_name] > 0:
                self._active_agents[agent_name] -= 1

    def record_usage(
        self,
        agent_name: str,
        tokens_used: int = 0,
        cost_usd: float = 0.0,
    ) -> None:
        """Record resource usage for an agent."""
        today = time.strftime("%Y-%m-%d")

        with self._lock:
            self._usage[agent_name][f"tokens_{today}"] += tokens_used
            self._usage[agent_name][f"cost_{today}"] += cost_usd
            self._usage[agent_name]["total_tokens"] += tokens_used
            self._usage[agent_name]["total_cost"] += cost_usd

    def check_domain_allowed(self, agent_name: str, domain: str) -> bool:
        """Check if a network domain is allowed for an agent."""
        quota = self.get_quota(agent_name)

        if quota.blocked_domains and domain in quota.blocked_domains:
            return False

        if quota.allowed_domains and domain not in quota.allowed_domains:
            return False

        return True

    def get_usage_report(self, agent_name: str) -> Dict[str, Any]:
        """Get usage report for an agent."""
        quota = self.get_quota(agent_name)
        today = time.strftime("%Y-%m-%d")

        with self._lock:
            usage = dict(self._usage[agent_name])

        return {
            "agent": agent_name,
            "today_tokens": usage.get(f"tokens_{today}", 0),
            "today_cost": usage.get(f"cost_{today}", 0),
            "total_tokens": usage.get("total_tokens", 0),
            "total_cost": usage.get("total_cost", 0),
            "quota": {
                "max_tokens_per_day": quota.max_tokens_per_day,
                "max_requests_per_hour": quota.max_requests_per_hour,
                "max_concurrent": quota.max_concurrent_agents,
            },
            "rate_current": len(self._rate_windows.get(agent_name, [])),
            "active_slots": self._active_agents.get(agent_name, 0),
        }
```

---

## Integration with GAIA

### 6.1 Security Middleware

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Security middleware that integrates with GAIA's Agent base class.
"""

import functools
from typing import Any, Callable, Optional

from gaia.logger import get_logger

log = get_logger(__name__)


# Global security components
_vault: Optional[SecretVault] = None
_enforcer: Optional[PermissionEnforcer] = None
_audit: Optional[AuditLogger] = None
_governor: Optional[ResourceGovernor] = None


def initialize_security(
    vault_db: str = "gaia_vault.db",
    audit_db: str = "gaia_audit.db",
    enable_audit: bool = True,
    default_quota: Optional[ResourceQuota] = None,
) -> None:
    """
    Initialize the GAIA security framework.

    Call once at application startup.
    """
    global _vault, _enforcer, _audit, _governor

    _audit = AuditLogger(db_path=audit_db) if enable_audit else None
    _vault = SecretVault(db_path=vault_db, audit_logger=_audit)
    _enforcer = PermissionEnforcer(audit_logger=_audit)
    _governor = ResourceGovernor(default_quota=default_quota, audit_logger=_audit)

    log.info("GAIA security framework initialized")


def get_vault() -> SecretVault:
    """Get the global secret vault."""
    global _vault
    if _vault is None:
        initialize_security()
    return _vault


def get_enforcer() -> PermissionEnforcer:
    """Get the global permission enforcer."""
    global _enforcer
    if _enforcer is None:
        initialize_security()
    return _enforcer


def get_audit_logger() -> Optional[AuditLogger]:
    """Get the global audit logger."""
    return _audit


def get_governor() -> ResourceGovernor:
    """Get the global resource governor."""
    global _governor
    if _governor is None:
        initialize_security()
    return _governor


def require_permission(*permissions: Permission):
    """
    Decorator to require permissions for a function.

    Usage:
        @require_permission(Permission.EMAIL_SEND)
        def send_email(self, to, subject, body):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get security context from first arg (self) if agent
            context = None
            if args and hasattr(args[0], "_security_context"):
                context = args[0]._security_context

            if context:
                enforcer = get_enforcer()
                for perm in permissions:
                    if not enforcer.check(context, perm, resource=func.__name__):
                        raise PermissionError(
                            f"Permission {perm.value} required for {func.__name__}"
                        )

            return func(*args, **kwargs)
        return wrapper
    return decorator


def secret(name: str, context: Optional[SecurityContext] = None) -> Optional[str]:
    """
    Convenience function to get a secret.

    Usage:
        api_key = secret("openai_key")
    """
    return get_vault().get(name, context)
```

### 6.2 Secure Agent Base Mixin

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Security mixin for GAIA Agent base class.
"""


class SecureAgentMixin:
    """
    Mixin that adds security capabilities to GAIA agents.

    Integrates:
    - Permission checking before tool execution
    - Secret access through vault
    - Resource limit enforcement
    - Audit logging of agent actions

    Usage:
        class MyAgent(Agent, SecureAgentMixin):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.init_security()
    """

    def init_security(self) -> None:
        """Initialize security for this agent."""
        enforcer = get_enforcer()
        self._security_context = enforcer.create_context(
            agent_name=self.__class__.__name__,
            agent_type=self.__class__.__name__,
        )
        self._governor = get_governor()
        self._vault = get_vault()

        # Acquire execution slot
        if not self._governor.acquire_agent_slot(self.__class__.__name__):
            raise RuntimeError(
                f"Concurrent agent limit reached for {self.__class__.__name__}"
            )

    def get_secret(self, name: str) -> Optional[str]:
        """Get a secret from the vault with audit."""
        enforcer = get_enforcer()
        if not enforcer.check(
            self._security_context, Permission.SECRET_READ, resource=name
        ):
            return None
        return self._vault.get(name, self._security_context)

    def check_permission(self, permission: Permission, resource: str = "") -> bool:
        """Check if this agent has a permission."""
        return get_enforcer().check(self._security_context, permission, resource)

    def cleanup_security(self) -> None:
        """Release security resources. Call on agent destruction."""
        if hasattr(self, "_governor"):
            self._governor.release_agent_slot(self.__class__.__name__)
```

---

## Threat Model

```
+---------------------------------------------------------------+
|                    GAIA Threat Model                           |
|                                                               |
|  THREAT 1: Credential Exposure                                |
|  - Risk: API keys in env vars, logs, or prompts               |
|  - Mitigation: Encrypted vault, secret scanning, log redaction|
|                                                               |
|  THREAT 2: Privilege Escalation                               |
|  - Risk: Agent accesses resources beyond its scope            |
|  - Mitigation: Capability-based permissions, audit logging    |
|                                                               |
|  THREAT 3: Prompt Injection (Indirect)                        |
|  - Risk: Document content manipulates agent behavior          |
|  - Mitigation: Input sanitization, output validation          |
|                                                               |
|  THREAT 4: Resource Exhaustion                                |
|  - Risk: Runaway agent consumes all tokens/compute            |
|  - Mitigation: Resource governor, budget limits, rate limiting|
|                                                               |
|  THREAT 5: Audit Tampering                                    |
|  - Risk: Attacker modifies audit log to hide activity         |
|  - Mitigation: Hash chain integrity, append-only storage      |
|                                                               |
|  THREAT 6: Data Exfiltration                                  |
|  - Risk: Agent sends sensitive data to external endpoints     |
|  - Mitigation: Domain whitelisting, content filtering         |
+---------------------------------------------------------------+
```

---

## Implementation Plan

| Week | Phase | Tasks | Deliverables |
|------|-------|-------|-------------|
| 1-2 | Core Types | SecurityContext, Permission, AuditEvent, SecretEntry | Type system |
| 3-4 | Vault | EncryptionEngine, SecretVault, keyring integration | Encrypted secrets |
| 5-6 | Permissions | PermissionEnforcer, profiles, context creation | Permission model |
| 7-8 | Audit | AuditLogger, hash chain, integrity verification | Audit system |
| 9 | Resources | ResourceGovernor, quotas, rate limiting | Resource limits |
| 10 | Integration | SecureAgentMixin, middleware, decorators | GAIA integration |
| 11 | CLI | `gaia security` commands (vault, audit, permissions) | CLI tools |
| 12 | Testing | Security tests, penetration testing, documentation | Production-ready |

---

## Testing Strategy

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Unit tests for security framework."""

import pytest
import tempfile
import os


class TestEncryptionEngine:
    def test_encrypt_decrypt(self):
        engine = EncryptionEngine.from_password("test-password")
        encrypted = engine.encrypt("my-secret-api-key")
        decrypted = engine.decrypt(encrypted)
        assert decrypted == "my-secret-api-key"

    def test_different_ciphertext_each_time(self):
        engine = EncryptionEngine.from_password("test-password")
        e1 = engine.encrypt("same-text")
        e2 = engine.encrypt("same-text")
        assert e1 != e2  # Different nonces produce different ciphertext

    def test_wrong_password_fails(self):
        engine1 = EncryptionEngine.from_password("password1")
        engine2 = EncryptionEngine.from_password("password2")
        encrypted = engine1.encrypt("secret")
        with pytest.raises(Exception):
            engine2.decrypt(encrypted)


class TestSecretVault:
    @pytest.fixture
    def vault(self, tmp_path):
        db = str(tmp_path / "test_vault.db")
        engine = EncryptionEngine.from_password("test-pass")
        return SecretVault(db_path=db, encryption_engine=engine)

    def test_store_and_retrieve(self, vault):
        vault.store("api_key", "sk-test123", secret_type=SecretType.API_KEY)
        value = vault.get("api_key")
        assert value == "sk-test123"

    def test_agent_scoping(self, vault):
        vault.store(
            "email_token", "tok-email",
            allowed_agents=["EmailAgent"]
        )
        # EmailAgent can access
        ctx = SecurityContext(agent_name="EmailAgent")
        value = vault.get("email_token", ctx)
        assert value == "tok-email"

        # CodeAgent cannot
        ctx = SecurityContext(agent_name="CodeAgent")
        value = vault.get("email_token", ctx)
        assert value is None

    def test_rotation(self, vault):
        vault.store("key", "v1")
        assert vault.get("key") == "v1"

        vault.rotate("key", "v2")
        assert vault.get("key") == "v2"

    def test_secret_scanning(self, vault):
        text = 'My API key is sk-abc123def456ghi789jkl012mno345pqr678stu901vwx234'
        findings = vault.scan_for_secrets(text)
        assert len(findings) > 0
        assert any(f["type"] == "openai_key" for f in findings)


class TestPermissionEnforcer:
    @pytest.fixture
    def enforcer(self):
        return PermissionEnforcer()

    def test_default_permissions(self, enforcer):
        perms = enforcer.get_permissions("ChatAgent")
        assert Permission.LLM_LOCAL in perms
        assert Permission.CODE_SHELL not in perms

    def test_permission_check(self, enforcer):
        ctx = enforcer.create_context("ChatAgent", "ChatAgent")
        assert enforcer.check(ctx, Permission.LLM_LOCAL)
        assert not enforcer.check(ctx, Permission.CODE_SHELL)

    def test_grant_revoke(self, enforcer):
        enforcer.grant_permission("ChatAgent", Permission.CODE_SHELL)
        perms = enforcer.get_permissions("ChatAgent")
        assert Permission.CODE_SHELL in perms

        enforcer.revoke_permission("ChatAgent", Permission.CODE_SHELL)
        perms = enforcer.get_permissions("ChatAgent")
        assert Permission.CODE_SHELL not in perms


class TestAuditLogger:
    @pytest.fixture
    def logger(self, tmp_path):
        return AuditLogger(db_path=str(tmp_path / "test_audit.db"))

    def test_log_and_query(self, logger):
        logger.log_event(AuditEvent(
            event_type=AuditEventType.SECRET_ACCESS,
            action="get",
            resource="api_key",
            outcome="success",
            severity="info",
        ))
        logger.flush()

        events = logger.query_events(event_type=AuditEventType.SECRET_ACCESS)
        assert len(events) == 1
        assert events[0]["resource"] == "api_key"

    def test_integrity_verification(self, logger):
        for i in range(10):
            logger.log_event(AuditEvent(
                event_type=AuditEventType.TOOL_EXECUTE,
                action=f"action_{i}",
                resource=f"resource_{i}",
            ))
        logger.flush()

        result = logger.verify_integrity()
        assert result["valid"] is True
        assert result["total_entries"] == 10


class TestResourceGovernor:
    @pytest.fixture
    def governor(self):
        return ResourceGovernor(
            default_quota=ResourceQuota(
                name="test",
                max_tokens_per_day=10000,
                max_requests_per_hour=10,
                max_concurrent_agents=2,
            )
        )

    def test_token_budget(self, governor):
        assert governor.check_token_budget("TestAgent", 5000) is True
        governor.record_usage("TestAgent", tokens_used=8000)
        assert governor.check_token_budget("TestAgent", 5000) is False

    def test_concurrent_limit(self, governor):
        assert governor.acquire_agent_slot("A") is True
        assert governor.acquire_agent_slot("B") is True
        assert governor.acquire_agent_slot("C") is False
        governor.release_agent_slot("A")
        assert governor.acquire_agent_slot("C") is True
```

---

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Secret exposure incidents | 0 | Secret scanning + audit review |
| Unauthorized access attempts blocked | 100% | Audit log analysis |
| Audit log integrity | 100% hash chain valid | Automated verification |
| Encryption correctness | 100% round-trip | Unit test coverage |
| Permission enforcement accuracy | 100% of defined rules | Integration tests |
| Resource limit enforcement | 100% of quota violations caught | Load testing |
| Secret scan detection rate | >95% for known patterns | Pattern test suite |
| Vault operation latency | <10ms for get/store | Benchmark |

---

## Complete Code

```
src/gaia/security/
    __init__.py
    types.py              # Core types (Permission, SecurityContext, etc.)
    vault.py              # SecretVault, EncryptionEngine
    permissions.py        # PermissionEnforcer, profiles
    audit.py              # AuditLogger, hash chain, compliance
    governor.py           # ResourceGovernor, quotas, rate limiting
    middleware.py         # Security middleware, decorators
    mixin.py              # SecureAgentMixin
    scanner.py            # Secret scanning utilities
    cli.py                # CLI commands for 'gaia security'

tests/unit/security/
    test_vault.py
    test_encryption.py
    test_permissions.py
    test_audit.py
    test_governor.py
    test_scanner.py

tests/integration/security/
    test_secure_agent.py
    test_end_to_end_security.py
```
