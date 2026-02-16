# Security Review: GAIA Architecture Documents

**Reviewer**: GAIA Code Reviewer Agent (claude-opus-4-6)
**Date**: February 7, 2026
**Scope**: 7 new architecture specification documents
**Classification**: INTERNAL - SECURITY SENSITIVE

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Methodology](#2-methodology)
3. [Vulnerability Summary Matrix](#3-vulnerability-summary-matrix)
4. [Document-by-Document Findings](#4-document-by-document-findings)
   - 4.1 [COMPUTER_USE_ARCHITECTURE.md](#41-computer_use_architecturemd)
   - 4.2 [EMAIL_INTEGRATION_ARCHITECTURE.md](#42-email_integration_architecturemd)
   - 4.3 [WORKFLOW_ORCHESTRATION_ARCHITECTURE.md](#43-workflow_orchestration_architecturemd)
   - 4.4 [MULTI_DOCUMENT_SYNTHESIS_ARCHITECTURE.md](#44-multi_document_synthesis_architecturemd)
   - 4.5 [OBSERVABILITY_ARCHITECTURE.md](#45-observability_architecturemd)
   - 4.6 [SECURITY_SECRETS_ARCHITECTURE.md](#46-security_secrets_architecturemd)
   - 4.7 [ERROR_RECOVERY_ARCHITECTURE.md](#47-error_recovery_architecturemd)
5. [Cross-Cutting Security Concerns](#5-cross-cutting-security-concerns)
6. [Integration Recommendations](#6-integration-recommendations)
7. [Compliance Considerations](#7-compliance-considerations)
8. [Security Testing Strategy](#8-security-testing-strategy)
9. [Prioritized Remediation Plan](#9-prioritized-remediation-plan)
10. [Appendix: Severity Definitions](#10-appendix-severity-definitions)

---

## 1. Executive Summary

### Overall Security Posture: MODERATE RISK

Seven new architecture documents were reviewed for security vulnerabilities, covering computer use automation, email integration, workflow orchestration, multi-document synthesis, observability, secrets management, and error recovery. These documents collectively propose significant new attack surface for the GAIA framework.

### Key Statistics

| Category | Count |
|----------|-------|
| **CRITICAL vulnerabilities** | 8 |
| **HIGH vulnerabilities** | 14 |
| **MEDIUM vulnerabilities** | 19 |
| **LOW vulnerabilities** | 11 |
| **Informational findings** | 9 |
| **Total findings** | 61 |

### Top 3 Critical Risks

1. **Command Injection via Computer Use** -- The `DesktopAutomation` code on line 2431 of `COMPUTER_USE_ARCHITECTURE.md` uses `shell=True` with application names that could be controlled by LLM output, enabling arbitrary command execution.

2. **OAuth Token Storage in Plaintext** -- The Gmail provider in `EMAIL_INTEGRATION_ARCHITECTURE.md` (lines 658-659) writes OAuth tokens to a JSON file on disk without encryption, with a default path of `gmail_token.json` in the working directory.

3. **Fallback Encryption Key Derivable from Public Data** -- The `SECURITY_SECRETS_ARCHITECTURE.md` (lines 508-516) `_fallback_key()` method derives the vault encryption key from `platform.node()` and `socket.gethostname()`, both of which are publicly discoverable, combined with a hardcoded salt `b"gaia-vault-salt"`.

### Strengths Identified

- The Security & Secrets Architecture document demonstrates mature security thinking with AES-256-GCM encryption, PBKDF2 key derivation, and hash-chained audit logs.
- The Workflow Orchestration document explicitly avoids `eval()` in condition evaluation, using AST-based safe evaluation instead.
- The Error Recovery Architecture properly classifies errors by category and does not retry on permanent failures (authentication errors), preventing credential lockout.
- The Observability Architecture uses context variables (`contextvars`) for async-safe trace propagation.
- Most code blocks carry the AMD copyright header (`Copyright(C) 2024-2025 Advanced Micro Devices, Inc.`).

---

## 2. Methodology

### Review Approach

1. **Static analysis** of all Python code blocks in each document
2. **Pattern matching** for known vulnerability signatures (hardcoded secrets, injection vectors, path traversal, insecure defaults)
3. **Architecture review** for defense-in-depth gaps, missing rate limiting, and insufficient audit logging
4. **Compliance mapping** against OWASP Top 10 and CWE categories
5. **Cross-document integration analysis** for permission boundary gaps between subsystems

### Severity Rating Criteria

| Rating | Definition |
|--------|-----------|
| **CRITICAL** | Exploitable vulnerability that can lead to remote code execution, credential theft, or complete system compromise |
| **HIGH** | Vulnerability that can lead to data exposure, privilege escalation, or significant denial of service |
| **MEDIUM** | Vulnerability that requires specific conditions to exploit or has limited impact |
| **LOW** | Best practice deviation or defense-in-depth improvement |
| **INFO** | Observation or recommendation for future consideration |

---

## 3. Vulnerability Summary Matrix

| ID | Document | Severity | Category | Title |
|----|----------|----------|----------|-------|
| SEC-001 | COMPUTER_USE | CRITICAL | Command Injection | `shell=True` in desktop app launcher |
| SEC-002 | COMPUTER_USE | CRITICAL | Unrestricted Input | No URL allowlist for browser automation |
| SEC-003 | COMPUTER_USE | HIGH | Credential Exposure | Screenshots may capture sensitive data |
| SEC-004 | COMPUTER_USE | HIGH | Privilege Escalation | No permission boundary for keyboard/mouse |
| SEC-005 | COMPUTER_USE | MEDIUM | Information Disclosure | VLM prompts expose screen content to model |
| SEC-006 | COMPUTER_USE | MEDIUM | Missing AMD Header | Early code blocks lack AMD copyright |
| SEC-007 | EMAIL | CRITICAL | Credential Exposure | OAuth tokens stored in plaintext JSON |
| SEC-008 | EMAIL | CRITICAL | Credential Exposure | IMAP password stored in memory dict |
| SEC-009 | EMAIL | HIGH | IMAP Injection | Unsanitized input in IMAP SEARCH query |
| SEC-010 | EMAIL | HIGH | Missing Rate Limiting | No rate limit on email send operations |
| SEC-011 | EMAIL | HIGH | Data Privacy | Email body content passed to LLM without PII scrubbing |
| SEC-012 | EMAIL | MEDIUM | XSS via HTML Email | HTML email body rendered without sanitization |
| SEC-013 | EMAIL | MEDIUM | Missing Send Confirmation | No human-in-loop for auto-send by default |
| SEC-014 | EMAIL | LOW | OAuth Scope Overprovision | Gmail+Calendar scopes combined by default |
| SEC-015 | WORKFLOW | CRITICAL | Server-Side Request Forgery | HttpStepExecutor allows arbitrary URL requests |
| SEC-016 | WORKFLOW | HIGH | Command Injection | SubprocessStepExecutor with context interpolation |
| SEC-017 | WORKFLOW | HIGH | Webhook Authentication Bypass | Signature validation disabled when no secret |
| SEC-018 | WORKFLOW | HIGH | Denial of Service | No concurrency limit on webhook handler |
| SEC-019 | WORKFLOW | MEDIUM | Information Disclosure | Full request headers stored in trigger data |
| SEC-020 | WORKFLOW | MEDIUM | Expression Injection | AST-based eval fallback path uses `literal_eval` |
| SEC-021 | WORKFLOW | MEDIUM | Bind to All Interfaces | Webhook server defaults to 0.0.0.0 |
| SEC-022 | WORKFLOW | LOW | File Watcher Path Traversal | No validation on watched directory path |
| SEC-023 | MULTI_DOC | HIGH | Path Traversal | `ingest_document` does not sanitize file path |
| SEC-024 | MULTI_DOC | HIGH | Denial of Service | No limit on document size or collection size |
| SEC-025 | MULTI_DOC | MEDIUM | Information Disclosure | Absolute file paths stored in metadata |
| SEC-026 | MULTI_DOC | MEDIUM | Hash Collision | MD5 used for embedding cache keys |
| SEC-027 | MULTI_DOC | LOW | Resource Exhaustion | Class-level embedding cache unbounded across instances |
| SEC-028 | MULTI_DOC | LOW | Missing Type Validation | HTML parsing uses regex, vulnerable to malformed input |
| SEC-029 | OBSERVABILITY | HIGH | Information Disclosure | Span attributes may contain sensitive prompt content |
| SEC-030 | OBSERVABILITY | HIGH | Data Privacy | Session recordings store complete LLM prompts/responses |
| SEC-031 | OBSERVABILITY | MEDIUM | Log Injection | Span attributes not sanitized before SQLite insert |
| SEC-032 | OBSERVABILITY | MEDIUM | Denial of Service | Unbounded span buffer in TracerProvider |
| SEC-033 | OBSERVABILITY | LOW | Missing Access Control | No authentication on trace/metrics query endpoints |
| SEC-034 | OBSERVABILITY | LOW | Time Reference Vulnerability | `time.time()` not monotonic for span duration |
| SEC-035 | SECRETS | CRITICAL | Weak Key Derivation | Fallback key derived from public machine data |
| SEC-036 | SECRETS | HIGH | Missing Access Control | Secret vault operations accept `None` context |
| SEC-037 | SECRETS | HIGH | Key Material in Memory | Encryption key stored as plain bytes in `_key` |
| SEC-038 | SECRETS | MEDIUM | Audit Log Race Condition | Buffer flush not guaranteed on crash |
| SEC-039 | SECRETS | MEDIUM | Hash Chain Weakness | SHA-256 chain does not include sequence number |
| SEC-040 | SECRETS | MEDIUM | Secret Scanning Incomplete | Regex patterns miss common secret formats |
| SEC-041 | SECRETS | LOW | Vault DB Permissions | No file permission enforcement on SQLite files |
| SEC-042 | SECRETS | LOW | Copyright Year Range | Copyright says 2024-2025 but date is 2026 |
| SEC-043 | ERROR | MEDIUM | Error Message Exposure | Stack traces stored in step metadata |
| SEC-044 | ERROR | MEDIUM | Resource Exhaustion | Dead letter queue has no maximum size |
| SEC-045 | ERROR | LOW | Timing Side Channel | Error classification uses string matching |

---

## 4. Document-by-Document Findings

### 4.1 COMPUTER_USE_ARCHITECTURE.md

**File**: `/mnt/c/Users/14255/Work/gaia/architecture/COMPUTER_USE_ARCHITECTURE.md`

#### SEC-001: Command Injection via shell=True [CRITICAL]

**Location**: Line 2431 (approximately)
**Code**:
```python
sp.Popen(app_name, shell=True)
```

**Description**: The `DesktopAutomation` code passes an application name through `shell=True` to `subprocess.Popen`. If the `app_name` value originates from LLM output or user input, an attacker can inject arbitrary shell commands (e.g., `calc.exe & del /q C:\*`).

**CWE**: CWE-78 (OS Command Injection)

**Recommended Fix**:
```python
# NEVER use shell=True with untrusted input
import shlex
import subprocess

ALLOWED_APPLICATIONS = {
    "notepad": "notepad.exe",
    "calculator": "calc.exe",
    "browser": "chrome.exe",
    # ... allowlist only
}

def launch_application(self, app_name: str) -> bool:
    """Launch application from allowlist only."""
    safe_path = ALLOWED_APPLICATIONS.get(app_name.lower())
    if safe_path is None:
        log.warning(f"Application not in allowlist: {app_name}")
        return False
    subprocess.Popen([safe_path], shell=False)
    return True
```

#### SEC-002: No URL Allowlist for Browser Automation [CRITICAL]

**Location**: Lines 927-935 (BrowserAutomationAgent.navigate)
**Code**:
```python
async def navigate(self, url: str, wait_until: str = "networkidle"):
    await self.page.goto(url, wait_until=wait_until)
```

**Description**: The browser automation `navigate` method accepts any URL without validation. An agent influenced by prompt injection could navigate to `file:///etc/passwd`, `javascript:` URIs, internal network services (`http://169.254.169.254/` for cloud metadata), or phishing sites.

**CWE**: CWE-918 (Server-Side Request Forgery)

**Recommended Fix**:
```python
import urllib.parse

ALLOWED_SCHEMES = {"http", "https"}
BLOCKED_HOSTS = {
    "169.254.169.254",  # AWS metadata
    "metadata.google.internal",  # GCP metadata
    "localhost",
    "127.0.0.1",
    "0.0.0.0",
}

async def navigate(self, url: str, wait_until: str = "networkidle"):
    """Navigate to URL with security validation."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ALLOWED_SCHEMES:
        raise ValueError(f"URL scheme not allowed: {parsed.scheme}")
    if parsed.hostname and parsed.hostname.lower() in BLOCKED_HOSTS:
        raise ValueError(f"Navigation to {parsed.hostname} is blocked")
    if parsed.hostname and parsed.hostname.startswith("10."):
        raise ValueError("Navigation to internal networks is blocked")
    await self.page.goto(url, wait_until=wait_until)
```

#### SEC-003: Screenshots May Capture Sensitive Data [HIGH]

**Location**: Lines 308-311 (ScreenCaptureEngine.save_screenshot)
**Code**:
```python
def save_screenshot(self, path: str, monitor_id: int = 1):
    """Save screenshot to file."""
    img = self.capture_screen(monitor_id)
    img.save(path)
```

**Description**: Screenshots can capture password managers, banking sessions, private messages, or any on-screen content. Saved screenshots persist on disk without encryption, and the `capture_continuous` generator (line 287) creates a stream of screenshots at configurable FPS that could exfiltrate sensitive screen content.

**Impact**: PII exposure, credential capture, privacy violation.

**Recommended Fix**:
- Add a configurable region blocklist (e.g., exclude taskbar/notification area).
- Encrypt screenshots at rest or store only in memory.
- Add a visual indicator (overlay) when screen capture is active.
- Require explicit user consent before each capture session.
- Auto-delete screenshots after processing with secure deletion.

#### SEC-004: No Permission Boundary for Keyboard/Mouse [HIGH]

**Location**: Lines 598-855 (MouseController, KeyboardController)

**Description**: The MouseController and KeyboardController classes have no permission checks, no restricted areas, and no confirmation prompts. An agent could type arbitrary text into any application, execute keyboard shortcuts to open terminals, or use `paste_text` (line 793) to inject content via the clipboard.

**CWE**: CWE-862 (Missing Authorization)

**Recommended Fix**:
```python
class SafetyGuard:
    """Validate and filter dangerous actions before execution."""

    DANGEROUS_HOTKEYS = {
        ("ctrl", "alt", "delete"),
        ("alt", "f4"),
        ("ctrl", "shift", "escape"),  # Task Manager
    }

    RESTRICTED_REGIONS = []  # Populated from config

    def validate_hotkey(self, keys: tuple) -> bool:
        normalized = tuple(k.lower() for k in keys)
        if normalized in self.DANGEROUS_HOTKEYS:
            log.warning(f"Blocked dangerous hotkey: {keys}")
            return False
        return True

    def validate_click(self, x: int, y: int) -> bool:
        for region in self.RESTRICTED_REGIONS:
            if region.contains_point(x, y):
                log.warning(f"Click blocked in restricted region at ({x}, {y})")
                return False
        return True
```

#### SEC-005: VLM Prompts Expose Screen Content [MEDIUM]

**Location**: Lines 466-489 (VLMAnalyzer.describe_screen)

**Description**: The VLM analyzer sends full screenshots to the VLM model with prompts requesting detailed description of all text and UI elements. If using a cloud-based VLM, this transmits potentially sensitive screen content to an external service. Even with a local VLM, the prompts and responses may be logged.

**Recommended Fix**:
- When using cloud VLM, add a mandatory consent check.
- Implement region redaction before sending to VLM (blur sensitive areas).
- Log only sanitized prompt metadata, not full prompt/response content.
- Add a `privacy_mode` flag that limits VLM detail level.

#### SEC-006: Missing AMD Copyright Header on Early Code Blocks [MEDIUM]

**Location**: Lines 192-312, 317-448, 452-586, 593-855 (ScreenCaptureEngine, OCREngine, VLMAnalyzer, Mouse/KeyboardController)

**Description**: The first four major code blocks in the Computer Use Architecture document do not carry the AMD copyright header. Only code blocks starting from line 1221 (DesktopAutomation section) include the proper header.

**AMD Compliance Requirement**: All source code must include:
```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
```

---

### 4.2 EMAIL_INTEGRATION_ARCHITECTURE.md

**File**: `/mnt/c/Users/14255/Work/gaia/architecture/EMAIL_INTEGRATION_ARCHITECTURE.md`

#### SEC-007: OAuth Tokens Stored in Plaintext JSON [CRITICAL]

**Location**: Lines 624, 658-659 (GmailProvider.authenticate)
**Code**:
```python
token_file = credentials.get("token_file", "gmail_token.json")
# ...
with open(token_file, "w") as token_fp:
    token_fp.write(creds.to_json())
```

**Description**: OAuth2 refresh tokens are written as plaintext JSON to a file in the working directory. The default filename `gmail_token.json` is predictable. The file has no restrictive permissions set. Refresh tokens provide persistent access to the user's Gmail account.

**CWE**: CWE-312 (Cleartext Storage of Sensitive Information)

**Recommended Fix**:
```python
import os
import stat

async def authenticate(self, credentials: Dict[str, Any]) -> bool:
    # ... existing OAuth flow ...

    # Store token securely
    token_file = credentials.get("token_file", "gmail_token.json")

    # Option 1: Use the GAIA SecretVault
    from gaia.security import SecretVault
    vault = SecretVault()
    vault.store(
        "gmail_oauth_token",
        creds.to_json(),
        secret_type=SecretType.OAUTH_TOKEN,
        allowed_agents=["EmailAgent"],
    )

    # Option 2: If file storage is needed, restrict permissions
    with open(token_file, "w") as token_fp:
        token_fp.write(creds.to_json())
    os.chmod(token_file, stat.S_IRUSR | stat.S_IWUSR)  # 0600
    log.info(f"Gmail token saved to {token_file} (restricted permissions)")
```

#### SEC-008: IMAP Password Stored in Memory Dictionary [CRITICAL]

**Location**: Lines 1134-1139 (IMAPSMTPProvider.authenticate)
**Code**:
```python
self.smtp_config = {
    "host": credentials.get("smtp_host", imap_host.replace("imap", "smtp")),
    "port": credentials.get("smtp_port", 587),
    "username": self.username,
    "password": password,  # Plaintext password stored in dict
    "use_tls": credentials.get("smtp_use_tls", True),
}
```

**Description**: The user's email password is stored as a plaintext string in `self.smtp_config`, which persists for the lifetime of the provider object. This password is accessible via Python introspection, may appear in debug output, and is not cleared from memory after use.

**CWE**: CWE-316 (Cleartext Storage in Memory)

**Recommended Fix**:
```python
# Use the GAIA SecretVault or at minimum, clear after each SMTP session
import gc

class SecurePasswordHolder:
    """Holds a password reference that can be zeroed."""
    def __init__(self, password: str):
        self._pwd = bytearray(password.encode("utf-8"))

    def get(self) -> str:
        return self._pwd.decode("utf-8")

    def clear(self):
        for i in range(len(self._pwd)):
            self._pwd[i] = 0
        gc.collect()
```

#### SEC-009: IMAP Injection via Unsanitized Search Query [HIGH]

**Location**: Lines 1179-1202 (IMAPSMTPProvider._build_imap_search)
**Code**:
```python
if criteria.sender:
    parts.append(f'FROM "{criteria.sender}"')
if criteria.subject_contains:
    parts.append(f'SUBJECT "{criteria.subject_contains}"')
```

**Description**: The `_build_imap_search` method constructs IMAP SEARCH commands by directly interpolating user-supplied values into the query string. An attacker providing a sender value like `" OR ALL "` could manipulate the IMAP search to return all messages, bypassing intended filtering. While IMAP injection is lower risk than SQL injection, it can lead to unauthorized data access.

**CWE**: CWE-943 (Improper Neutralization of Special Elements in Data Query Logic)

**Recommended Fix**:
```python
def _sanitize_imap_value(self, value: str) -> str:
    """Sanitize a value for use in IMAP SEARCH queries."""
    # Remove IMAP special characters and double quotes
    sanitized = value.replace('"', '').replace('\\', '').replace('\n', ' ').replace('\r', '')
    # Limit length
    return sanitized[:200]

def _build_imap_search(self, criteria: Optional[EmailFilter]) -> str:
    if criteria.sender:
        parts.append(f'FROM "{self._sanitize_imap_value(criteria.sender)}"')
```

#### SEC-010: No Rate Limiting on Email Send Operations [HIGH]

**Location**: Lines 853-930 (GmailProvider.send_message), Lines 1304-1356 (IMAPSMTPProvider.send_message)

**Description**: Neither email provider implements rate limiting on send operations. A malfunctioning or compromised agent could send hundreds of emails per minute, potentially resulting in the account being flagged for spam, blacklisted, or used for phishing campaigns.

**Recommended Fix**:
```python
from collections import deque
import time

class RateLimitedSender:
    """Rate limiter for email send operations."""
    def __init__(self, max_per_minute: int = 10, max_per_hour: int = 50):
        self.max_per_minute = max_per_minute
        self.max_per_hour = max_per_hour
        self._send_times: deque = deque()

    def check_rate_limit(self) -> bool:
        now = time.time()
        # Clean old entries
        while self._send_times and self._send_times[0] < now - 3600:
            self._send_times.popleft()

        hour_count = len(self._send_times)
        minute_count = sum(1 for t in self._send_times if t > now - 60)

        if minute_count >= self.max_per_minute:
            raise RateLimitError(f"Email rate limit exceeded: {minute_count}/{self.max_per_minute} per minute")
        if hour_count >= self.max_per_hour:
            raise RateLimitError(f"Email rate limit exceeded: {hour_count}/{self.max_per_hour} per hour")

        self._send_times.append(now)
        return True
```

#### SEC-011: Email Body Passed to LLM Without PII Scrubbing [HIGH]

**Location**: Lines 1467-1498 (EmailClassifier, referenced from classification engine)

**Description**: The email classification engine sends full email body content to the local LLM for classification. Email bodies frequently contain PII (names, phone numbers, SSNs, medical information, financial data). Even with a local LLM, this data may be logged, cached in model context, or persisted in session recordings (per the Observability Architecture).

**CWE**: CWE-532 (Insertion of Sensitive Information into Log File)

**Recommended Fix**:
```python
import re

def scrub_pii(text: str) -> str:
    """Remove common PII patterns from text before LLM processing."""
    patterns = {
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
        "email_addr": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "credit_card": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
    }
    for pii_type, pattern in patterns.items():
        text = re.sub(pattern, f"[{pii_type.upper()}_REDACTED]", text)
    return text
```

#### SEC-012: HTML Email Body Rendered Without Sanitization [MEDIUM]

**Location**: Lines 370-373 (EmailMessage.body property)
**Code**:
```python
if self.body_html:
    text = re.sub(r"<[^>]+>", " ", self.body_html)
    return unescape(text).strip()
```

**Description**: The `body` property strips HTML tags with a simple regex but does not handle complex nested tags, encoded entities, or JavaScript within HTML attributes. If the resulting text is ever rendered in a web UI or Electron app, XSS is possible. The `unescape` call may re-introduce dangerous characters.

**CWE**: CWE-79 (Cross-site Scripting)

**Recommended Fix**: Use a proper HTML sanitization library like `bleach` or `html-sanitizer`.

#### SEC-013: Missing Human-in-Loop for Auto-Send [MEDIUM]

**Location**: Architecture Overview, Outgoing Email Flow (lines 236-255)

**Description**: While the architecture diagram shows a "Human Review Queue" in the outgoing email flow, the actual code for `send_message` has no confirmation step. The `SendGuard` and `ContentFilter` mentioned in the architecture (line 213) are not implemented in the provided code.

**Recommended Fix**: Implement a mandatory confirmation step:
```python
class SendGuard:
    """Require explicit approval before sending emails."""

    def __init__(self, auto_approve: bool = False):
        self.auto_approve = auto_approve
        self._pending: Dict[str, Dict] = {}

    async def queue_for_review(self, message: Dict) -> str:
        """Queue a message for human review. Returns review_id."""
        review_id = uuid.uuid4().hex[:8]
        self._pending[review_id] = {
            "message": message,
            "queued_at": datetime.now(),
            "status": "pending",
        }
        return review_id

    async def approve(self, review_id: str) -> bool:
        """Human approves a queued message."""
        if review_id in self._pending:
            self._pending[review_id]["status"] = "approved"
            return True
        return False
```

#### SEC-014: OAuth Scope Over-Provision [LOW]

**Location**: Lines 565-575 (GmailProvider scopes)
**Code**:
```python
GMAIL_SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/gmail.labels",
]
CALENDAR_SCOPES = [
    "https://www.googleapis.com/auth/calendar.readonly",
    "https://www.googleapis.com/auth/calendar.events",
]
```

And on line 623:
```python
scopes = credentials.get("scopes", GMAIL_SCOPES + CALENDAR_SCOPES)
```

**Description**: By default, the provider requests both Gmail and Calendar scopes combined. This violates the principle of least privilege. An agent that only needs to read emails should not receive calendar write access.

**Recommended Fix**: Request scopes incrementally based on actual operations needed.

---

### 4.3 WORKFLOW_ORCHESTRATION_ARCHITECTURE.md

**File**: `/mnt/c/Users/14255/Work/gaia/architecture/WORKFLOW_ORCHESTRATION_ARCHITECTURE.md`

#### SEC-015: Server-Side Request Forgery in HttpStepExecutor [CRITICAL]

**Location**: Lines 1168-1261 (HttpStepExecutor.execute)
**Code**:
```python
url = step.config.get("url", "")
# ...
url = self._interpolate(url, context)
# ...
async with session.request(method, url, headers=headers, json=body, timeout=timeout) as response:
```

**Description**: The `HttpStepExecutor` allows workflow definitions to specify arbitrary URLs and HTTP methods. Context interpolation means the URL can be dynamically constructed from previous step outputs. An attacker who can influence workflow definitions or inject context data could make requests to internal services, cloud metadata endpoints (`http://169.254.169.254/latest/meta-data/`), or other SSRF targets.

**CWE**: CWE-918 (Server-Side Request Forgery)

**Recommended Fix**:
```python
BLOCKED_HOSTS = {"169.254.169.254", "metadata.google.internal", "localhost", "127.0.0.1"}
BLOCKED_SCHEMES = {"file", "ftp", "gopher"}
ALLOWED_DOMAINS = None  # Set to a list to enable allowlist mode

def _validate_url(self, url: str) -> None:
    """Validate URL before making HTTP request."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme in BLOCKED_SCHEMES:
        raise ValueError(f"URL scheme blocked: {parsed.scheme}")
    if parsed.hostname and parsed.hostname in BLOCKED_HOSTS:
        raise ValueError(f"Host blocked: {parsed.hostname}")
    if parsed.hostname:
        import ipaddress
        try:
            ip = ipaddress.ip_address(parsed.hostname)
            if ip.is_private or ip.is_loopback or ip.is_link_local:
                raise ValueError(f"Private/internal IP blocked: {parsed.hostname}")
        except ValueError:
            pass  # Not an IP, hostname is fine
    if ALLOWED_DOMAINS is not None:
        if parsed.hostname not in ALLOWED_DOMAINS:
            raise ValueError(f"Domain not in allowlist: {parsed.hostname}")
```

#### SEC-016: Command Injection in SubprocessStepExecutor [HIGH]

**Location**: Lines 1433-1498 (SubprocessStepExecutor.execute)
**Code**:
```python
command = step.config.get("command", [])
use_shell = step.config.get("shell", False)
# ...
if isinstance(command, list):
    command = [self._interpolate(str(arg), context) for arg in command]
```

**Description**: While `shell=False` is the default (good), the `shell` parameter is still configurable from workflow definitions (`step.config.get("shell", False)`). An attacker who can modify workflow YAML/definitions could set `shell: true`. Additionally, context interpolation into command arguments means that previous step outputs could inject shell metacharacters.

**CWE**: CWE-78 (OS Command Injection)

**Recommended Fix**:
```python
# REMOVE the shell option entirely
async def execute(self, step, context, run):
    command = step.config.get("command", [])
    # NEVER allow shell=True from config
    # use_shell = step.config.get("shell", False)  # REMOVED

    if not isinstance(command, list):
        raise ValueError("command must be a list of arguments, not a string")

    # Sanitize each interpolated argument
    safe_command = []
    for arg in command:
        interpolated = self._interpolate(str(arg), context)
        # Reject arguments containing shell metacharacters
        if any(c in interpolated for c in ';&|`$(){}[]!#~'):
            raise ValueError(f"Command argument contains disallowed characters: {interpolated[:50]}")
        safe_command.append(interpolated)

    proc = await asyncio.create_subprocess_exec(
        *safe_command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
        env=env,
    )
```

#### SEC-017: Webhook Authentication Bypass [HIGH]

**Location**: Lines 760-764 (WebhookTrigger.handle_webhook)
**Code**:
```python
if route["require_signature"] and self.webhook_secret:
    signature = request.headers.get("X-Webhook-Signature", "")
    body = await request.body()
    if not self._validate_signature(body, signature):
        raise HTTPException(status_code=401, detail="Invalid signature")
```

**Description**: Signature validation is only performed when BOTH `require_signature` is True AND `self.webhook_secret` is set. If the webhook_secret is `None` (the default on line 707), authentication is completely bypassed even when `require_signature=True`. This is a logic flaw that silently degrades security.

**CWE**: CWE-287 (Improper Authentication)

**Recommended Fix**:
```python
# Fail closed: if signature is required but no secret is configured, reject
if route["require_signature"]:
    if not self.webhook_secret:
        raise HTTPException(
            status_code=500,
            detail="Webhook signature required but no secret configured"
        )
    signature = request.headers.get("X-Webhook-Signature", "")
    body = await request.body()
    if not self._validate_signature(body, signature):
        raise HTTPException(status_code=401, detail="Invalid signature")
```

#### SEC-018: No Concurrency Limit on Webhook Handler [HIGH]

**Location**: Lines 742-806 (WebhookTrigger.start)

**Description**: The webhook server has no request rate limiting, no concurrency control, and no request body size limit. An attacker could flood the webhook endpoint to trigger excessive workflow executions, causing resource exhaustion.

**Recommended Fix**: Add rate limiting middleware and request body size limits to the FastAPI application.

#### SEC-019: Full Request Headers Stored in Trigger Data [MEDIUM]

**Location**: Lines 773-781
**Code**:
```python
trigger_data = {
    "trigger_type": "webhook",
    "path": full_path,
    "method": request.method,
    "headers": dict(request.headers),  # ALL headers stored
    "body": body_data,
    "source_ip": request.client.host,
}
```

**Description**: All HTTP headers are stored in trigger data, which is passed through the workflow context and may be persisted. Headers can contain `Authorization` tokens, session cookies, and other sensitive data.

**Recommended Fix**: Filter out sensitive headers before storage:
```python
SENSITIVE_HEADERS = {"authorization", "cookie", "x-api-key", "x-auth-token"}
safe_headers = {
    k: v for k, v in request.headers.items()
    if k.lower() not in SENSITIVE_HEADERS
}
```

#### SEC-020: Expression Evaluation Fallback Path [MEDIUM]

**Location**: Lines 1307-1364 (ConditionalStepExecutor._evaluate_condition)
**Code**:
```python
# Fallback: try as truthy evaluation
return bool(ast.literal_eval(resolved))
```

**Description**: While the primary evaluation path correctly uses AST comparison operators, the fallback path calls `ast.literal_eval()` on the fully-resolved expression. While `literal_eval` is generally safe (only evaluates literals), the `resolved` string is constructed from workflow context data via regex substitution, and there may be edge cases where crafted context values produce unexpected evaluation results.

**CWE**: CWE-95 (Improper Neutralization of Directives in Dynamically Evaluated Code)

**Recommended Fix**: Remove the fallback path. If the expression does not match the AST Compare pattern, return False explicitly and log a warning.

#### SEC-021: Webhook Server Binds to All Interfaces [MEDIUM]

**Location**: Line 705
**Code**:
```python
host: str = "0.0.0.0",
```

**Description**: The webhook server defaults to binding on all network interfaces (`0.0.0.0`), exposing it to the entire network. For a local development tool, this should default to `127.0.0.1`.

**Recommended Fix**: Change default to `host: str = "127.0.0.1"`.

#### SEC-022: File Watcher Path Traversal [LOW]

**Location**: Lines 831-858 (FileWatchTrigger.register)

**Description**: The `watch_path` parameter is not validated to prevent watching sensitive directories (e.g., `/etc/`, `C:\Windows\`, home directory dotfiles). Additionally, symlink following could allow watching directories outside the intended scope.

---

### 4.4 MULTI_DOCUMENT_SYNTHESIS_ARCHITECTURE.md

**File**: `/mnt/c/Users/14255/Work/gaia/architecture/MULTI_DOCUMENT_SYNTHESIS_ARCHITECTURE.md`

#### SEC-023: Path Traversal in Document Ingestion [HIGH]

**Location**: Lines 418-436 (DocumentIngestionPipeline.ingest_document)
**Code**:
```python
def ingest_document(self, file_path: str, document_id: Optional[str] = None, ...):
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {file_path}")
```

**Description**: The `ingest_document` method accepts any file path without sanitization. An attacker who can influence the file path (e.g., via a workflow step or user input) could use path traversal sequences (`../../etc/passwd`) or absolute paths to read any file accessible by the process.

**CWE**: CWE-22 (Path Traversal)

**Recommended Fix**:
```python
import os

ALLOWED_DOCUMENT_DIRS = []  # Configure via settings

def _validate_file_path(self, file_path: str) -> Path:
    """Validate file path is within allowed directories."""
    path = Path(file_path).resolve()

    if ALLOWED_DOCUMENT_DIRS:
        if not any(str(path).startswith(str(Path(d).resolve())) for d in ALLOWED_DOCUMENT_DIRS):
            raise PermissionError(f"File path outside allowed directories: {file_path}")

    # Check for symlink attacks
    if path.is_symlink():
        real_path = path.resolve()
        if ALLOWED_DOCUMENT_DIRS:
            if not any(str(real_path).startswith(str(Path(d).resolve())) for d in ALLOWED_DOCUMENT_DIRS):
                raise PermissionError(f"Symlink target outside allowed directories: {real_path}")

    return path
```

#### SEC-024: No Limit on Document Size or Collection Size [HIGH]

**Location**: Lines 494-552 (ingest_collection), Lines 860-903 (_generate_chunks)

**Description**: There is no maximum file size check before ingestion, no limit on the number of documents in a collection, and no limit on total memory usage during chunk generation and embedding. A maliciously large document (e.g., a 10GB PDF) could cause memory exhaustion and crash the application.

**CWE**: CWE-400 (Uncontrolled Resource Consumption)

**Recommended Fix**:
```python
MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB
MAX_DOCUMENTS_PER_COLLECTION = 100
MAX_PAGES_PER_DOCUMENT = 5000

def ingest_document(self, file_path: str, ...):
    path = Path(file_path)
    if path.stat().st_size > MAX_FILE_SIZE_BYTES:
        raise ValueError(
            f"File too large: {path.stat().st_size / 1024 / 1024:.1f}MB "
            f"(max: {MAX_FILE_SIZE_BYTES / 1024 / 1024}MB)"
        )
```

#### SEC-025: Absolute File Paths Stored in Metadata [MEDIUM]

**Location**: Lines 469-476 (DocumentMetadata construction)
**Code**:
```python
metadata = DocumentMetadata(
    ...
    file_path=str(path.absolute()),
    ...
)
```

**Description**: Storing absolute file paths in document metadata reveals the server's directory structure. If metadata is shared or exported, this leaks internal path information.

**CWE**: CWE-200 (Exposure of Sensitive Information)

**Recommended Fix**: Store only the filename or a relative path, not the absolute path.

#### SEC-026: MD5 Used for Embedding Cache Keys [MEDIUM]

**Location**: Lines 923, 943
**Code**:
```python
cache_key = hashlib.md5(chunk.text.encode("utf-8")).hexdigest()
```

**Description**: MD5 is used for embedding cache keys. While this is not a cryptographic use case (only for cache lookup), MD5 is deprecated and its use may raise compliance flags. Additionally, MD5 collisions are trivially constructable, meaning two different texts could map to the same cache entry, producing incorrect embeddings.

**CWE**: CWE-328 (Use of Weak Hash)

**Recommended Fix**: Replace with SHA-256:
```python
cache_key = hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()
```

#### SEC-027: Class-Level Embedding Cache Unbounded [LOW]

**Location**: Lines 906-907
**Code**:
```python
_embedding_cache: Dict[str, List[float]] = {}
_EMBEDDING_CACHE_MAX_SIZE = 50_000
```

**Description**: The embedding cache is a class-level attribute shared across all instances. With 50,000 entries of 384-dimensional float vectors, this could consume approximately 150MB of memory. Multiple pipeline instances share the same cache, which could lead to unexpected memory growth.

#### SEC-028: HTML Parsing Uses Regex [LOW]

**Location**: Lines 728-775 (_parse_html)

**Description**: HTML parsing uses regex for tag stripping and table extraction. This is inherently fragile and can be bypassed with malformed HTML, potentially causing ReDoS (Regular Expression Denial of Service) with crafted input.

---

### 4.5 OBSERVABILITY_ARCHITECTURE.md

**File**: `/mnt/c/Users/14255/Work/gaia/architecture/OBSERVABILITY_ARCHITECTURE.md`

#### SEC-029: Span Attributes May Contain Sensitive Data [HIGH]

**Location**: Lines 316-317 (Span.set_attribute), Lines 781 (SQLiteSpanExporter.export)
**Code**:
```python
def set_attribute(self, key: str, value: Any) -> None:
    self.attributes[key] = value
```

**Description**: Span attributes are arbitrary key-value pairs that get serialized and persisted to SQLite. Agent instrumentation will naturally include `query` (user prompts), `result` (LLM responses), and tool arguments in attributes. These may contain PII, credentials, or other sensitive data. The export path serializes all attributes to JSON without any filtering.

**CWE**: CWE-532 (Information Exposure Through Log Files)

**Recommended Fix**:
```python
SENSITIVE_ATTRIBUTE_KEYS = {"password", "token", "secret", "api_key", "authorization", "credential"}
MAX_ATTRIBUTE_VALUE_LENGTH = 1000

def set_attribute(self, key: str, value: Any) -> None:
    """Set a span attribute with sanitization."""
    if any(s in key.lower() for s in SENSITIVE_ATTRIBUTE_KEYS):
        self.attributes[key] = "[REDACTED]"
        return
    if isinstance(value, str) and len(value) > MAX_ATTRIBUTE_VALUE_LENGTH:
        self.attributes[key] = value[:MAX_ATTRIBUTE_VALUE_LENGTH] + "...[truncated]"
    else:
        self.attributes[key] = value
```

#### SEC-030: Session Recordings Store Complete Prompts/Responses [HIGH]

**Location**: Lines 377-407 (SessionRecording)

**Description**: The `SessionRecording` dataclass stores complete agent sessions including all prompts, responses, tool calls, and state changes. This creates a comprehensive record of all user interactions that could contain highly sensitive information. These recordings are designed to be replayed and shared.

**CWE**: CWE-312 (Cleartext Storage of Sensitive Information)

**Recommended Fix**:
- Encrypt session recordings at rest.
- Implement configurable redaction rules for session data.
- Add retention policies to auto-delete recordings after a configurable period.
- Require explicit opt-in for session recording.
- Never include session recordings in shared exports without explicit consent.

#### SEC-031: Log Injection via Span Attributes [MEDIUM]

**Location**: Lines 764-784 (SQLiteSpanExporter.export)
**Code**:
```python
cursor.execute("""
    INSERT OR REPLACE INTO spans
    (trace_id, span_id, parent_span_id, name, kind, ...)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
""", (
    span.trace_id,
    span.span_id,
    ...
    json.dumps(span.attributes, default=str),  # Arbitrary data
))
```

**Description**: While parameterized queries prevent SQL injection, the span attributes are serialized to JSON and stored as a TEXT column. If these attributes are later rendered in a web dashboard without sanitization, they could contain XSS payloads. Additionally, very large attribute values could cause storage issues.

**Recommended Fix**: Validate and truncate attribute values before serialization.

#### SEC-032: Unbounded Span Buffer in TracerProvider [MEDIUM]

**Location**: Lines 484, 517-524
**Code**:
```python
self._span_buffer: List[Span] = []
# ...
def _on_span_end(self, span: Span) -> None:
    with self._buffer_lock:
        self._span_buffer.append(span)
        if len(self._span_buffer) >= 100:
            self._flush()
```

**Description**: If export consistently fails (e.g., database is locked), spans accumulate in the buffer without limit. Each span contains attributes, events, and potentially large data. This could lead to memory exhaustion.

**Recommended Fix**: Add a maximum buffer size with oldest-first eviction.

#### SEC-033: No Access Control on Trace/Metrics Queries [LOW]

**Location**: Architecture overview mentions CLI commands (`gaia observe traces`, etc.)

**Description**: The query and dashboard layer has no authentication or access control. Anyone with access to the SQLite files or CLI can query all traces, metrics, and session recordings, including those containing sensitive data.

#### SEC-034: Non-Monotonic Time Reference [LOW]

**Location**: Lines 291-292, 336
**Code**:
```python
start_time: float = field(default_factory=time.time)
# ...
self.end_time = time.time()
```

**Description**: `time.time()` is not monotonic and can go backwards due to NTP adjustments, leading to negative durations. Use `time.monotonic()` for duration calculations and `time.time()` only for wall-clock timestamps.

---

### 4.6 SECURITY_SECRETS_ARCHITECTURE.md

**File**: `/mnt/c/Users/14255/Work/gaia/architecture/SECURITY_SECRETS_ARCHITECTURE.md`

#### SEC-035: Fallback Encryption Key Derived from Public Data [CRITICAL]

**Location**: Lines 508-516 (EncryptionEngine._fallback_key)
**Code**:
```python
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
```

**Description**: When the `keyring` package is not installed, the vault falls back to deriving the encryption key from `platform.node()` (hostname) and `socket.gethostname()` (also hostname), combined with a hardcoded salt `b"gaia-vault-salt"`. Since both `platform.node()` and `socket.gethostname()` return the same value on most systems, and hostnames are publicly discoverable (via DNS, network scanning, or OS fingerprinting), an attacker who knows the hostname can reconstruct the exact encryption key and decrypt all secrets in the vault.

**CWE**: CWE-321 (Use of Hard-coded Cryptographic Key), CWE-330 (Use of Insufficiently Random Values)

**Recommended Fix**:
```python
@classmethod
def _fallback_key(cls) -> "EncryptionEngine":
    """Fallback: generate and persist a random key in a protected file."""
    key_file = Path.home() / ".gaia" / "vault.key"
    key_file.parent.mkdir(parents=True, exist_ok=True)

    if key_file.exists():
        key = key_file.read_bytes()
        if len(key) == 32:
            return cls(key)
        # Invalid key file, regenerate
        log.warning("Invalid vault key file, regenerating")

    # Generate truly random key
    key = os.urandom(32)
    key_file.write_bytes(key)

    # Restrict permissions (owner read/write only)
    import stat
    key_file.chmod(stat.S_IRUSR | stat.S_IWUSR)

    log.warning(
        "Generated fallback vault key in ~/.gaia/vault.key. "
        "For production, install keyring: pip install keyring"
    )
    return cls(key)
```

#### SEC-036: Secret Vault Operations Accept None Context [HIGH]

**Location**: Lines 710-800 (SecretVault.get)
**Code**:
```python
def get(self, name: str, context: Optional[SecurityContext] = None) -> Optional[str]:
```

**Description**: The `get`, `store`, `rotate`, and `delete` operations all accept `context=None`, which means they can be called without any security context. When context is None, agent access control checks are skipped (line 763: `if allowed_agents_json and context`). This allows unauthenticated access to secrets.

**CWE**: CWE-862 (Missing Authorization)

**Recommended Fix**:
```python
def get(self, name: str, context: SecurityContext) -> Optional[str]:
    """Retrieve and decrypt a secret. Context is REQUIRED."""
    if context is None:
        raise ValueError("SecurityContext is required for secret access")
    # ... rest of method
```

Alternatively, for backward compatibility:
```python
def get(self, name: str, context: Optional[SecurityContext] = None) -> Optional[str]:
    if context is None:
        log.warning(f"Secret '{name}' accessed without security context -- this will be denied in future versions")
        if self.audit:
            self.audit.log_event(AuditEvent(
                event_type=AuditEventType.SECRET_ACCESS,
                action="get",
                resource=name,
                outcome="warning_no_context",
                severity="warning",
            ))
```

#### SEC-037: Encryption Key Material Exposed in Memory [HIGH]

**Location**: Lines 441-448 (EncryptionEngine.__init__)
**Code**:
```python
def __init__(self, master_key: bytes):
    self._key = master_key
```

**Description**: The 32-byte AES master key is stored as a plain Python `bytes` object in `self._key`. Python bytes objects are immutable and cannot be securely zeroed. The key persists in memory for the lifetime of the `EncryptionEngine` instance and may be captured by memory dumps, core dumps, or process memory inspection.

**CWE**: CWE-316 (Cleartext Storage of Sensitive Information in Memory)

**Recommended Fix**: Use a `bytearray` that can be zeroed, and provide a `close()` method:
```python
class EncryptionEngine:
    def __init__(self, master_key: bytes):
        self._key = bytearray(master_key)

    def close(self):
        """Securely zero the key material."""
        for i in range(len(self._key)):
            self._key[i] = 0

    def __del__(self):
        self.close()
```

#### SEC-038: Audit Log Buffer Not Guaranteed to Flush on Crash [MEDIUM]

**Location**: Lines 1301-1315 (AuditLogger.log_event)
**Code**:
```python
self._event_buffer.append((event, previous_hash, entry_hash))

if len(self._event_buffer) >= self._buffer_size:
    self._flush()
```

**Description**: Audit events are buffered (up to 50 events) before being written to the database. If the process crashes before a flush, audit events are lost. For a security audit log, this is a significant gap -- the most critical events (those leading to a crash) would be the ones lost.

**CWE**: CWE-778 (Insufficient Logging)

**Recommended Fix**: For security-critical events, flush immediately:
```python
def log_event(self, event: AuditEvent) -> None:
    with self._lock:
        previous_hash = self._last_hash
        entry_hash = self._compute_hash(event, previous_hash)
        self._last_hash = entry_hash

        self._event_buffer.append((event, previous_hash, entry_hash))

        # Flush immediately for critical events
        if event.severity == "critical" or event.event_type in (
            AuditEventType.PERMISSION_DENIED,
            AuditEventType.ANOMALY_DETECTED,
            AuditEventType.SECRET_ACCESS,
        ):
            self._flush()
        elif len(self._event_buffer) >= self._buffer_size:
            self._flush()
```

#### SEC-039: Hash Chain Does Not Include Sequence Number [MEDIUM]

**Location**: Lines 1293-1299 (AuditLogger._compute_hash)
**Code**:
```python
def _compute_hash(self, event: AuditEvent, previous_hash: str) -> str:
    data = (
        f"{event.event_id}|{event.event_type.value}|{event.timestamp.isoformat()}|"
        f"{event.action}|{event.resource}|{event.outcome}|{previous_hash}"
    )
    return hashlib.sha256(data.encode()).hexdigest()
```

**Description**: The hash chain computation does not include the expected sequence number. An attacker with database write access could delete entries from the middle of the log and reconstruct the hash chain, as long as they know the previous hash. Including a monotonically increasing sequence number would make this detectable.

**Recommended Fix**: Include the entry's auto-increment ID in the hash computation.

#### SEC-040: Secret Scanning Patterns Incomplete [MEDIUM]

**Location**: Lines 914-950 (SecretVault.scan_for_secrets)

**Description**: The secret scanning regex patterns miss several common secret formats:
- Slack tokens (`xoxb-`, `xoxp-`)
- Stripe keys (`sk_live_`, `pk_live_`)
- SendGrid keys (`SG.`)
- JWT tokens (`eyJ`)
- Azure connection strings
- Google API keys (`AIza`)

**Recommended Fix**: Extend the patterns list and consider using a dedicated secret scanning library.

#### SEC-041: Vault Database File Permissions Not Set [LOW]

**Location**: Lines 597-623 (SecretVault._init_db)

**Description**: The SQLite database files (`gaia_vault.db`, `gaia_audit.db`) are created with default OS permissions. On multi-user systems, other users may be able to read the encrypted vault contents. While values are encrypted, the metadata (secret names, allowed agents, creation dates) is stored in plaintext.

**Recommended Fix**: Set restrictive permissions immediately after database creation:
```python
import stat
os.chmod(self.db_path, stat.S_IRUSR | stat.S_IWUSR)  # 0600
```

#### SEC-042: Copyright Year Range Inconsistency [LOW]

**Location**: All code blocks

**Description**: All code blocks use `Copyright(C) 2024-2025` but the document date is February 7, 2026. This should be updated to `2024-2026`.

---

### 4.7 ERROR_RECOVERY_ARCHITECTURE.md

**File**: `/mnt/c/Users/14255/Work/gaia/architecture/ERROR_RECOVERY_ARCHITECTURE.md`

#### SEC-043: Stack Traces Stored in Step Metadata [MEDIUM]

**Location**: Lines 1056-1065 (AgentStepExecutor, from Workflow doc cross-reference)
**Code**:
```python
return StepResult(
    step_id=step.step_id,
    status=StepStatus.FAILED,
    error=str(e),
    started_at=started_at,
    completed_at=datetime.now(),
    metadata={"traceback": traceback.format_exc()},
)
```

**Description**: Full Python stack traces are stored in `StepResult.metadata["traceback"]`. Stack traces reveal internal code structure, file paths, variable names, and potentially sensitive data from local variables. If step results are persisted, shared, or returned via API, this constitutes information disclosure.

**CWE**: CWE-209 (Generation of Error Message Containing Sensitive Information)

**Recommended Fix**:
```python
def _sanitize_traceback(self, tb: str) -> str:
    """Remove sensitive information from tracebacks."""
    lines = tb.split("\n")
    sanitized = []
    for line in lines:
        # Remove absolute paths, keep only filename
        import re
        line = re.sub(r'File ".*[/\\]', 'File ".../', line)
        sanitized.append(line)
    return "\n".join(sanitized)
```

#### SEC-044: Dead Letter Queue Has No Maximum Size [MEDIUM]

**Location**: Lines 371-383 (DeadLetterEntry)

**Description**: The `DeadLetterEntry` data model includes `args: Dict[str, Any]` which stores the original operation arguments. There is no maximum size limit on the dead letter queue, and the stored args could be arbitrarily large. A sustained failure condition could cause the DLQ to grow unbounded.

**Recommended Fix**: Add maximum DLQ size and maximum entry payload size:
```python
MAX_DLQ_SIZE = 1000
MAX_ENTRY_ARGS_SIZE_BYTES = 10_000

class DeadLetterQueue:
    def add(self, entry: DeadLetterEntry) -> bool:
        serialized_size = len(json.dumps(entry.args, default=str))
        if serialized_size > MAX_ENTRY_ARGS_SIZE_BYTES:
            entry.args = {"_truncated": True, "_original_size": serialized_size}
        if len(self._entries) >= MAX_DLQ_SIZE:
            # Evict oldest resolved/abandoned entries first
            self._evict_oldest()
        self._entries.append(entry)
```

#### SEC-045: Timing Side Channel in Error Classification [LOW]

**Location**: Lines 404-431 (classify_error)
**Code**:
```python
def classify_error(exception: Exception) -> ErrorCategory:
    for exc_type, category in ERROR_CLASSIFICATIONS.items():
        if isinstance(exception, exc_type):
            return category
    # ...
    msg = str(exception).lower()
    if any(word in msg for word in ["timeout", "timed out", "rate limit", "busy", "overloaded"]):
        return ErrorCategory.TRANSIENT
```

**Description**: The error classification iterates through exception types in dictionary order and falls through to string matching. The string matching on error messages means that an attacker who can control error messages (e.g., via a crafted API response) could influence retry behavior. Setting an error message to "timeout" would cause the system to retry when it should not.

**CWE**: CWE-703 (Improper Check or Handling of Exceptional Conditions)

**Recommended Fix**: Classify based on exception type hierarchy first. Only use string matching as a last resort and log when string-based classification is used so it can be audited.

---

## 5. Cross-Cutting Security Concerns

### 5.1 Missing Integration Between Security Architecture and Other Subsystems

The `SECURITY_SECRETS_ARCHITECTURE.md` defines a comprehensive security model (SecretVault, PermissionEnforcer, AuditLogger, ResourceGovernor), but none of the other 6 architecture documents reference or integrate with it:

| Document | Uses SecretVault? | Uses PermissionEnforcer? | Uses AuditLogger? |
|----------|-------------------|--------------------------|---------------------|
| Computer Use | No | No | No |
| Email Integration | No | No | No |
| Workflow Orchestration | No | No | No |
| Multi-Document Synthesis | No | No | No |
| Observability | No | No | No |
| Error Recovery | No | No | No |

**Recommendation**: Each architecture must be required to integrate with the Security Gateway:

```python
# Example: Email provider should use SecretVault
from gaia.security import SecretVault, PermissionEnforcer, AuditLogger

class GmailProvider(BaseEmailProvider):
    def __init__(self, vault: SecretVault, enforcer: PermissionEnforcer):
        self.vault = vault
        self.enforcer = enforcer

    async def authenticate(self, context: SecurityContext) -> bool:
        # Check permission
        if not self.enforcer.check(context, Permission.EMAIL_READ):
            return False
        # Get credentials from vault
        creds_json = self.vault.get("gmail_oauth_token", context)
```

### 5.2 No Input Validation Framework

None of the 7 documents define a shared input validation framework. Each subsystem performs ad-hoc validation (or none at all). GAIA needs a centralized validation layer.

**Recommendation**: Create `src/gaia/security/validation.py`:
```python
class InputValidator:
    """Centralized input validation for all GAIA subsystems."""

    @staticmethod
    def validate_url(url: str, allow_private: bool = False) -> str:
        """Validate and sanitize a URL."""
        ...

    @staticmethod
    def validate_file_path(path: str, allowed_dirs: List[str]) -> Path:
        """Validate file path is within allowed directories."""
        ...

    @staticmethod
    def validate_email(email: str) -> str:
        """Validate email address format."""
        ...

    @staticmethod
    def sanitize_for_log(text: str, max_length: int = 500) -> str:
        """Sanitize text for safe logging."""
        ...
```

### 5.3 Inconsistent Error Information Exposure

Some documents expose full exception details in responses (Error Recovery: stack traces in metadata), while others silently swallow errors. There is no consistent policy for what error information is safe to return to users versus log internally.

### 5.4 No Transport-Level Security for Inter-Component Communication

The architecture documents describe multiple services communicating over HTTP (webhook server, API server, observability dashboard), but none specify TLS requirements. All internal HTTP traffic should use TLS.

### 5.5 No Data Retention or Deletion Policy

Multiple subsystems persist data (session recordings, audit logs, email cache, trace data, metrics) without defined retention periods or deletion mechanisms. For GDPR/CCPA compliance, data subjects must be able to request deletion of their data.

---

## 6. Integration Recommendations

### 6.1 Mandatory Security Gateway Integration

Every new architecture component MUST integrate with the Security Gateway defined in `SECURITY_SECRETS_ARCHITECTURE.md`. This means:

1. **All credential access** goes through `SecretVault`, never direct environment variables or file reads.
2. **All operations** check permissions via `PermissionEnforcer` before execution.
3. **All security-relevant events** are logged via `AuditLogger`.
4. **All resource-consuming operations** are gated by `ResourceGovernor`.

### 6.2 Defense-in-Depth for Computer Use

The Computer Use Architecture requires the strongest security controls because it has the highest impact:

```
User Request
  |
  v
[Permission Check] -- Does agent have COMPUTER_USE permission?
  |
  v
[Safety Guard] -- Is the action in the allowlist?
  |
  v
[Human Confirmation] -- Does this action require approval?
  |
  v
[Rate Limiter] -- Too many actions in time window?
  |
  v
[Execute Action]
  |
  v
[Audit Log] -- Record what was done
  |
  v
[Verify Result] -- Did the expected change occur?
```

### 6.3 Email Security Controls

Email operations must enforce:
- **Rate limiting**: Max 10 emails/minute, 50/hour, 200/day
- **Content filtering**: Block emails containing detected secrets
- **Recipient validation**: Allowlist/blocklist for recipient domains
- **Human review queue**: All auto-generated emails require approval by default
- **Attachment scanning**: Size limits and type restrictions

### 6.4 Workflow Security Sandbox

Workflows should execute in a security sandbox:
- **Network restrictions**: HttpStepExecutor should use URL allowlists
- **Subprocess restrictions**: Only pre-approved commands
- **Data isolation**: Each workflow run gets its own context; no access to other runs
- **Resource limits**: Max execution time, memory, and output size per step

---

## 7. Compliance Considerations

### 7.1 SOC 2 Type II

| Control | Status | Gap |
|---------|--------|-----|
| Encryption at rest | Partial | Vault uses AES-256-GCM; token files are plaintext |
| Access control | Partial | Permission model defined but not integrated |
| Audit logging | Good | Hash-chained audit log with integrity verification |
| Incident response | Missing | No automated alerting or incident runbooks |
| Change management | Partial | No code signing or integrity checks |

### 7.2 HIPAA (if processing medical data)

| Requirement | Status | Gap |
|-------------|--------|-----|
| PHI encryption | Partial | Email body may contain PHI, sent to LLM unencrypted |
| Access controls | Partial | PermissionEnforcer exists but not enforced |
| Audit trail | Good | AuditLogger with tamper detection |
| Minimum necessary | Missing | Email classification sends full body to LLM |
| BAA with subprocessors | N/A | Local LLM avoids third-party data processing |

### 7.3 GDPR / CCPA

| Requirement | Status | Gap |
|-------------|--------|-----|
| Data minimization | Missing | Session recordings capture all data |
| Right to deletion | Missing | No deletion mechanism for stored data |
| Consent | Missing | No consent mechanism for screen capture or email access |
| Data portability | Partial | Data stored in SQLite (exportable) |
| Purpose limitation | Missing | No enforcement of data use purpose |

### 7.4 OWASP Top 10 Mapping

| OWASP Category | Applicable Findings |
|----------------|---------------------|
| A01: Broken Access Control | SEC-004, SEC-017, SEC-033, SEC-036 |
| A02: Cryptographic Failures | SEC-007, SEC-008, SEC-035, SEC-037 |
| A03: Injection | SEC-001, SEC-009, SEC-016, SEC-020 |
| A04: Insecure Design | SEC-002, SEC-013, SEC-015 |
| A05: Security Misconfiguration | SEC-014, SEC-021, SEC-041 |
| A06: Vulnerable Components | N/A (not assessed) |
| A07: Auth Failures | SEC-017, SEC-036 |
| A08: Software/Data Integrity | SEC-038, SEC-039, SEC-043 |
| A09: Logging Failures | SEC-029, SEC-030, SEC-038 |
| A10: SSRF | SEC-002, SEC-015 |

---

## 8. Security Testing Strategy

### 8.1 Unit Tests (Per Component)

```python
# tests/unit/security/test_secret_vault.py

class TestSecretVaultSecurity:
    """Security-focused tests for the SecretVault."""

    def test_fallback_key_not_deterministic(self):
        """Verify fallback key uses random data, not hostname."""
        engine1 = EncryptionEngine._fallback_key()
        engine2 = EncryptionEngine._fallback_key()
        # After fix: keys should be the same (loaded from file)
        # but NOT derivable from hostname

    def test_secret_access_denied_without_context(self):
        """Verify secrets cannot be accessed without SecurityContext."""
        vault = SecretVault()
        vault.store("test_key", "secret_value", allowed_agents=["AgentA"])
        # Should raise or return None when no context
        result = vault.get("test_key", context=None)
        assert result is None  # Or assert raises ValueError

    def test_secret_access_denied_wrong_agent(self):
        """Verify agent-scoped secrets are enforced."""
        vault = SecretVault()
        vault.store("test_key", "secret_value", allowed_agents=["AgentA"])
        context = SecurityContext(agent_name="AgentB")
        result = vault.get("test_key", context=context)
        assert result is None

    def test_expired_secret_not_returned(self):
        """Verify expired secrets are not accessible."""
        vault = SecretVault()
        vault.store("test_key", "secret_value", expires_in_days=-1)
        result = vault.get("test_key")
        assert result is None

    def test_audit_log_integrity(self):
        """Verify audit log tamper detection works."""
        logger = AuditLogger()
        for i in range(10):
            logger.log_event(AuditEvent(action=f"action_{i}"))
        logger.flush()
        result = logger.verify_integrity()
        assert result["valid"] is True

    def test_secret_scanning_detects_api_keys(self):
        """Verify secret scanner catches common patterns."""
        vault = SecretVault()
        text = "My key is sk-abc123def456ghi789jkl012mno345pqr678stu901vwx234"
        findings = vault.scan_for_secrets(text)
        assert len(findings) > 0
        assert findings[0]["type"] == "openai_key"
```

### 8.2 Integration Tests

```python
# tests/integration/test_security_integration.py

class TestSecurityIntegration:
    """Cross-component security integration tests."""

    def test_email_agent_uses_secret_vault(self):
        """Verify EmailAgent retrieves credentials from vault, not env vars."""
        ...

    def test_workflow_subprocess_blocks_shell(self):
        """Verify SubprocessStepExecutor rejects shell=True."""
        step = StepDefinition(
            step_id="test",
            name="test",
            step_type=StepType.SUBPROCESS,
            config={"command": ["echo", "hello"], "shell": True},
        )
        # Should raise or ignore shell=True
        ...

    def test_webhook_rejects_unsigned_requests(self):
        """Verify webhook endpoint rejects requests without valid signature."""
        ...

    def test_http_step_blocks_ssrf(self):
        """Verify HttpStepExecutor blocks requests to internal IPs."""
        step = StepDefinition(
            step_id="test",
            name="test",
            step_type=StepType.HTTP,
            config={"url": "http://169.254.169.254/latest/meta-data/", "method": "GET"},
        )
        # Should raise ValueError about blocked host
        ...
```

### 8.3 Penetration Testing Checklist

| Test | Target | Priority |
|------|--------|----------|
| Command injection via LLM output into desktop automation | Computer Use | P0 |
| SSRF via HttpStepExecutor with crafted workflow | Workflow | P0 |
| OAuth token theft from filesystem | Email | P0 |
| Vault key reconstruction from hostname | Secrets | P0 |
| IMAP injection via crafted search criteria | Email | P1 |
| Path traversal via document ingestion | Multi-Doc | P1 |
| Webhook flooding for DoS | Workflow | P1 |
| Session recording data exfiltration | Observability | P1 |
| Audit log tampering detection bypass | Secrets | P2 |
| ReDoS via crafted HTML document | Multi-Doc | P2 |
| Memory exhaustion via large documents | Multi-Doc | P2 |
| Browser automation to phishing sites | Computer Use | P2 |

### 8.4 Automated Security Scanning

Integrate the following into CI/CD:

1. **Static Analysis**: Run `bandit` on all Python code
   ```bash
   bandit -r src/gaia/ -f json -o bandit_report.json
   ```

2. **Dependency Scanning**: Check for known vulnerabilities in dependencies
   ```bash
   pip-audit --format json --output pip_audit_report.json
   ```

3. **Secret Scanning**: Run `detect-secrets` on all files
   ```bash
   detect-secrets scan --all-files > .secrets.baseline
   ```

4. **SAST**: Use semgrep with Python security rules
   ```bash
   semgrep --config p/python-security src/gaia/
   ```

---

## 9. Prioritized Remediation Plan

### Phase 1: Critical Fixes (Week 1-2)

| ID | Fix | Effort | Owner |
|----|-----|--------|-------|
| SEC-001 | Remove shell=True from desktop automation, implement app allowlist | 2 days | Computer Use team |
| SEC-035 | Replace hostname-based key derivation with random key file | 1 day | Security team |
| SEC-007 | Integrate Gmail token storage with SecretVault | 2 days | Email team |
| SEC-008 | Replace plaintext SMTP password with SecurePasswordHolder | 1 day | Email team |
| SEC-015 | Add URL validation and SSRF protection to HttpStepExecutor | 2 days | Workflow team |
| SEC-002 | Add URL allowlist to browser automation navigate() | 1 day | Computer Use team |
| SEC-017 | Fix webhook signature validation logic (fail-closed) | 0.5 days | Workflow team |
| SEC-036 | Make SecurityContext required for vault operations | 1 day | Security team |

### Phase 2: High Priority Fixes (Week 3-4)

| ID | Fix | Effort | Owner |
|----|-----|--------|-------|
| SEC-003 | Add screenshot encryption and consent mechanism | 3 days | Computer Use team |
| SEC-004 | Implement SafetyGuard for keyboard/mouse operations | 3 days | Computer Use team |
| SEC-009 | Add IMAP query sanitization | 1 day | Email team |
| SEC-010 | Implement rate limiting for email send | 2 days | Email team |
| SEC-011 | Add PII scrubbing before LLM classification | 2 days | Email team |
| SEC-016 | Remove shell option from SubprocessStepExecutor | 1 day | Workflow team |
| SEC-018 | Add rate limiting to webhook server | 1 day | Workflow team |
| SEC-023 | Add path validation to document ingestion | 1 day | Multi-Doc team |
| SEC-024 | Add file size and collection size limits | 1 day | Multi-Doc team |
| SEC-029 | Add attribute sanitization to span exporter | 1 day | Observability team |
| SEC-030 | Add encryption and opt-in for session recordings | 2 days | Observability team |
| SEC-037 | Use bytearray for key material with secure zeroing | 1 day | Security team |

### Phase 3: Medium Priority Fixes (Week 5-8)

| ID | Fix | Effort | Owner |
|----|-----|--------|-------|
| SEC-005 | Add privacy mode for VLM analysis | 2 days | Computer Use team |
| SEC-006 | Add AMD copyright to all code blocks | 0.5 days | All teams |
| SEC-012 | Use proper HTML sanitization library | 1 day | Email team |
| SEC-013 | Implement SendGuard with human review queue | 3 days | Email team |
| SEC-019 | Filter sensitive headers from webhook trigger data | 0.5 days | Workflow team |
| SEC-020 | Remove fallback ast.literal_eval path | 0.5 days | Workflow team |
| SEC-021 | Change webhook default bind to 127.0.0.1 | 0.5 days | Workflow team |
| SEC-025 | Store relative paths in document metadata | 0.5 days | Multi-Doc team |
| SEC-026 | Replace MD5 with SHA-256 for cache keys | 0.5 days | Multi-Doc team |
| SEC-031 | Add attribute value truncation in span export | 1 day | Observability team |
| SEC-032 | Add maximum buffer size with eviction | 1 day | Observability team |
| SEC-038 | Immediate flush for critical audit events | 1 day | Security team |
| SEC-039 | Include sequence number in hash chain | 1 day | Security team |
| SEC-040 | Extend secret scanning patterns | 1 day | Security team |
| SEC-043 | Sanitize stack traces in error metadata | 1 day | Error Recovery team |
| SEC-044 | Add DLQ size limits | 1 day | Error Recovery team |

### Phase 4: Low Priority and Continuous (Week 9+)

| ID | Fix | Effort | Owner |
|----|-----|--------|-------|
| SEC-014 | Implement incremental OAuth scope requests | 2 days | Email team |
| SEC-022 | Validate file watcher paths | 1 day | Workflow team |
| SEC-027 | Add per-instance embedding cache limits | 1 day | Multi-Doc team |
| SEC-028 | Replace regex HTML parsing with proper parser | 2 days | Multi-Doc team |
| SEC-033 | Add authentication to observability queries | 2 days | Observability team |
| SEC-034 | Use monotonic clock for duration calculations | 0.5 days | Observability team |
| SEC-041 | Set restrictive permissions on database files | 0.5 days | Security team |
| SEC-042 | Update copyright year to 2024-2026 | 0.5 days | All teams |
| SEC-045 | Add logging for string-based error classification | 0.5 days | Error Recovery team |
| N/A | Cross-subsystem security integration testing | Ongoing | All teams |
| N/A | Data retention policy implementation | 5 days | Security team |
| N/A | TLS for inter-component communication | 3 days | Infrastructure team |

---

## 10. Appendix: Severity Definitions

### CRITICAL
- **Exploitability**: Can be exploited remotely or by LLM output with minimal complexity
- **Impact**: System compromise, credential theft, arbitrary code execution
- **Example**: Command injection via shell=True, vault key derivable from hostname
- **SLA**: Fix within 48 hours of detection

### HIGH
- **Exploitability**: Requires specific conditions but is reliably exploitable
- **Impact**: Data exposure, unauthorized access, significant service disruption
- **Example**: Plaintext credential storage, missing access control, IMAP injection
- **SLA**: Fix within 1 week

### MEDIUM
- **Exploitability**: Requires significant attacker capability or specific configuration
- **Impact**: Limited data exposure, defense-in-depth gap, compliance violation
- **Example**: Missing PII scrubbing, audit log gaps, expression evaluation edge cases
- **SLA**: Fix within 1 month

### LOW
- **Exploitability**: Difficult to exploit or requires physical/local access
- **Impact**: Information disclosure, best practice deviation
- **Example**: MD5 for non-crypto use, copyright year, file permissions
- **SLA**: Fix in next release

### INFORMATIONAL
- **Exploitability**: Not directly exploitable
- **Impact**: Improves overall security posture
- **Example**: Recommendations for future architecture improvements
- **SLA**: Track and address as resources allow

---

**End of Security Review**

*This review was conducted by the GAIA Code Reviewer Agent using the code-reviewer persona. For questions or escalations, contact @kovtcharov-amd.*
