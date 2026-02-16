# Category Enablement Gaps - Missing Architectures

**Date**: February 7, 2026
**Purpose**: Identify missing architectures needed to enable four core agent categories
**Scope**: Gaps beyond existing 18 architecture documents

---

## Target Categories

1. **Workflow Automation** - Email triage, scheduling, task orchestration
2. **Coding Assistance** - Code analysis, debugging, generation (mostly covered)
3. **Computer Use Agents** - UI automation, desktop control, system interaction
4. **Knowledge Assistants** - Large document analysis, research, synthesis

---

## Coverage Analysis

### ✅ Well Covered (Existing Docs)

| Category | Coverage | Existing Docs |
|----------|----------|---------------|
| **Coding Assistance** | 85% | LSP integration, manifest, state machine, tools, memory, RAG |
| **Knowledge Assistants** | 70% | RAG, memory, task-centric, learning loops |
| **Workflow Automation** | 40% | Task queue, continuous execution, state machine |
| **Computer Use** | 10% | (Only basic bash execution) |

### ❌ Critical Gaps

---

## 1. Computer Use Agents Architecture ★★★★★

**Impact**: CRITICAL for computer automation category
**Urgency**: High (missing entire category)
**Complexity**: Very High (8-12 weeks)
**Dependencies**: None (new subsystem)

### The Problem

GAIA has **no architecture** for computer use agents that need to:
- Understand visual UI (screenshots, screen regions)
- Control mouse/keyboard
- Navigate applications
- Click buttons, fill forms
- Verify UI state
- Handle multi-monitor setups

**Current state**: Agents can only interact via:
- Bash commands (blind execution)
- File operations
- API calls

**Can't do**: "Open Chrome, navigate to Gmail, triage my emails" or "Click the 'Submit' button in the form"

### Required Components

#### 1.1 Screen Capture & Vision System

```python
class ScreenCaptureAgent:
    """Capture and understand screen content."""

    def capture_screen(self, monitor: int = 0) -> Image:
        """Capture screenshot of specified monitor."""
        pass

    def capture_region(self, x: int, y: int, width: int, height: int) -> Image:
        """Capture specific screen region."""
        pass

    def find_element_by_image(self, template: Image) -> Tuple[int, int]:
        """Find UI element using image template matching."""
        pass

    def find_element_by_text(self, text: str, ocr_engine: str = "tesseract") -> List[Tuple[int, int]]:
        """Find text on screen using OCR."""
        pass

    def describe_screen(self, vision_model: VLM) -> str:
        """Describe current screen content using VLM."""
        pass

    def find_clickable_elements(self, vision_model: VLM) -> List[Dict]:
        """Detect buttons, links, inputs using VLM."""
        # Returns: [{"type": "button", "text": "Submit", "bbox": (x,y,w,h)}, ...]
        pass
```

**Technologies**:
- Screenshot: `mss`, `pyautogui`, `Pillow`
- OCR: `tesseract`, `easyocr`, `PaddleOCR`
- Vision: GAIA's VLM support (Qwen2.5-VL)
- Template matching: OpenCV

#### 1.2 Input Control System

```python
class InputController:
    """Control mouse and keyboard."""

    # Mouse operations
    def move_mouse(self, x: int, y: int, duration: float = 0.5):
        """Move mouse to position smoothly."""
        pass

    def click(self, x: int, y: int, button: str = "left", clicks: int = 1):
        """Click at position."""
        pass

    def drag(self, start_x: int, start_y: int, end_x: int, end_y: int):
        """Drag from start to end."""
        pass

    def scroll(self, clicks: int, direction: str = "down"):
        """Scroll mouse wheel."""
        pass

    # Keyboard operations
    def type_text(self, text: str, interval: float = 0.05):
        """Type text with human-like timing."""
        pass

    def press_key(self, key: str):
        """Press keyboard key (e.g., 'enter', 'ctrl+c')."""
        pass

    def hotkey(self, *keys: str):
        """Press key combination (e.g., 'ctrl', 'shift', 's')."""
        pass
```

**Technologies**:
- Cross-platform: `pynput`, `pyautogui`
- macOS: `pyobjc` for native APIs
- Windows: `pywin32`, `ctypes`
- Linux: `python-xlib`, `evdev`

#### 1.3 Browser Automation Framework

```python
class BrowserAutomationAgent:
    """High-level browser automation (headless or visible)."""

    def __init__(self, browser: str = "chromium", headless: bool = False):
        # Uses Playwright or Selenium
        pass

    async def navigate(self, url: str):
        """Navigate to URL."""
        pass

    async def click_element(self, selector: str):
        """Click element by CSS selector."""
        pass

    async def fill_form(self, selector: str, value: str):
        """Fill input field."""
        pass

    async def wait_for_element(self, selector: str, timeout: float = 30):
        """Wait for element to appear."""
        pass

    async def extract_text(self, selector: str) -> str:
        """Extract text from element."""
        pass

    async def take_screenshot(self, path: str = None) -> bytes:
        """Screenshot of current page."""
        pass

    async def evaluate_js(self, script: str) -> Any:
        """Execute JavaScript in browser context."""
        pass
```

**Technologies**:
- Modern: `playwright` (best for AI agents)
- Alternative: `selenium`
- Lightweight: `requests-html`, `httpx` + BeautifulSoup

#### 1.4 Desktop Application Automation

```python
class DesktopAutomationAgent:
    """Automate native desktop applications."""

    def get_window_list(self) -> List[Dict]:
        """List all open windows."""
        # Returns: [{"pid": 1234, "title": "VSCode", "rect": (x,y,w,h)}, ...]
        pass

    def activate_window(self, title: str):
        """Bring window to foreground."""
        pass

    def get_ui_tree(self, window_title: str) -> Dict:
        """Get accessibility UI tree (macOS/Windows)."""
        # Uses: NSAccessibility (macOS), UI Automation (Windows), AT-SPI (Linux)
        pass

    def click_ui_element(self, window_title: str, element_name: str):
        """Click UI element by accessibility name."""
        pass

    def get_text_from_element(self, window_title: str, element_name: str) -> str:
        """Extract text from UI element."""
        pass
```

**Technologies**:
- macOS: `pyobjc` + NSAccessibility
- Windows: `pywinauto`, UI Automation
- Linux: AT-SPI (`pyatspi2`)
- Cross-platform: `accessibility-insights`

#### 1.5 Computer Use Safety Framework

```python
class ComputerUseSafetyGuard:
    """Safety guardrails for computer control."""

    def __init__(self, config: SafetyConfig):
        self.allowed_apps = config.allowed_apps
        self.forbidden_actions = config.forbidden_actions
        self.confirm_before = config.confirm_before
        pass

    def check_action(self, action: ComputerAction) -> Tuple[bool, str]:
        """
        Validate action is safe to execute.

        Returns: (is_safe, reason_if_not)
        """
        # Block: closing system apps, deleting files, running sudo, etc.
        pass

    def require_confirmation(self, action: ComputerAction) -> bool:
        """Check if action requires human confirmation."""
        # Confirm: file deletion, sending email, financial transactions
        pass

    def create_checkpoint(self):
        """Create system restore point (Windows/macOS)."""
        pass

    def rollback_to_checkpoint(self):
        """Restore system to checkpoint."""
        pass
```

**Safety rules**:
- ❌ Block: `rm -rf /`, `sudo` commands, system file modifications
- ⚠️ Confirm: Email sending, file deletion, financial forms, password entry
- ✅ Allow: Web browsing, document editing, data entry

### Integration Example

```python
from gaia.computer_use import ScreenCaptureAgent, InputController, BrowserAutomationAgent

class EmailTriageAgent(Agent):
    """Email triage using computer use."""

    def __init__(self):
        super().__init__()
        self.screen = ScreenCaptureAgent()
        self.input = InputController()
        self.browser = BrowserAutomationAgent()

    async def triage_gmail(self):
        """Open Gmail and triage emails."""

        # Open Gmail
        await self.browser.navigate("https://mail.google.com")
        await self.browser.wait_for_element("div[role='main']")

        # Find unread emails
        unread = await self.browser.evaluate_js("""
            Array.from(document.querySelectorAll('tr.zE')).map(row => ({
                sender: row.querySelector('.yW span').textContent,
                subject: row.querySelector('.y6 span').textContent,
                snippet: row.querySelector('.y2').textContent
            }))
        """)

        for email in unread:
            # Classify using LLM
            category = self.classify_email(email)

            if category == "spam":
                # Click checkbox, then delete button
                # ... (implementation)
                pass
            elif category == "urgent":
                # Star the email
                # ... (implementation)
                pass
            elif category == "newsletter":
                # Archive
                pass
```

---

## 2. Workflow Automation Architecture ★★★★☆

**Impact**: HIGH (enables email, calendar, automation category)
**Urgency**: Medium
**Complexity**: High (6-8 weeks)
**Dependencies**: Task queue, continuous execution

### The Problem

GAIA can execute tasks but lacks:
- **Triggers**: Cron-like scheduling, event-driven workflows
- **Connectors**: Email (IMAP/SMTP), Calendar (Google/Outlook), Webhooks
- **Orchestration**: Multi-step workflows with conditional logic
- **Notifications**: Push notifications, alerts, reminders

**Can't do**: "Check my email every hour and categorize urgent messages" or "When I get a calendar invite, automatically block focus time before the meeting"

### Required Components

#### 2.1 Email Integration Framework

```python
class EmailAgent:
    """Email reading, writing, and automation."""

    def __init__(self, provider: str = "gmail"):
        # Supports: Gmail API, Outlook Graph API, IMAP/SMTP
        pass

    def fetch_unread_emails(self, folder: str = "INBOX", limit: int = 50) -> List[Email]:
        """Fetch unread emails."""
        pass

    def classify_email(self, email: Email) -> Dict[str, Any]:
        """
        Classify email using LLM.

        Returns:
            {
                "category": "work" | "personal" | "spam" | "newsletter",
                "priority": "urgent" | "normal" | "low",
                "sentiment": "positive" | "neutral" | "negative",
                "action_required": bool,
                "summary": str
            }
        """
        pass

    def draft_reply(self, email: Email, context: str = None) -> str:
        """Generate email reply using LLM."""
        pass

    def send_email(self, to: str, subject: str, body: str, cc: List[str] = None):
        """Send email."""
        pass

    def move_to_folder(self, email_id: str, folder: str):
        """Move email to folder (Archive, Trash, Custom)."""
        pass

    def apply_label(self, email_id: str, label: str):
        """Apply label (Gmail) or category (Outlook)."""
        pass

    def auto_respond(self, email: Email, template: str = "out_of_office"):
        """Send auto-response."""
        pass
```

**Technologies**:
- Gmail: `google-api-python-client` (Gmail API)
- Outlook: `msal` + Microsoft Graph API
- Generic: `imaplib`, `smtplib` (IMAP/SMTP)
- Parsing: `email`, `beautifulsoup4` (HTML emails)

#### 2.2 Calendar Integration Framework

```python
class CalendarAgent:
    """Calendar reading, scheduling, and automation."""

    def __init__(self, provider: str = "google"):
        # Supports: Google Calendar, Outlook Calendar, iCal
        pass

    def get_events(self, start: datetime, end: datetime) -> List[Event]:
        """Fetch events in time range."""
        pass

    def find_free_slots(self, duration_minutes: int, start: datetime, end: datetime) -> List[datetime]:
        """Find available time slots."""
        pass

    def schedule_meeting(self, title: str, start: datetime, duration_minutes: int,
                        attendees: List[str] = None, description: str = None) -> Event:
        """Create calendar event."""
        pass

    def suggest_meeting_time(self, attendees: List[str], duration_minutes: int,
                            preferred_hours: Tuple[int, int] = (9, 17)) -> List[datetime]:
        """
        Suggest meeting times that work for all attendees.
        Uses LLM to analyze calendars and preferences.
        """
        pass

    def block_focus_time(self, hours_per_week: int = 10):
        """Automatically block focus time on calendar."""
        pass

    def auto_decline_conflicts(self, priority_keywords: List[str]):
        """Auto-decline meetings that conflict with high-priority events."""
        pass
```

**Technologies**:
- Google Calendar: `google-api-python-client`
- Outlook Calendar: Microsoft Graph API
- iCal: `icalendar`, `caldav`

#### 2.3 Workflow Orchestration Engine

```python
class WorkflowEngine:
    """Define and execute multi-step workflows with triggers."""

    def define_workflow(self, name: str, trigger: Trigger, actions: List[Action]):
        """
        Define workflow.

        Example:
            trigger = EmailReceivedTrigger(from_domain="important-client.com")
            actions = [
                ClassifyEmailAction(),
                ConditionalAction(
                    condition="priority == 'urgent'",
                    if_true=[
                        NotifySlackAction(channel="#urgent"),
                        CreateTaskAction(priority="high")
                    ],
                    if_false=[ArchiveEmailAction()]
                )
            ]
        """
        pass

    def schedule_workflow(self, workflow_name: str, cron: str):
        """Schedule workflow with cron expression."""
        # Example: "0 */1 * * *" = every hour
        pass

    def register_webhook(self, workflow_name: str, url: str):
        """Trigger workflow via webhook."""
        pass
```

**Example workflows**:
```python
# Workflow 1: Email triage every hour
workflow = WorkflowEngine()
workflow.define_workflow(
    name="email_triage",
    trigger=ScheduleTrigger(cron="0 */1 * * *"),  # Every hour
    actions=[
        FetchUnreadEmailsAction(limit=20),
        ClassifyEmailsAction(),
        ApplyLabelsAction(),
        ArchiveLowPriorityAction()
    ]
)

# Workflow 2: Meeting prep automation
workflow.define_workflow(
    name="meeting_prep",
    trigger=CalendarEventTrigger(minutes_before=60),  # 1 hour before meeting
    actions=[
        FetchMeetingContextAction(),  # Related emails, docs
        SummarizeContextAction(),  # LLM summary
        CreateBriefingDocAction(),  # Generate agenda
        NotifyAction(message="Meeting brief ready")
    ]
)

# Workflow 3: Weekly report generation
workflow.define_workflow(
    name="weekly_report",
    trigger=ScheduleTrigger(cron="0 9 * * FRI"),  # Every Friday 9am
    actions=[
        FetchTasksAction(completed_this_week=True),
        FetchEmailStatsAction(),
        FetchCalendarStatsAction(),
        GenerateReportAction(template="weekly_summary"),
        SendEmailAction(to="manager@company.com", subject="Weekly Report")
    ]
)
```

#### 2.4 Notification & Alert System

```python
class NotificationAgent:
    """Send notifications via multiple channels."""

    def notify_slack(self, channel: str, message: str, urgent: bool = False):
        """Send Slack notification."""
        pass

    def notify_email(self, to: str, subject: str, body: str):
        """Send email notification."""
        pass

    def notify_webhook(self, url: str, payload: Dict):
        """POST to webhook URL."""
        pass

    def notify_desktop(self, title: str, message: str):
        """Show desktop notification (macOS/Windows/Linux)."""
        pass

    def notify_mobile(self, device_token: str, message: str):
        """Send push notification to mobile (via Firebase, APNs)."""
        pass
```

**Technologies**:
- Slack: `slack_sdk`
- Desktop: `plyer`, `win10toast` (Windows), `pync` (macOS)
- Mobile: `firebase-admin` (FCM), `pyapns` (APNs)

---

## 3. Knowledge Assistant Enhancements ★★★★☆

**Impact**: HIGH (improves document analysis category)
**Urgency**: Medium
**Complexity**: Medium (4-6 weeks)
**Dependencies**: RAG (existing)

### Current State vs Gaps

✅ **Existing** (from current GAIA):
- RAG with PDF support
- Vector search (FAISS)
- Basic document Q&A

❌ **Missing**:
- Multi-document synthesis (compare/contrast across docs)
- Citation tracking (which doc, which page)
- Advanced document preprocessing (DOCX, PPT, Excel, images)
- Table extraction and analysis
- Hierarchical chunking for long documents
- Cross-reference detection

### Required Components

#### 3.1 Multi-Document Synthesis Engine

```python
class MultiDocumentSynthesizer:
    """Synthesize information across multiple documents."""

    def compare_documents(self, doc_ids: List[str], aspect: str) -> str:
        """
        Compare documents on specific aspect.

        Example: compare_documents([doc1, doc2, doc3], "pricing strategy")
        Returns: "Doc1 focuses on value-based pricing, while Doc2 recommends
                  cost-plus, and Doc3 suggests freemium model..."
        """
        pass

    def find_contradictions(self, doc_ids: List[str]) -> List[Dict]:
        """Find contradictory statements across documents."""
        # Returns: [{"doc1": doc_id, "doc2": doc_id, "statement1": str,
        #            "statement2": str, "contradiction": str}, ...]
        pass

    def synthesize_answer(self, query: str, doc_ids: List[str]) -> Dict:
        """
        Answer query using information from multiple documents.

        Returns:
            {
                "answer": str,
                "sources": [
                    {"doc_id": str, "doc_name": str, "page": int, "excerpt": str},
                    ...
                ],
                "confidence": float
            }
        """
        pass

    def create_summary_from_multiple(self, doc_ids: List[str], max_length: int = 500) -> str:
        """Create unified summary from multiple documents."""
        pass
```

#### 3.2 Citation Tracking System

```python
class CitationTracker:
    """Track which document/page information came from."""

    def extract_with_citations(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Extract passages with precise citations.

        Returns:
            [
                {
                    "text": "The quarterly revenue was $5.2M",
                    "source": {
                        "document": "Q4_2025_Report.pdf",
                        "page": 12,
                        "section": "Financial Summary",
                        "confidence": 0.95,
                        "bounding_box": {"page": 12, "x": 100, "y": 200, "w": 400, "h": 50}
                    }
                },
                ...
            ]
        """
        pass

    def format_citations(self, citations: List[Dict], style: str = "apa") -> str:
        """Format citations in academic style (APA, MLA, Chicago)."""
        pass

    def generate_bibliography(self, doc_ids: List[str]) -> str:
        """Generate bibliography from used documents."""
        pass
```

#### 3.3 Advanced Document Preprocessing

```python
class DocumentPreprocessor:
    """Extract and structure content from various document formats."""

    def process_docx(self, file_path: str) -> StructuredDocument:
        """Extract text, tables, images from DOCX."""
        # Preserves: headings, lists, tables, embedded images
        pass

    def process_pptx(self, file_path: str) -> StructuredDocument:
        """Extract slides, notes, images from PowerPoint."""
        # Per-slide extraction with speaker notes
        pass

    def process_excel(self, file_path: str) -> StructuredDocument:
        """Extract sheets, tables, charts from Excel."""
        # Each sheet becomes a section, tables preserved
        pass

    def process_image_pdf(self, file_path: str) -> StructuredDocument:
        """OCR scanned PDF."""
        # Uses Tesseract or PaddleOCR
        pass

    def extract_tables(self, file_path: str, format: str) -> List[pd.DataFrame]:
        """Extract tables from document as DataFrames."""
        # Uses: camelot, tabula, pdfplumber
        pass

    def extract_images(self, file_path: str) -> List[Tuple[Image, str]]:
        """
        Extract images with captions.

        Returns: [(image_data, caption), ...]
        """
        pass

    def detect_structure(self, text: str) -> Dict:
        """
        Detect document structure (sections, subsections).

        Returns:
            {
                "title": str,
                "sections": [
                    {"heading": str, "level": int, "content": str, "subsections": [...]},
                    ...
                ]
            }
        """
        pass
```

**Technologies**:
- DOCX: `python-docx`
- PPTX: `python-pptx`
- Excel: `openpyxl`, `pandas`
- PDF tables: `camelot-py`, `tabula-py`, `pdfplumber`
- OCR: `tesseract`, `easyocr`, `paddleocr`
- Images: `Pillow`, `pdf2image`

#### 3.4 Table Analysis Agent

```python
class TableAnalysisAgent:
    """Analyze and query structured tables."""

    def parse_table(self, table: pd.DataFrame) -> TableSchema:
        """Infer table schema and semantics."""
        # Detects: column types, units, relationships
        pass

    def query_table(self, table: pd.DataFrame, query: str) -> Dict:
        """
        Answer questions about table using LLM + SQL.

        Example: "What was the highest revenue month?"
        Generates SQL, executes, returns natural language answer.
        """
        pass

    def compare_tables(self, table1: pd.DataFrame, table2: pd.DataFrame) -> str:
        """Compare two tables and describe differences."""
        pass

    def visualize_table(self, table: pd.DataFrame, chart_type: str = "auto") -> Image:
        """Generate visualization from table data."""
        # Uses: matplotlib, seaborn, plotly
        pass
```

#### 3.5 Hierarchical Document Chunking

```python
class HierarchicalChunker:
    """Chunk long documents preserving structure."""

    def chunk_document(self, doc: StructuredDocument, strategy: str = "semantic") -> List[Chunk]:
        """
        Chunk document hierarchically.

        Strategies:
        - "semantic": Use section boundaries + semantic similarity
        - "sliding_window": Overlapping fixed-size chunks
        - "recursive": Split until max chunk size, preserve hierarchy
        """
        pass

    def create_chunk_index(self, chunks: List[Chunk]) -> ChunkIndex:
        """
        Create index with parent-child relationships.

        Enables: "Show me the section that contains information about X"
        """
        pass

    def retrieve_with_context(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieve chunks with surrounding context.

        Returns chunks + parent section + neighboring chunks
        """
        pass
```

---

## 4. Observability & Debugging Architecture ★★★★☆

**Impact**: HIGH (critical for production agents)
**Urgency**: Medium
**Complexity**: Medium (4-5 weeks)
**Dependencies**: Universal memory, execution provenance

### The Problem

GAIA agents are **black boxes** in production:
- Can't debug why agent made a decision
- Can't replay agent behavior
- Can't monitor performance metrics
- Can't trace errors across multi-step workflows
- Can't understand cost per operation

**Needed for**: Production deployment, debugging, cost optimization, compliance

### Required Components

#### 4.1 Distributed Tracing System

```python
class AgentTracer:
    """OpenTelemetry-compatible distributed tracing."""

    def start_span(self, name: str, attributes: Dict = None) -> Span:
        """Start a new trace span."""
        pass

    def end_span(self, span: Span, status: str = "ok"):
        """End span and record duration."""
        pass

    @contextmanager
    def trace_operation(self, operation: str, **attributes):
        """
        Context manager for automatic tracing.

        Usage:
            with tracer.trace_operation("tool_execution", tool_name="read_file"):
                result = read_file("data.txt")
        """
        pass

    def trace_llm_call(self, model: str, prompt: str, response: str,
                      tokens_used: int, duration: float):
        """Specialized tracing for LLM calls."""
        pass

    def export_trace(self, trace_id: str, format: str = "jaeger") -> str:
        """Export trace in standard format."""
        pass
```

**Technologies**:
- OpenTelemetry Python SDK
- Exporters: Jaeger, Zipkin, DataDog, Honeycomb
- Visualization: Jaeger UI, Grafana Tempo

#### 4.2 Metrics Collection System

```python
class AgentMetrics:
    """Prometheus-compatible metrics."""

    def record_counter(self, name: str, value: float = 1.0, labels: Dict = None):
        """Increment counter (e.g., total_tasks_completed)."""
        pass

    def record_histogram(self, name: str, value: float, labels: Dict = None):
        """Record histogram value (e.g., task_duration_seconds)."""
        pass

    def record_gauge(self, name: str, value: float, labels: Dict = None):
        """Set gauge value (e.g., active_tasks)."""
        pass

    def get_metrics_summary(self) -> Dict:
        """Get current metrics snapshot."""
        # Returns: {
        #     "total_tasks": 1234,
        #     "avg_task_duration_seconds": 45.2,
        #     "success_rate": 0.92,
        #     "total_llm_tokens": 1500000,
        #     "total_cost_usd": 125.50
        # }
        pass
```

**Key metrics**:
- **Task metrics**: total_tasks, successful_tasks, failed_tasks, task_duration
- **LLM metrics**: total_tokens, tokens_per_task, llm_calls, llm_latency
- **Cost metrics**: total_cost, cost_per_task, cost_by_model
- **Tool metrics**: tool_invocations, tool_failures, tool_duration
- **Memory metrics**: memory_searches, memory_size, retrieval_latency

**Technologies**:
- Prometheus Python client
- StatsD for metrics aggregation
- Grafana for visualization

#### 4.3 Time-Travel Debugging

```python
class AgentReplay:
    """Replay agent execution for debugging."""

    def record_execution(self, task_id: str):
        """Record all state changes during execution."""
        # Stores: LLM prompts, responses, tool calls, results, state transitions
        pass

    def replay_execution(self, task_id: str, stop_at_step: int = None):
        """
        Replay recorded execution.

        Can stop at specific step to inspect state.
        """
        pass

    def compare_executions(self, task_id1: str, task_id2: str) -> str:
        """
        Compare two execution traces.

        Shows: Where they diverged, different decisions made
        """
        pass

    def export_execution_trace(self, task_id: str, format: str = "json") -> str:
        """Export full execution trace for external analysis."""
        pass
```

**Use case**: "Why did the agent fail on this task but succeed on a similar one?"

---

## 5. Security & Secrets Management ★★★★☆

**Impact**: HIGH (required for production)
**Urgency**: Medium (needed before production deployment)
**Complexity**: Medium (3-4 weeks)

### The Problem

Agents need access to:
- API keys (OpenAI, Google Calendar, Slack, etc.)
- Credentials (email passwords, database credentials)
- OAuth tokens
- SSH keys

**Current state**: Hardcoded in code or env vars (insecure, not audited)

### Required Components

#### 5.1 Secrets Vault Integration

```python
class SecretsManager:
    """Secure secrets storage and retrieval."""

    def store_secret(self, name: str, value: str, metadata: Dict = None):
        """Store secret securely."""
        # Encrypted at rest, access logged
        pass

    def get_secret(self, name: str) -> str:
        """Retrieve secret."""
        # Logs access for audit
        pass

    def rotate_secret(self, name: str, new_value: str):
        """Rotate secret and invalidate old value."""
        pass

    def grant_access(self, secret_name: str, agent_id: str, duration: int = 3600):
        """Grant temporary access to secret."""
        # Auto-revoke after duration
        pass
```

**Technologies**:
- HashiCorp Vault
- AWS Secrets Manager
- Azure Key Vault
- Google Cloud Secret Manager
- Local: `keyring` (OS keychain integration)

#### 5.2 Permission Model

```python
class PermissionManager:
    """Define what agents can/cannot do."""

    def define_permissions(self, agent_id: str, permissions: List[Permission]):
        """
        Set agent permissions.

        Example:
            permissions = [
                FilePermission(allowed_paths=["/home/user/projects"], read=True, write=True),
                NetworkPermission(allowed_domains=["api.github.com"], protocols=["https"]),
                ToolPermission(allowed_tools=["read_file", "write_file"], forbidden=["bash"]),
                SecretsPermission(allowed_secrets=["github_token"])
            ]
        """
        pass

    def check_permission(self, agent_id: str, action: Action) -> Tuple[bool, str]:
        """Check if agent can perform action."""
        # Returns: (allowed, reason_if_denied)
        pass
```

---

## 6. Plugin & Extension System ★★★☆☆

**Impact**: MEDIUM (enables ecosystem)
**Urgency**: Low (nice-to-have)
**Complexity**: Medium (3-4 weeks)

### Required Components

```python
class PluginManager:
    """Load and manage third-party plugins."""

    def register_plugin(self, plugin_path: str):
        """Register plugin from .py file or package."""
        pass

    def load_plugins(self):
        """Load all registered plugins."""
        # Sandboxed execution for safety
        pass

    def get_plugin_tools(self, plugin_name: str) -> List[Tool]:
        """Get tools provided by plugin."""
        pass
```

**Plugin interface**:
```python
class GaiaPlugin:
    """Base class for GAIA plugins."""

    def register_tools(self) -> List[Tool]:
        """Return tools provided by this plugin."""
        pass

    def register_hooks(self) -> Dict[str, Callable]:
        """
        Register lifecycle hooks.

        Hooks:
        - on_task_start
        - on_tool_execute
        - on_state_change
        - on_task_complete
        """
        pass
```

---

## Priority Recommendations

### Immediate (Next 4 weeks)
1. **Computer Use Architecture** (Week 1-2)
   - Screen capture + OCR
   - Input control (mouse/keyboard)
   - Safety guardrails

2. **Email Integration** (Week 3-4)
   - IMAP/Gmail API
   - Email classification
   - Auto-response

### Short-term (Weeks 5-8)
3. **Browser Automation** (Week 5-6)
   - Playwright integration
   - DOM interaction

4. **Multi-Document Synthesis** (Week 7-8)
   - Citation tracking
   - Cross-document analysis

### Medium-term (Weeks 9-12)
5. **Workflow Orchestration** (Week 9-10)
   - Cron triggers
   - Webhook support

6. **Observability** (Week 11-12)
   - Distributed tracing
   - Metrics collection

---

## Integration with Existing Timeline

These fit into **AI_ACCELERATED_TIMELINE.md** as follows:

- **Week 10-12** (Production phase): Add Computer Use + Email + Observability
- **Week 13-15** (Post-GA): Add Workflow Orchestration + Multi-Doc Synthesis
- **Week 16+** (Ecosystem): Plugin system

---

## Success Metrics by Category

### Workflow Automation
- ✅ Can triage 100+ emails per hour
- ✅ Can schedule meetings automatically
- ✅ Can run workflows on cron schedules
- ✅ 90%+ automation rate for routine tasks

### Coding Assistance
- ✅ (Already strong - 85% covered)
- Enhance: Debugger integration, test generation

### Computer Use
- ✅ Can control browser (navigate, click, fill forms)
- ✅ Can control desktop apps (open, close, interact)
- ✅ Can understand UI via VLM
- ✅ Safety: 0 destructive actions without confirmation

### Knowledge Assistants
- ✅ Can analyze 100+ page documents
- ✅ Can synthesize across 10+ documents
- ✅ Can extract and analyze tables
- ✅ Citation accuracy: 95%+

---

## Summary

**What's covered well**: Coding assistance (85%)
**What's partially covered**: Knowledge assistants (70%), Workflow automation (40%)
**What's critically missing**: Computer use (10%)

**Top 3 priorities**:
1. Computer Use Architecture (enables entire category)
2. Email/Calendar Integration (completes workflow automation)
3. Multi-Document Synthesis (completes knowledge assistants)

Implementing these three would bring all four categories to 80%+ coverage.

---

*Category Enablement Gaps - Ensuring GAIA supports all four target agent types out of the box.*
