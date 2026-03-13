# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
GAIA Code: The World's Most Autonomous Coding Agent

Implements the Recursive Agent Composition (RAC) architecture with:
- Context-lean design (never fills context window)
- Continuous execution (no step limits, quality-driven completion)
- Quality gates (verifies output works)
- Checkpoint/resume (survives interruptions)
- Persistent memory (learns across sessions)
- Recursive decomposition via agent_query()

This is the complete M0-M3 implementation:
- M0: Prompting Foundation + RLM patterns
- M1: SharedAgentState + agent_query() + 7 databases
- M2: Quality gates + escalation ladder + continuous execution
- M3: Checkpoint/resume + audit log + time awareness
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import os

logger = logging.getLogger(__name__)


def _format_memory_age(stored_at: Optional[str], now: datetime) -> str:
    """Return a human-readable age string for a stored_at timestamp."""
    if not stored_at:
        return "unknown time"
    try:
        t = datetime.strptime(stored_at, "%Y-%m-%d %H:%M:%S")
        delta = now - t
        days = delta.days
        if days == 0:
            hours = delta.seconds // 3600
            if hours == 0:
                mins = delta.seconds // 60
                return "just now" if mins < 2 else f"{mins}m ago"
            return f"{hours}h ago"
        if days == 1:
            return "yesterday"
        if days < 7:
            return f"{days} days ago"
        if days < 30:
            return f"{days // 7}w ago"
        return f"{days // 30}mo ago"
    except Exception:
        return stored_at[:10] if stored_at else "unknown time"


from gaia.agents.base.agent import Agent
from gaia.agents.base.console import AgentConsole, SilentConsole

# Import CodeAgent tool mixins that are actually available
try:
    from gaia.agents.code.tools.file_io import FileIOToolsMixin
    from gaia.agents.code.tools.testing import TestingMixin
    from gaia.agents.code.tools.external_tools import ExternalToolsMixin
    from gaia.agents.code.tools.code_formatting import CodeFormattingMixin
    from gaia.agents.code.tools.code_tools import CodeToolsMixin
    from gaia.agents.code.tools.project_management import ProjectManagementMixin
    from gaia.agents.code.tools.error_fixing import ErrorFixingMixin
    from gaia.agents.code.tools.typescript_tools import TypeScriptToolsMixin
    from gaia.agents.code.tools.validation_parsing import ValidationAndParsingMixin
    from gaia.agents.code.tools.validation_tools import ValidationToolsMixin
    from gaia.agents.code.tools.web_dev_tools import WebToolsMixin
    from gaia.agents.code.tools.cli_tools import CLIToolsMixin

    CODEAGENT_TOOLS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Some CodeAgent tools not available: {e}")
    # Define empty mixins as fallback
    class FileIOToolsMixin:
        def register_file_io_tools(self): pass
    class TestingMixin:
        def register_testing_tools(self): pass
    class ExternalToolsMixin:
        def register_external_tools(self): pass
    class CodeFormattingMixin:
        def register_code_formatting_tools(self): pass
    class CodeToolsMixin:
        def register_code_tools(self): pass
    class ProjectManagementMixin:
        def register_project_management_tools(self): pass
    class ErrorFixingMixin:
        def register_error_fixing_tools(self): pass
    class TypeScriptToolsMixin:
        def register_typescript_tools(self): pass
    class ValidationAndParsingMixin:
        def register_validation_parsing_tools(self): pass
    class ValidationToolsMixin:
        def register_validation_tools(self): pass
    class WebToolsMixin:
        def register_web_tools(self): pass
    class CLIToolsMixin:
        def register_cli_tools(self): pass

    CODEAGENT_TOOLS_AVAILABLE = False

from .persona import PersonaEngine, create_persona
from .quality_gates import EscalationLadder, QualityGateRunner
from .system_prompt import (
    get_core_system_prompt,
    get_error_recovery_prompts,
    get_tool_usage_guidelines,
)
from .tools import GaiaCodeTools
from .tui import GaiaCodeSimpleTUI, create_tui


class GaiaCodeAgent(
    Agent,
    # CodeAgent tool mixins (70+ essential tools)
    FileIOToolsMixin,          # read, write, edit files
    TestingMixin,              # pytest, jest, coverage
    ExternalToolsMixin,        # web_search, search_docs
    CLIToolsMixin,             # shell execution
    CodeFormattingMixin,       # black, prettier
    ProjectManagementMixin,    # project management
    ErrorFixingMixin,          # error fixing
    TypeScriptToolsMixin,      # TypeScript/Node tools
    ValidationAndParsingMixin, # validation helpers
    ValidationToolsMixin,      # validation tools
    WebToolsMixin,             # Next.js, React tools
    CodeToolsMixin,            # core code tools
    # GAIA Code RAC tools
    GaiaCodeTools,             # RAC + M7 tools
):
    """
    The world's most autonomous coding agent.

    Key differentiators:
    1. **Context-Lean**: Never fills context window. Stores knowledge externally.
    2. **Continuous Execution**: No step limits. Runs until quality gates pass.
    3. **Quality-First**: Verifies output works before declaring "done".
    4. **Recursive**: Decomposes complex tasks via agent_query().
    5. **Persistent**: Learns across sessions via knowledge DB.

    Usage:
        agent = GaiaCodeAgent()
        agent.process_query("Build a REST API with authentication")
        # Agent plans, codes, tests, fixes, and verifies automatically
    """

    def __init__(
        self,
        workspace_dir: Optional[Path] = None,
        enable_quality_gates: bool = True,
        enable_continuous_execution: bool = True,
        tui_mode: str = "simple",
        persona: str = "pike",  # Default to Pike (simplicity advocate)
        allowed_paths: Optional[List[str]] = None,
        target_dir: Optional[str] = None,
        specialist_name: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize GAIA Code agent.

        Args:
            workspace_dir: Directory for agent workspace (default: ~/.gaia/workspace)
            enable_quality_gates: Enable quality gates (default: True)
            enable_continuous_execution: Remove step limits (default: True)
            tui_mode: TUI mode - "full", "simple", "minimal", "off" (default: "simple")
            persona: Personality profile - "direct", "collaborative", "socratic", "mentor", "pragmatic", "friendly" (default: "collaborative")
            allowed_paths: Additional paths the agent is allowed to read/write
            target_dir: Directory to write generated/translated code to (default: project_dir).
                        Set this when the output location differs from where gaia-code was invoked,
                        e.g. translating Python in /src/python → C++ in /src/cpp.
                        Distinct from workspace_dir (the ~/.gaia/workspace DB store).
            **kwargs: Agent initialization parameters
        """
        self._extra_allowed_paths = allowed_paths or []
        self._specialist_name = specialist_name
        # Set defaults for GAIA Code
        if "max_steps" not in kwargs:
            # Continuous execution: high limit (quality-driven, not step-driven)
            kwargs["max_steps"] = 1000 if enable_continuous_execution else 100

        # Default to Claude Sonnet 4.6 for best speed/cost balance
        if "use_claude" not in kwargs and "use_chatgpt" not in kwargs:
            kwargs["use_claude"] = True
            if "claude_model" not in kwargs:
                kwargs["claude_model"] = "claude-sonnet-4-6"

        if "max_plan_iterations" not in kwargs:
            # Allow many plan iterations for complex tasks
            kwargs["max_plan_iterations"] = 100

        # Claude Sonnet supports 200K context — give GaiaCodeAgent 100K input budget
        # (keeps 84K reserve for output tokens and API overhead)
        if "max_input_tokens" not in kwargs:
            kwargs["max_input_tokens"] = 100000

        # Override model_id AFTER defaults to ensure Claude model is used
        if kwargs.get("use_claude"):
            kwargs["model_id"] = kwargs.get("claude_model", "claude-sonnet-4-6")

        # Check credentials before initializing
        from .credentials import check_and_setup_credentials

        # Create console for credential prompts if TUI enabled
        cred_console = None
        if tui_mode != "off" and not kwargs.get("silent_mode"):
            try:
                from rich.console import Console
                cred_console = Console()
            except ImportError:
                pass

        # Check and setup credentials
        cred_success, api_key = check_and_setup_credentials(
            use_claude=kwargs.get("use_claude", True),
            console=cred_console,
            interactive=not kwargs.get("silent_mode", False),
        )

        if not cred_success:
            raise RuntimeError(
                "API key required but not provided. "
                "Set ANTHROPIC_API_KEY environment variable or run interactively to configure."
            )

        # If API key was provided interactively, use it
        if api_key and kwargs.get("use_claude", True):
            os.environ["ANTHROPIC_API_KEY"] = api_key

        # Enable RAC features from base Agent
        kwargs["enable_shared_state"] = True
        kwargs["workspace_dir"] = str(workspace_dir) if workspace_dir else None
        kwargs["enable_audit_log"] = True

        # Initialize base agent (with RAC: shared_state, audit_log, time tracking)
        super().__init__(**kwargs)

        # Initialize workspace: register core tools and specialist agents into DBs.
        # Runs on every startup to keep databases current across sessions.
        from .integration import initialize_workspace
        _ws = Path(self.shared_state.workspace_dir) if (
            hasattr(self, 'shared_state') and self.shared_state and self.shared_state.workspace_dir
        ) else None
        initialize_workspace(_ws)

        # Suppress noisy warnings from chat/LLM subsystems unless debug mode.
        # Must be AFTER super().__init__() since Agent.__init__ calls basicConfig.
        # NOTE: do NOT suppress gaia.agents — that namespace includes agent.py
        # itself, and silencing it prevents execution logs from reaching the DB handler.
        if not kwargs.get("debug"):
            for name in ["gaia.chat", "gaia.chat.prompts", "gaia.chat.sdk",
                         "gaia.llm"]:
                logging.getLogger(name).setLevel(logging.ERROR)
            # Also suppress via GAIA's custom logger system
            try:
                from gaia.logger import log_manager
                log_manager.set_level("gaia.chat", logging.ERROR)
            except (ImportError, AttributeError):
                pass

        # Capture the project directory (where gaia-code was invoked from).
        # This is used to namespace active_state facts and other per-project data
        # so that facts from project-a don't pollute context when working in project-b.
        _cwd = str(Path.cwd())
        self.project_dir = _cwd
        # Track whether the caller explicitly set project_dir (not just the default CWD).
        # Used to suppress reconnaissance hints that would otherwise scan the wrong dir.
        self._project_dir_explicit = False

        # target_dir: where generated/translated code is written.
        # Defaults to project_dir — only differs for tasks like code translation
        # where the output lives in a separate directory from the source.
        self.target_dir = str(Path(target_dir).resolve()) if target_dir else self.project_dir

        # Track the active plan for this session (created lazily in process_query)
        self._current_plan_id: Optional[str] = None

        # Initialize persona engine (coding-specific)
        self.persona = create_persona(persona, workspace_dir)

        # Initialize quality gates (from base module)
        self.quality_gates = QualityGateRunner()
        if not enable_quality_gates:
            for gate_name in self.quality_gates.gates:
                self.quality_gates.disable_gate(gate_name)

        # Initialize escalation ladder (from base module)
        self.escalation_ladder = EscalationLadder()

        # Initialize TUI (coding-specific)
        self.tui_mode = tui_mode
        if tui_mode != "off" and not kwargs.get("silent_mode"):
            self.tui = create_tui(mode=tui_mode)
        else:
            self.tui = None

        # Initialize validators (required by CodeAgent tools)
        from gaia.security import PathValidator
        from gaia.agents.code.validators import SyntaxValidator, ASTAnalyzer, AntipatternChecker, RequirementsValidator

        # Build allowed paths: workspace dir + source/target dirs + user-specified paths
        all_allowed = list(self._extra_allowed_paths)
        if hasattr(self, 'shared_state') and self.shared_state and self.shared_state.workspace_dir:
            all_allowed.append(str(self.shared_state.workspace_dir))
        # Always allow project_dir and target_dir so the agent can read source and write target
        for d in (self.project_dir, self.target_dir):
            if d and d not in all_allowed:
                all_allowed.append(d)
        self.path_validator = PathValidator(all_allowed if all_allowed else None)
        self.syntax_validator = SyntaxValidator()
        self.ast_analyzer = ASTAnalyzer()
        self.antipattern_checker = AntipatternChecker()
        self.requirements_validator = RequirementsValidator()

        # IMPORTANT: Rebuild system prompt to include tools and persona
        # Uses _format_tools_for_prompt() override which only includes essential tools
        self.rebuild_system_prompt()

        # Inject specialist system prompt if this is a specialist sub-agent
        if self._specialist_name:
            from .integration import get_specialist_system_prompt
            spec_prompt = get_specialist_system_prompt(self._specialist_name)
            if spec_prompt:
                self.system_prompt += (
                    f"\n\n## Specialist Role ({self._specialist_name})\n{spec_prompt}"
                )

        # Sync all _TOOL_REGISTRY tools → tools.db so find_tool() can discover them
        # Must be AFTER rebuild_system_prompt() (which triggers _register_tools())
        self._sync_tools_to_db()

        # Load tools created by the agent in previous sessions back into _TOOL_REGISTRY
        self._load_learned_tools()

        logger.debug("GAIA Code Agent initialized")
        logger.debug(f"Workspace: {self.shared_state.workspace_dir}")
        logger.debug(f"Persona: {self.persona.profile.name}")
        logger.debug(f"Quality gates: {enable_quality_gates}")
        logger.debug(f"Continuous execution: {enable_continuous_execution}")
        logger.debug(f"TUI mode: {tui_mode}")
        logger.debug(f"Model: Claude Opus 4.6" if kwargs.get("use_claude") else f"Model: {kwargs.get('model_id', 'local')}")

    def _get_system_prompt(self, _user_input: Optional[str] = None) -> str:
        """
        Get the BASE system prompt for GAIA Code (without tools).

        The base Agent's rebuild_system_prompt() will add:
        - Tools from _TOOL_REGISTRY (compact format via _format_tools_for_prompt override)
        - Response format instructions

        This method provides our custom prompt content.
        """
        # Core system prompt with RLM patterns
        prompt = get_core_system_prompt()

        # Add tool usage guidelines
        prompt += "\n\n" + get_tool_usage_guidelines()

        # Add persona-specific communication style (single active profile only)
        prompt += "\n\n" + self.persona.get_system_prompt_addition()

        return prompt

    # Essential tools always present in the system prompt — the minimum needed
    # to bootstrap any task. All other tools are discovered on-demand via find_tool().
    _ESSENTIAL_TOOLS = {
        "read_file", "write_file", "edit_file",          # file I/O
        "run_shell_command",                               # execution
        "glob_search", "grep_content",                    # search
        "agent_query",                                     # recursive decomposition
        "find_tool", "recall", "remember",                # memory / tool discovery
        "search_conversations",                            # past conversation recall
    }

    def _format_tools_for_prompt(self) -> str:
        """
        Include only essential bootstrap tools in the system prompt.

        All other tools (~60+) are discovered on demand via find_tool(query).
        This reduces the tool section from ~7,100 tokens to ~150 tokens.

        Usage pattern:
          1. Agent calls find_tool("run python tests") → tools.db returns run_pytest definition
          2. Agent calls run_pytest(path="tests/") using the returned signature
        """
        from gaia.agents.base.tools import _TOOL_REGISTRY

        tool_lines = []
        for name in sorted(self._ESSENTIAL_TOOLS):
            if name not in _TOOL_REGISTRY:
                continue
            tool_info = _TOOL_REGISTRY[name]
            params_str = ", ".join(
                f"{pname}{'' if pinfo['required'] else '?'}: {pinfo['type']}"
                for pname, pinfo in tool_info["parameters"].items()
            )
            desc = tool_info["description"].split("\n")[0].strip()  # first line only
            tool_lines.append(f"- {name}({params_str}): {desc}")

        # Add a note about tool discovery
        tool_lines.append("")
        tool_lines.append("All other tools are in tools.db — use find_tool(query) to discover them.")
        tool_lines.append("find_tool returns: name, parameters, description — enough to call the tool immediately.")

        return "\n".join(tool_lines)

    def _sync_tools_to_db(self) -> None:
        """
        Sync all tools from _TOOL_REGISTRY into tools.db so find_tool() can discover them.

        Called once after all tools are registered. The tools.db entry includes the full
        parameter schema so find_tool() results have enough info to call the tool.
        """
        from gaia.agents.base.tools import _TOOL_REGISTRY

        if not self.shared_state:
            return

        # Infer category from tool name
        def _infer_category(name: str) -> str:
            if name.startswith("git_"):
                return "git"
            if name in ("run_pytest", "run_jest", "check_coverage"):
                return "testing"
            if name in ("check_syntax", "run_linter", "check_imports", "format_code"):
                return "quality"
            if name in ("run_python", "run_shell_command", "run_background"):
                return "execution"
            if name in ("read_file", "write_file", "edit_file", "glob_search",
                        "grep_content", "list_files", "delete_file", "copy_file", "move_file"):
                return "file_io"
            if name in ("recall", "store_insight", "find_tool", "agent_query", "remember"):
                return "memory"
            if any(kw in name for kw in ("npm", "next", "react", "css", "html", "web")):
                return "web"
            if any(kw in name for kw in ("typescript", "_ts_", "eslint")):
                return "typescript"
            return "utility"

        synced = 0
        for name, tool_info in _TOOL_REGISTRY.items():
            try:
                # Convert _TOOL_REGISTRY parameter format to a clean schema dict
                params = {
                    pname: {"type": pinfo["type"], "required": pinfo["required"]}
                    for pname, pinfo in tool_info["parameters"].items()
                }
                self.shared_state.tools.register_tool(
                    name=name,
                    category=_infer_category(name),
                    description=tool_info["description"].strip(),
                    source="registry",
                    parameters=params,
                )
                synced += 1
            except Exception as e:
                logger.debug(f"[GaiaCode] skip sync for {name}: {e}")

        logger.info(f"[GaiaCode] synced {synced} tools to tools.db")

    def _load_learned_tools(self) -> int:
        """
        Load tools created by the agent in previous sessions into _TOOL_REGISTRY.

        Called once at startup. Each learned tool's .py wrapper is imported via
        importlib and registered so it's callable this session without rediscovery.

        Returns:
            Number of learned tools loaded
        """
        import importlib.util
        from gaia.agents.base.tools import tool as tool_decorator

        if not self.shared_state:
            return 0

        rows = self.shared_state.tools.conn.execute(
            "SELECT name, code_path FROM tools WHERE source='learned' AND enabled=TRUE"
        ).fetchall()

        loaded = 0
        for name, code_path in rows:
            if not code_path:
                continue
            path = Path(code_path)
            if not path.exists():
                logger.warning(f"[GaiaCode] learned tool '{name}' missing file: {code_path}")
                continue
            try:
                spec = importlib.util.spec_from_file_location(name, path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                func = getattr(module, name)
                tool_decorator(func)
                loaded += 1
                logger.debug(f"[GaiaCode] loaded learned tool: {name}")
            except Exception as e:
                logger.warning(f"[GaiaCode] failed to load learned tool '{name}': {e}")

        if loaded:
            logger.info(f"[GaiaCode] loaded {loaded} learned tools from previous sessions")
        return loaded

    def _register_tools(self) -> None:
        """Register all tools: CodeAgent tools + GAIA Code RAC tools."""
        # Register all CodeAgent tools (70+ tools)
        if CODEAGENT_TOOLS_AVAILABLE:
            try:
                self.register_code_tools()           # Core code generation
            except AttributeError:
                pass

            try:
                self.register_file_io_tools()        # File I/O
            except AttributeError:
                pass

            try:
                self.register_testing_tools()        # pytest, jest
            except AttributeError:
                pass

            try:
                self.register_external_tools()       # web_search, search_docs
            except AttributeError:
                pass

            try:
                self.register_cli_tools()            # shell execution
            except AttributeError:
                pass

            try:
                self.register_code_formatting_tools()  # black, prettier
            except AttributeError:
                pass

            try:
                self.register_project_management_tools()  # project management
            except AttributeError:
                pass

            try:
                self.register_error_fixing_tools()   # error fixing
            except AttributeError:
                pass

            try:
                self.register_typescript_tools()     # TypeScript/Node
            except AttributeError:
                pass

            try:
                self.register_validation_tools()     # validation
            except AttributeError:
                pass

            try:
                self.register_web_tools()            # Next.js, React
            except AttributeError:
                pass

        # Register GAIA Code RAC tools (agent_query, recall, etc.)
        self.register_gaia_code_tools()

    def _create_console(self):
        """Create console for agent output."""
        if self.silent_mode:
            return SilentConsole()
        return AgentConsole()

    def process_query(
        self, query: str, create_plan: bool = True
    ) -> Dict[str, Any]:
        """
        Process a user query with full RAC capabilities.

        This is the main entry point for GAIA Code. It:
        1. Creates a plan (if complex task)
        2. Executes the plan recursively
        3. Runs quality gates
        4. Returns verified result

        Args:
            query: User's coding task
            create_plan: Whether to create a plan first (default: True)

        Returns:
            Dict with result and status
        """
        self.task_start = datetime.now()
        self._current_query = query  # made available to tools via self._agent reference
        self._log_audit("TASK_START", {"query": query})

        # Inject current working memories into the task context
        # This makes the agent aware of what it stored in previous calls
        if self.shared_state:
            context_blocks = []

            # 1. Working memory (session key/value facts)
            try:
                memories = self.shared_state.memory.recall_memories(
                    limit=20, source_dir=self.project_dir
                )
                if memories:
                    now = datetime.now()
                    same = [m for m in memories if m["same_project"]]
                    other = [m for m in memories if not m["same_project"]]

                    def _fmt(m: dict) -> str:
                        age = _format_memory_age(m.get("stored_at"), now)
                        ctx = f', while: "{m["query_context"][:60]}"' if m.get("query_context") else ""
                        return f"  [{m['key']}] {m['value']}  (stored {age}{ctx})"

                    # Show target_dir only when it differs from project_dir
                    if self.target_dir == self.project_dir:
                        dir_header = f"project: {self.project_dir}"
                    else:
                        dir_header = f"project: {self.project_dir}, target: {self.target_dir}"
                    lines = [f"## Working Memory ({dir_header}, as of {now.strftime('%Y-%m-%d %H:%M')})"]
                    if same:
                        lines += [_fmt(m) for m in same]
                    if other:
                        lines.append("  -- from other projects (may not be relevant) --")
                        for m in other:
                            src = m.get("source_dir") or "unknown"
                            age = _format_memory_age(m.get("stored_at"), now)
                            ctx = f', while: "{m["query_context"][:60]}"' if m.get("query_context") else ""
                            lines.append(f"  [{m['key']}] {m['value']}  (from {src}, stored {age}{ctx})")

                    context_blocks.append("\n".join(lines))
                    logger.debug("[GaiaCode] injected %d memories (%d same-project, %d other) into task",
                                 len(memories), len(same), len(other))
            except Exception:
                pass

            # 2. Relevant knowledge insights (cross-session learnings)
            try:
                insights = self.shared_state.knowledge.recall(query[:200], top_k=5)
                if insights:
                    ins_lines = "\n".join(
                        f"  - [{i.get('category', '?')}] {i.get('content', '')[:200]}"
                        for i in insights
                    )
                    context_blocks.append(f"## Relevant Knowledge (from past sessions)\n{ins_lines}")
                    logger.debug("[GaiaCode] injected %d insights into task", len(insights))
            except Exception:
                pass

            # 3. Available skills (learned workflows)
            try:
                skills = self.shared_state.skills.find_skills()
                if skills:
                    sk_lines = "\n".join(
                        f"  - {s['name']} ({s['category']}): {s['description']}"
                        for s in skills[:10]
                    )
                    context_blocks.append(f"## Available Skills (reusable workflows)\n{sk_lines}")
                    logger.debug("[GaiaCode] injected %d skills into task", len(skills))
            except Exception:
                pass

            # 4. Relevant tools from tools.db (semantic search — no extra LLM call)
            # Injects the top-K matching tool definitions BEFORE the LLM sees the query.
            # Scales to thousands of tools: only relevant ones enter context.
            try:
                relevant_tools = self.shared_state.tools.find_tools(query[:300], top_k=12)
                if relevant_tools:
                    tool_lines = []
                    for t in relevant_tools:
                        params = ""
                        if t.get("parameters"):
                            params = ", ".join(
                                f"{k}: {v['type']}" for k, v in t["parameters"].items()
                            )
                        tool_lines.append(f"  - {t['name']}({params}): {t['description'][:120]}")
                    context_blocks.append(
                        "## Contextually Relevant Tools (pre-fetched from tools.db)\n"
                        + "\n".join(tool_lines)
                    )
                    logger.debug("[GaiaCode] injected %d relevant tools into task", len(relevant_tools))
            except Exception:
                pass

            if context_blocks:
                query = query + "\n\n" + "\n\n".join(context_blocks)

        # Auto-detect output directories from the query and add to PathValidator
        self._auto_add_query_paths(query)

        # Start TUI if available
        if self.tui:
            self.tui.start(query)

        # Create plan + root milestone in master plan
        self._current_task_id = None
        if create_plan:
            plan_id = self.shared_state.plan.create_plan(
                query[:200],
                project_dir=self.project_dir,
                target_dir=self.target_dir,
            )
            task_id = self.shared_state.plan.create_task(plan_id, title=query[:200], depth=0)
            self._current_plan_id = plan_id
            self._current_task_id = task_id
            self._log_audit("PLAN_CREATE", {"plan_id": plan_id, "task_id": task_id})

            # Mark root task as in-progress so the dashboard shows activity
            try:
                self.shared_state.plan.start_task(task_id)
            except Exception:
                pass

            # Show plan in TUI if available
            if self.tui and hasattr(self.tui, 'show_plan'):
                tasks = self.shared_state.plan.get_plan_tasks(plan_id)
                plan_data = [{"description": t["title"], "status": t["status"]} for t in tasks]
                self.tui.show_plan(plan_data)
        root_task = None  # kept for API compat with _execute_with_quality_gates

        # Detect tasks that involve an existing codebase and prepend a
        # reconnaissance reminder so the agent reads source files before planning.
        _recon_keywords = (
            "convert", "migrate", "port", "translate",
            "refactor", "rewrite", "update", "fix", "improve",
            "analyze", "review", "audit",
        )
        _query_lower = query.lower()
        _needs_recon = any(kw in _query_lower for kw in _recon_keywords)
        # Only add recon hint when project_dir was explicitly set by the user.
        # If it's just the default CWD, scanning it would glob the wrong directory
        # (e.g. the GAIA source tree instead of the user's project).
        if _needs_recon and self._project_dir_explicit:
            _recon_hint = (
                "\n\n⚠️  RECONNAISSANCE REQUIRED BEFORE PLANNING:\n"
                f"1. Call `glob_search('**/*')` to discover ALL files in `{self.project_dir}`\n"
                "2. Call `read_file` on every source file you will work with\n"
                "3. ONLY THEN create your plan — no guessing at file names or APIs\n"
            )
            query = query + _recon_hint

        try:
            # Execute the task
            result = self._execute_with_quality_gates(query, root_task)

            # Log completion
            elapsed = (datetime.now() - self.task_start).total_seconds()
            self._log_audit(
                "TASK_COMPLETE",
                {"elapsed_seconds": elapsed, "success": result["success"]},
            )

            # Update plan task status to match execution outcome
            if self._current_task_id:
                try:
                    if result.get("success"):
                        self.shared_state.plan.complete_task(
                            self._current_task_id,
                            result=result.get("result", "")[:500],
                        )
                    else:
                        self.shared_state.plan.fail_task(
                            self._current_task_id,
                            error=result.get("error", "Task did not succeed"),
                        )
                except Exception:
                    pass

            # Auto-register a skill pattern for successful tasks
            if result.get("success"):
                try:
                    self._auto_register_skill(query, result)
                except Exception:
                    pass

            # Complete TUI (guard against double-complete is inside each TUI class)
            if self.tui:
                message = result.get("result") or "Task completed"
                self.tui.complete(success=result["success"], message=message)

            return result

        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            self._log_audit("TASK_INTERRUPTED", {"query": query})

            if self._current_task_id:
                try:
                    self.shared_state.plan.fail_task(self._current_task_id, error="Interrupted by user")
                except Exception:
                    pass

            if self.tui:
                self.tui.complete(success=False, message="Interrupted by user")

            # Create checkpoint before exiting
            self.checkpoint()

            raise

        except Exception as e:
            self._log_audit("TASK_ERROR", {"error": str(e)})

            if self._current_task_id:
                try:
                    self.shared_state.plan.fail_task(self._current_task_id, error=str(e))
                except Exception:
                    pass

            if self.tui:
                self.tui.complete(success=False, message=f"Error: {str(e)}")

            raise

    def _auto_add_query_paths(self, query: str) -> None:
        """Auto-detect absolute paths in the query and add them to PathValidator."""
        import re
        # Match absolute paths: /mnt/... or C:\... or C:/...
        path_patterns = [
            r'(/mnt/[^\s"\']+)',           # WSL paths
            r'(/[a-z]/[^\s"\']+)',          # Short WSL paths like /c/Users/...
            r'([A-Z]:\\[^\s"\']+)',         # Windows backslash paths
            r'([A-Z]:/[^\s"\']+)',          # Windows forward-slash paths
        ]
        for pattern in path_patterns:
            for match in re.finditer(pattern, query):
                path_str = match.group(1)
                try:
                    p = Path(path_str)
                    # Add the directory (or parent of file) to allowed paths
                    target = p if p.suffix == '' else p.parent
                    self.path_validator.add_allowed_path(str(target))
                    logger.debug(f"Auto-added allowed path from query: {target}")
                except Exception:
                    pass

    def _execute_with_quality_gates(
        self, query: str, root_task: Optional[Any]
    ) -> Dict[str, Any]:
        """
        Execute a task with quality gate verification.

        This implements the M2 behavior:
        - Execute task
        - Run quality gates
        - If gates fail: retry → decompose → escalate → ask user
        - If gates pass: return result
        """
        attempt = 0
        max_attempts = 10  # Safety limit

        # Directories to skip — build artifacts, cache, and downloaded deps
        # are not authored files and would cause false completeness failures.
        _SKIP_DIRS = {
            "build", ".build", "_build",
            ".pytest_cache", ".benchmarks", "__pycache__",
            "node_modules", ".git",
            "_deps",          # CMake FetchContent deps
            "CMakeFiles",     # CMake internal dir
            ".cache",
        }

        while attempt < max_attempts:
            attempt += 1

            # Update TUI if available
            if self.tui:
                self.tui.update(stage="Executing", current=query[:50], percent=attempt * 10)

            # Snapshot files that already exist BEFORE this attempt executes.
            # Quality gates only apply to NEW files written in THIS attempt — not
            # pre-existing files from earlier steps which have already been verified.
            pre_existing: set = set()
            if self.target_dir and self.target_dir != self.project_dir:
                try:
                    _target_pre = Path(self.target_dir)
                    if _target_pre.exists():
                        for _p in _target_pre.rglob("*"):
                            if _p.is_file() and _p.suffix not in (".db",):
                                if not any(part in _SKIP_DIRS for part in _p.parts):
                                    pre_existing.add(str(_p))
                except Exception:
                    pass

            # Execute the task
            result = self._execute_task(query, root_task)

            # Check if files were written to target_dir (more reliable than manifest tracking).
            # Only scan when target_dir was explicitly set to a different location than project_dir
            # (e.g. code-translation tasks).  Scanning the default project_dir (cwd) would walk
            # the entire repo on every call, causing unacceptable latency.
            # IMPORTANT: only include files that are NEW since this attempt started — pre-existing
            # files were verified in earlier steps and must not trigger a full retry here.
            if not result.get("files") and self.target_dir and self.target_dir != self.project_dir:
                try:
                    target = Path(self.target_dir)
                    if target.exists():
                        on_disk = []
                        for p in target.rglob("*"):
                            if not p.is_file():
                                continue
                            if p.suffix in (".db",):
                                continue
                            # Skip any path that contains a build/cache directory
                            if any(part in _SKIP_DIRS for part in p.parts):
                                continue
                            # Only include files that are NEW in this attempt
                            if str(p) not in pre_existing:
                                on_disk.append(str(p))
                        if on_disk:
                            result["files"] = on_disk
                            logger.info("[GaiaCode] found %d NEW files in target_dir this attempt", len(on_disk))
                except Exception:
                    pass

            # Skip quality gates if no files were created
            if not result.get("files"):
                return result

            # Run quality gates on created files
            file_paths = result.get("files", [])
            gate_results = self.quality_gates.run_all(file_paths)
            all_passed = self.quality_gates.all_passed(gate_results)

            # Update TUI with gate results
            if self.tui:
                gates_status = {name: result.passed for name, result in gate_results.items()}
                self.tui.update_quality_gates(gates_status)

            self._log_audit(
                "QUALITY_GATES",
                {
                    "attempt": attempt,
                    "passed": all_passed,
                    "results": [
                        {"gate": name, "passed": r.passed, "message": r.message}
                        for name, r in gate_results.items()
                    ],
                },
            )

            # If all gates passed, we're done
            if all_passed:
                result["quality_gates"] = list(gate_results.values())
                result["attempts"] = attempt

                # Update TUI completion
                if self.tui:
                    self.tui.complete(success=True, message=result.get("result"))

                return result

            # Gates failed - escalate
            action = self.escalation_ladder.get_action()
            self._log_audit(
                "ESCALATION",
                {"action": action, "attempt": attempt},
            )

            if action == "retry":
                # Retry: just loop again
                self.escalation_ladder.increment()
                continue

            elif action == "decompose":
                # Decompose into smaller subtasks using agent_query()
                result = self._decompose_task(query, list(gate_results.values()))
                self.escalation_ladder.reset()
                return result

            elif action == "alternative":
                # Try different approach
                logger.info("Trying alternative approach...")
                self.escalation_ladder.escalate()
                continue

            elif action == "ask_user":
                # Ask user for help
                result = self._ask_user_for_help(query, list(gate_results.values()))
                self.escalation_ladder.reset()
                return result

        # Max attempts reached
        if self.tui:
            self.tui.complete(success=False, message="Max attempts reached")

        return {
            "success": False,
            "result": None,
            "error": f"Max attempts ({max_attempts}) reached without passing quality gates",
        }

    def _execute_task(self, query: str, root_task: Optional[Any]) -> Dict[str, Any]:
        """
        Execute task using BASE AGENT's process_query (REUSES GAIA'S TOOL LOOP).

        The base Agent.process_query() handles:
        - Full conversation loop
        - Tool call parsing from JSON
        - Tool execution via _execute_tool()
        - Multi-step plans
        - Everything!

        We just wrap it with our enhancements.

        Args:
            query: Task to execute
            root_task: Optional root task node

        Returns:
            Dict with execution result
        """
        try:
            # Track files before execution
            files_before = set(self.shared_state.manifest.list_files())

            logger.debug(f"Executing via base Agent.process_query: {query}")

            # Call base Agent's process_query which has the full tool execution loop!
            # This is the REAL execution method that handles tools
            base_result = super().process_query(user_input=query)

            # base_result should be a dict, but guard against edge cases
            # (e.g. if the LLM returned a JSON array instead of an object the
            # base agent may propagate it as a list).
            if isinstance(base_result, dict):
                # The base Agent returns {"result": "...", "steps_taken": N, ...}
                # It does NOT include a "status": "success" key.  Checking for
                # that key means success is ALWAYS False, which breaks plan
                # tracking and the benchmark runner.  Use "result" non-empty as
                # the success signal instead, and only treat "status": "error"
                # as an explicit failure.
                result_text = base_result.get("result", str(base_result))
                success = bool(result_text) and base_result.get("status") != "error"
            elif isinstance(base_result, list):
                # Extract the last string item as the result (conversation history)
                result_text = next(
                    (str(item) for item in reversed(base_result) if item), ""
                )
                success = bool(result_text)
            else:
                result_text = str(base_result) if base_result else ""
                success = bool(result_text)

            # Check what files were created
            files_after = set(self.shared_state.manifest.list_files())
            new_files = list(files_after - files_before)

            logger.debug(f"Base Agent completed successfully")

            return {
                "success": success,
                "result": result_text,
                "files": new_files,
                "project_dir": ".",
            }

        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            return {
                "success": False,
                "result": None,
                "error": str(e),
                "files": [],
            }


    def _on_plan_created(self, plan_steps: list) -> None:
        """
        Pre-register future plan steps as 'pending' tasks in memory.db.

        This ensures the planning panel shows queued/upcoming work, not just
        past and current tasks. Only agent_query steps are pre-registered since
        those represent meaningful sub-agent dispatches; low-level tool calls
        (write_file, run_cli_command etc.) are not tracked at task level.
        """
        if not self._current_plan_id:
            return
        try:
            state = self.shared_state
            root = state.memory.conn.execute(
                "SELECT id FROM plan_tasks WHERE plan_id=? AND depth=0 LIMIT 1",
                (self._current_plan_id,),
            ).fetchone()
            parent_id = root[0] if root else None
            for step in plan_steps:
                tool_name = step.get("tool", "")
                if tool_name not in ("agent_query",):
                    continue
                task_desc = (
                    step.get("description")
                    or step.get("tool_args", {}).get("task", "")[:120]
                    or f"agent_query"
                )
                state.plan.create_task(
                    self._current_plan_id,
                    title=task_desc[:160],
                    depth=1,
                    parent_id=parent_id,
                )
        except Exception as e:
            logger.debug("[GaiaCode] _on_plan_created: %s", e)

    def _decompose_task(
        self, query: str, gate_results: List["GateResult"]
    ) -> Dict[str, Any]:
        """
        Decompose task into smaller subtasks using agent_query().

        This is the core RAC pattern: when stuck, decompose and recurse.
        """
        # Analyze gate failures to determine decomposition strategy
        failed_gates = [r for r in gate_results if not r.passed]

        if not failed_gates:
            return {"success": True, "result": "No failed gates to decompose"}

        # Map gate failures to specialist agents
        gate_to_specialist = {
            "completeness": "ArchitectureAgent",
            "syntax":  "DebuggerAgent",
            "imports": "DebuggerAgent",
            "tests":   "TestingAgent",
        }

        # Create subtasks for each failed gate, with specialist routing
        subtasks = []
        for gate in failed_gates:
            gate_name = getattr(gate, 'gate_name', getattr(gate, 'name', 'unknown')).lower()
            specialist = gate_to_specialist.get(gate_name)
            errors = getattr(gate, 'errors', [])
            error_summary = "; ".join(errors[:5]) if errors else ""
            if gate_name == "completeness":
                subtasks.append((
                    f"The following files are missing or empty — rewrite them with full content: {error_summary}",
                    specialist,
                ))
            elif gate_name == "syntax":
                subtasks.append(("Fix all syntax errors in the files you just created", specialist))
            elif gate_name == "imports":
                subtasks.append(("Fix all import errors in the files you just created", specialist))
            elif gate_name == "tests":
                subtasks.append(("Fix all failing tests", specialist))

        # Execute each subtask via agent_query()
        results = []
        for subtask, specialist in subtasks:
            subtask_result = self.tool_agent_query(task=subtask, specialist=specialist)
            results.append(subtask_result)

        # Check if all subtasks succeeded
        all_succeeded = all(r["success"] for r in results)

        return {
            "success": all_succeeded,
            "result": "Decomposed and executed subtasks",
            "subtask_results": results,
        }

    def _escalate_to_cloud(self, query: str) -> Dict[str, Any]:
        """
        Escalate to cloud LLM.

        Since Claude Opus 4.6 is already the default, this is already using cloud.
        Just retry the task.
        """
        logger.info("Already using Claude Opus 4.6 - retrying task")
        return self._execute_task(query, None)

    def _ask_user_for_help(
        self, query: str, gate_results: List["GateResult"]
    ) -> Dict[str, Any]:
        """
        Ask user for help via message queue.

        Last resort when automated approaches fail.
        """
        # Send message to user
        failed_gates = [r for r in gate_results if not r.passed]
        gate_names = [getattr(g, 'gate_name', getattr(g, 'name', 'unknown')) for g in failed_gates]
        message = f"I'm stuck on the task: '{query}'. Failed quality gates: {', '.join(gate_names)}. What should I do?"

        msg_result = self.tool_send_message(content=message, priority="Question")

        return {
            "success": False,
            "result": None,
            "error": "Waiting for user input",
            "message_id": msg_result["message_id"],
        }

    # _log_audit, get_audit_log, get_progress are inherited from base Agent

    def _execute_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> Any:
        """
        Override base _execute_tool to integrate with memory.db and tools.db.

        Before executing:
        - For read_file: check file_cache first (context-lean — skip I/O if cached)

        After executing:
        - Store result in tool_results (session memory)
        - Cache file content on read_file calls
        - Record tool usage duration+success in tools.db
        """
        start_ms = int(time.time() * 1000)

        # --- Normalize path aliases: LLMs sometimes use "path" instead of "file_path" ---
        if tool_name in ("write_file", "edit_file", "read_file") and "path" in tool_args and "file_path" not in tool_args:
            tool_args = dict(tool_args)
            tool_args["file_path"] = tool_args.pop("path")

        # --- Normalize list_files: LLMs call it with dir_path= or directory= instead of path= ---
        if tool_name == "list_files":
            if "dir_path" in tool_args and "path" not in tool_args:
                tool_args = dict(tool_args)
                tool_args["path"] = tool_args.pop("dir_path")
            elif "directory" in tool_args and "path" not in tool_args:
                tool_args = dict(tool_args)
                tool_args["path"] = tool_args.pop("directory")

        # --- Normalize tool name aliases: tools.db recipes vs actual registered functions ---
        # LLMs commonly use "run_shell_command" (from tools.db recipes) but the callable is run_cli_command
        _TOOL_NAME_ALIASES = {
            "run_shell_command": "run_cli_command",
            "shell_command": "run_cli_command",
            "execute_command": "run_cli_command",
            "bash": "run_cli_command",
        }
        if tool_name in _TOOL_NAME_ALIASES:
            tool_name = _TOOL_NAME_ALIASES[tool_name]

        # --- Normalize kwarg aliases for run_cli_command (LLMs use 'cwd', actual param is 'working_dir') ---
        if tool_name == "run_cli_command" and "cwd" in tool_args and "working_dir" not in tool_args:
            tool_args = dict(tool_args)
            tool_args["working_dir"] = tool_args.pop("cwd")

        # --- Redirect read_file(directory) to list_files to avoid "is a directory" error ---
        if tool_name in ("read_file", "read"):
            fp = tool_args.get("file_path") or tool_args.get("path") or tool_args.get("filename", "")
            if fp:
                import os
                if os.path.isdir(fp):
                    logger.debug("[GaiaCode] read_file on directory → redirecting to list_files: %s", fp)
                    return self._execute_tool("list_files", {"path": fp})

        # --- Pre-execution: serve read_file from file cache if available ---
        if self.shared_state and tool_name in ("read_file", "read"):
            file_path = (
                tool_args.get("file_path")
                or tool_args.get("path")
                or tool_args.get("filename", "")
            )
            if file_path:
                try:
                    cached = self.shared_state.memory.get_file(file_path)
                    if cached is not None:
                        logger.debug("[GaiaCode] file cache hit: %s", file_path)
                        # Record as a zero-duration cache hit, skip actual I/O
                        try:
                            self.shared_state.tools.record_usage(
                                tool_name=tool_name,
                                success=True,
                                duration_ms=0,
                                context="cache_hit",
                            )
                        except Exception:
                            pass
                        return cached
                except Exception:
                    pass  # Fall through to actual tool execution

        result = super()._execute_tool(tool_name, tool_args)

        # Only track if shared state is available
        if not self.shared_state:
            return result

        duration_ms = int(time.time() * 1000) - start_ms
        success = not (isinstance(result, dict) and result.get("status") == "error")
        error_msg = result.get("error") if isinstance(result, dict) else None

        # --- memory.db: store_tool_result (session history) ---
        try:
            result_preview = (
                result if isinstance(result, str) else json.dumps(result, default=str)
            )
            self.shared_state.memory.store_tool_result(
                tool_name=tool_name,
                args=tool_args,
                result=result_preview[:4000],
            )
        except Exception:
            pass

        # --- memory.db: cache file content for future cache-hits ---
        if tool_name in ("read_file", "read") and success:
            file_path = (
                tool_args.get("file_path")
                or tool_args.get("path")
                or tool_args.get("filename", "")
            )
            if file_path:
                content = result if isinstance(result, str) else str(result)
                try:
                    self.shared_state.memory.cache_file(file_path, content[:50000])
                except Exception:
                    pass

        # --- manifest: register written files so quality gates can inspect them ---
        if tool_name in ("write_file", "edit_file") and success:
            file_path = (
                tool_args.get("file_path")
                or tool_args.get("path")
                or tool_args.get("filename", "")
            )
            content = tool_args.get("content", "")
            if file_path:
                try:
                    self.shared_state.manifest.add_file(file_path, content[:5000])
                    logger.debug("[GaiaCode] registered in manifest: %s", file_path)
                except Exception:
                    pass

        # --- tools.db: record_usage ---
        try:
            self.shared_state.tools.record_usage(
                tool_name=tool_name,
                success=success,
                duration_ms=duration_ms,
                context=None,
                error=error_msg,
            )
        except Exception:
            pass

        return result

    def _auto_register_skill(self, task: str, result: Dict[str, Any]) -> None:
        """
        Automatically register a skill pattern from a successfully completed task.

        Extracts a concise skill name and category from the task description,
        then upserts it into skills.db so the dashboard shows learned patterns.
        Uses OR IGNORE so re-running the same task won't create duplicates.
        """
        if not self.shared_state:
            return

        # Derive a short skill name (first sentence / 80 chars)
        name = task.split(".")[0].split("\n")[0].strip()[:80]
        if not name:
            return

        # Infer category from task keywords
        task_lower = task.lower()
        if any(w in task_lower for w in ["test", "pytest", "coverage", "jest"]):
            category = "testing"
        elif any(w in task_lower for w in ["fix", "debug", "error", "bug", "traceback"]):
            category = "debugging"
        elif any(w in task_lower for w in ["refactor", "clean", "improve", "simplif"]):
            category = "refactoring"
        elif any(w in task_lower for w in ["document", "readme", "docstring", "comment"]):
            category = "documentation"
        elif any(w in task_lower for w in ["security", "vuln", "injection", "auth"]):
            category = "security"
        elif any(w in task_lower for w in ["git", "commit", "branch", "pull request", "pr"]):
            category = "git"
        elif any(w in task_lower for w in ["search", "find", "index", "analyse", "analyze"]):
            category = "analysis"
        else:
            category = "coding"

        steps = [
            {"step": 1, "action": "execute", "description": name},
        ]

        try:
            # Check if skill with this name already exists to avoid duplicates
            with self.shared_state.skills.lock:
                existing = self.shared_state.skills.conn.execute(
                    "SELECT id FROM skills WHERE name = ?", (name,)
                ).fetchone()
                if existing:
                    # Update success_count on the existing skill
                    self.shared_state.skills.conn.execute(
                        "UPDATE skills SET success_count = success_count + 1, "
                        "last_used = CURRENT_TIMESTAMP WHERE name = ?",
                        (name,),
                    )
                    self.shared_state.skills.conn.commit()
                    logger.debug("[GaiaCodeAgent] updated existing skill: %s", name)
                else:
                    from uuid import uuid4
                    skill_id = str(uuid4())
                    self.shared_state.skills.conn.execute(
                        """
                        INSERT INTO skills
                            (id, name, description, category, steps, success_count, confidence)
                        VALUES (?, ?, ?, ?, ?, 1, 1.0)
                        """,
                        (skill_id, name, task[:500], category, json.dumps(steps)),
                    )
                    self.shared_state.skills.conn.commit()
                    logger.debug("[GaiaCodeAgent] auto-registered skill: %s (%s)", name, category)
        except Exception as e:
            logger.debug("[GaiaCodeAgent] skill auto-register skipped: %s", e)

    def checkpoint(self) -> Dict[str, Any]:
        """
        Create a checkpoint of current state.

        M3: Checkpoint/Resume - This enables crash recovery.
        """
        checkpoint_data = {
            "timestamp": datetime.now().isoformat(),
            "session_start": self.session_start.isoformat(),
            "task_start": self.task_start.isoformat() if self.task_start else None,
            "plan_tasks": (
                self.shared_state.plan.get_plan_tasks(self._current_plan_id)
                if self._current_plan_id else []
            ),
            "plan_id": self._current_plan_id,
            "audit_log": self.audit_log,
            "escalation_ladder": {
                "retry_count": self.escalation_ladder.retry_count,
            },
        }

        # Save to workspace
        checkpoint_path = self.shared_state.workspace_dir / "checkpoint.json"
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint_data, f, indent=2)

        return {
            "success": True,
            "checkpoint_path": str(checkpoint_path),
            "timestamp": checkpoint_data["timestamp"],
        }

    def resume_from_checkpoint(self) -> bool:
        """
        Resume from a previous checkpoint.

        M3: Checkpoint/Resume - This enables crash recovery.
        """
        checkpoint_path = self.shared_state.workspace_dir / "checkpoint.json"

        if not checkpoint_path.exists():
            return False

        try:
            with open(checkpoint_path, "r") as f:
                checkpoint_data = json.load(f)

            # Restore state
            self.session_start = datetime.fromisoformat(
                checkpoint_data["session_start"]
            )
            if checkpoint_data["task_start"]:
                self.task_start = datetime.fromisoformat(checkpoint_data["task_start"])

            self.audit_log = checkpoint_data["audit_log"]
            self.escalation_ladder.retry_count = checkpoint_data["escalation_ladder"][
                "retry_count"
            ]

            logger.info(f"Resumed from checkpoint: {checkpoint_data['timestamp']}")
            return True

        except Exception as e:
            logger.error(f"Failed to resume from checkpoint: {e}")
            return False
