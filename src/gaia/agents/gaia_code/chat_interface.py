# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
Advanced Chat Interface with Autocomplete

Beautiful interactive chat interface using prompt_toolkit with:
- Path autocomplete
- Command autocomplete
- Syntax highlighting
- Multi-line input
- History
- Better UX than gaia chat
"""

import os
from pathlib import Path
from typing import List, Optional

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
    from prompt_toolkit.completion import Completer, Completion, PathCompleter
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.lexers import PygmentsLexer
    from prompt_toolkit.styles import Style
    from pygments.lexers.python import PythonLexer
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.syntax import Syntax

    PROMPT_TOOLKIT_AVAILABLE = True
except ImportError:
    PROMPT_TOOLKIT_AVAILABLE = False


class GAIACodeCompleter(Completer):
    """
    Custom completer for GAIA Code chat.

    Provides autocomplete for:
    - Special commands (/help, /plan, /status, etc.)
    - File paths
    - Common coding terms
    - Personas
    """

    def __init__(self):
        self.commands = [
            "/help",
            "/plan",
            "/status",
            "/persona",
            "/tools",
            "/clear",
            "/checkpoint",
            "/audit",
            "/exit",
            "/quit",
        ]

        self.personas = [
            "torvalds",
            "knuth",
            "pike",
            "carmack",
            "hickey",
            "kay",
            "thompson",
            "hopper",
        ]

        self.common_tasks = [
            "Create a ",
            "Build a ",
            "Fix ",
            "Debug ",
            "Refactor ",
            "Test ",
            "Optimize ",
            "Explain ",
            "Analyze ",
            "Review ",
        ]

        self.path_completer = PathCompleter(expanduser=True)

    def get_completions(self, document, complete_event):
        """Get completions for current input."""
        text = document.text_before_cursor

        # Command completions (starts with /)
        if text.startswith("/"):
            for cmd in self.commands:
                if cmd.startswith(text):
                    yield Completion(
                        cmd,
                        start_position=-len(text),
                        display=cmd,
                        display_meta="command",
                    )

        # Persona completions (after --persona or "use persona")
        elif "--persona" in text or "persona" in text.lower():
            word = text.split()[-1] if text.split() else ""
            for persona in self.personas:
                if persona.startswith(word.lower()):
                    yield Completion(
                        persona,
                        start_position=-len(word),
                        display=persona,
                        display_meta="personality",
                    )

        # Path completions (if looks like a path)
        elif "/" in text or "." in text or text.endswith("py"):
            for completion in self.path_completer.get_completions(document, complete_event):
                yield completion

        # Common task starters
        elif len(text) < 10 and not text.startswith("/"):
            for task in self.common_tasks:
                if task.lower().startswith(text.lower()):
                    yield Completion(
                        task,
                        start_position=-len(text),
                        display=task,
                        display_meta="suggestion",
                    )


class AdvancedChatInterface:
    """
    Advanced chat interface with autocomplete and beautiful formatting.

    Better than gaia chat with:
    - Path autocomplete
    - Command autocomplete
    - Syntax highlighting
    - Multi-line input (Alt+Enter)
    - History (up/down arrows)
    - Vi/Emacs keybindings
    """

    def __init__(self, agent, persona_name: str = "pike"):
        self.agent = agent
        self.persona_name = persona_name
        self.console = Console()

        if not PROMPT_TOOLKIT_AVAILABLE:
            raise ImportError(
                "prompt_toolkit required for advanced chat interface.\n"
                "Install with: pip install prompt-toolkit pygments"
            )

        # Create prompt session with all features
        history_file = Path.home() / ".gaia" / "cache" / "chat_history.txt"
        history_file.parent.mkdir(parents=True, exist_ok=True)

        # Custom style
        self.style = Style.from_dict({
            'prompt': '#ed1c24 bold',
            'input': '#e0e0e0',
        })

        # Key bindings
        kb = KeyBindings()

        @kb.add('c-d')  # Ctrl+D to exit
        def _(event):
            event.app.exit()

        # Create session
        self.session = PromptSession(
            history=FileHistory(str(history_file)),
            auto_suggest=AutoSuggestFromHistory(),
            completer=GAIACodeCompleter(),
            complete_while_typing=True,
            lexer=PygmentsLexer(PythonLexer),
            style=self.style,
            key_bindings=kb,
        )

        self.conversation_active = True

    def start(self):
        """Start the chat interface."""
        self._show_welcome()

        while self.conversation_active:
            try:
                # Get user input with autocomplete
                user_input = self.session.prompt(
                    [
                        ('class:prompt', '\n❯ '),
                        ('class:input', ''),
                    ],
                    multiline=False,
                )

                if not user_input.strip():
                    continue

                # Handle commands
                if user_input.startswith("/"):
                    self._handle_command(user_input)
                else:
                    self._handle_message(user_input)

            except KeyboardInterrupt:
                continue  # Ctrl+C just cancels current input
            except EOFError:
                break  # Ctrl+D exits

        self._show_goodbye()

    def _show_welcome(self):
        """Show welcome with rich formatting."""
        welcome = Panel(
            f"[bold cyan]GAIA Code Interactive Chat[/bold cyan]\n\n"
            f"Persona: [yellow]{self.persona_name.title()}[/yellow]\n"
            f"Model: [green]Claude Opus 4.6[/green]\n\n"
            f"Features:\n"
            f"  • [cyan]Path autocomplete[/cyan] - Type file paths, press Tab\n"
            f"  • [cyan]Command autocomplete[/cyan] - Type /, see commands\n"
            f"  • [cyan]History[/cyan] - Up/Down arrows\n"
            f"  • [cyan]Multi-line[/cyan] - Alt+Enter for new line\n\n"
            f"Commands: [dim]/help /plan /status /persona /exit[/dim]\n"
            f"Shortcuts: [dim]Ctrl+D=exit, Ctrl+C=cancel, Tab=autocomplete[/dim]",
            border_style="cyan",
            padding=(1, 2),
        )
        self.console.print()
        self.console.print(welcome)

    def _handle_message(self, message: str):
        """Handle user message."""
        # Show user message
        self.console.print()

        # Get agent response
        try:
            result = self.agent.process_query(message, create_plan=False)
            response = result.get("result", "Task completed")

            # Show agent response with formatting
            self._show_agent_response(response)

        except Exception as e:
            self.console.print(f"[red]✗ Error: {e}[/red]")

    def _show_agent_response(self, response: str):
        """Show agent response with nice formatting."""
        # Check if response contains code
        if "```" in response or "def " in response or "class " in response:
            # Render as markdown (handles code blocks)
            md = Markdown(response)
            panel = Panel(
                md,
                title=f"[bold green]{self.persona_name.title()}[/bold green]",
                border_style="green",
                padding=(1, 2),
            )
            self.console.print(panel)
        else:
            # Plain text
            panel = Panel(
                response,
                title=f"[bold green]{self.persona_name.title()}[/bold green]",
                border_style="green",
                padding=(1, 2),
            )
            self.console.print(panel)

    def _handle_command(self, command: str):
        """Handle special commands."""
        if command in ("/exit", "/quit"):
            self.conversation_active = False
        elif command == "/help":
            self._show_help()
        elif command == "/clear":
            self.console.clear()
            self._show_welcome()
        elif command == "/status":
            self._show_status()
        elif command == "/persona":
            self._show_persona_menu()
        else:
            self.console.print(f"[yellow]Unknown command: {command}[/yellow]")
            self.console.print("[dim]Type /help for available commands[/dim]")

    def _show_help(self):
        """Show help."""
        help_text = """
**Commands:**
- `/help` - Show this help
- `/plan` - Show execution plan
- `/status` - Agent status
- `/persona` - Change personality
- `/tools` - List available tools
- `/clear` - Clear screen
- `/exit` - End session

**Shortcuts:**
- `Tab` - Autocomplete paths/commands
- `↑↓` - Navigate history
- `Ctrl+C` - Cancel input
- `Ctrl+D` - Exit session

**Tips:**
- Just type naturally: "Create a REST API"
- Paths autocomplete: "Read file ./sr[Tab]" → "./src/"
- Commands autocomplete: "/he[Tab]" → "/help"
        """
        md = Markdown(help_text)
        self.console.print()
        self.console.print(Panel(md, title="[cyan]Help[/cyan]", border_style="cyan"))

    def _show_status(self):
        """Show agent status."""
        progress = self.agent.get_progress()
        self.console.print()
        self.console.print(f"[bold]Status:[/bold]")
        self.console.print(f"  Tasks: {progress['completed']}/{progress['total_tasks']}")
        self.console.print(f"  Progress: {progress['progress_percent']}%")
        self.console.print(f"  Persona: {self.persona_name.title()}")

    def _show_persona_menu(self):
        """Show persona selection."""
        from prompt_toolkit.shortcuts import radiolist_dialog

        personas = [
            ("torvalds", "Torvalds - Brutally honest"),
            ("knuth", "Knuth - Thorough teacher"),
            ("pike", "Pike - Simplicity advocate"),
            ("carmack", "Carmack - Performance-obsessed"),
            ("hickey", "Hickey - Thoughtful questioner"),
            ("kay", "Kay - Visionary thinker"),
            ("thompson", "Thompson - Minimalist"),
            ("hopper", "Hopper - Practical problem-solver"),
        ]

        result = radiolist_dialog(
            title="Select Persona",
            text="Choose your agent's personality:",
            values=personas,
        ).run()

        if result:
            self.persona_name = result
            from gaia.agents.gaia_code.persona import create_persona
            self.agent.persona = create_persona(result, self.agent.shared_state.workspace_dir)
            self.console.print(f"\n[green]✓[/green] Persona changed to [yellow]{result.title()}[/yellow]")

    def _show_goodbye(self):
        """Show goodbye."""
        self.console.print()
        self.console.print("[bold cyan]Thanks for using GAIA Code! 👋[/bold cyan]")
        self.console.print()


def start_advanced_chat(
    agent,
    persona_name: str = "pike",
):
    """
    Start advanced chat interface.

    Args:
        agent: GaiaCodeAgent instance
        persona_name: Persona name

    Requires:
        pip install prompt-toolkit pygments
    """
    if not PROMPT_TOOLKIT_AVAILABLE:
        print("Advanced chat interface requires prompt-toolkit")
        print("Install with: pip install prompt-toolkit pygments")
        print()
        print("Falling back to simple interactive mode...")
        from .interactive_session import InteractiveSession
        session = InteractiveSession(agent)
        session.start()
        return

    interface = AdvancedChatInterface(agent, persona_name)
    interface.start()
