# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
Interactive Chat Session for GAIA Code

Provides a Claude Code-style interactive REPL where you can chat with the agent,
ask questions, request tasks, and iterate on solutions.

Features:
- Multi-turn conversation
- Beautiful chat UI with Rich
- Special commands (/help, /plan, /status, /persona, etc.)
- Conversation history
- Context awareness
- Interrupt and resume
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.prompt import Prompt
    from rich.syntax import Syntax
    from rich.text import Text

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class InteractiveSession:
    """
    Interactive chat session with GAIA Code agent.

    Similar to Claude Code's conversational interface.
    """

    def __init__(self, agent, console=None):
        """
        Initialize interactive session.

        Args:
            agent: GaiaCodeAgent instance
            console: Rich console (optional)
        """
        self.agent = agent
        self.console = console or (Console() if RICH_AVAILABLE else None)
        self.conversation_history = []
        self.session_active = True

        # Special commands
        self.commands = {
            "/help": self.cmd_help,
            "/plan": self.cmd_plan,
            "/status": self.cmd_status,
            "/persona": self.cmd_persona,
            "/tools": self.cmd_tools,
            "/clear": self.cmd_clear,
            "/checkpoint": self.cmd_checkpoint,
            "/audit": self.cmd_audit,
            "/exit": self.cmd_exit,
            "/quit": self.cmd_exit,
        }

    def start(self):
        """Start the interactive session."""
        self._show_welcome()

        while self.session_active:
            try:
                # Get user input
                user_input = self._get_user_input()

                if not user_input or not user_input.strip():
                    continue

                # Check for special commands
                if user_input.startswith("/"):
                    self._handle_command(user_input)
                    continue

                # Regular conversation
                self._handle_message(user_input)

            except KeyboardInterrupt:
                self._handle_interrupt()
            except EOFError:
                break
            except Exception as e:
                logger.error(f"Error in interactive session: {e}")
                if self.console:
                    self.console.print(f"[red]✗ Error: {e}[/red]")
                else:
                    print(f"✗ Error: {e}")

        self._show_goodbye()

    def _show_welcome(self):
        """Show welcome message."""
        if self.console:
            welcome = Panel(
                f"[bold cyan]GAIA Code Interactive Session[/bold cyan]\n\n"
                f"Persona: [yellow]{self.agent.persona.profile.name}[/yellow]\n"
                f"Model: [green]Claude Opus 4.6[/green]\n\n"
                f"Chat with me about your code! I can:\n"
                f"  • Answer questions about code\n"
                f"  • Implement features\n"
                f"  • Debug issues\n"
                f"  • Refactor code\n"
                f"  • Analyze architecture\n\n"
                f"Type [cyan]/help[/cyan] for commands or just chat naturally.",
                border_style="cyan",
                padding=(1, 2),
            )
            self.console.print()
            self.console.print(welcome)
            self.console.print()
        else:
            print("\n" + "=" * 70)
            print("GAIA CODE INTERACTIVE SESSION")
            print("=" * 70)
            print(f"\nPersona: {self.agent.persona.profile.name}")
            print(f"Model: Claude Opus 4.6")
            print("\nType /help for commands or just chat naturally.")
            print("=" * 70)
            print()

    def _get_user_input(self) -> str:
        """Get user input with nice prompt."""
        if self.console:
            return Prompt.ask("\n[bold cyan]You[/bold cyan]")
        else:
            return input("\nYou: ")

    def _handle_message(self, message: str):
        """Handle a regular chat message."""
        # DON'T add user message here - base Agent.process_query() handles
        # appending both user and assistant messages to conversation_history.
        # We just need to make sure prior history is synced to the agent.
        # The agent.conversation_history already has prior messages from
        # previous process_query calls.

        # Get agent response
        try:
            # Use the agent's process_query with the full conversation context
            # process_query reads self.conversation_history for prior context,
            # adds the new user message, gets LLM response, and appends both
            # user + assistant to self.conversation_history at the end.
            result = self.agent.process_query(message, create_plan=False)

            response = result.get("result", "Task completed")

            # Sync agent's conversation_history back to our local copy
            # (agent.conversation_history now includes the new user+assistant pair)
            self.conversation_history = list(self.agent.conversation_history)

            # Show agent's response
            self._show_agent_response(response)

        except Exception as e:
            error_msg = f"Error processing message: {e}"
            logger.error(error_msg)

            if self.console:
                self.console.print(f"[red]✗ {error_msg}[/red]")
            else:
                print(f"✗ {error_msg}")

    def _show_agent_response(self, response: str):
        """Show agent's response with nice formatting."""
        if self.console:
            # Check if response contains code
            if "```" in response:
                # Contains code blocks - render as markdown
                md = Markdown(response)
                panel = Panel(
                    md,
                    title=f"[bold green]{self.agent.persona.profile.name}[/bold green]",
                    border_style="green",
                    padding=(1, 2),
                )
                self.console.print(panel)
            else:
                # Plain text
                text = Text(response)
                panel = Panel(
                    text,
                    title=f"[bold green]{self.agent.persona.profile.name}[/bold green]",
                    border_style="green",
                    padding=(1, 2),
                )
                self.console.print(panel)
        else:
            print(f"\n{self.agent.persona.profile.name}: {response}")

    def _handle_command(self, command: str):
        """Handle a special command."""
        cmd_parts = command.split()
        cmd_name = cmd_parts[0].lower()

        if cmd_name in self.commands:
            self.commands[cmd_name]()
        else:
            if self.console:
                self.console.print(f"[red]Unknown command: {cmd_name}[/red]")
                self.console.print("[dim]Type /help for available commands[/dim]")
            else:
                print(f"Unknown command: {cmd_name}")
                print("Type /help for available commands")

    def cmd_help(self):
        """Show help."""
        if self.console:
            help_text = """
**Available Commands:**

`/help` - Show this help message
`/plan` - Show current execution plan
`/status` - Show agent status and progress
`/persona` - Show current persona or change it
`/tools` - List available tools
`/clear` - Clear conversation history
`/checkpoint` - Create a checkpoint
`/audit` - Show audit log
`/exit` or `/quit` - Exit interactive session

**Tips:**

• Just type naturally - "Create a REST API with auth"
• Ask questions - "How do I fix this error?"
• Request changes - "Make it simpler" or "Add error handling"
• Review code - "Explain what this code does"
• Iterate - "That's good, now add tests"
            """
            md = Markdown(help_text)
            panel = Panel(
                md,
                title="[bold cyan]Help[/bold cyan]",
                border_style="cyan",
            )
            self.console.print()
            self.console.print(panel)
        else:
            print("\nAvailable Commands:")
            print("  /help - Show this help")
            print("  /plan - Show current plan")
            print("  /status - Show status")
            print("  /persona - Show/change persona")
            print("  /tools - List tools")
            print("  /exit - Exit session")

    def cmd_plan(self):
        """Show current plan."""
        progress = self.agent.get_progress()

        if self.console:
            from rich.table import Table

            table = Table(title="Current Plan", show_header=True)
            table.add_column("Task", style="cyan")
            table.add_column("Status", justify="center")

            tasks = self.agent.shared_state.plan.get_all_tasks()
            for task in tasks:
                status_icon = {
                    "completed": "[green]✓[/green]",
                    "in_progress": "[yellow]⟳[/yellow]",
                    "failed": "[red]✗[/red]",
                    "pending": "[dim]◯[/dim]",
                }.get(task.status, "◯")

                table.add_row(task.description[:50], status_icon)

            self.console.print()
            self.console.print(table)
            self.console.print()
            self.console.print(f"Progress: {progress['progress_percent']}%")
        else:
            print("\nCurrent Plan:")
            tasks = self.agent.shared_state.plan.get_all_tasks()
            for i, task in enumerate(tasks, 1):
                status = task.status
                icon = {"completed": "✓", "in_progress": "⟳", "failed": "✗", "pending": "◯"}.get(status, "◯")
                print(f"  {icon} {i}. {task.description}")
            print(f"\nProgress: {progress['progress_percent']}%")

    def cmd_status(self):
        """Show agent status."""
        progress = self.agent.get_progress()

        if self.console:
            status_text = f"""
**Session Info:**
• Total tasks: {progress['total_tasks']}
• Completed: {progress['completed']}
• In progress: {progress['in_progress']}
• Pending: {progress['pending']}
• Progress: {progress['progress_percent']}%

**Agent Config:**
• Persona: {self.agent.persona.profile.name}
• Quality gates: {'Enabled' if self.agent.quality_gates.gates else 'Disabled'}
• Workspace: {self.agent.shared_state.workspace_dir}
            """
            md = Markdown(status_text)
            self.console.print()
            self.console.print(md)
        else:
            print(f"\nStatus:")
            print(f"  Tasks: {progress['completed']}/{progress['total_tasks']}")
            print(f"  Progress: {progress['progress_percent']}%")
            print(f"  Persona: {self.agent.persona.profile.name}")

    def cmd_persona(self):
        """Show or change persona."""
        if self.console:
            from gaia.agents.gaia_code.persona import list_personas

            personas = list_personas()

            self.console.print()
            self.console.print(f"[bold]Current Persona:[/bold] [yellow]{self.agent.persona.profile.name}[/yellow]")
            self.console.print(f"[dim]{self.agent.persona.profile.description}[/dim]")
            self.console.print()

            change = Prompt.ask(
                "Change persona?",
                choices=["yes", "no"],
                default="no"
            )

            if change == "yes":
                self.console.print("\nAvailable Personas:")
                for i, p in enumerate(personas, 1):
                    self.console.print(f"  {i}. [cyan]{p['name']}[/cyan] - {p['description'][:50]}...")

                choice = Prompt.ask("\nSelect persona (1-8)", default="1")
                try:
                    idx = int(choice) - 1
                    if 0 <= idx < len(personas):
                        new_persona_name = personas[idx]['name'].lower()
                        from gaia.agents.gaia_code.persona import create_persona
                        self.agent.persona = create_persona(new_persona_name, self.agent.shared_state.workspace_dir)
                        self.console.print(f"\n[green]✓[/green] Persona changed to [yellow]{self.agent.persona.profile.name}[/yellow]")
                except:
                    self.console.print("[red]Invalid choice[/red]")
        else:
            print(f"\nCurrent Persona: {self.agent.persona.profile.name}")
            print(f"{self.agent.persona.profile.description}")

    def cmd_tools(self):
        """List available tools."""
        from gaia.agents.base.tools import _TOOL_REGISTRY

        if self.console:
            from rich.table import Table

            table = Table(title=f"Available Tools ({len(_TOOL_REGISTRY)})", show_header=True)
            table.add_column("Tool Name", style="cyan")
            table.add_column("Description")

            for tool_name, tool_info in list(_TOOL_REGISTRY.items())[:20]:  # Show first 20
                desc = tool_info.get("description", "No description")[:60]
                table.add_row(tool_name, desc)

            if len(_TOOL_REGISTRY) > 20:
                table.add_row("[dim]...[/dim]", f"[dim]... and {len(_TOOL_REGISTRY) - 20} more tools[/dim]")

            self.console.print()
            self.console.print(table)
        else:
            print(f"\nAvailable Tools ({len(_TOOL_REGISTRY)}):")
            for tool_name in list(_TOOL_REGISTRY.keys())[:20]:
                print(f"  - {tool_name}")
            if len(_TOOL_REGISTRY) > 20:
                print(f"  ... and {len(_TOOL_REGISTRY) - 20} more")

    def cmd_clear(self):
        """Clear conversation history."""
        self.conversation_history.clear()
        self.agent.conversation_history.clear()

        if self.console:
            self.console.clear()
            self.console.print("[green]✓[/green] Conversation history cleared")
        else:
            print("\n✓ Conversation history cleared")

    def cmd_checkpoint(self):
        """Create checkpoint."""
        result = self.agent.checkpoint()

        if self.console:
            self.console.print()
            self.console.print(f"[green]✓[/green] Checkpoint created")
            self.console.print(f"[dim]Location: {result['checkpoint_path']}[/dim]")
        else:
            print(f"\n✓ Checkpoint created: {result['checkpoint_path']}")

    def cmd_audit(self):
        """Show audit log."""
        log = self.agent.get_audit_log()

        if self.console:
            self.console.print()
            self.console.print(f"[bold]Audit Log[/bold] ({len(log)} entries)")
            self.console.print()

            for entry in log[-10:]:  # Show last 10
                timestamp = entry.get("timestamp", "")
                action = entry.get("action_type", "")
                self.console.print(f"[dim]{timestamp}[/dim] [cyan]{action}[/cyan]")

            if len(log) > 10:
                self.console.print(f"\n[dim]... and {len(log) - 10} more entries[/dim]")
        else:
            print(f"\nAudit Log ({len(log)} entries):")
            for entry in log[-10:]:
                print(f"  {entry.get('timestamp', '')} - {entry.get('action_type', '')}")

    def cmd_exit(self):
        """Exit the session."""
        self.session_active = False

    def _handle_interrupt(self):
        """Handle Ctrl+C."""
        if self.console:
            self.console.print("\n\n[yellow]Interrupted[/yellow]")
            choice = Prompt.ask(
                "What would you like to do?",
                choices=["continue", "checkpoint", "exit"],
                default="continue"
            )

            if choice == "exit":
                self.session_active = False
            elif choice == "checkpoint":
                self.cmd_checkpoint()
        else:
            print("\n\nInterrupted")
            choice = input("Continue, checkpoint, or exit? [continue/checkpoint/exit]: ")
            if choice == "exit":
                self.session_active = False
            elif choice == "checkpoint":
                self.cmd_checkpoint()

    def _show_goodbye(self):
        """Show goodbye message."""
        if self.console:
            self.console.print()
            self.console.print("[bold cyan]Thanks for using GAIA Code! 👋[/bold cyan]")
            self.console.print()
        else:
            print("\nThanks for using GAIA Code! 👋\n")


def start_interactive_session(
    persona: str = "pike",
    workspace_dir: Optional[Path] = None,
    tui_mode: str = "simple",
):
    """
    Start an interactive chat session with GAIA Code.

    Args:
        persona: Personality to use
        workspace_dir: Workspace directory
        tui_mode: TUI mode for task execution

    Returns:
        None (runs until user exits)
    """
    from .agent import GaiaCodeAgent

    # Create console
    console = Console() if RICH_AVAILABLE else None

    # Show loading message
    if console:
        with console.status("[cyan]Initializing GAIA Code...[/cyan]", spinner="dots"):
            agent = GaiaCodeAgent(
                workspace_dir=workspace_dir,
                persona=persona,
                tui_mode=tui_mode,
                silent_mode=False,
            )
    else:
        print("Initializing GAIA Code...")
        agent = GaiaCodeAgent(
            workspace_dir=workspace_dir,
            persona=persona,
            tui_mode=tui_mode,
            silent_mode=False,
        )

    # Start interactive session
    session = InteractiveSession(agent, console)
    session.start()
