# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
Interactive TUI: Planning Questions with Selectable Options

Integrates interactive planning into the TUI for beautiful UX.

Features:
- Display questions with numbered options
- Keyboard selection (1-9 for options)
- Always includes "Custom answer" option
- Real-time preview of plan as user answers
- Beautiful formatting with Rich
"""

from typing import Any, Dict, List, Optional

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt
    from rich.table import Table
    from rich.text import Text

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    Console = object


class InteractivePlanningTUI:
    """
    Interactive TUI for the planning phase.

    Shows questions with selectable options, builds plan interactively.
    """

    def __init__(self, console: Optional[Console] = None):
        self.console = console or (Console() if RICH_AVAILABLE else None)
        self.answers = {}

    def ask_question(
        self,
        question: str,
        options: Optional[List[str]] = None,
        importance: str = "important",
        category: str = "requirements",
        default: Optional[str] = None,
    ) -> str:
        """
        Ask a planning question with selectable options.

        Args:
            question: The question to ask
            options: List of predefined options (None for free text)
            importance: "critical", "important", or "optional"
            category: Question category
            default: Default option (if any)

        Returns:
            User's answer
        """
        if not self.console:
            # Fallback to simple input
            print(f"\n{question}")
            if options:
                for i, opt in enumerate(options, 1):
                    print(f"  {i}. {opt}")
                print(f"  {len(options) + 1}. Custom answer")
            return input("Your answer: ")

        # Rich TUI version
        self.console.print()

        # Importance indicator
        importance_markers = {
            "critical": "[bold red]❗ CRITICAL[/bold red]",
            "important": "[yellow]⚠️ IMPORTANT[/yellow]",
            "optional": "[blue]ℹ️ OPTIONAL[/blue]",
        }
        marker = importance_markers.get(importance, "")

        # Create question panel
        question_text = Text()
        if marker:
            question_text.append(marker + "\n\n")
        question_text.append(question, style="bold white")

        panel = Panel(
            question_text,
            title=f"[bold cyan]Planning Question[/bold cyan] [{category}]",
            border_style="cyan",
        )
        self.console.print(panel)

        if options:
            # Create options table
            table = Table(show_header=False, box=None, padding=(0, 2))
            table.add_column("Option", style="cyan")
            table.add_column("Description", style="white")

            for i, option in enumerate(options, 1):
                # Split on dash if present (for descriptions)
                if " - " in option:
                    name, desc = option.split(" - ", 1)
                    table.add_row(f"{i}.", f"[bold]{name}[/bold] - {desc}")
                else:
                    table.add_row(f"{i}.", option)

            # Always add custom option
            table.add_row(
                f"{len(options) + 1}.",
                "[dim]Custom answer (enter your own)[/dim]"
            )

            self.console.print(table)
            self.console.print()

            # Get selection
            while True:
                choice = Prompt.ask(
                    "[cyan]Your choice[/cyan]",
                    default=str(default) if default else "1",
                )

                try:
                    choice_num = int(choice)
                    if 1 <= choice_num <= len(options):
                        # Selected a predefined option
                        answer = options[choice_num - 1]
                        # Remove description part if present
                        if " - " in answer:
                            answer = answer.split(" - ")[0]
                        break
                    elif choice_num == len(options) + 1:
                        # Custom answer
                        answer = Prompt.ask("[cyan]Enter your answer[/cyan]")
                        break
                    else:
                        self.console.print("[red]Invalid choice. Try again.[/red]")
                except ValueError:
                    # Not a number - treat as custom answer
                    answer = choice
                    break

        else:
            # Free text question (no options)
            answer = Prompt.ask("[cyan]Your answer[/cyan]", default=default or "")

        # Show confirmation
        self.console.print(f"  ✓ Answered: [green]{answer}[/green]\n")

        return answer

    def show_plan_preview(self, plan_data: Dict):
        """
        Show a preview of the current plan being built.

        Args:
            plan_data: Plan data dictionary
        """
        if not self.console:
            return

        self.console.print()
        self.console.print("[bold cyan]═══ PLAN PREVIEW ═══[/bold cyan]")
        self.console.print()

        # Show goal
        self.console.print(f"[bold]Goal:[/bold] {plan_data.get('goal', 'Not set')}")
        self.console.print()

        # Show tasks
        tasks = plan_data.get("tasks", [])
        if tasks:
            self.console.print(f"[bold]Tasks ({len(tasks)}):[/bold]")
            for i, task in enumerate(tasks, 1):
                desc = task.get("description", "")
                complexity = task.get("complexity", "medium")
                complexity_color = {
                    "low": "green",
                    "medium": "yellow",
                    "high": "red",
                }.get(complexity, "white")

                self.console.print(f"  {i}. {desc} [{complexity_color}]{complexity}[/{complexity_color}]")

                if task.get("specialist"):
                    self.console.print(f"     [dim]→ {task['specialist']}[/dim]")

                if task.get("subtasks"):
                    for subtask in task["subtasks"][:3]:  # Show first 3
                        self.console.print(f"     [dim]• {subtask}[/dim]")
                    if len(task.get("subtasks", [])) > 3:
                        self.console.print(f"     [dim]• ... and {len(task['subtasks']) - 3} more[/dim]")

            self.console.print()

        # Show assumptions
        assumptions = plan_data.get("assumptions", [])
        if assumptions:
            self.console.print(f"[bold]Assumptions ({len(assumptions)}):[/bold]")
            for assumption in assumptions[:5]:  # Show first 5
                self.console.print(f"  • [dim]{assumption}[/dim]")
            if len(assumptions) > 5:
                self.console.print(f"  • [dim]... and {len(assumptions) - 5} more[/dim]")
            self.console.print()

        self.console.print("[bold cyan]═══════════════════[/bold cyan]")
        self.console.print()

    def ask_for_plan_approval(self, plan_data: Dict) -> bool:
        """
        Show full plan and ask for approval.

        Args:
            plan_data: Complete plan data

        Returns:
            True if approved, False if needs revision
        """
        if not self.console:
            return input("\nApprove this plan? (yes/no): ").lower().startswith("y")

        # Show complete plan
        self.console.print()
        self.console.print("[bold green]═" * 35 + "[/bold green]")
        self.console.print("[bold green]COMPLETE PLAN - READY FOR YOUR APPROVAL[/bold green]")
        self.console.print("[bold green]═" * 35 + "[/bold green]")
        self.console.print()

        self.show_plan_preview(plan_data)

        # Ask for approval
        self.console.print()
        response = Prompt.ask(
            "[bold cyan]Do you approve this plan?[/bold cyan]",
            choices=["yes", "no", "revise"],
            default="yes",
        )

        if response == "yes":
            self.console.print("[green]✓ Plan approved! Starting execution...[/green]")
            return True
        elif response == "no":
            self.console.print("[yellow]✗ Plan rejected. Let's start over.[/yellow]")
            return False
        else:  # revise
            revision_notes = Prompt.ask(
                "[cyan]What changes would you like?[/cyan]"
            )
            self.console.print(f"[yellow]✓ Will revise plan based on: {revision_notes}[/yellow]")
            return False

    def show_planning_progress(self, current_phase: str, progress: int):
        """
        Show progress through planning phases.

        Args:
            current_phase: Current planning phase
            progress: Progress percentage (0-100)
        """
        if not self.console:
            return

        phases = {
            "understanding": "1️⃣ Understanding Requirements",
            "decomposition": "2️⃣ Decomposing Tasks",
            "validation": "3️⃣ Validating Plan",
            "approved": "✓ Plan Approved",
        }

        phase_text = phases.get(current_phase, current_phase)

        self.console.print(f"\n[bold cyan]Planning Phase:[/bold cyan] {phase_text} ({progress}%)")

    def show_adaptive_replan_notice(self, reason: str):
        """
        Show notice that plan is being adapted during execution.

        Args:
            reason: Why plan is being adapted
        """
        if not self.console:
            return

        panel = Panel(
            f"[yellow]Plan being adapted:[/yellow]\n\n{reason}\n\n[dim]Replanning in progress...[/dim]",
            title="[yellow]⟳ ADAPTIVE REPLANNING[/yellow]",
            border_style="yellow",
        )

        self.console.print()
        self.console.print(panel)
        self.console.print()


def create_interactive_planning_tui(console: Optional[Console] = None) -> InteractivePlanningTUI:
    """
    Create an interactive planning TUI.

    Args:
        console: Optional Rich console

    Returns:
        InteractivePlanningTUI instance
    """
    return InteractivePlanningTUI(console)
