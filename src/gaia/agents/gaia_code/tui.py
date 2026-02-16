# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
GAIA Code TUI: World-Class Terminal User Interface

A beautiful, informative, non-overwhelming TUI that exceeds all existing CLIs.

Design Principles:
1. **Clarity over Verbosity** - Show what matters, hide noise
2. **Visual Hierarchy** - Most important info is most prominent
3. **Real-time Updates** - Live progress without spam
4. **Scannable** - User can glance and understand instantly
5. **Delightful** - Smooth animations, clear indicators, satisfying feedback

Uses Rich library for:
- Live updating layouts
- Progress bars
- Clean tables
- Syntax highlighting
- Panels and boxes
- Spinners and status
"""

import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

# Import Rich with fallback
try:
    from rich.align import Align
    from rich.console import Console, Group
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TaskID,
        TextColumn,
        TimeElapsedColumn,
    )
    from rich.status import Status
    from rich.syntax import Syntax
    from rich.table import Table
    from rich.text import Text

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    # Simple fallback classes for when Rich is not available
    class Console:
        def print(self, *args, **kwargs):
            print(*args)

    class Status:
        def __init__(self, *args, **kwargs):
            pass
        def start(self):
            pass
        def stop(self):
            pass
        def update(self, *args):
            pass

    class Progress:
        def __init__(self, *args, **kwargs):
            self.tasks = []
        def add_task(self, *args, **kwargs):
            return 0
        def update(self, *args, **kwargs):
            pass
        def start(self):
            pass
        def stop(self):
            pass

    TaskID = int


@dataclass
class TaskProgress:
    """Track progress of a task."""

    description: str
    status: str = "pending"  # pending | running | completed | failed
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[str] = None
    error: Optional[str] = None


class GaiaCodeTUI:
    """
    World-class TUI for GAIA Code.

    Features:
    - Clean, minimal design
    - Real-time progress updates
    - Quality gate status
    - Task tree visualization
    - Agent activity log (last 5 only)
    - Time tracking
    - No spam, no overwhelm
    """

    def __init__(self, console: Optional[Console] = None):
        """
        Initialize the TUI.

        Args:
            console: Rich console instance (creates new if not provided)
        """
        self.console = console or Console()

        # State
        self.current_task: Optional[str] = None
        self.task_progress: List[TaskProgress] = []
        self.quality_gates: Dict[str, str] = {}  # gate -> status
        self.activity_log: List[str] = []  # Last 5 activities
        self.started_at: Optional[datetime] = None
        self.current_step: int = 0
        self.total_steps: int = 0

        # Rich components
        self.layout = self._create_layout()
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=self.console,
        )
        self.task_id: Optional[TaskID] = None
        self.live: Optional[Live] = None

    def _create_layout(self) -> Layout:
        """Create the main layout."""
        layout = Layout()

        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3),
        )

        layout["body"].split_row(
            Layout(name="main", ratio=2),
            Layout(name="sidebar", ratio=1),
        )

        return layout

    def start(self, task: str, estimated_steps: int = 0):
        """
        Start the TUI for a new task.

        Args:
            task: Task description
            estimated_steps: Estimated number of steps (0 = unknown)
        """
        self.current_task = task
        self.started_at = datetime.now()
        self.total_steps = estimated_steps

        # Create progress bar
        if estimated_steps > 0:
            self.task_id = self.progress.add_task(
                task, total=estimated_steps
            )

        # Start live display
        self.live = Live(
            self._render(),
            console=self.console,
            refresh_per_second=2,
            transient=False,
        )
        self.live.start()

    def update_step(self, step: int, description: str):
        """
        Update current step.

        Args:
            step: Current step number
            description: Step description
        """
        self.current_step = step

        if self.task_id is not None:
            self.progress.update(self.task_id, completed=step)

        self._add_activity(f"Step {step}: {description}")

        if self.live:
            self.live.update(self._render())

    def update_task_progress(self, tasks: List[TaskProgress]):
        """
        Update task progress.

        Args:
            tasks: List of task progress objects
        """
        self.task_progress = tasks

        if self.live:
            self.live.update(self._render())

    def update_quality_gates(self, gates: Dict[str, str]):
        """
        Update quality gate status.

        Args:
            gates: Dict mapping gate name to status ("pass" | "fail" | "running" | "pending")
        """
        self.quality_gates = gates

        if self.live:
            self.live.update(self._render())

    def add_activity(self, message: str):
        """
        Add an activity message.

        Args:
            message: Activity message
        """
        self._add_activity(message)

        if self.live:
            self.live.update(self._render())

    def _add_activity(self, message: str):
        """Internal: Add activity and keep only last 5."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.activity_log.append(f"[dim]{timestamp}[/dim] {message}")

        # Keep only last 5
        if len(self.activity_log) > 5:
            self.activity_log = self.activity_log[-5:]

    def complete(self, success: bool = True, message: Optional[str] = None):
        """
        Complete the task and stop the TUI.

        Args:
            success: Whether task succeeded
            message: Optional completion message
        """
        if self.live:
            self.live.stop()

        # Show completion panel
        if success:
            self._show_success(message)
        else:
            self._show_failure(message)

    def _show_success(self, message: Optional[str]):
        """Show success message."""
        elapsed = self._elapsed_time()

        content = Text()
        content.append("✓ Task completed successfully\n", style="bold green")

        if message:
            content.append(f"\n{message}\n", style="white")

        content.append(f"\nTime: {elapsed}", style="dim")

        panel = Panel(
            content,
            title="[bold green]Success[/bold green]",
            border_style="green",
        )

        self.console.print(panel)

    def _show_failure(self, message: Optional[str]):
        """Show failure message."""
        elapsed = self._elapsed_time()

        content = Text()
        content.append("✗ Task failed\n", style="bold red")

        if message:
            content.append(f"\n{message}\n", style="white")

        content.append(f"\nTime: {elapsed}", style="dim")

        panel = Panel(
            content,
            title="[bold red]Failed[/bold red]",
            border_style="red",
        )

        self.console.print(panel)

    def _render(self) -> Layout:
        """Render the current state to layout."""
        # Header
        self.layout["header"].update(self._render_header())

        # Main content
        self.layout["main"].update(self._render_main())

        # Sidebar
        self.layout["sidebar"].update(self._render_sidebar())

        # Footer
        self.layout["footer"].update(self._render_footer())

        return self.layout

    def _render_header(self) -> Panel:
        """Render header with task and time."""
        elapsed = self._elapsed_time()

        # Progress indicator
        if self.total_steps > 0:
            progress_text = f"{self.current_step}/{self.total_steps} steps"
            percent = int((self.current_step / self.total_steps) * 100)
            progress_bar = "█" * (percent // 5) + "░" * (20 - percent // 5)
            progress = f"{progress_bar} {percent}%"
        else:
            progress_text = f"{self.current_step} steps"
            progress = "⟳ Running"

        header_text = Text()
        header_text.append("GAIA Code", style="bold cyan")
        header_text.append(f"  •  {progress_text}  •  {elapsed}", style="dim")

        return Panel(
            Align.center(header_text),
            style="cyan",
            box=None,
        )

    def _render_main(self) -> Panel:
        """Render main content area."""
        # Current task
        task_text = Text()
        task_text.append("Task: ", style="bold")
        task_text.append(self.current_task or "Initializing...", style="white")

        # Task progress (if any)
        if self.task_progress:
            progress_table = self._create_task_table()
            content = Group(task_text, Text(), progress_table)
        else:
            content = task_text

        return Panel(
            content,
            title="[bold]Current Task[/bold]",
            border_style="blue",
            padding=(1, 2),
        )

    def _render_sidebar(self) -> Panel:
        """Render sidebar with quality gates and activity."""
        # Quality gates
        gates_content = self._render_quality_gates()

        # Recent activity
        activity_content = self._render_activity()

        content = Group(gates_content, Text(), activity_content)

        return Panel(
            content,
            title="[bold]Status[/bold]",
            border_style="magenta",
            padding=(1, 2),
        )

    def _render_quality_gates(self) -> Table:
        """Render quality gates status."""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Gate", style="bold")
        table.add_column("Status")

        if not self.quality_gates:
            table.add_row("Quality Gates", "[dim]Not started[/dim]")
        else:
            for gate, status in self.quality_gates.items():
                if status == "pass":
                    icon = "✓"
                    style = "green"
                elif status == "fail":
                    icon = "✗"
                    style = "red"
                elif status == "running":
                    icon = "⟳"
                    style = "yellow"
                else:  # pending
                    icon = "◯"
                    style = "dim"

                table.add_row(
                    gate.capitalize(),
                    f"[{style}]{icon} {status.capitalize()}[/{style}]",
                )

        return table

    def _render_activity(self) -> Group:
        """Render recent activity log."""
        if not self.activity_log:
            return Group(
                Text("Recent Activity", style="bold"),
                Text("[dim]No activity yet[/dim]"),
            )

        lines = [Text("Recent Activity", style="bold")]

        for activity in self.activity_log:
            lines.append(Text.from_markup(activity))

        return Group(*lines)

    def _render_footer(self) -> Panel:
        """Render footer with help text."""
        help_text = Text()
        help_text.append("Press ", style="dim")
        help_text.append("Ctrl+C", style="bold yellow")
        help_text.append(" to interrupt  •  ", style="dim")
        help_text.append("gaia code --status", style="bold")
        help_text.append(" for details", style="dim")

        return Panel(
            Align.center(help_text),
            style="dim",
            box=None,
        )

    def _create_task_table(self) -> Table:
        """Create task progress table."""
        table = Table(show_header=True, box=None, padding=(0, 1))
        table.add_column("Task", style="white")
        table.add_column("Status", justify="center")

        for task in self.task_progress[:10]:  # Show max 10
            # Status icon
            if task.status == "completed":
                icon = "[green]✓[/green]"
            elif task.status == "running":
                icon = "[yellow]⟳[/yellow]"
            elif task.status == "failed":
                icon = "[red]✗[/red]"
            else:  # pending
                icon = "[dim]◯[/dim]"

            # Task description (truncate if too long)
            desc = task.description
            if len(desc) > 50:
                desc = desc[:47] + "..."

            table.add_row(desc, icon)

        if len(self.task_progress) > 10:
            table.add_row(
                f"[dim]... and {len(self.task_progress) - 10} more[/dim]",
                "[dim]...[/dim]",
            )

        return table

    def _elapsed_time(self) -> str:
        """Get elapsed time string."""
        if not self.started_at:
            return "0s"

        elapsed = datetime.now() - self.started_at
        total_seconds = int(elapsed.total_seconds())

        if total_seconds < 60:
            return f"{total_seconds}s"
        elif total_seconds < 3600:
            minutes = total_seconds // 60
            seconds = total_seconds % 60
            return f"{minutes}m {seconds}s"
        else:
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            return f"{hours}h {minutes}m"

    # ========================================================================
    # Simple Mode: For Single-Line Updates
    # ========================================================================

    def show_spinner(self, message: str):
        """Show a spinner with message (for quick operations)."""
        return Status(message, spinner="dots", console=self.console)

    def show_progress_bar(self, description: str, total: int) -> TaskID:
        """
        Show a simple progress bar.

        Args:
            description: What's being done
            total: Total steps

        Returns:
            Task ID for updating
        """
        return self.progress.add_task(description, total=total)

    def print_success(self, message: str):
        """Print a success message."""
        self.console.print(f"[green]✓[/green] {message}")

    def print_error(self, message: str):
        """Print an error message."""
        self.console.print(f"[red]✗[/red] {message}")

    def print_info(self, message: str):
        """Print an info message."""
        self.console.print(f"[blue]ℹ[/blue] {message}")

    def print_warning(self, message: str):
        """Print a warning message."""
        self.console.print(f"[yellow]⚠[/yellow] {message}")


class GaiaCodeSimpleTUI:
    """
    Simplified TUI for GAIA Code - Clean and Minimal.

    Shows only what matters:
    - Current task (1 line)
    - Progress bar (1 line)
    - Current step (1 line)
    - Quality gates summary (1 line)
    - Elapsed time (1 line)

    Total: ~5 lines, updates in place.
    """

    def __init__(self, console: Optional[Console] = None):
        """Initialize simple TUI."""
        self.console = console or Console()
        self.progress = Progress(
            SpinnerColumn(spinner_name="dots"),
            TextColumn("[bold blue]{task.fields[stage]}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("•"),
            TextColumn("{task.fields[current]}"),
            TextColumn("•"),
            TimeElapsedColumn(),
            console=self.console,
            transient=True,
        )
        self.task_id: Optional[TaskID] = None
        self.started_at: Optional[datetime] = None

    def start(self, task: str):
        """Start a task."""
        self.started_at = datetime.now()

        # Show task
        self.console.print()
        self.console.print(f"[bold cyan]▶ GAIA Code[/bold cyan]")
        self.console.print(f"  [white]{task}[/white]")
        self.console.print()

        # Start progress
        self.progress.start()
        self.task_id = self.progress.add_task(
            "",
            total=100,
            stage="Initializing",
            current="Starting...",
        )

    def update(
        self,
        stage: str,
        current: str,
        percent: int = None,
        completed: int = None,
    ):
        """
        Update progress.

        Args:
            stage: Current stage (e.g., "Planning", "Executing", "Testing")
            current: Current activity (e.g., "Creating models")
            percent: Progress percentage (0-100)
            completed: Completed steps (if percent not provided)
        """
        if self.task_id is not None:
            if percent is not None:
                self.progress.update(
                    self.task_id,
                    completed=percent,
                    stage=stage,
                    current=current,
                )
            elif completed is not None:
                self.progress.update(
                    self.task_id,
                    completed=completed,
                    stage=stage,
                    current=current,
                )

    def update_quality_gates(self, gates: Dict[str, bool]):
        """
        Update quality gates.

        Args:
            gates: Dict of gate_name -> passed
        """
        # Show gates status inline
        gate_status = []
        for gate, passed in gates.items():
            if passed:
                gate_status.append(f"[green]✓ {gate}[/green]")
            else:
                gate_status.append(f"[red]✗ {gate}[/red]")

        status_text = " ".join(gate_status)

        self.update(
            stage="Quality Gates",
            current=status_text,
        )

    def complete(self, success: bool = True, message: Optional[str] = None):
        """Complete the task."""
        if self.progress.tasks:
            self.progress.stop()

        self.console.print()

        if success:
            self.console.print(f"[bold green]✓ Success[/bold green]")
        else:
            self.console.print(f"[bold red]✗ Failed[/bold red]")

        if message:
            self.console.print(f"  {message}")

        if self.started_at:
            elapsed = datetime.now() - self.started_at
            self.console.print(f"  [dim]Time: {self._format_time(elapsed)}[/dim]")

        self.console.print()

    def show_quality_gates_detail(self, results: List[Dict]):
        """
        Show detailed quality gate results.

        Args:
            results: List of gate result dicts
        """
        self.console.print()
        self.console.print("[bold]Quality Gate Results[/bold]")
        self.console.print()

        table = Table(show_header=True, box=None)
        table.add_column("Gate", style="bold")
        table.add_column("Status", justify="center")
        table.add_column("Details")

        for result in results:
            if result.get("passed"):
                status = "[green]✓ Pass[/green]"
            else:
                status = "[red]✗ Fail[/red]"

            details = result.get("message", "")
            table.add_row(result.get("name", ""), status, details)

        self.console.print(table)

    def show_plan(self, tasks: List[Dict]):
        """
        Show the task plan.

        Args:
            tasks: List of task dicts with description and status
        """
        self.console.print()
        self.console.print("[bold]Execution Plan[/bold]")
        self.console.print()

        for i, task in enumerate(tasks, 1):
            status = task.get("status", "pending")

            if status == "completed":
                icon = "[green]✓[/green]"
                style = "dim"
            elif status == "in_progress":
                icon = "[yellow]▶[/yellow]"
                style = "bold"
            elif status == "failed":
                icon = "[red]✗[/red]"
                style = "red"
            else:  # pending
                icon = "[dim]◯[/dim]"
                style = "dim"

            desc = task.get("description", "")
            self.console.print(f"{icon} [{style}]{i}. {desc}[/{style}]")

        self.console.print()

    def _format_time(self, delta: timedelta) -> str:
        """Format time delta."""
        total_seconds = int(delta.total_seconds())

        if total_seconds < 60:
            return f"{total_seconds}s"
        elif total_seconds < 3600:
            minutes = total_seconds // 60
            seconds = total_seconds % 60
            return f"{minutes}m {seconds}s"
        else:
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            return f"{hours}h {minutes}m"


class GaiaCodeMinimalTUI:
    """
    Minimal TUI - Just a single line that updates.

    Perfect for: Users who want the least distraction.

    Shows:
    ⟳ Step 5/12 • Creating auth endpoints • 42% • 2m 34s
    """

    def __init__(self, console: Optional[Console] = None):
        """Initialize minimal TUI."""
        self.console = console or Console()
        self.status: Optional[Status] = None
        self.started_at: Optional[datetime] = None

    def start(self, task: str):
        """Start task."""
        self.started_at = datetime.now()
        self.console.print(f"\n[bold cyan]GAIA Code:[/bold cyan] {task}\n")

    def update(self, step: int, total: int, current: str, percent: int = None):
        """Update status line."""
        if percent is None and total > 0:
            percent = int((step / total) * 100)

        elapsed = self._elapsed()
        status_text = f"Step {step}/{total} • {current} • {percent}% • {elapsed}"

        if self.status:
            self.status.update(status_text)
        else:
            self.status = Status(status_text, spinner="dots", console=self.console)
            self.status.start()

    def complete(self, success: bool = True, message: str = None):
        """Complete task."""
        if self.status:
            self.status.stop()

        icon = "[green]✓[/green]" if success else "[red]✗[/red]"
        result = "Success" if success else "Failed"

        self.console.print(f"\n{icon} [bold]{result}[/bold]")

        if message:
            self.console.print(f"  {message}")

        if self.started_at:
            self.console.print(f"  [dim]{self._elapsed()}[/dim]\n")

    def _elapsed(self) -> str:
        """Get elapsed time."""
        if not self.started_at:
            return "0s"

        delta = datetime.now() - self.started_at
        total_seconds = int(delta.total_seconds())

        if total_seconds < 60:
            return f"{total_seconds}s"
        else:
            minutes = total_seconds // 60
            seconds = total_seconds % 60
            return f"{minutes}m {seconds}s"


def create_tui(mode: str = "simple", console: Optional[Console] = None):
    """
    Create a TUI instance.

    Args:
        mode: TUI mode - "full", "simple", "minimal"
        console: Optional console instance

    Returns:
        TUI instance
    """
    if not RICH_AVAILABLE:
        # Fallback to minimal mode if Rich not available
        mode = "minimal"

    if mode == "full":
        return GaiaCodeTUI(console)
    elif mode == "minimal":
        return GaiaCodeMinimalTUI(console)
    else:  # simple (default)
        return GaiaCodeSimpleTUI(console)
