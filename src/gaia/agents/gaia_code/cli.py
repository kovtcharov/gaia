# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
CLI commands for GAIA Code autonomous agent.

Commands:
- gaia code <task>: Execute a coding task
- gaia code status: Show current progress
- gaia code audit: Show audit log
- gaia code resume: Resume from checkpoint
- gaia code checkpoint: Create checkpoint
"""

import argparse
import json
import sys
from pathlib import Path

from .agent import GaiaCodeAgent


def cmd_gaia_code(args):
    """Execute GAIA Code command or start interactive session."""
    import logging

    # Set logging level based on debug flag
    if not (hasattr(args, 'debug') and args.debug):
        logging.getLogger().setLevel(logging.ERROR)
        logging.getLogger("gaia").setLevel(logging.ERROR)

    # Check for specific flags first
    if hasattr(args, 'status') and args.status:
        return cmd_status(args)
    elif hasattr(args, 'audit') and args.audit:
        return cmd_audit(args)
    elif hasattr(args, 'resume') and args.resume:
        return cmd_resume(args)
    elif hasattr(args, 'checkpoint') and args.checkpoint:
        return cmd_checkpoint(args)
    elif hasattr(args, 'task') and args.task:
        return cmd_execute(args)
    elif hasattr(args, 'interactive') and args.interactive:
        return cmd_interactive(args)
    else:
        # No task provided - start interactive session by default
        return cmd_interactive(args)


def cmd_interactive(args):
    """Start interactive chat session."""
    from .interactive_session import start_interactive_session

    workspace_dir = Path(args.workspace) if hasattr(args, 'workspace') and args.workspace else None
    persona = args.persona if hasattr(args, 'persona') else 'pike'
    tui_mode = args.tui if hasattr(args, 'tui') else 'simple'

    try:
        start_interactive_session(
            persona=persona,
            workspace_dir=workspace_dir,
            tui_mode=tui_mode,
        )
        return 0
    except KeyboardInterrupt:
        print("\n\n👋 Session ended")
        return 0
    except Exception as e:
        print(f"\n✗ Error: {e}")
        if hasattr(args, 'debug') and args.debug:
            import traceback
            traceback.print_exc()
        return 1


def cmd_execute(args):
    """Execute a coding task."""
    workspace_dir = Path(args.workspace) if args.workspace else None
    persona = getattr(args, 'persona', 'pike')

    # Build kwargs - only pass use_claude/use_chatgpt if explicitly requested
    # This lets the agent's default (Claude Opus) take effect when neither is specified
    agent_kwargs = dict(
        workspace_dir=workspace_dir,
        enable_quality_gates=not args.no_quality_gates,
        enable_continuous_execution=not args.no_continuous,
        tui_mode=args.tui,
        persona=persona,
        silent_mode=args.silent,
        debug=args.debug,
    )
    if args.claude:
        agent_kwargs["use_claude"] = True
    if args.chatgpt:
        agent_kwargs["use_chatgpt"] = True

    # Create agent
    agent = GaiaCodeAgent(**agent_kwargs)

    # Execute task - TUI handles all output
    try:
        result = agent.process_query(args.task, create_plan=not args.no_plan)
        return 0 if result["success"] else 1

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted. Creating checkpoint...")
        agent.checkpoint()
        print("Checkpoint saved. Resume with: gaia code --resume")
        return 130

    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.debug:
            import traceback

            traceback.print_exc()
        return 1


def cmd_status(args):
    """Show current progress."""
    workspace_dir = Path(args.workspace) if args.workspace else None

    agent = GaiaCodeAgent(
        workspace_dir=workspace_dir,
        silent_mode=True,
        skip_lemonade=True,
    )

    progress = agent.get_progress()

    print("📊 GAIA Code Status")
    print("=" * 50)
    print(f"Total tasks: {progress['total_tasks']}")
    print(f"Completed: {progress['completed']}")
    print(f"In progress: {progress['in_progress']}")
    print(f"Pending: {progress['pending']}")
    print(f"Failed: {progress['failed']}")
    print(f"Progress: {progress['progress_percent']}%")

    if progress["elapsed_seconds"]:
        elapsed_min = int(progress["elapsed_seconds"] / 60)
        print(f"Elapsed: {elapsed_min} minutes")

    print("=" * 50)

    return 0


def cmd_audit(args):
    """Show audit log."""
    workspace_dir = Path(args.workspace) if args.workspace else None

    agent = GaiaCodeAgent(
        workspace_dir=workspace_dir,
        silent_mode=True,
        skip_lemonade=True,
    )

    if agent.resume_from_checkpoint():
        log = agent.get_audit_log()

        print("📋 GAIA Code Audit Log")
        print("=" * 50)

        for entry in log:
            timestamp = entry["timestamp"]
            action_type = entry["action_type"]
            details = entry.get("details", {})

            print(f"[{timestamp}] {action_type}")
            if details:
                for key, value in details.items():
                    print(f"  {key}: {value}")
            print()

        print("=" * 50)
        print(f"Total entries: {len(log)}")
    else:
        print("No checkpoint found. Start a task with: gaia code <task>")
        return 1

    return 0


def cmd_resume(args):
    """Resume from checkpoint."""
    workspace_dir = Path(args.workspace) if args.workspace else None

    agent = GaiaCodeAgent(
        workspace_dir=workspace_dir,
        enable_quality_gates=not args.no_quality_gates,
        enable_continuous_execution=not args.no_continuous,
        silent_mode=args.silent,
        debug=args.debug,
    )

    if agent.resume_from_checkpoint():
        print("✅ Resumed from checkpoint")

        # Show progress
        progress = agent.get_progress()
        print(f"\nProgress: {progress['progress_percent']}%")
        print(f"Tasks: {progress['completed']}/{progress['total_tasks']} completed")

        # Continue execution would go here
        # For now, just show status
        return 0
    else:
        print("❌ No checkpoint found")
        return 1


def cmd_checkpoint(args):
    """Create checkpoint."""
    workspace_dir = Path(args.workspace) if args.workspace else None

    agent = GaiaCodeAgent(
        workspace_dir=workspace_dir,
        silent_mode=True,
        skip_lemonade=True,
    )

    result = agent.checkpoint()

    if result["success"]:
        print("✅ Checkpoint created")
        print(f"Path: {result['checkpoint_path']}")
        print(f"Time: {result['timestamp']}")
        return 0
    else:
        print("❌ Failed to create checkpoint")
        return 1


def add_gaia_code_parser(subparsers):
    """Add GAIA Code parser to CLI."""
    parser = subparsers.add_parser(
        "code",
        help="GAIA Code: Autonomous coding agent with RAC",
        description="Execute coding tasks with recursive decomposition, quality gates, and persistent memory.",
    )

    # Task argument
    parser.add_argument(
        "task",
        nargs="?",
        help="Coding task to execute (e.g., 'Build a REST API with auth')",
    )

    # Command flags
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show current progress",
    )

    parser.add_argument(
        "--audit",
        action="store_true",
        help="Show audit log",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint",
    )

    parser.add_argument(
        "--checkpoint",
        action="store_true",
        help="Create checkpoint",
    )

    # Interactive mode
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Start interactive chat session",
    )

    # Persona selection
    parser.add_argument(
        "--persona",
        type=str,
        choices=["torvalds", "knuth", "pike", "carmack", "hickey", "kay", "thompson", "hopper"],
        default="pike",
        help="Agent persona (default: pike)",
    )

    # Configuration flags
    parser.add_argument(
        "--workspace",
        type=str,
        help="Workspace directory (default: ~/.gaia/workspace)",
    )

    parser.add_argument(
        "--no-quality-gates",
        action="store_true",
        help="Disable quality gates",
    )

    parser.add_argument(
        "--no-continuous",
        action="store_true",
        help="Disable continuous execution (use step limits)",
    )

    parser.add_argument(
        "--no-plan",
        action="store_true",
        help="Disable plan creation (direct execution)",
    )

    # LLM selection
    parser.add_argument(
        "--claude",
        action="store_true",
        help="Use Claude API instead of local LLM",
    )

    parser.add_argument(
        "--chatgpt",
        action="store_true",
        help="Use ChatGPT/OpenAI API instead of local LLM",
    )

    # Debug flags
    parser.add_argument(
        "--silent",
        action="store_true",
        help="Silent mode (no console output)",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output",
    )

    # TUI configuration
    parser.add_argument(
        "--tui",
        type=str,
        choices=["full", "simple", "minimal", "off"],
        default="simple",
        help="TUI mode: full (detailed), simple (default, clean), minimal (one line), off (verbose logs)",
    )

    parser.set_defaults(func=cmd_gaia_code)

    return parser


def main():
    """Standalone CLI entry point for gaia-code-rac."""
    parser = argparse.ArgumentParser(
        description="GAIA Code: Autonomous coding agent with RAC architecture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  gaia-code-rac "Build a REST API with authentication"
  gaia-code-rac "Create a calculator with tests" --tui simple
  gaia-code-rac -i                       # Interactive mode
  gaia-code-rac --status                 # Show progress
  gaia-code-rac --resume                 # Resume from checkpoint
""",
    )

    # Reuse the same argument setup
    parser.add_argument("task", nargs="?", help="Coding task to execute")
    parser.add_argument("--status", action="store_true", help="Show current progress")
    parser.add_argument("--audit", action="store_true", help="Show audit log")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--checkpoint", action="store_true", help="Create checkpoint")
    parser.add_argument("-i", "--interactive", action="store_true", help="Interactive chat session")
    parser.add_argument("--persona", type=str, default="pike",
                        choices=["torvalds", "knuth", "pike", "carmack", "hickey", "kay", "thompson", "hopper"],
                        help="Agent persona (default: pike)")
    parser.add_argument("--workspace", type=str, help="Workspace directory")
    parser.add_argument("--no-quality-gates", action="store_true", help="Disable quality gates")
    parser.add_argument("--no-continuous", action="store_true", help="Disable continuous execution")
    parser.add_argument("--no-plan", action="store_true", help="Disable plan creation")
    parser.add_argument("--claude", action="store_true", help="Use Claude API")
    parser.add_argument("--chatgpt", action="store_true", help="Use ChatGPT/OpenAI API")
    parser.add_argument("--silent", action="store_true", help="Silent mode")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--tui", type=str, default="simple",
                        choices=["full", "simple", "minimal", "off"],
                        help="TUI mode (default: simple)")

    args = parser.parse_args()
    sys.exit(cmd_gaia_code(args))
