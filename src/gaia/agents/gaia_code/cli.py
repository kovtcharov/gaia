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
    """Execute GAIA Code command."""
    if args.status:
        return cmd_status(args)
    elif args.audit:
        return cmd_audit(args)
    elif args.resume:
        return cmd_resume(args)
    elif args.checkpoint:
        return cmd_checkpoint(args)
    elif args.task:
        return cmd_execute(args)
    else:
        print("Usage: gaia code <task> | --status | --audit | --resume | --checkpoint")
        return 1


def cmd_execute(args):
    """Execute a coding task."""
    workspace_dir = Path(args.workspace) if args.workspace else None

    # Create agent
    agent = GaiaCodeAgent(
        workspace_dir=workspace_dir,
        enable_quality_gates=not args.no_quality_gates,
        enable_continuous_execution=not args.no_continuous,
        silent_mode=args.silent,
        debug=args.debug,
        use_claude=args.claude,
        use_chatgpt=args.chatgpt,
    )

    # Execute task
    print(f"🚀 GAIA Code: {args.task}")
    print(f"📁 Workspace: {agent.shared_state.workspace_dir}")
    print()

    try:
        result = agent.process_query(args.task, create_plan=not args.no_plan)

        if result["success"]:
            print("\n✅ Task completed successfully!")
            if result.get("result"):
                print(f"\nResult: {result['result']}")

            # Show quality gate results
            if result.get("quality_gates"):
                print("\nQuality Gates:")
                for gate in result["quality_gates"]:
                    status = "✅" if gate.passed else "❌"
                    print(f"  {status} {gate.gate_name}: {gate.message}")

            # Show attempts
            if result.get("attempts"):
                print(f"\nAttempts: {result['attempts']}")

            return 0
        else:
            print("\n❌ Task failed!")
            if result.get("error"):
                print(f"\nError: {result['error']}")

            return 1

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

    parser.set_defaults(func=cmd_gaia_code)

    return parser
