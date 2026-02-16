# Add interactive command
def cmd_interactive(args):
    """Start interactive chat session with modern TUI."""
    from .interactive_session import start_interactive_session
    from pathlib import Path

    workspace_dir = Path(args.workspace) if hasattr(args, 'workspace') and args.workspace else None
    persona = args.persona if hasattr(args, 'persona') else 'pike'
    tui_mode = args.tui if hasattr(args, 'tui') else 'simple'

    print("\n🚀 Starting GAIA Code Interactive Session...")
    print(f"   Persona: {persona.title()}")
    print(f"   TUI: {tui_mode}")
    print("   Type /help for commands or just chat naturally\n")

    try:
        start_interactive_session(
            persona=persona,
            workspace_dir=workspace_dir,
            tui_mode=tui_mode,
        )
        return 0
    except KeyboardInterrupt:
        print("\n\n👋 Session ended gracefully")
        return 0
    except Exception as e:
        print(f"\n✗ Error: {e}")
        if hasattr(args, 'debug') and args.debug:
            import traceback
            traceback.print_exc()
        return 1

# Update cmd_gaia_code to handle no arguments
def cmd_gaia_code_updated(args):
    """Execute GAIA Code command or start interactive session."""
    # Check if interactive mode or no task provided
    if hasattr(args, 'interactive') and args.interactive:
        return cmd_interactive(args)
    elif hasattr(args, 'status') and args.status:
        return cmd_status(args)
    elif hasattr(args, 'audit') and args.audit:
        return cmd_audit(args)
    elif hasattr(args, 'resume') and args.resume:
        return cmd_resume(args)
    elif hasattr(args, 'checkpoint') and args.checkpoint:
        return cmd_checkpoint(args)
    elif hasattr(args, 'task') and args.task:
        return cmd_execute(args)
    else:
        # No arguments - start interactive session by default
        print("\n💡 No task specified - starting interactive session")
        print("   (Use 'gaia code \"task\"' for one-off execution)\n")
        return cmd_interactive(args)

# Add to parser
def add_interactive_argument(parser):
    """Add interactive mode argument."""
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Start interactive chat session (like Claude Code). Also starts if no task provided.",
    )
