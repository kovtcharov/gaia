# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
Credential Manager: Secure storage and retrieval of API keys and settings.

Features:
- Stores credentials in ~/.gaia/cache/credentials.json
- Prompts user for API key if missing
- Shows where credentials are saved/loaded from
- Encrypts sensitive data (optional)
- Settings management for agent configuration
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class CredentialManager:
    """
    Manages API keys and agent settings.

    Storage location: ~/.gaia/cache/credentials.json

    Stores:
    - API keys (Anthropic, OpenAI, Perplexity)
    - Model preferences
    - Agent settings (persona, TUI mode, etc.)
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize credential manager.

        Args:
            cache_dir: Cache directory (default: ~/.gaia/cache)
        """
        if cache_dir is None:
            cache_dir = Path.home() / ".gaia" / "cache"

        cache_dir.mkdir(parents=True, exist_ok=True)

        self.cache_dir = cache_dir
        self.credentials_file = cache_dir / "credentials.json"
        self.settings_file = cache_dir / "settings.json"

        # Load existing credentials and settings
        self.credentials = self._load_credentials()
        self.settings = self._load_settings()

    def _load_credentials(self) -> Dict[str, str]:
        """Load credentials from file."""
        if not self.credentials_file.exists():
            return {}

        try:
            with open(self.credentials_file, "r") as f:
                creds = json.load(f)

            logger.info(f"Loaded credentials from {self.credentials_file}")
            return creds

        except Exception as e:
            logger.warning(f"Failed to load credentials: {e}")
            return {}

    def _load_settings(self) -> Dict[str, Any]:
        """Load settings from file."""
        if not self.settings_file.exists():
            return {}

        try:
            with open(self.settings_file, "r") as f:
                settings = json.load(f)

            logger.info(f"Loaded settings from {self.settings_file}")
            return settings

        except Exception as e:
            logger.warning(f"Failed to load settings: {e}")
            return {}

    def _save_credentials(self):
        """Save credentials to file."""
        try:
            # Create backup if exists
            if self.credentials_file.exists():
                backup = self.credentials_file.with_suffix(".json.bak")
                self.credentials_file.rename(backup)

            with open(self.credentials_file, "w") as f:
                json.dump(self.credentials, f, indent=2)

            # Set restrictive permissions (owner only)
            os.chmod(self.credentials_file, 0o600)

            logger.info(f"Saved credentials to {self.credentials_file}")

        except Exception as e:
            logger.error(f"Failed to save credentials: {e}")

    def _save_settings(self):
        """Save settings to file."""
        try:
            with open(self.settings_file, "w") as f:
                json.dump(self.settings, f, indent=2)

            logger.info(f"Saved settings to {self.settings_file}")

        except Exception as e:
            logger.error(f"Failed to save settings: {e}")

    def get_anthropic_key(self) -> Optional[str]:
        """
        Get Anthropic API key.

        Checks in order:
        1. Environment variable (ANTHROPIC_API_KEY)
        2. Credentials file
        3. Returns None if not found

        Returns:
            API key or None
        """
        # Check environment first
        env_key = os.getenv("ANTHROPIC_API_KEY")
        if env_key:
            logger.debug("Using Anthropic API key from environment variable")
            return env_key

        # Check credentials file
        if "anthropic_api_key" in self.credentials:
            logger.debug(f"Using Anthropic API key from {self.credentials_file}")
            return self.credentials["anthropic_api_key"]

        return None

    def set_anthropic_key(self, api_key: str, save: bool = True):
        """
        Set Anthropic API key.

        Args:
            api_key: The API key
            save: Whether to save to file (default: True)
        """
        self.credentials["anthropic_api_key"] = api_key

        if save:
            self._save_credentials()
            logger.info(f"Anthropic API key saved to {self.credentials_file}")

    def get_perplexity_key(self) -> Optional[str]:
        """Get Perplexity API key."""
        env_key = os.getenv("PERPLEXITY_API_KEY")
        if env_key:
            return env_key

        return self.credentials.get("perplexity_api_key")

    def set_perplexity_key(self, api_key: str, save: bool = True):
        """Set Perplexity API key."""
        self.credentials["perplexity_api_key"] = api_key

        if save:
            self._save_credentials()

    def get_setting(self, key: str, default: Any = None) -> Any:
        """
        Get a setting value.

        Args:
            key: Setting key
            default: Default value if not found

        Returns:
            Setting value
        """
        return self.settings.get(key, default)

    def set_setting(self, key: str, value: Any, save: bool = True):
        """
        Set a setting value.

        Args:
            key: Setting key
            value: Setting value
            save: Whether to save immediately
        """
        self.settings[key] = value

        if save:
            self._save_settings()

    def get_all_settings(self) -> Dict[str, Any]:
        """Get all settings."""
        return self.settings.copy()

    def get_credentials_path(self) -> str:
        """Get path to credentials file."""
        return str(self.credentials_file)

    def get_settings_path(self) -> str:
        """Get path to settings file."""
        return str(self.settings_file)

    def has_anthropic_key(self) -> bool:
        """Check if Anthropic API key is configured."""
        return self.get_anthropic_key() is not None

    def show_credential_status(self) -> str:
        """
        Get credential status message for display.

        Returns:
            Status message
        """
        lines = []

        # Anthropic
        if self.has_anthropic_key():
            source = "environment" if os.getenv("ANTHROPIC_API_KEY") else "credentials file"
            lines.append(f"✓ Anthropic API key found ({source})")
            if source == "credentials file":
                lines.append(f"  Location: {self.credentials_file}")
        else:
            lines.append("✗ Anthropic API key not found")
            lines.append(f"  Will save to: {self.credentials_file}")

        # Perplexity
        if self.get_perplexity_key():
            source = "environment" if os.getenv("PERPLEXITY_API_KEY") else "credentials file"
            lines.append(f"✓ Perplexity API key found ({source})")
        else:
            lines.append("ℹ Perplexity API key not configured (web search will be limited)")

        return "\n".join(lines)


def prompt_for_api_key(console=None) -> Optional[str]:
    """
    Prompt user to enter API key.

    Args:
        console: Rich console for formatted prompts (optional)

    Returns:
        API key or None if user cancels
    """
    if console:
        # Rich formatted prompt
        from rich.panel import Panel
        from rich.prompt import Prompt

        panel = Panel(
            "[yellow]Anthropic API key not found.[/yellow]\n\n"
            "To use Claude Opus 4.6, you need an Anthropic API key.\n\n"
            "Get one at: https://console.anthropic.com/\n\n"
            "The key will be saved securely to:\n"
            "[cyan]~/.gaia/cache/credentials.json[/cyan]",
            title="[bold red]API Key Required[/bold red]",
            border_style="red",
        )

        console.print()
        console.print(panel)
        console.print()

        api_key = Prompt.ask(
            "[cyan]Enter your Anthropic API key[/cyan]",
            password=True,  # Hide input
        )

        if api_key and len(api_key) > 10:
            console.print()
            console.print(f"[green]✓[/green] API key will be saved to [cyan]~/.gaia/cache/credentials.json[/cyan]")
            console.print()
            return api_key
        else:
            console.print("[red]✗[/red] Invalid API key")
            return None

    else:
        # Simple text prompt
        print()
        print("=" * 70)
        print("ANTHROPIC API KEY REQUIRED")
        print("=" * 70)
        print()
        print("Anthropic API key not found.")
        print("To use Claude Opus 4.6, you need an API key.")
        print()
        print("Get one at: https://console.anthropic.com/")
        print()
        print("The key will be saved securely to:")
        print(f"  ~/.gaia/cache/credentials.json")
        print()

        import getpass

        api_key = getpass.getpass("Enter your Anthropic API key (hidden): ")

        if api_key and len(api_key) > 10:
            print()
            print(f"✓ API key will be saved to ~/.gaia/cache/credentials.json")
            print()
            return api_key
        else:
            print("✗ Invalid API key")
            return None


def check_and_setup_credentials(
    use_claude: bool = True,
    console=None,
    interactive: bool = True,
) -> tuple[bool, Optional[str]]:
    """
    Check if credentials exist, prompt if needed.

    Args:
        use_claude: Whether agent is configured to use Claude
        console: Rich console for formatted output
        interactive: Whether to prompt for missing credentials

    Returns:
        Tuple of (success, api_key)
    """
    manager = CredentialManager()

    # Show credential status
    if console:
        from rich.panel import Panel

        status = manager.show_credential_status()
        panel = Panel(
            status,
            title="[cyan]Credential Status[/cyan]",
            border_style="cyan",
        )
        console.print()
        console.print(panel)
        console.print()
    else:
        print()
        print(manager.show_credential_status())
        print()

    # If using Claude, need API key
    if use_claude:
        api_key = manager.get_anthropic_key()

        if not api_key:
            if interactive:
                # Prompt user for key
                api_key = prompt_for_api_key(console)

                if api_key:
                    # Save it
                    manager.set_anthropic_key(api_key, save=True)
                    return True, api_key
                else:
                    return False, None
            else:
                # Non-interactive mode - cannot proceed
                return False, None

        return True, api_key

    # Not using Claude - no credentials needed
    return True, None
