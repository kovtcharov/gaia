# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
ExecutionObserver: Run, Observe, Debug, Fix Until Functional

Critical capability: GAIA Code doesn't just generate code - it:
1. RUNS the code it creates
2. OBSERVES the output/behavior
3. DEBUGS issues found
4. FIXES problems
5. REPEATS until fully functional

For web apps:
- Takes screenshots
- Uses Playwright to interact
- Verifies UI behaves correctly
- Tests user flows end-to-end

This is the difference between "code generator" and "autonomous developer."
"""

import json
import logging
import select
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Try to import pexpect for better interactive support
try:
    import pexpect
    PEXPECT_AVAILABLE = True
except ImportError:
    PEXPECT_AVAILABLE = False
    logger.info("pexpect not available. Install with: pip install pexpect")


@dataclass
class ExecutionResult:
    """Result of running code."""

    success: bool
    output: str
    error: str
    returncode: int
    duration_seconds: float
    screenshots: List[str] = None  # Paths to screenshots taken


@dataclass
class ObservationResult:
    """Result of observing running application."""

    is_running: bool
    port: Optional[int] = None
    url: Optional[str] = None
    screenshots: List[str] = None
    console_logs: List[str] = None
    errors: List[str] = None
    performance_metrics: Optional[Dict] = None


class CodeExecutor:
    """
    Executes code and captures output.

    Handles:
    - Python scripts
    - Web servers (Flask, FastAPI, Next.js)
    - CLI applications
    - Background processes
    """

    def __init__(self, workspace_dir: Path):
        self.workspace_dir = workspace_dir
        self.running_processes = {}

    def run_python_script(
        self, script_path: str, args: List[str] = None, timeout: int = 30
    ) -> ExecutionResult:
        """
        Run a Python script and capture output.

        Args:
            script_path: Path to Python script
            args: Command line arguments
            timeout: Timeout in seconds

        Returns:
            ExecutionResult with output and errors
        """
        start_time = time.time()

        cmd = [sys.executable, script_path]
        if args:
            cmd.extend(args)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self.workspace_dir),
            )

            duration = time.time() - start_time

            return ExecutionResult(
                success=result.returncode == 0,
                output=result.stdout,
                error=result.stderr,
                returncode=result.returncode,
                duration_seconds=duration,
            )

        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False,
                output="",
                error=f"Execution timed out after {timeout}s",
                returncode=-1,
                duration_seconds=timeout,
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                output="",
                error=str(e),
                returncode=-1,
                duration_seconds=time.time() - start_time,
            )

    def start_web_server(
        self, command: str, port: int = 8000, wait_time: int = 5
    ) -> Dict[str, Any]:
        """
        Start a web server in background and wait for it to be ready.

        Args:
            command: Command to start server (e.g., "uvicorn main:app")
            port: Expected port
            wait_time: How long to wait for server to start

        Returns:
            Dict with process info and readiness status
        """
        try:
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(self.workspace_dir),
            )

            self.running_processes[port] = process

            # Wait for server to be ready
            time.sleep(wait_time)

            # Check if it's running
            if process.poll() is None:
                return {
                    "success": True,
                    "pid": process.pid,
                    "port": port,
                    "url": f"http://localhost:{port}",
                }
            else:
                stdout, stderr = process.communicate()
                return {
                    "success": False,
                    "error": f"Server failed to start: {stderr}",
                }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def stop_process(self, port: int):
        """Stop a running process."""
        if port in self.running_processes:
            process = self.running_processes[port]
            process.terminate()
            process.wait(timeout=5)
            del self.running_processes[port]

    def cleanup_all(self):
        """Stop all running processes."""
        for port in list(self.running_processes.keys()):
            self.stop_process(port)


class WebAppObserver:
    """
    Observes web applications using Playwright.

    Capabilities:
    - Take screenshots
    - Interact with UI (click, type, navigate)
    - Verify elements exist
    - Test user flows
    - Capture console logs
    - Detect errors
    """

    def __init__(self):
        self.browser = None
        self.context = None
        self.page = None
        self._playwright_available = False

        # Try to import Playwright
        try:
            from playwright.sync_api import sync_playwright

            self.playwright = sync_playwright()
            self._playwright_available = True
        except ImportError:
            logger.warning(
                "Playwright not installed. Web app observation limited. "
                "Install with: pip install playwright && playwright install"
            )

    def start_browser(self, headless: bool = True):
        """Start browser for observation."""
        if not self._playwright_available:
            return False

        try:
            pw = self.playwright.start()
            self.browser = pw.chromium.launch(headless=headless)
            self.context = self.browser.new_context(
                viewport={"width": 1920, "height": 1080}
            )
            self.page = self.context.new_page()
            return True
        except Exception as e:
            logger.error(f"Failed to start browser: {e}")
            return False

    def observe_web_app(
        self, url: str, screenshot_path: Optional[str] = None
    ) -> ObservationResult:
        """
        Observe a web application.

        Args:
            url: URL to observe
            screenshot_path: Path to save screenshot

        Returns:
            ObservationResult with observations
        """
        if not self._playwright_available or not self.page:
            return ObservationResult(
                is_running=False,
                errors=["Playwright not available"],
            )

        try:
            # Navigate to app
            response = self.page.goto(url, wait_until="networkidle", timeout=10000)

            # Capture console logs
            console_logs = []
            self.page.on("console", lambda msg: console_logs.append(msg.text()))

            # Capture errors
            errors = []
            self.page.on("pageerror", lambda err: errors.append(str(err)))

            # Wait a bit for page to render
            time.sleep(2)

            # Take screenshot
            screenshots = []
            if screenshot_path:
                self.page.screenshot(path=screenshot_path)
                screenshots.append(screenshot_path)

            return ObservationResult(
                is_running=response.ok if response else False,
                port=None,  # Extract from URL
                url=url,
                screenshots=screenshots,
                console_logs=console_logs,
                errors=errors,
            )

        except Exception as e:
            return ObservationResult(
                is_running=False,
                errors=[str(e)],
            )

    def interact(self, actions: List[Dict]) -> List[Dict]:
        """
        Perform interactions with web app.

        Args:
            actions: List of actions like:
                [
                    {"type": "click", "selector": "#submit-button"},
                    {"type": "fill", "selector": "#email", "value": "test@example.com"},
                    {"type": "screenshot", "path": "after_login.png"}
                ]

        Returns:
            List of action results
        """
        if not self.page:
            return [{"error": "No page loaded"}]

        results = []

        for action in actions:
            try:
                action_type = action.get("type")

                if action_type == "click":
                    self.page.click(action["selector"])
                    results.append({"success": True, "action": "click"})

                elif action_type == "fill":
                    self.page.fill(action["selector"], action["value"])
                    results.append({"success": True, "action": "fill"})

                elif action_type == "screenshot":
                    self.page.screenshot(path=action["path"])
                    results.append({"success": True, "action": "screenshot", "path": action["path"]})

                elif action_type == "wait":
                    time.sleep(action.get("seconds", 1))
                    results.append({"success": True, "action": "wait"})

                elif action_type == "assert_text":
                    text = self.page.text_content(action["selector"])
                    expected = action["expected"]
                    success = expected in text
                    results.append({
                        "success": success,
                        "action": "assert_text",
                        "found": text,
                        "expected": expected,
                    })

                elif action_type == "assert_visible":
                    visible = self.page.is_visible(action["selector"])
                    results.append({
                        "success": visible,
                        "action": "assert_visible",
                        "selector": action["selector"],
                    })

            except Exception as e:
                results.append({"success": False, "action": action_type, "error": str(e)})

        return results

    def cleanup(self):
        """Close browser."""
        if self.page:
            self.page.close()
        if self.context:
            self.context.close()
        if self.browser:
            self.browser.close()
        if self._playwright_available:
            self.playwright.stop()


class ExecutionObserver:
    """
    Combines execution and observation for complete feedback loop.

    Workflow:
    1. EXECUTE code
    2. OBSERVE output/behavior
    3. ANALYZE results
    4. IDENTIFY issues
    5. FIX problems
    6. REPEAT until functional
    """

    def __init__(self, workspace_dir: Path):
        self.workspace_dir = workspace_dir
        self.executor = CodeExecutor(workspace_dir)
        self.web_observer = WebAppObserver()

    def run_and_observe(
        self, project_type: str, entry_point: str
    ) -> Dict[str, Any]:
        """
        Run project and observe behavior.

        Args:
            project_type: "script", "web_api", "web_app", "cli"
            entry_point: Main file or command

        Returns:
            Dict with execution results and observations
        """
        if project_type == "script":
            return self._run_script(entry_point)

        elif project_type in ("web_api", "web_app"):
            return self._run_and_observe_web(entry_point, project_type == "web_app")

        elif project_type == "cli":
            return self._run_cli_app(entry_point)

        else:
            return {"success": False, "error": f"Unknown project type: {project_type}"}

    def _run_script(self, script_path: str) -> Dict[str, Any]:
        """Run a Python script and return results."""
        result = self.executor.run_python_script(script_path)

        return {
            "success": result.success,
            "type": "script",
            "output": result.output,
            "error": result.error,
            "duration": result.duration_seconds,
            "observations": {
                "ran_successfully": result.success,
                "produced_output": len(result.output) > 0,
                "had_errors": len(result.error) > 0,
            },
        }

    def _run_and_observe_web(self, command: str, is_ui_app: bool) -> Dict[str, Any]:
        """Run web server and observe it."""
        # Start server
        server_result = self.executor.start_web_server(command)

        if not server_result["success"]:
            return {
                "success": False,
                "error": server_result["error"],
            }

        url = server_result["url"]

        # For UI apps, use Playwright to observe
        if is_ui_app:
            self.web_observer.start_browser(headless=True)
            screenshot_path = str(self.workspace_dir / "app_screenshot.png")
            observation = self.web_observer.observe_web_app(url, screenshot_path)

            return {
                "success": observation.is_running,
                "type": "web_app",
                "url": url,
                "observations": {
                    "server_started": server_result["success"],
                    "page_loaded": observation.is_running,
                    "screenshots": observation.screenshots,
                    "console_logs": observation.console_logs,
                    "errors": observation.errors,
                },
            }
        else:
            # For APIs, just check if server responds
            try:
                import urllib.request

                urllib.request.urlopen(url, timeout=5)

                return {
                    "success": True,
                    "type": "web_api",
                    "url": url,
                    "observations": {
                        "server_started": True,
                        "responds_to_requests": True,
                    },
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Server started but not responding: {e}",
                }

    def _run_cli_app(self, command: str) -> Dict[str, Any]:
        """Run CLI app and capture interaction."""
        # For CLI apps, run with test inputs
        result = self.executor.run_python_script(command)

        return {
            "success": result.success,
            "type": "cli",
            "output": result.output,
            "error": result.error,
            "observations": {
                "interactive_prompts": "input(" in result.output or "?" in result.output,
                "produced_output": len(result.output) > 0,
            },
        }

    def cleanup(self):
        """Cleanup all running processes and browsers."""
        self.executor.cleanup_all()
        self.web_observer.cleanup()


class InteractiveCLIExecutor:
    """
    Executes and interacts with CLI tools like a human would.

    Capabilities:
    - Run interactive CLI applications
    - Observe output in real-time
    - Detect prompts and questions
    - Provide input automatically or via LLM decision
    - Handle yes/no questions, text input, selections
    - Capture full interaction transcript
    """

    def __init__(self):
        self.use_pexpect = PEXPECT_AVAILABLE

    def run_interactive(
        self,
        command: str,
        interactions: List[Dict],
        timeout: int = 300,
    ) -> Dict[str, Any]:
        """
        Run a CLI tool with interactive responses.

        Args:
            command: Command to run
            interactions: List of expected prompts and responses
                Example:
                [
                    {"expect": "Enter name:", "respond": "John Doe"},
                    {"expect": "Confirm? (y/n)", "respond": "y"},
                    {"expect": "Password:", "respond": "secret123"},
                ]
            timeout: Total timeout in seconds

        Returns:
            Dict with full transcript and success status
        """
        if self.use_pexpect:
            return self._run_with_pexpect(command, interactions, timeout)
        else:
            return self._run_with_subprocess(command, interactions, timeout)

    def _run_with_pexpect(
        self,
        command: str,
        interactions: List[Dict],
        timeout: int,
    ) -> Dict[str, Any]:
        """Run with pexpect (better for interactive)."""
        try:
            child = pexpect.spawn(command, timeout=timeout, encoding='utf-8')

            transcript = []
            all_output = []

            for interaction in interactions:
                pattern = interaction.get("expect")
                response = interaction.get("respond")

                # Wait for expected prompt
                try:
                    index = child.expect([pattern, pexpect.EOF, pexpect.TIMEOUT])

                    # Capture what we saw
                    before = child.before or ""
                    all_output.append(before)

                    if index == 0:
                        # Found the expected pattern
                        matched = child.after or ""
                        transcript.append({
                            "output": before,
                            "prompt": matched,
                            "response": response,
                        })

                        # Send response
                        child.sendline(response)

                    elif index == 1:
                        # EOF - process ended
                        break
                    else:
                        # Timeout waiting for prompt
                        transcript.append({
                            "error": f"Timeout waiting for pattern: {pattern}",
                            "output": before,
                        })
                        break

                except pexpect.EOF:
                    break
                except pexpect.TIMEOUT:
                    transcript.append({
                        "error": f"Timeout waiting for: {pattern}",
                    })
                    break

            # Wait for process to finish
            child.expect(pexpect.EOF)
            remaining_output = child.before or ""
            all_output.append(remaining_output)

            return {
                "success": child.exitstatus == 0 if child.exitstatus is not None else True,
                "transcript": transcript,
                "full_output": "".join(all_output),
                "exit_code": child.exitstatus,
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "transcript": transcript if 'transcript' in locals() else [],
            }

    def _run_with_subprocess(
        self,
        command: str,
        interactions: List[Dict],
        timeout: int,
    ) -> Dict[str, Any]:
        """Fallback: Run with subprocess (limited interactivity)."""
        try:
            process = subprocess.Popen(
                command,
                shell=True,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            transcript = []
            output_buffer = ""

            # Send all responses upfront (simplified approach)
            for interaction in interactions:
                response = interaction.get("respond", "")
                process.stdin.write(response + "\n")
                process.stdin.flush()

            # Wait for completion
            stdout, stderr = process.communicate(timeout=timeout)

            return {
                "success": process.returncode == 0,
                "transcript": interactions,
                "full_output": stdout,
                "error_output": stderr,
                "exit_code": process.returncode,
                "note": "Used subprocess (limited). Install pexpect for better interactive support.",
            }

        except subprocess.TimeoutExpired:
            process.kill()
            return {
                "success": False,
                "error": f"Process timed out after {timeout}s",
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    def auto_interact(
        self,
        command: str,
        prompt_handler: Optional[Callable[[str], str]] = None,
        timeout: int = 300,
    ) -> Dict[str, Any]:
        """
        Run CLI tool and automatically respond to prompts.

        Uses LLM (via prompt_handler) to decide how to respond to prompts.

        Args:
            command: Command to run
            prompt_handler: Function that takes prompt text and returns response
                           If None, uses default responses
            timeout: Timeout in seconds

        Returns:
            Dict with transcript and results
        """
        if not self.use_pexpect:
            return {
                "success": False,
                "error": "pexpect required for auto_interact. Install with: pip install pexpect"
            }

        try:
            child = pexpect.spawn(command, timeout=timeout, encoding='utf-8')

            transcript = []
            output = []

            while True:
                try:
                    # Read until we see something that looks like a prompt
                    index = child.expect([
                        r'.*\? *$',  # Question (ends with ?)
                        r'.*: *$',   # Prompt (ends with :)
                        r'\[.*\] *$',  # Selection (e.g., [Y/n])
                        pexpect.EOF,
                        pexpect.TIMEOUT,
                    ], timeout=5)

                    before = child.before or ""
                    output.append(before)

                    if index in (0, 1, 2):
                        # Looks like a prompt
                        prompt_text = (child.before or "") + (child.after or "")

                        # Decide how to respond
                        if prompt_handler:
                            response = prompt_handler(prompt_text)
                        else:
                            response = self._default_response(prompt_text)

                        transcript.append({
                            "prompt": prompt_text,
                            "response": response,
                        })

                        # Send response
                        child.sendline(response)

                    elif index == 3:
                        # EOF - process ended
                        break
                    else:
                        # Timeout - might be done or might be stuck
                        break

                except pexpect.TIMEOUT:
                    # No more prompts - process might be done
                    break
                except pexpect.EOF:
                    break

            # Get any remaining output
            try:
                child.expect(pexpect.EOF, timeout=1)
                output.append(child.before or "")
            except:
                pass

            return {
                "success": True,
                "transcript": transcript,
                "full_output": "".join(output),
                "interactions": len(transcript),
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "transcript": transcript if 'transcript' in locals() else [],
            }

    def _default_response(self, prompt: str) -> str:
        """
        Default response strategy for common prompts.

        Args:
            prompt: The prompt text

        Returns:
            Appropriate response
        """
        prompt_lower = prompt.lower()

        # Yes/no questions
        if "[y/n]" in prompt_lower or "(y/n)" in prompt_lower:
            # Default to yes for most things
            if "delete" in prompt_lower or "remove" in prompt_lower:
                return "n"  # Be safe with destructive actions
            return "y"

        # Selection prompts
        if "select" in prompt_lower or "choose" in prompt_lower:
            return "1"  # Default to first option

        # Name/text prompts
        if "name" in prompt_lower:
            return "GaiaCodeApp"

        if "email" in prompt_lower:
            return "dev@example.com"

        if "password" in prompt_lower:
            return "temporary_password_123"

        if "port" in prompt_lower:
            return "8000"

        # Default: empty (just press enter)
        return ""
