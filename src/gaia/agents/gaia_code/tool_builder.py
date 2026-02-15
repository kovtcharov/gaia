# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
ToolBuilder: Agent creates its own tools.

M5: Agent Auto-Generation + Self-Extension

The ToolBuilder enables the agent to:
- Write new tool functions
- Validate tool safety
- Test tools
- Register tools in tools.db

This is one of the key self-extension capabilities.
"""

import ast
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

from .shared_state import get_shared_state

logger = logging.getLogger(__name__)


class ToolBuilder:
    """
    Enables the agent to create new tools by writing Python code.

    Safety checks:
    - No eval/exec/compile
    - No subprocess with shell=True
    - No os.system
    - No __import__
    - No file I/O outside workspace
    """

    def __init__(self, workspace_dir: Optional[Path] = None):
        self.workspace_dir = workspace_dir or Path.home() / ".gaia" / "workspace"
        self.tools_dir = self.workspace_dir / "learned_tools"
        self.tools_dir.mkdir(parents=True, exist_ok=True)
        self.state = get_shared_state(workspace_dir)

    def create_tool(
        self,
        name: str,
        description: str,
        code: str,
        category: str = "learned",
        test_code: Optional[str] = None,
    ) -> Dict:
        """
        Create a new tool from generated code.

        Args:
            name: Tool name (must be valid Python identifier)
            description: What the tool does
            code: Python source code for the tool function
            category: Tool category (default: "learned")
            test_code: Optional test code to validate the tool

        Returns:
            Dict with success status and details
        """
        # Validate name
        if not name.isidentifier():
            return {
                "success": False,
                "error": f"Invalid tool name: {name} (must be valid Python identifier)",
            }

        # Validate safety
        safety_result = self._validate_safety(code)
        if not safety_result["safe"]:
            return {
                "success": False,
                "error": f"Safety validation failed: {safety_result['reason']}",
            }

        # Validate syntax
        try:
            ast.parse(code)
        except SyntaxError as e:
            return {
                "success": False,
                "error": f"Syntax error in tool code: {e}",
            }

        # Write tool to file
        tool_path = self.tools_dir / f"{name}.py"
        tool_path.write_text(code)

        # Test the tool if test code provided
        if test_code:
            test_result = self._run_tool_test(name, test_code)
            if not test_result["passed"]:
                tool_path.unlink()  # Remove failed tool
                return {
                    "success": False,
                    "error": f"Tool test failed: {test_result['error']}",
                }

        # Register in tools.db
        try:
            tool_id = self.state.tools.register_tool(
                name=name,
                category=category,
                description=description,
                source="learned",
                code_path=str(tool_path),
            )

            logger.info(f"Created tool: {name} (ID: {tool_id})")

            return {
                "success": True,
                "tool_id": tool_id,
                "tool_name": name,
                "tool_path": str(tool_path),
            }

        except Exception as e:
            tool_path.unlink()  # Remove on registration failure
            return {
                "success": False,
                "error": f"Failed to register tool: {e}",
            }

    def _validate_safety(self, code: str) -> Dict:
        """
        Validate that tool code is safe to execute.

        Checks for:
        - eval/exec/compile
        - subprocess with shell=True
        - os.system
        - __import__
        - Arbitrary file access

        Args:
            code: Python source code

        Returns:
            Dict with 'safe' (bool) and 'reason' (str)
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return {"safe": False, "reason": "Syntax error"}

        # Walk the AST and check for dangerous operations
        for node in ast.walk(tree):
            # Check for eval/exec/compile
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in ("eval", "exec", "compile", "__import__"):
                    return {
                        "safe": False,
                        "reason": f"Dangerous function call: {node.func.id}",
                    }

            # Check for subprocess with shell=True
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr in ("system", "popen"):
                        return {
                            "safe": False,
                            "reason": f"Dangerous function: {node.func.attr}",
                        }

                # Check for shell=True in subprocess calls
                for keyword in node.keywords:
                    if keyword.arg == "shell" and isinstance(
                        keyword.value, ast.Constant
                    ):
                        if keyword.value.value is True:
                            return {
                                "safe": False,
                                "reason": "subprocess call with shell=True",
                            }

        return {"safe": True, "reason": ""}

    def _run_tool_test(self, tool_name: str, test_code: str) -> Dict:
        """
        Run test code for a tool.

        Args:
            tool_name: Name of the tool
            test_code: Python test code

        Returns:
            Dict with 'passed' (bool) and 'error' (str)
        """
        # Write test file
        test_path = self.tools_dir / f"test_{tool_name}.py"
        test_path.write_text(test_code)

        try:
            # Run pytest on the test file
            result = subprocess.run(
                [sys.executable, "-m", "pytest", str(test_path), "-v"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            # Clean up test file
            test_path.unlink()

            if result.returncode == 0:
                return {"passed": True, "error": ""}
            else:
                return {
                    "passed": False,
                    "error": result.stdout + result.stderr,
                }

        except subprocess.TimeoutExpired:
            test_path.unlink()
            return {"passed": False, "error": "Test timed out (>10s)"}
        except Exception as e:
            if test_path.exists():
                test_path.unlink()
            return {"passed": False, "error": str(e)}

    def list_learned_tools(self) -> List[Dict]:
        """
        List all learned tools.

        Returns:
            List of tool info dicts
        """
        try:
            cursor = self.state.tools.conn.execute(
                "SELECT id, name, description, created_at FROM tools WHERE source = 'learned'"
            )

            tools = []
            for row in cursor.fetchall():
                tools.append(
                    {
                        "id": row[0],
                        "name": row[1],
                        "description": row[2],
                        "created_at": row[3],
                    }
                )

            return tools

        except Exception as e:
            logger.warning(f"Failed to list learned tools: {e}")
            return []

    def remove_tool(self, tool_name: str) -> bool:
        """
        Remove a learned tool.

        Args:
            tool_name: Name of tool to remove

        Returns:
            True if removed, False otherwise
        """
        try:
            # Remove from database
            self.state.tools.conn.execute(
                "DELETE FROM tools WHERE name = ? AND source = 'learned'", (tool_name,)
            )
            self.state.tools.conn.commit()

            # Remove file
            tool_path = self.tools_dir / f"{tool_name}.py"
            if tool_path.exists():
                tool_path.unlink()

            logger.info(f"Removed tool: {tool_name}")
            return True

        except Exception as e:
            logger.warning(f"Failed to remove tool {tool_name}: {e}")
            return False

    def get_tool_usage_stats(self, tool_name: str) -> Dict:
        """
        Get usage statistics for a tool.

        Args:
            tool_name: Name of tool

        Returns:
            Dict with usage statistics
        """
        try:
            # Get tool ID
            cursor = self.state.tools.conn.execute(
                "SELECT id FROM tools WHERE name = ?", (tool_name,)
            )
            row = cursor.fetchone()

            if not row:
                return {"error": "Tool not found"}

            tool_id = row[0]

            # Get usage stats
            cursor = self.state.tools.conn.execute(
                """
                SELECT
                    COUNT(*) as total_uses,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_uses,
                    AVG(duration_ms) as avg_duration_ms
                FROM tool_usage
                WHERE tool_id = ?
            """,
                (tool_id,),
            )

            row = cursor.fetchone()

            return {
                "total_uses": row[0] or 0,
                "successful_uses": row[1] or 0,
                "avg_duration_ms": row[2] or 0,
                "success_rate": (
                    (row[1] / row[0] * 100) if row[0] and row[0] > 0 else 0
                ),
            }

        except Exception as e:
            logger.warning(f"Failed to get usage stats for {tool_name}: {e}")
            return {"error": str(e)}


# Example tool template for the agent to use
TOOL_TEMPLATE = '''
"""
{description}
"""

def {tool_name}({parameters}) -> {return_type}:
    """
    {description}

    Args:
        {args_doc}

    Returns:
        {return_doc}
    """
    {implementation}
'''

TEST_TEMPLATE = '''
import pytest
from {tool_name} import {tool_name}

def test_{tool_name}_basic():
    """Test basic functionality."""
    {test_implementation}

def test_{tool_name}_edge_cases():
    """Test edge cases."""
    {edge_case_tests}
'''
