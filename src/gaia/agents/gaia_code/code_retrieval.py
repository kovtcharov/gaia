# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
CodeRetrievalPipeline: Multi-layer code retrieval for the GAIA Code agent.

4-Layer Architecture:
  Layer 1: Structural Index — AST-parsed symbols, deps, file hashes, code chunks
  Layer 2: Semantic Annotations — Summaries/concepts per chunk (AST + optional LLM)
  Layer 3: Retrieval Index — FAISS embeddings + FTS5 keyword search
  Layer 4: Context Assembler — Query classification, hierarchical zoom, impact analysis

Features:
  - Incremental indexing via SHA-256 file hashing
  - Background file watcher for auto-updating index
  - Hybrid search (FAISS semantic + FTS5 keyword)
  - Smart file filtering (skip binaries/lock files from annotation)
  - LLM-enriched annotations (cloud by default, local opt-in)
  - Impact analysis with transitive dependency traversal
"""

import ast
import hashlib
import json
import logging
import os
import pickle
import re
import sqlite3
import threading
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ============================================================================
# Constants
# ============================================================================

# File extensions to fully index (AST parse + annotate)
INDEXABLE_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx",
    ".yaml", ".yml", ".json", ".toml",
    ".md", ".rst", ".txt",
    ".html", ".css", ".scss",
    ".sh", ".bash",
    ".sql",
    ".cfg", ".ini",
}

# Files to track existence only (no annotation)
TRACK_ONLY_EXTENSIONS = {
    ".lock", ".svg", ".png", ".jpg", ".jpeg", ".gif", ".ico",
    ".woff", ".woff2", ".ttf", ".eot",
    ".pyc", ".pyo", ".so", ".dll", ".dylib",
    ".zip", ".tar", ".gz", ".bz2", ".xz",
    ".exe", ".bin", ".o", ".a",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx",
    ".db", ".sqlite", ".sqlite3",
    ".faiss", ".pkl", ".pickle", ".npy", ".npz",
    ".mp3", ".mp4", ".wav", ".avi",
    ".egg-info",
}

# Filenames to skip entirely (lock files, generated)
SKIP_FILENAMES = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
    "poetry.lock", "Pipfile.lock", "uv.lock",
    "composer.lock", "Gemfile.lock", "Cargo.lock",
}

# Directories to always exclude
EXCLUDE_DIRS = {
    "__pycache__", ".git", ".tox", "node_modules", ".pytest_cache",
    "build", "dist", ".eggs", "venv", ".venv", "env", "virtualenv",
    ".mypy_cache", ".ruff_cache", ".hypothesis", "htmlcov",
    ".next", ".nuxt", ".cache", ".parcel-cache",
}

# Maximum lines for a single chunk before splitting
MAX_CHUNK_LINES = 200

# File watcher debounce interval (ms)
WATCHER_DEBOUNCE_MS = 500

# Default context budget in tokens (rough estimate: 4 chars per token)
DEFAULT_CONTEXT_BUDGET = 8000

CHARS_PER_TOKEN = 4


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class CodeChunk:
    """A semantically meaningful chunk of code."""
    chunk_id: str
    file_path: str
    relative_path: str
    module_path: str
    chunk_type: str  # module_docstring, import_block, class, function, method, standalone_block
    symbol_name: str
    parent_symbol: Optional[str] = None
    start_line: int = 0
    end_line: int = 0
    source_code: str = ""
    docstring: Optional[str] = None
    decorators: List[str] = field(default_factory=list)
    signature: Optional[str] = None
    complexity: int = 0
    dependencies: List[str] = field(default_factory=list)
    content_hash: str = ""


@dataclass
class CodeAnnotation:
    """Semantic annotation for a code chunk."""
    chunk_id: str
    summary: str = ""
    purpose: str = ""
    concepts: List[str] = field(default_factory=list)
    relationships: List[str] = field(default_factory=list)
    search_text: str = ""
    annotation_source: str = "ast"  # "ast" or "llm"
    generated_at: str = ""


@dataclass
class SearchResult:
    """A single search result from the retrieval pipeline."""
    chunk_id: str
    file_path: str
    relative_path: str
    module_path: str
    symbol_name: str
    chunk_type: str
    start_line: int
    end_line: int
    source_code: str
    summary: str
    score: float = 0.0
    match_type: str = ""  # "semantic", "keyword", "hybrid", "structural"


@dataclass
class ImpactResult:
    """Result of an impact analysis."""
    target_file: str
    target_symbol: str
    direct_dependents: List[Dict[str, Any]] = field(default_factory=list)
    symbol_references: List[Dict[str, Any]] = field(default_factory=list)
    transitive_impact: List[str] = field(default_factory=list)
    risk_level: str = "LOW"  # LOW, MEDIUM, HIGH


# ============================================================================
# Layer 1: Structural Index
# ============================================================================


def compute_file_hash(file_path: str) -> str:
    """Compute SHA-256 hash of a file's contents."""
    sha256 = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for block in iter(lambda: f.read(8192), b""):
                sha256.update(block)
        return sha256.hexdigest()
    except (OSError, IOError):
        return ""


def _get_decorator_name(decorator) -> str:
    """Extract decorator name from AST node."""
    if isinstance(decorator, ast.Name):
        return decorator.id
    elif isinstance(decorator, ast.Attribute):
        return ast.dump(decorator)
    elif isinstance(decorator, ast.Call):
        if isinstance(decorator.func, ast.Name):
            return decorator.func.id
        elif isinstance(decorator.func, ast.Attribute):
            return ast.dump(decorator.func)
    return "unknown"


def _calculate_complexity(node: ast.AST) -> int:
    """Calculate cyclomatic complexity of an AST node."""
    complexity = 1
    for child in ast.walk(node):
        if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
            complexity += 1
        elif isinstance(child, ast.ExceptHandler):
            complexity += 1
        elif isinstance(child, ast.BoolOp):
            complexity += len(child.values) - 1
    return complexity


def _get_function_signature(node) -> str:
    """Extract function signature string from AST node."""
    args = node.args
    parts = []
    # Regular args
    for arg in args.args:
        name = arg.arg
        if arg.annotation:
            name += f": {ast.dump(arg.annotation)}"
        parts.append(name)
    # *args
    if args.vararg:
        parts.append(f"*{args.vararg.arg}")
    # **kwargs
    if args.kwarg:
        parts.append(f"**{args.kwarg.arg}")
    sig = f"({', '.join(parts)})"
    # Return annotation
    if node.returns:
        sig += f" -> {ast.dump(node.returns)}"
    return sig


def _get_node_source(source_lines: List[str], node: ast.AST) -> str:
    """Extract source code for an AST node."""
    start = node.lineno - 1
    end = getattr(node, "end_lineno", node.lineno)
    return "\n".join(source_lines[start:end])


def extract_code_chunks(
    file_path: str,
    relative_path: str,
    module_path: str,
    source: str,
) -> List[CodeChunk]:
    """
    Extract semantically meaningful code chunks from a Python file.

    Chunks at natural code boundaries:
    - module_docstring: File-level docstring
    - import_block: All imports grouped as one chunk
    - class: Entire class (for classes ≤200 lines)
    - function: Each top-level function
    - method: Each method in large classes (>200 lines)
    - standalone_block: Module-level code (assignments, if-name-main)
    """
    chunks = []
    source_lines = source.splitlines()

    try:
        tree = ast.parse(source, filename=file_path)
    except SyntaxError:
        # Fall back to single-file chunk
        chunk_id = f"{module_path}::__file__"
        chunks.append(CodeChunk(
            chunk_id=chunk_id,
            file_path=file_path,
            relative_path=relative_path,
            module_path=module_path,
            chunk_type="file",
            symbol_name="__file__",
            start_line=1,
            end_line=len(source_lines),
            source_code=source[:5000],  # Truncate large files
            content_hash=hashlib.sha256(source.encode()).hexdigest()[:16],
        ))
        return chunks

    # 1. Module docstring
    module_docstring = ast.get_docstring(tree)
    if module_docstring:
        chunks.append(CodeChunk(
            chunk_id=f"{module_path}::__doc__",
            file_path=file_path,
            relative_path=relative_path,
            module_path=module_path,
            chunk_type="module_docstring",
            symbol_name="__doc__",
            start_line=1,
            end_line=module_docstring.count("\n") + 2,
            source_code=f'"""{module_docstring}"""',
            docstring=module_docstring,
            content_hash=hashlib.sha256(module_docstring.encode()).hexdigest()[:16],
        ))

    # 2. Import block — gather all imports
    import_lines = []
    import_deps = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            import_lines.append((node.lineno, node.end_lineno or node.lineno))
            for alias in node.names:
                import_deps.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            import_lines.append((node.lineno, node.end_lineno or node.lineno))
            if node.module:
                import_deps.append(node.module)

    if import_lines:
        start = min(s for s, _ in import_lines)
        end = max(e for _, e in import_lines)
        import_source = "\n".join(source_lines[start - 1:end])
        chunks.append(CodeChunk(
            chunk_id=f"{module_path}::__imports__",
            file_path=file_path,
            relative_path=relative_path,
            module_path=module_path,
            chunk_type="import_block",
            symbol_name="__imports__",
            start_line=start,
            end_line=end,
            source_code=import_source,
            dependencies=import_deps,
            content_hash=hashlib.sha256(import_source.encode()).hexdigest()[:16],
        ))

    # 3. Classes and functions
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            class_lines = (node.end_lineno or node.lineno) - node.lineno + 1

            if class_lines <= MAX_CHUNK_LINES:
                # Small class → one chunk
                class_source = _get_node_source(source_lines, node)
                chunks.append(CodeChunk(
                    chunk_id=f"{module_path}::{node.name}",
                    file_path=file_path,
                    relative_path=relative_path,
                    module_path=module_path,
                    chunk_type="class",
                    symbol_name=node.name,
                    start_line=node.lineno,
                    end_line=node.end_lineno or node.lineno,
                    source_code=class_source,
                    docstring=ast.get_docstring(node),
                    decorators=[_get_decorator_name(d) for d in node.decorator_list],
                    complexity=_calculate_complexity(node),
                    content_hash=hashlib.sha256(class_source.encode()).hexdigest()[:16],
                ))
            else:
                # Large class → split into methods
                # First, add class signature chunk
                class_sig_end = node.lineno
                for child in ast.iter_child_nodes(node):
                    if hasattr(child, "lineno"):
                        class_sig_end = child.lineno - 1
                        break
                class_sig_source = "\n".join(source_lines[node.lineno - 1:class_sig_end])
                chunks.append(CodeChunk(
                    chunk_id=f"{module_path}::{node.name}::__sig__",
                    file_path=file_path,
                    relative_path=relative_path,
                    module_path=module_path,
                    chunk_type="class",
                    symbol_name=node.name,
                    start_line=node.lineno,
                    end_line=class_sig_end,
                    source_code=class_sig_source,
                    docstring=ast.get_docstring(node),
                    decorators=[_get_decorator_name(d) for d in node.decorator_list],
                    content_hash=hashlib.sha256(class_sig_source.encode()).hexdigest()[:16],
                ))

                # Then each method
                for child in ast.iter_child_nodes(node):
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        method_source = _get_node_source(source_lines, child)
                        chunks.append(CodeChunk(
                            chunk_id=f"{module_path}::{node.name}::{child.name}",
                            file_path=file_path,
                            relative_path=relative_path,
                            module_path=module_path,
                            chunk_type="method",
                            symbol_name=child.name,
                            parent_symbol=node.name,
                            start_line=child.lineno,
                            end_line=child.end_lineno or child.lineno,
                            source_code=method_source,
                            docstring=ast.get_docstring(child),
                            decorators=[_get_decorator_name(d) for d in child.decorator_list],
                            signature=_get_function_signature(child),
                            complexity=_calculate_complexity(child),
                            content_hash=hashlib.sha256(method_source.encode()).hexdigest()[:16],
                        ))

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_source = _get_node_source(source_lines, node)
            chunks.append(CodeChunk(
                chunk_id=f"{module_path}::{node.name}",
                file_path=file_path,
                relative_path=relative_path,
                module_path=module_path,
                chunk_type="function",
                symbol_name=node.name,
                start_line=node.lineno,
                end_line=node.end_lineno or node.lineno,
                source_code=func_source,
                docstring=ast.get_docstring(node),
                decorators=[_get_decorator_name(d) for d in node.decorator_list],
                signature=_get_function_signature(node),
                complexity=_calculate_complexity(node),
                content_hash=hashlib.sha256(func_source.encode()).hexdigest()[:16],
            ))

    # 4. Standalone blocks (assignments, if __name__ == "__main__", etc.)
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            block_source = _get_node_source(source_lines, node)
            # Use line number for uniqueness
            chunks.append(CodeChunk(
                chunk_id=f"{module_path}::__block_L{node.lineno}",
                file_path=file_path,
                relative_path=relative_path,
                module_path=module_path,
                chunk_type="standalone_block",
                symbol_name=f"__block_L{node.lineno}",
                start_line=node.lineno,
                end_line=node.end_lineno or node.lineno,
                source_code=block_source,
                content_hash=hashlib.sha256(block_source.encode()).hexdigest()[:16],
            ))
        elif isinstance(node, ast.If):
            # Check for if __name__ == "__main__"
            block_source = _get_node_source(source_lines, node)
            if "__name__" in block_source and "__main__" in block_source:
                chunks.append(CodeChunk(
                    chunk_id=f"{module_path}::__main__",
                    file_path=file_path,
                    relative_path=relative_path,
                    module_path=module_path,
                    chunk_type="standalone_block",
                    symbol_name="__main__",
                    start_line=node.lineno,
                    end_line=node.end_lineno or node.lineno,
                    source_code=block_source,
                    content_hash=hashlib.sha256(block_source.encode()).hexdigest()[:16],
                ))

    return chunks


def extract_non_python_chunks(
    file_path: str,
    relative_path: str,
    module_path: str,
    source: str,
    language: str,
) -> List[CodeChunk]:
    """Extract chunks from non-Python files using structure-aware splitting."""
    chunks = []
    source_lines = source.splitlines()

    if language in ("yaml", "json", "toml"):
        # One chunk per file for config files
        chunks.append(CodeChunk(
            chunk_id=f"{module_path}::__file__",
            file_path=file_path,
            relative_path=relative_path,
            module_path=module_path,
            chunk_type="file",
            symbol_name="__file__",
            start_line=1,
            end_line=len(source_lines),
            source_code=source[:5000],
            content_hash=hashlib.sha256(source.encode()).hexdigest()[:16],
        ))

    elif language in ("markdown", "rst"):
        # Chunk by header sections
        current_start = 1
        current_name = "__header__"
        for i, line in enumerate(source_lines, 1):
            if line.startswith("#") or (language == "rst" and re.match(r'^[=\-~^]+$', line)):
                if i > current_start:
                    section_source = "\n".join(source_lines[current_start - 1:i - 1])
                    if section_source.strip():
                        chunks.append(CodeChunk(
                            chunk_id=f"{module_path}::{current_name}",
                            file_path=file_path,
                            relative_path=relative_path,
                            module_path=module_path,
                            chunk_type="section",
                            symbol_name=current_name,
                            start_line=current_start,
                            end_line=i - 1,
                            source_code=section_source,
                            content_hash=hashlib.sha256(section_source.encode()).hexdigest()[:16],
                        ))
                current_start = i
                current_name = line.strip().lstrip("#").strip()[:50] or f"section_L{i}"

        # Last section
        if current_start <= len(source_lines):
            section_source = "\n".join(source_lines[current_start - 1:])
            if section_source.strip():
                chunks.append(CodeChunk(
                    chunk_id=f"{module_path}::{current_name}",
                    file_path=file_path,
                    relative_path=relative_path,
                    module_path=module_path,
                    chunk_type="section",
                    symbol_name=current_name,
                    start_line=current_start,
                    end_line=len(source_lines),
                    source_code=section_source,
                    content_hash=hashlib.sha256(section_source.encode()).hexdigest()[:16],
                ))

    elif language in ("javascript", "typescript", "tsx", "jsx"):
        # Blank-line-separated blocks
        block_start = 1
        block_num = 0
        in_block = False
        for i, line in enumerate(source_lines, 1):
            if line.strip():
                if not in_block:
                    block_start = i
                    in_block = True
            else:
                if in_block:
                    block_source = "\n".join(source_lines[block_start - 1:i - 1])
                    block_num += 1
                    chunks.append(CodeChunk(
                        chunk_id=f"{module_path}::block_{block_num}",
                        file_path=file_path,
                        relative_path=relative_path,
                        module_path=module_path,
                        chunk_type="block",
                        symbol_name=f"block_{block_num}",
                        start_line=block_start,
                        end_line=i - 1,
                        source_code=block_source,
                        content_hash=hashlib.sha256(block_source.encode()).hexdigest()[:16],
                    ))
                    in_block = False

        # Last block
        if in_block:
            block_source = "\n".join(source_lines[block_start - 1:])
            block_num += 1
            chunks.append(CodeChunk(
                chunk_id=f"{module_path}::block_{block_num}",
                file_path=file_path,
                relative_path=relative_path,
                module_path=module_path,
                chunk_type="block",
                symbol_name=f"block_{block_num}",
                start_line=block_start,
                end_line=len(source_lines),
                source_code=block_source,
                content_hash=hashlib.sha256(block_source.encode()).hexdigest()[:16],
            ))

    else:
        # Fixed-size blocks (500 lines)
        block_size = 500
        for i in range(0, len(source_lines), block_size):
            block_source = "\n".join(source_lines[i:i + block_size])
            block_num = i // block_size + 1
            chunks.append(CodeChunk(
                chunk_id=f"{module_path}::block_{block_num}",
                file_path=file_path,
                relative_path=relative_path,
                module_path=module_path,
                chunk_type="block",
                symbol_name=f"block_{block_num}",
                start_line=i + 1,
                end_line=min(i + block_size, len(source_lines)),
                source_code=block_source,
                content_hash=hashlib.sha256(block_source.encode()).hexdigest()[:16],
            ))

    return chunks


def _detect_language(file_path: str) -> str:
    """Detect programming language from file extension."""
    ext = Path(file_path).suffix.lower()
    lang_map = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".tsx": "tsx",
        ".jsx": "jsx",
        ".yaml": "yaml", ".yml": "yaml",
        ".json": "json",
        ".toml": "toml",
        ".md": "markdown",
        ".rst": "rst",
        ".html": "html",
        ".css": "css", ".scss": "css",
        ".sh": "shell", ".bash": "shell",
        ".sql": "sql",
        ".cfg": "config", ".ini": "config",
        ".txt": "text",
    }
    return lang_map.get(ext, "unknown")


# ============================================================================
# Layer 2: Semantic Annotations
# ============================================================================


def generate_ast_annotation(chunk: CodeChunk) -> CodeAnnotation:
    """Generate annotation from AST information only (no LLM)."""
    # Build summary
    if chunk.chunk_type == "module_docstring":
        summary = f"Module docstring for {chunk.module_path}"
    elif chunk.chunk_type == "import_block":
        summary = f"Imports in {chunk.module_path}"
    elif chunk.chunk_type == "class":
        if chunk.parent_symbol:
            summary = f"class '{chunk.symbol_name}' (signature) in {chunk.module_path}"
        else:
            summary = f"class '{chunk.symbol_name}' in {chunk.module_path}"
    elif chunk.chunk_type == "method":
        summary = f"method '{chunk.symbol_name}' of {chunk.parent_symbol} in {chunk.module_path}"
    elif chunk.chunk_type == "function":
        summary = f"function '{chunk.symbol_name}' in {chunk.module_path}"
    elif chunk.chunk_type == "standalone_block":
        summary = f"Module-level code at line {chunk.start_line} in {chunk.module_path}"
    elif chunk.chunk_type == "section":
        summary = f"Section '{chunk.symbol_name}' in {chunk.relative_path}"
    else:
        summary = f"{chunk.chunk_type} '{chunk.symbol_name}' in {chunk.module_path}"

    # Purpose from docstring
    purpose = chunk.docstring or ""

    # Extract concepts from docstring + decorators
    concepts = list(chunk.decorators)
    if chunk.docstring:
        # Extract nouns (simple heuristic: capitalized words and technical terms)
        words = re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)*\b', chunk.docstring)
        concepts.extend(words[:10])
        # Also extract key technical terms
        tech_terms = re.findall(r'\b(?:API|REST|HTTP|SQL|JSON|XML|HTML|CSS|JWT|OAuth|NPU|GPU|CPU|LLM|RAG|AST|MCP|CLI|TUI|FAISS|FTS5|AMD)\b', chunk.docstring, re.IGNORECASE)
        concepts.extend(tech_terms)

    # Extract relationships from source and dependencies
    relationships = []
    if chunk.source_code:
        # Inheritance
        inherit_match = re.findall(r'class\s+\w+\(([^)]+)\)', chunk.source_code)
        for bases in inherit_match:
            for base in bases.split(","):
                base = base.strip()
                if base and base != "object":
                    relationships.append(f"inherits from {base}")

    # Imports used (from dependencies field, independent of source_code)
    for dep in chunk.dependencies:
        relationships.append(f"imports {dep}")

    # Unique concepts
    concepts = list(dict.fromkeys(concepts))

    # Build search text
    search_text = f"{summary} | {purpose} | {' '.join(concepts)}"

    return CodeAnnotation(
        chunk_id=chunk.chunk_id,
        summary=summary,
        purpose=purpose,
        concepts=concepts,
        relationships=relationships,
        search_text=search_text,
        annotation_source="ast",
        generated_at=datetime.now().isoformat(),
    )


def generate_llm_annotations_batch(
    chunks: List[CodeChunk],
    llm_client,
    batch_size: int = 10,
) -> List[CodeAnnotation]:
    """
    Generate LLM-enriched annotations for a batch of chunks.

    Args:
        chunks: Code chunks to annotate
        llm_client: LLM client (Claude API or Lemonade)
        batch_size: Number of chunks per LLM call

    Returns:
        List of CodeAnnotation objects
    """
    annotations = []

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]

        # Build prompt with all chunks in batch
        prompt_parts = [
            "For each code snippet below, provide a JSON array with one object per snippet.",
            "Each object must have: summary (1-2 sentences), purpose (what it does), concepts (list of key terms).",
            "Return ONLY valid JSON. No markdown, no explanation.",
            "",
        ]

        for j, chunk in enumerate(batch):
            code_preview = chunk.source_code[:1500]  # Limit code size
            prompt_parts.append(f"--- Snippet {j + 1}: {chunk.chunk_type} '{chunk.symbol_name}' ---")
            prompt_parts.append(code_preview)
            prompt_parts.append("")

        prompt = "\n".join(prompt_parts)

        try:
            # Call LLM
            response = llm_client.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=2000,
            )

            # Parse response
            response_text = response if isinstance(response, str) else response.get("content", "")

            # Try to extract JSON from response
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                for j, chunk in enumerate(batch):
                    if j < len(parsed):
                        item = parsed[j]
                        concepts = item.get("concepts", [])
                        summary = item.get("summary", "")
                        purpose = item.get("purpose", "")
                        search_text = f"{summary} | {purpose} | {' '.join(concepts)}"

                        annotations.append(CodeAnnotation(
                            chunk_id=chunk.chunk_id,
                            summary=summary,
                            purpose=purpose,
                            concepts=concepts if isinstance(concepts, list) else [],
                            relationships=[],
                            search_text=search_text,
                            annotation_source="llm",
                            generated_at=datetime.now().isoformat(),
                        ))
                    else:
                        # Fall back to AST annotation
                        annotations.append(generate_ast_annotation(chunk))
            else:
                # JSON parse failed, fall back to AST
                logger.warning("[CodeRetrieval] LLM response not valid JSON, falling back to AST annotations")
                for chunk in batch:
                    annotations.append(generate_ast_annotation(chunk))

        except Exception as e:
            logger.warning(f"[CodeRetrieval] LLM annotation failed: {e}, falling back to AST")
            for chunk in batch:
                annotations.append(generate_ast_annotation(chunk))

    return annotations


# ============================================================================
# Layer 3 & 4: CodeRetrievalPipeline (Main Class)
# ============================================================================


class CodeRetrievalPipeline:
    """
    Multi-layer code retrieval pipeline for the GAIA Code agent.

    Provides:
    - Incremental structural indexing (Layer 1)
    - Semantic annotations with optional LLM enrichment (Layer 2)
    - Hybrid FAISS + FTS5 search (Layer 3)
    - Query classification and context assembly (Layer 4)
    - Background file watcher for auto-updating
    - Impact analysis with transitive dependency traversal
    """

    def __init__(
        self,
        root_path: str,
        workspace_dir: Optional[Path] = None,
        llm_provider: str = "cloud",
        llm_client=None,
        enable_watcher: bool = True,
        extensions: Optional[Set[str]] = None,
    ):
        """
        Initialize the code retrieval pipeline.

        Args:
            root_path: Root directory of codebase to index
            workspace_dir: Workspace directory for databases
            llm_provider: "cloud" (default), "local", or "none" for annotations
            llm_client: Pre-configured LLM client (optional)
            enable_watcher: Enable background file watcher (default: True)
            extensions: File extensions to index (default: INDEXABLE_EXTENSIONS)
        """
        self.root = Path(root_path).resolve()
        self.workspace_dir = workspace_dir or Path.home() / ".gaia" / "workspace"
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.llm_provider = llm_provider
        self.llm_client = llm_client
        self.extensions = extensions or INDEXABLE_EXTENSIONS

        # Database path
        self.db_path = self.workspace_dir / "code_index.db"
        self.conn = self._init_database()

        # FAISS index (lazy loaded)
        self._faiss_index = None
        self._faiss_mapping: List[str] = []  # chunk_id list
        self._embedding_engine = None

        # File watcher
        self._watcher_thread = None
        self._watcher_stop = threading.Event()
        self._pending_changes: Dict[str, str] = {}  # path -> change_type
        self._pending_lock = threading.Lock()

        # Statistics
        self.stats = {
            "files_indexed": 0,
            "chunks_created": 0,
            "annotations_created": 0,
            "last_index_time": None,
        }

        # Start watcher if enabled
        if enable_watcher:
            self._start_watcher()

        logger.info(f"[CodeRetrieval] Initialized pipeline for {self.root}")

    # ====================================================================
    # Database Setup
    # ====================================================================

    def _init_database(self) -> sqlite3.Connection:
        """Initialize SQLite database with schema."""
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")

        conn.executescript("""
            CREATE TABLE IF NOT EXISTS repositories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                root_path TEXT UNIQUE,
                last_full_index TEXT,
                last_incremental TEXT,
                total_files INTEGER DEFAULT 0,
                total_chunks INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS files (
                file_path TEXT PRIMARY KEY,
                repo_id INTEGER,
                relative_path TEXT,
                module_path TEXT,
                content_hash TEXT,
                indexed_at TEXT,
                line_count INTEGER DEFAULT 0,
                size_bytes INTEGER DEFAULT 0,
                language TEXT,
                docstring TEXT,
                is_track_only INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS code_chunks (
                chunk_id TEXT PRIMARY KEY,
                file_path TEXT,
                chunk_type TEXT,
                symbol_name TEXT,
                parent_symbol TEXT,
                start_line INTEGER,
                end_line INTEGER,
                source_code TEXT,
                docstring TEXT,
                signature TEXT,
                decorators TEXT,
                complexity INTEGER DEFAULT 0,
                dependencies TEXT,
                content_hash TEXT,
                FOREIGN KEY (file_path) REFERENCES files(file_path)
            );

            CREATE TABLE IF NOT EXISTS dependencies (
                source_file TEXT,
                target_file TEXT,
                import_type TEXT,
                symbols_imported TEXT,
                FOREIGN KEY (source_file) REFERENCES files(file_path)
            );

            CREATE TABLE IF NOT EXISTS annotations (
                chunk_id TEXT PRIMARY KEY,
                summary TEXT,
                purpose TEXT,
                concepts TEXT,
                relationships TEXT,
                search_text TEXT,
                annotation_source TEXT,
                generated_at TEXT,
                FOREIGN KEY (chunk_id) REFERENCES code_chunks(chunk_id)
            );

            CREATE TABLE IF NOT EXISTS summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                level TEXT,
                scope TEXT,
                summary TEXT,
                generated_at TEXT
            );

            -- Indices for fast lookups
            CREATE INDEX IF NOT EXISTS idx_chunks_file ON code_chunks(file_path);
            CREATE INDEX IF NOT EXISTS idx_chunks_type ON code_chunks(chunk_type);
            CREATE INDEX IF NOT EXISTS idx_chunks_symbol ON code_chunks(symbol_name);
            CREATE INDEX IF NOT EXISTS idx_deps_source ON dependencies(source_file);
            CREATE INDEX IF NOT EXISTS idx_deps_target ON dependencies(target_file);
            CREATE INDEX IF NOT EXISTS idx_files_hash ON files(content_hash);
            CREATE INDEX IF NOT EXISTS idx_annotations_source ON annotations(annotation_source);
        """)

        # FTS5 virtual table for keyword search
        try:
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS code_chunks_fts USING fts5(
                    chunk_id,
                    symbol_name,
                    module_path,
                    source_code,
                    summary,
                    concepts,
                    tokenize='porter unicode61'
                )
            """)
        except sqlite3.OperationalError:
            # FTS5 table already exists or not available
            pass

        conn.commit()
        return conn

    # ====================================================================
    # Layer 1: Structural Indexing
    # ====================================================================

    def index_repository(
        self,
        force_full: bool = False,
        include_tests: bool = True,
    ) -> Dict[str, Any]:
        """
        Index the repository with incremental support.

        Args:
            force_full: Force full re-index (default: False, incremental)
            include_tests: Include test files (default: True)

        Returns:
            Dict with index statistics
        """
        start_time = time.time()
        logger.info(f"[CodeRetrieval] Starting {'full' if force_full else 'incremental'} index of {self.root}")

        # Ensure repo record exists
        self.conn.execute(
            "INSERT OR IGNORE INTO repositories (root_path) VALUES (?)",
            (str(self.root),)
        )
        repo_row = self.conn.execute(
            "SELECT id FROM repositories WHERE root_path = ?",
            (str(self.root),)
        ).fetchone()
        repo_id = repo_row[0]

        # Get existing file hashes
        existing_hashes = {}
        if not force_full:
            cursor = self.conn.execute(
                "SELECT file_path, content_hash FROM files WHERE repo_id = ?",
                (repo_id,)
            )
            existing_hashes = {row[0]: row[1] for row in cursor.fetchall()}

        # Walk source files and classify changes
        current_files = {}
        new_files = []
        modified_files = []
        unchanged_files = []
        track_only_files = []

        for file_path in self._walk_source_files(include_tests):
            path_str = str(file_path)
            ext = file_path.suffix.lower()

            if ext in TRACK_ONLY_EXTENSIONS or file_path.name in SKIP_FILENAMES:
                track_only_files.append(file_path)
                current_files[path_str] = "__track_only__"
                continue

            content_hash = compute_file_hash(path_str)
            current_files[path_str] = content_hash

            if force_full or path_str not in existing_hashes:
                new_files.append(file_path)
            elif existing_hashes[path_str] != content_hash:
                modified_files.append(file_path)
            else:
                unchanged_files.append(file_path)

        # Detect deleted files
        deleted_files = set(existing_hashes.keys()) - set(current_files.keys())

        logger.info(
            f"[CodeRetrieval] File status: {len(new_files)} new, "
            f"{len(modified_files)} modified, {len(unchanged_files)} unchanged, "
            f"{len(deleted_files)} deleted, {len(track_only_files)} track-only"
        )

        # Process deletions
        for file_path in deleted_files:
            self._remove_file_from_index(file_path)

        # Process track-only files (record existence, no chunks)
        for file_path in track_only_files:
            self._index_track_only_file(file_path, repo_id)

        # Process new and modified files
        files_to_index = new_files + modified_files
        all_chunks = []
        for file_path in files_to_index:
            chunks = self._index_single_file(file_path, repo_id)
            all_chunks.extend(chunks)

        # Rebuild dependency graph for affected files
        if files_to_index:
            self._rebuild_dependencies(files_to_index, repo_id)

        # Generate annotations for new chunks
        annotations = self._generate_annotations(all_chunks)

        # Rebuild FAISS and FTS5 indices
        if files_to_index or deleted_files:
            self._rebuild_fts5_index()
            self._rebuild_faiss_index()

        # Update repo record
        elapsed = time.time() - start_time
        now = datetime.now().isoformat()
        total_files = self.conn.execute(
            "SELECT COUNT(*) FROM files WHERE repo_id = ?", (repo_id,)
        ).fetchone()[0]
        total_chunks = self.conn.execute(
            "SELECT COUNT(*) FROM code_chunks c JOIN files f ON c.file_path = f.file_path WHERE f.repo_id = ?",
            (repo_id,)
        ).fetchone()[0]

        if force_full:
            self.conn.execute(
                "UPDATE repositories SET last_full_index = ?, total_files = ?, total_chunks = ? WHERE id = ?",
                (now, total_files, total_chunks, repo_id)
            )
        else:
            self.conn.execute(
                "UPDATE repositories SET last_incremental = ?, total_files = ?, total_chunks = ? WHERE id = ?",
                (now, total_files, total_chunks, repo_id)
            )
        self.conn.commit()

        self.stats = {
            "files_indexed": len(files_to_index),
            "files_unchanged": len(unchanged_files),
            "files_deleted": len(deleted_files),
            "files_track_only": len(track_only_files),
            "chunks_created": len(all_chunks),
            "annotations_created": len(annotations),
            "total_files": total_files,
            "total_chunks": total_chunks,
            "index_time_seconds": round(elapsed, 2),
            "last_index_time": now,
        }

        logger.info(
            f"[CodeRetrieval] Indexed {len(files_to_index)} files, "
            f"{len(all_chunks)} chunks in {elapsed:.2f}s"
        )

        return self.stats

    def _walk_source_files(self, include_tests: bool = True) -> List[Path]:
        """Walk and collect all source files, respecting exclusions."""
        files = []
        all_extensions = self.extensions | TRACK_ONLY_EXTENSIONS

        for root, dirs, filenames in os.walk(str(self.root)):
            # Prune excluded directories (modifying dirs in-place)
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.endswith(".egg-info")]

            for filename in filenames:
                file_path = Path(root) / filename

                # Skip by filename
                if filename in SKIP_FILENAMES:
                    continue

                # Check extension
                ext = file_path.suffix.lower()
                if ext not in all_extensions:
                    continue

                # Skip test files if requested
                if not include_tests and ("test_" in filename or filename.endswith("_test.py")):
                    continue

                files.append(file_path)

        return files

    def _index_single_file(self, file_path: Path, repo_id: int) -> List[CodeChunk]:
        """Index a single file and return its chunks."""
        path_str = str(file_path)
        try:
            relative_path = str(file_path.relative_to(self.root))
        except ValueError:
            relative_path = file_path.name
        module_path = self._path_to_module(relative_path)
        language = _detect_language(path_str)

        try:
            with open(path_str, "r", encoding="utf-8", errors="replace") as f:
                source = f.read()
        except (OSError, IOError) as e:
            logger.warning(f"[CodeRetrieval] Cannot read {path_str}: {e}")
            return []

        content_hash = hashlib.sha256(source.encode()).hexdigest()
        source_lines = source.splitlines()

        # Extract file-level docstring for Python
        file_docstring = None
        if language == "python":
            try:
                tree = ast.parse(source)
                file_docstring = ast.get_docstring(tree)
            except SyntaxError:
                pass

        # Upsert file record
        self.conn.execute("""
            INSERT OR REPLACE INTO files
            (file_path, repo_id, relative_path, module_path, content_hash, indexed_at,
             line_count, size_bytes, language, docstring, is_track_only)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
        """, (
            path_str, repo_id, relative_path, module_path, content_hash,
            datetime.now().isoformat(), len(source_lines),
            file_path.stat().st_size if file_path.exists() else 0,
            language, file_docstring,
        ))

        # Remove old chunks for this file
        self.conn.execute("DELETE FROM code_chunks WHERE file_path = ?", (path_str,))
        self.conn.execute("DELETE FROM annotations WHERE chunk_id IN (SELECT chunk_id FROM code_chunks WHERE file_path = ?)", (path_str,))

        # Extract chunks
        if language == "python":
            chunks = extract_code_chunks(path_str, relative_path, module_path, source)
        else:
            chunks = extract_non_python_chunks(path_str, relative_path, module_path, source, language)

        # Store chunks in database
        for chunk in chunks:
            self.conn.execute("""
                INSERT OR REPLACE INTO code_chunks
                (chunk_id, file_path, chunk_type, symbol_name, parent_symbol,
                 start_line, end_line, source_code, docstring, signature,
                 decorators, complexity, dependencies, content_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                chunk.chunk_id, chunk.file_path, chunk.chunk_type,
                chunk.symbol_name, chunk.parent_symbol, chunk.start_line,
                chunk.end_line, chunk.source_code, chunk.docstring,
                chunk.signature, json.dumps(chunk.decorators),
                chunk.complexity, json.dumps(chunk.dependencies),
                chunk.content_hash,
            ))

        self.conn.commit()
        return chunks

    def _index_track_only_file(self, file_path: Path, repo_id: int):
        """Record file existence without creating chunks."""
        path_str = str(file_path)
        try:
            relative_path = str(file_path.relative_to(self.root))
        except ValueError:
            relative_path = file_path.name
        module_path = self._path_to_module(relative_path)
        language = _detect_language(path_str)
        size = file_path.stat().st_size if file_path.exists() else 0

        self.conn.execute("""
            INSERT OR REPLACE INTO files
            (file_path, repo_id, relative_path, module_path, content_hash, indexed_at,
             line_count, size_bytes, language, docstring, is_track_only)
            VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?, NULL, 1)
        """, (
            path_str, repo_id, relative_path, module_path, "__track_only__",
            datetime.now().isoformat(), size, language,
        ))

    def _remove_file_from_index(self, file_path: str):
        """Remove a file and all its chunks from the index."""
        self.conn.execute(
            "DELETE FROM annotations WHERE chunk_id IN (SELECT chunk_id FROM code_chunks WHERE file_path = ?)",
            (file_path,)
        )
        self.conn.execute("DELETE FROM code_chunks WHERE file_path = ?", (file_path,))
        self.conn.execute("DELETE FROM dependencies WHERE source_file = ?", (file_path,))
        self.conn.execute("DELETE FROM files WHERE file_path = ?", (file_path,))
        self.conn.commit()

    def _rebuild_dependencies(self, files: List[Path], repo_id: int):
        """Rebuild dependency edges for given files."""
        for file_path in files:
            path_str = str(file_path)
            self.conn.execute("DELETE FROM dependencies WHERE source_file = ?", (path_str,))

            # Get chunks with import deps
            cursor = self.conn.execute(
                "SELECT dependencies FROM code_chunks WHERE file_path = ? AND chunk_type = 'import_block'",
                (path_str,)
            )
            row = cursor.fetchone()
            if row and row[0]:
                deps = json.loads(row[0])
                for dep in deps:
                    # Try to resolve to a file in the repo
                    target = self._resolve_import(dep)
                    self.conn.execute(
                        "INSERT INTO dependencies (source_file, target_file, import_type, symbols_imported) VALUES (?, ?, 'import', '')",
                        (path_str, target or dep)
                    )

        self.conn.commit()

    def _resolve_import(self, module: str) -> Optional[str]:
        """Resolve an import to a file path in the database."""
        parts = module.split(".")
        # Try common patterns
        candidates = [
            "/".join(parts[:-1]) + f"/{parts[-1]}.py" if len(parts) > 1 else f"{parts[0]}.py",
            "/".join(parts) + "/__init__.py",
            "src/" + "/".join(parts[:-1]) + f"/{parts[-1]}.py" if len(parts) > 1 else f"src/{parts[0]}.py",
            "src/" + "/".join(parts) + "/__init__.py",
        ]

        for candidate in candidates:
            full_path = str(self.root / candidate)
            row = self.conn.execute(
                "SELECT file_path FROM files WHERE file_path = ?", (full_path,)
            ).fetchone()
            if row:
                return row[0]

        return None

    def _path_to_module(self, relative_path: str) -> str:
        """Convert file path to Python module path."""
        parts = Path(relative_path).parts
        parts_list = list(parts)

        # Remove extension
        if parts_list and parts_list[-1].endswith(".py"):
            parts_list[-1] = parts_list[-1][:-3]
        elif parts_list:
            # Keep non-Python extension in module path for uniqueness
            parts_list[-1] = parts_list[-1].replace(".", "_")

        # Remove __init__
        if parts_list and parts_list[-1] == "__init__":
            parts_list = parts_list[:-1]

        # Skip src/ if present
        if parts_list and parts_list[0] == "src":
            parts_list = parts_list[1:]

        return ".".join(parts_list)

    # ====================================================================
    # Layer 2: Semantic Annotations
    # ====================================================================

    def _generate_annotations(self, chunks: List[CodeChunk]) -> List[CodeAnnotation]:
        """Generate annotations for chunks based on configured provider."""
        if not chunks:
            return []

        annotations = []

        if self.llm_provider == "none" or self.llm_client is None:
            # AST-only mode
            for chunk in chunks:
                ann = generate_ast_annotation(chunk)
                annotations.append(ann)
        else:
            # LLM-enriched mode
            # Only annotate meaningful chunks (skip imports, standalone blocks)
            llm_chunks = [c for c in chunks if c.chunk_type in ("class", "function", "method", "module_docstring")]
            ast_chunks = [c for c in chunks if c.chunk_type not in ("class", "function", "method", "module_docstring")]

            # AST annotations for non-meaningful chunks
            for chunk in ast_chunks:
                annotations.append(generate_ast_annotation(chunk))

            # LLM annotations for meaningful chunks
            if llm_chunks:
                llm_annotations = generate_llm_annotations_batch(
                    llm_chunks, self.llm_client, batch_size=10
                )
                annotations.extend(llm_annotations)

        # Store annotations
        for ann in annotations:
            self.conn.execute("""
                INSERT OR REPLACE INTO annotations
                (chunk_id, summary, purpose, concepts, relationships,
                 search_text, annotation_source, generated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                ann.chunk_id, ann.summary, ann.purpose,
                json.dumps(ann.concepts), json.dumps(ann.relationships),
                ann.search_text, ann.annotation_source, ann.generated_at,
            ))

        self.conn.commit()
        return annotations

    def generate_summaries(self) -> Dict[str, str]:
        """Generate hierarchical summaries (file → module → architecture)."""
        summaries = {}

        # File-level summaries
        cursor = self.conn.execute("""
            SELECT f.file_path, f.relative_path, f.module_path,
                   GROUP_CONCAT(a.summary, ' | ')
            FROM files f
            JOIN code_chunks c ON c.file_path = f.file_path
            JOIN annotations a ON a.chunk_id = c.chunk_id
            WHERE f.is_track_only = 0
            GROUP BY f.file_path
        """)
        file_summaries = {}
        for row in cursor.fetchall():
            file_path, rel_path, module_path, chunk_summaries = row
            file_summary = f"File {rel_path}: {chunk_summaries}"
            file_summaries[module_path] = file_summary

            self.conn.execute("""
                INSERT OR REPLACE INTO summaries (level, scope, summary, generated_at)
                VALUES ('file', ?, ?, ?)
            """, (module_path, file_summary, datetime.now().isoformat()))

        # Module-level summaries
        modules = defaultdict(list)
        for module_path, summary in file_summaries.items():
            parts = module_path.split(".")
            if len(parts) >= 2:
                module_name = ".".join(parts[:2])
            else:
                module_name = parts[0]
            modules[module_name].append(summary)

        for module_name, file_sums in modules.items():
            module_summary = f"Module {module_name}: " + " | ".join(file_sums[:5])
            summaries[module_name] = module_summary

            self.conn.execute("""
                INSERT OR REPLACE INTO summaries (level, scope, summary, generated_at)
                VALUES ('module', ?, ?, ?)
            """, (module_name, module_summary, datetime.now().isoformat()))

        # Architecture summary
        arch_parts = [f"- {name}: {summary[:200]}" for name, summary in list(summaries.items())[:15]]
        arch_summary = "Architecture Overview:\n" + "\n".join(arch_parts)
        summaries["__architecture__"] = arch_summary

        self.conn.execute("""
            INSERT OR REPLACE INTO summaries (level, scope, summary, generated_at)
            VALUES ('architecture', '__all__', ?, ?)
        """, (arch_summary, datetime.now().isoformat()))

        self.conn.commit()
        return summaries

    # ====================================================================
    # Layer 3: Retrieval Index
    # ====================================================================

    def _get_embedding_engine(self):
        """Lazy-load the embedding engine."""
        if self._embedding_engine is None:
            try:
                from .embedding_engine import EmbeddingEngine
                self._embedding_engine = EmbeddingEngine()
            except ImportError:
                logger.warning("[CodeRetrieval] sentence-transformers not available, FAISS search disabled")
                return None
        return self._embedding_engine

    def _rebuild_faiss_index(self):
        """Rebuild the FAISS index from all annotations."""
        engine = self._get_embedding_engine()
        if engine is None:
            return

        try:
            import faiss
        except ImportError:
            logger.warning("[CodeRetrieval] FAISS not available, skipping index build")
            return

        # Get all search texts
        cursor = self.conn.execute(
            "SELECT chunk_id, search_text FROM annotations WHERE search_text IS NOT NULL AND search_text != ''"
        )
        rows = cursor.fetchall()

        if not rows:
            logger.info("[CodeRetrieval] No annotations to index")
            return

        chunk_ids = [row[0] for row in rows]
        search_texts = [row[1] for row in rows]

        # Generate embeddings
        logger.info(f"[CodeRetrieval] Generating embeddings for {len(search_texts)} chunks...")
        embeddings = engine.batch_embed(search_texts)

        # Normalize for inner product search
        faiss.normalize_L2(embeddings)

        # Build index
        dimension = embeddings.shape[1]
        if len(chunk_ids) > 50000:
            # Use IVF for large indices
            nlist = min(int(len(chunk_ids) ** 0.5), 256)
            quantizer = faiss.IndexFlatIP(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
            index.train(embeddings)
            index.add(embeddings)
            index.nprobe = min(nlist // 4, 16)
        else:
            index = faiss.IndexFlatIP(dimension)
            index.add(embeddings)

        self._faiss_index = index
        self._faiss_mapping = chunk_ids

        # Persist to disk
        faiss_path = self.workspace_dir / "code_chunks.faiss"
        mapping_path = self.workspace_dir / "code_chunks_mapping.pkl"
        faiss.write_index(index, str(faiss_path))
        with open(mapping_path, "wb") as f:
            pickle.dump(chunk_ids, f)

        logger.info(f"[CodeRetrieval] Built FAISS index: {len(chunk_ids)} vectors, dim={dimension}")

    def _load_faiss_index(self) -> bool:
        """Load FAISS index from disk."""
        try:
            import faiss
        except ImportError:
            return False

        faiss_path = self.workspace_dir / "code_chunks.faiss"
        mapping_path = self.workspace_dir / "code_chunks_mapping.pkl"

        if not faiss_path.exists() or not mapping_path.exists():
            return False

        try:
            self._faiss_index = faiss.read_index(str(faiss_path))
            with open(mapping_path, "rb") as f:
                self._faiss_mapping = pickle.load(f)
            logger.debug(f"[CodeRetrieval] Loaded FAISS index: {len(self._faiss_mapping)} vectors")
            return True
        except Exception as e:
            logger.warning(f"[CodeRetrieval] Failed to load FAISS index: {e}")
            return False

    def _rebuild_fts5_index(self):
        """Rebuild the FTS5 full-text search index by dropping and recreating."""
        try:
            # Drop and recreate FTS5 table (cleanest way to rebuild)
            self.conn.execute("DROP TABLE IF EXISTS code_chunks_fts")
            self.conn.execute("""
                CREATE VIRTUAL TABLE code_chunks_fts USING fts5(
                    chunk_id,
                    symbol_name,
                    module_path,
                    source_code,
                    summary,
                    concepts,
                    tokenize='porter unicode61'
                )
            """)

            # Populate from chunks + annotations
            cursor = self.conn.execute("""
                SELECT c.chunk_id, c.symbol_name, c.file_path, c.source_code,
                       COALESCE(a.summary, ''), COALESCE(a.concepts, '')
                FROM code_chunks c
                LEFT JOIN annotations a ON a.chunk_id = c.chunk_id
            """)

            for row in cursor.fetchall():
                chunk_id, symbol_name, file_path, source_code, summary, concepts = row
                # Get module_path from file
                module_row = self.conn.execute(
                    "SELECT module_path FROM files WHERE file_path = ?", (file_path,)
                ).fetchone()
                module_path = module_row[0] if module_row else ""

                # Truncate source for FTS (keep first 2000 chars)
                source_preview = (source_code or "")[:2000]

                self.conn.execute("""
                    INSERT INTO code_chunks_fts (chunk_id, symbol_name, module_path, source_code, summary, concepts)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (chunk_id, symbol_name, module_path, source_preview, summary, concepts))

            self.conn.commit()
            logger.debug("[CodeRetrieval] Rebuilt FTS5 index")

        except sqlite3.OperationalError as e:
            logger.warning(f"[CodeRetrieval] FTS5 rebuild failed: {e}")

    def _search_faiss(self, query: str, top_k: int = 20) -> List[Tuple[str, float]]:
        """Search using FAISS semantic similarity."""
        if self._faiss_index is None:
            if not self._load_faiss_index():
                return []

        engine = self._get_embedding_engine()
        if engine is None:
            return []

        try:
            import faiss
        except ImportError:
            return []

        query_emb = engine.embed(query).reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query_emb)

        k = min(top_k, self._faiss_index.ntotal)
        if k == 0:
            return []

        scores, indices = self._faiss_index.search(query_emb, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self._faiss_mapping):
                results.append((self._faiss_mapping[idx], float(score)))

        return results

    def _search_fts5(self, query: str, top_k: int = 20) -> List[Tuple[str, float]]:
        """Search using FTS5 keyword matching."""
        # Sanitize query for FTS5
        safe_query = re.sub(r'[^\w\s]', ' ', query)
        terms = safe_query.split()
        if not terms:
            return []

        # Build FTS5 query with OR
        fts_query = " OR ".join(terms)

        try:
            cursor = self.conn.execute("""
                SELECT chunk_id, rank
                FROM code_chunks_fts
                WHERE code_chunks_fts MATCH ?
                ORDER BY rank
                LIMIT ?
            """, (fts_query, top_k))

            results = []
            for row in cursor.fetchall():
                # Normalize rank to 0-1 score (FTS5 rank is negative, lower is better)
                score = 1.0 / (1.0 + abs(row[1]))
                results.append((row[0], score))

            return results

        except sqlite3.OperationalError as e:
            logger.debug(f"[CodeRetrieval] FTS5 search failed: {e}")
            return []

    def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        alpha: float = 0.6,
    ) -> List[SearchResult]:
        """
        Hybrid search combining FAISS semantic and FTS5 keyword search.

        Args:
            query: Search query (natural language or keywords)
            top_k: Number of results to return
            alpha: Weight for FAISS (0.0 = FTS5 only, 1.0 = FAISS only)

        Returns:
            List of SearchResult objects sorted by combined score
        """
        # Get results from both sources
        faiss_results = self._search_faiss(query, top_k=top_k * 2)
        fts5_results = self._search_fts5(query, top_k=top_k * 2)

        # Combine scores
        combined: Dict[str, float] = {}

        # Normalize FAISS scores
        if faiss_results:
            max_faiss = max(s for _, s in faiss_results) or 1.0
            for chunk_id, score in faiss_results:
                normalized = score / max_faiss
                combined[chunk_id] = combined.get(chunk_id, 0.0) + normalized * alpha

        # Normalize FTS5 scores
        if fts5_results:
            max_fts = max(s for _, s in fts5_results) or 1.0
            for chunk_id, score in fts5_results:
                normalized = score / max_fts
                combined[chunk_id] = combined.get(chunk_id, 0.0) + normalized * (1 - alpha)

        # Sort by combined score
        sorted_ids = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_k]

        # Fetch full chunk data
        results = []
        for chunk_id, score in sorted_ids:
            result = self._get_search_result(chunk_id, score, "hybrid")
            if result:
                results.append(result)

        return results

    def _get_search_result(self, chunk_id: str, score: float, match_type: str) -> Optional[SearchResult]:
        """Fetch full search result data from database."""
        cursor = self.conn.execute("""
            SELECT c.chunk_id, c.file_path, c.chunk_type, c.symbol_name,
                   c.start_line, c.end_line, c.source_code,
                   f.relative_path, f.module_path,
                   COALESCE(a.summary, '')
            FROM code_chunks c
            JOIN files f ON c.file_path = f.file_path
            LEFT JOIN annotations a ON a.chunk_id = c.chunk_id
            WHERE c.chunk_id = ?
        """, (chunk_id,))

        row = cursor.fetchone()
        if not row:
            return None

        return SearchResult(
            chunk_id=row[0],
            file_path=row[1],
            chunk_type=row[2],
            symbol_name=row[3],
            start_line=row[4],
            end_line=row[5],
            source_code=row[6],
            relative_path=row[7],
            module_path=row[8],
            summary=row[9],
            score=score,
            match_type=match_type,
        )

    # ====================================================================
    # Layer 4: Context Assembly
    # ====================================================================

    def query(
        self,
        query_text: str,
        top_k: int = 10,
        context_budget: int = DEFAULT_CONTEXT_BUDGET,
    ) -> Dict[str, Any]:
        """
        High-level query interface with automatic query classification and context assembly.

        Args:
            query_text: Natural language query
            top_k: Max number of results
            context_budget: Max tokens for context assembly

        Returns:
            Dict with results, context, and metadata
        """
        # Classify query
        query_type = self._classify_query(query_text)
        logger.debug(f"[CodeRetrieval] Query type: {query_type} for '{query_text}'")

        if query_type == "structural":
            results = self._query_structural(query_text, top_k)
        elif query_type == "locational":
            results = self._query_locational(query_text, top_k)
        elif query_type == "impact":
            return self._query_impact(query_text)
        elif query_type == "summary":
            return self._query_summary(query_text)
        else:
            # Default: semantic query with hybrid search
            results = self._query_semantic(query_text, top_k)

        # Assemble context within budget
        context = self._assemble_context(results, query_text, context_budget)

        return {
            "query": query_text,
            "query_type": query_type,
            "results": [asdict(r) for r in results],
            "context": context,
            "result_count": len(results),
        }

    def _classify_query(self, query: str) -> str:
        """Classify query type using rule-based heuristics.

        Order matters: impact and structural checks run before locational
        because queries like 'what breaks if I change base.py' should be
        classified as impact, not locational.
        """
        q = query.lower()

        # Impact queries (check FIRST — may contain file paths)
        impact_patterns = [
            "what breaks", "what depends", "impact", "who uses",
            "what imports", "what references", "affected by",
        ]
        if any(p in q for p in impact_patterns):
            return "impact"

        # Structural queries
        structural_patterns = [
            "find all", "list all", "which classes", "which functions",
            "show all", "what classes", "what functions",
            "that inherit", "that extend", "subclass",
        ]
        if any(p in q for p in structural_patterns):
            return "structural"

        # Summary queries
        summary_patterns = [
            "summarize", "overview", "architecture", "describe",
            "what does .* do", "explain",
        ]
        if any(re.search(p, q) for p in summary_patterns):
            return "summary"

        # Locational queries (contain file paths or specific symbol names)
        if re.search(r'[\\/][\w.]+\.\w+', query) or re.search(r'\b\w+\.py\b', q):
            return "locational"
        if re.search(r'line\s+\d+', q):
            return "locational"

        # Default: semantic
        return "semantic"

    def _query_structural(self, query: str, top_k: int) -> List[SearchResult]:
        """Handle structural queries using FTS5 and SQL filters."""
        # Determine what type of symbol to search for
        q = query.lower()
        chunk_type_filter = None
        if "class" in q:
            chunk_type_filter = "class"
        elif "function" in q or "method" in q:
            chunk_type_filter = ("function", "method")

        # Check for inheritance patterns
        inherit_match = re.search(r'(?:inherit|extend|subclass)\w*\s+(?:from\s+)?(\w+)', q)
        if inherit_match:
            base_class = inherit_match.group(1)
            return self._find_subclasses(base_class, top_k)

        # Use FTS5-heavy hybrid search
        results = self.hybrid_search(query, top_k=top_k * 2, alpha=0.2)

        # Filter by chunk type if detected
        if chunk_type_filter:
            if isinstance(chunk_type_filter, tuple):
                results = [r for r in results if r.chunk_type in chunk_type_filter]
            else:
                results = [r for r in results if r.chunk_type == chunk_type_filter]

        return results[:top_k]

    def _find_subclasses(self, base_class: str, top_k: int) -> List[SearchResult]:
        """Find all classes that inherit from a given base class."""
        # Search source code for inheritance patterns
        pattern = f"class %({base_class}%"
        cursor = self.conn.execute("""
            SELECT c.chunk_id, c.file_path, c.chunk_type, c.symbol_name,
                   c.start_line, c.end_line, c.source_code,
                   f.relative_path, f.module_path,
                   COALESCE(a.summary, '')
            FROM code_chunks c
            JOIN files f ON c.file_path = f.file_path
            LEFT JOIN annotations a ON a.chunk_id = c.chunk_id
            WHERE c.chunk_type = 'class'
              AND c.source_code LIKE ?
            LIMIT ?
        """, (f"%({base_class}%", top_k))

        results = []
        for row in cursor.fetchall():
            # Verify it actually inherits (not just mentions)
            if re.search(rf'class\s+\w+\([^)]*{re.escape(base_class)}', row[6]):
                results.append(SearchResult(
                    chunk_id=row[0], file_path=row[1], chunk_type=row[2],
                    symbol_name=row[3], start_line=row[4], end_line=row[5],
                    source_code=row[6], relative_path=row[7], module_path=row[8],
                    summary=row[9], score=1.0, match_type="structural",
                ))

        return results

    def _query_locational(self, query: str, top_k: int) -> List[SearchResult]:
        """Handle queries about specific files or locations."""
        # Extract file path from query
        file_match = re.search(r'([\w/\\]+\.(?:py|js|ts|tsx|md|yaml|json))', query)
        if file_match:
            file_pattern = file_match.group(1)
            cursor = self.conn.execute("""
                SELECT c.chunk_id, c.file_path, c.chunk_type, c.symbol_name,
                       c.start_line, c.end_line, c.source_code,
                       f.relative_path, f.module_path,
                       COALESCE(a.summary, '')
                FROM code_chunks c
                JOIN files f ON c.file_path = f.file_path
                LEFT JOIN annotations a ON a.chunk_id = c.chunk_id
                WHERE f.relative_path LIKE ?
                ORDER BY c.start_line
                LIMIT ?
            """, (f"%{file_pattern}%", top_k))

            results = []
            for row in cursor.fetchall():
                results.append(SearchResult(
                    chunk_id=row[0], file_path=row[1], chunk_type=row[2],
                    symbol_name=row[3], start_line=row[4], end_line=row[5],
                    source_code=row[6], relative_path=row[7], module_path=row[8],
                    summary=row[9], score=1.0, match_type="locational",
                ))
            return results

        # Extract line number
        line_match = re.search(r'line\s+(\d+)', query)
        if line_match and file_match:
            line_num = int(line_match.group(1))
            # Find chunk containing that line
            results = [r for r in results if r.start_line <= line_num <= r.end_line]
            return results[:top_k]

        # Fall back to hybrid search
        return self.hybrid_search(query, top_k=top_k, alpha=0.5)

    def _query_impact(self, query: str) -> Dict[str, Any]:
        """Handle impact analysis queries."""
        # Extract file/symbol from query
        file_match = re.search(r'([\w/\\]+\.py)', query)
        symbol_match = re.search(r'(?:function|class|method)?\s*["\']?(\w+)["\']?', query)

        file_path = file_match.group(1) if file_match else None
        symbol_name = symbol_match.group(1) if symbol_match else None

        if file_path:
            impact = self.impact_analysis(file_path, symbol_name)
            return {
                "query": query,
                "query_type": "impact",
                "impact": asdict(impact),
                "context": self._format_impact_context(impact),
                "result_count": len(impact.direct_dependents) + len(impact.transitive_impact),
            }

        # Fall back to hybrid search
        results = self.hybrid_search(query, top_k=10, alpha=0.5)
        return {
            "query": query,
            "query_type": "impact",
            "results": [asdict(r) for r in results],
            "context": self._assemble_context(results, query, DEFAULT_CONTEXT_BUDGET),
            "result_count": len(results),
        }

    def _query_semantic(self, query: str, top_k: int) -> List[SearchResult]:
        """Handle semantic queries with FAISS-heavy hybrid search."""
        return self.hybrid_search(query, top_k=top_k, alpha=0.8)

    def _query_summary(self, query: str) -> Dict[str, Any]:
        """Handle summary/overview queries."""
        q = query.lower()

        # Check for architecture-level summary
        if "architecture" in q or "overview" in q or "codebase" in q:
            cursor = self.conn.execute(
                "SELECT summary FROM summaries WHERE level = 'architecture' ORDER BY generated_at DESC LIMIT 1"
            )
            row = cursor.fetchone()
            if row:
                return {
                    "query": query,
                    "query_type": "summary",
                    "context": row[0],
                    "result_count": 1,
                }

        # Check for module-level summary
        module_match = re.search(r'(?:module|package)\s+(\w+(?:\.\w+)*)', q)
        if module_match:
            module_name = module_match.group(1)
            cursor = self.conn.execute(
                "SELECT summary FROM summaries WHERE level = 'module' AND scope LIKE ? ORDER BY generated_at DESC LIMIT 1",
                (f"%{module_name}%",)
            )
            row = cursor.fetchone()
            if row:
                return {
                    "query": query,
                    "query_type": "summary",
                    "context": row[0],
                    "result_count": 1,
                }

        # Fall back to hybrid search
        results = self.hybrid_search(query, top_k=10, alpha=0.5)
        return {
            "query": query,
            "query_type": "summary",
            "results": [asdict(r) for r in results],
            "context": self._assemble_context(results, query, DEFAULT_CONTEXT_BUDGET),
            "result_count": len(results),
        }

    def _assemble_context(
        self,
        results: List[SearchResult],
        query: str,
        budget: int,
    ) -> str:
        """Assemble context within token budget."""
        lines = [f"## Search Results for: {query}", ""]
        token_count = 0

        for i, result in enumerate(results):
            # Build result block
            block_lines = [
                f"### Result {i + 1}: {result.symbol_name} ({result.chunk_type})",
                f"**File:** {result.relative_path}:{result.start_line}-{result.end_line}",
                f"**Module:** {result.module_path}",
            ]
            if result.summary:
                block_lines.append(f"**Summary:** {result.summary}")
            block_lines.append(f"```\n{result.source_code[:3000]}\n```")
            block_lines.append("")

            block_text = "\n".join(block_lines)
            block_tokens = len(block_text) // CHARS_PER_TOKEN

            if token_count + block_tokens > budget:
                # Budget exhausted, add truncation notice
                lines.append(f"*({len(results) - i} more results omitted due to context budget)*")
                break

            lines.extend(block_lines)
            token_count += block_tokens

        return "\n".join(lines)

    def _format_impact_context(self, impact: ImpactResult) -> str:
        """Format impact analysis result as context string."""
        lines = [
            f"## Impact Analysis: {impact.target_symbol} in {impact.target_file}",
            f"**Risk Level:** {impact.risk_level}",
            "",
        ]

        if impact.direct_dependents:
            lines.append("### Direct Dependents")
            for dep in impact.direct_dependents:
                lines.append(f"- {dep.get('file', 'unknown')} (imports {dep.get('symbols', '')})")
            lines.append("")

        if impact.symbol_references:
            lines.append("### Symbol References")
            for ref in impact.symbol_references:
                lines.append(f"- {ref.get('file', 'unknown')}:{ref.get('line', '?')} - {ref.get('context', '')}")
            lines.append("")

        if impact.transitive_impact:
            lines.append("### Transitive Impact")
            for path in impact.transitive_impact:
                lines.append(f"- {path}")
            lines.append("")

        return "\n".join(lines)

    # ====================================================================
    # Impact Analysis
    # ====================================================================

    def impact_analysis(
        self,
        file_path: str,
        symbol_name: Optional[str] = None,
        max_depth: int = 3,
    ) -> ImpactResult:
        """
        Analyze the impact of changing a file or symbol.

        Args:
            file_path: Path to the file (can be relative)
            symbol_name: Specific symbol to analyze (optional)
            max_depth: Maximum transitive dependency depth

        Returns:
            ImpactResult with dependents and risk assessment
        """
        # Resolve file path
        resolved_path = self._resolve_file_path(file_path)

        # Get direct dependents (files that import from this file)
        cursor = self.conn.execute(
            "SELECT source_file, symbols_imported FROM dependencies WHERE target_file = ? OR target_file LIKE ?",
            (resolved_path or file_path, f"%{file_path}%")
        )
        direct_deps = []
        for row in cursor.fetchall():
            direct_deps.append({
                "file": row[0],
                "symbols": row[1] or "",
            })

        # Find symbol references if a specific symbol is given
        symbol_refs = []
        if symbol_name:
            cursor = self.conn.execute("""
                SELECT c.file_path, c.start_line, c.source_code
                FROM code_chunks c
                WHERE c.source_code LIKE ?
                  AND c.file_path != ?
                LIMIT 50
            """, (f"%{symbol_name}%", resolved_path or file_path))

            for row in cursor.fetchall():
                # Verify it's an actual reference (not just a substring match)
                if re.search(rf'\b{re.escape(symbol_name)}\b', row[2]):
                    context_line = ""
                    for line in row[2].splitlines():
                        if symbol_name in line:
                            context_line = line.strip()[:100]
                            break
                    symbol_refs.append({
                        "file": row[0],
                        "line": row[1],
                        "context": context_line,
                    })

        # BFS for transitive dependents
        transitive = []
        visited = {resolved_path or file_path}
        queue = deque([(dep["file"], 1) for dep in direct_deps])

        while queue:
            dep_file, depth = queue.popleft()
            if dep_file in visited or depth > max_depth:
                continue
            visited.add(dep_file)
            transitive.append(dep_file)

            # Get dependents of this dependent
            cursor = self.conn.execute(
                "SELECT source_file FROM dependencies WHERE target_file = ?",
                (dep_file,)
            )
            for row in cursor.fetchall():
                if row[0] not in visited:
                    queue.append((row[0], depth + 1))

        # Assess risk
        total_impact = len(direct_deps) + len(transitive)
        if total_impact > 20:
            risk = "HIGH"
        elif total_impact > 5:
            risk = "MEDIUM"
        else:
            risk = "LOW"

        return ImpactResult(
            target_file=file_path,
            target_symbol=symbol_name or "__all__",
            direct_dependents=direct_deps,
            symbol_references=symbol_refs,
            transitive_impact=transitive,
            risk_level=risk,
        )

    def _resolve_file_path(self, file_path: str) -> Optional[str]:
        """Resolve a potentially relative file path to an absolute one in the index."""
        # Try direct match
        row = self.conn.execute(
            "SELECT file_path FROM files WHERE file_path = ?", (file_path,)
        ).fetchone()
        if row:
            return row[0]

        # Try relative path match
        row = self.conn.execute(
            "SELECT file_path FROM files WHERE relative_path LIKE ?", (f"%{file_path}%",)
        ).fetchone()
        if row:
            return row[0]

        return None

    # ====================================================================
    # Structural Search (for search_code_structure tool)
    # ====================================================================

    def search_structure(
        self,
        symbol_pattern: Optional[str] = None,
        chunk_type: Optional[str] = None,
        module_pattern: Optional[str] = None,
        top_k: int = 50,
    ) -> List[SearchResult]:
        """
        Structural search using LIKE/GLOB patterns.

        Args:
            symbol_pattern: Pattern to match symbol names (LIKE syntax)
            chunk_type: Filter by chunk type (class, function, method, etc.)
            module_pattern: Pattern to match module paths (LIKE syntax)
            top_k: Max results

        Returns:
            List of SearchResult
        """
        conditions = []
        params = []

        if symbol_pattern:
            conditions.append("c.symbol_name LIKE ?")
            params.append(symbol_pattern)

        if chunk_type:
            conditions.append("c.chunk_type = ?")
            params.append(chunk_type)

        if module_pattern:
            conditions.append("f.module_path LIKE ?")
            params.append(module_pattern)

        if not conditions:
            return []

        where_clause = " AND ".join(conditions)
        cursor = self.conn.execute(f"""
            SELECT c.chunk_id, c.file_path, c.chunk_type, c.symbol_name,
                   c.start_line, c.end_line, c.source_code,
                   f.relative_path, f.module_path,
                   COALESCE(a.summary, '')
            FROM code_chunks c
            JOIN files f ON c.file_path = f.file_path
            LEFT JOIN annotations a ON a.chunk_id = c.chunk_id
            WHERE {where_clause}
            ORDER BY c.symbol_name
            LIMIT ?
        """, params + [top_k])

        results = []
        for row in cursor.fetchall():
            results.append(SearchResult(
                chunk_id=row[0], file_path=row[1], chunk_type=row[2],
                symbol_name=row[3], start_line=row[4], end_line=row[5],
                source_code=row[6], relative_path=row[7], module_path=row[8],
                summary=row[9], score=1.0, match_type="structural",
            ))

        return results

    # ====================================================================
    # File Watcher (Auto-Update)
    # ====================================================================

    def _start_watcher(self):
        """Start background file watcher thread."""
        self._watcher_thread = threading.Thread(
            target=self._watcher_loop,
            daemon=True,
            name="CodeRetrievalWatcher",
        )
        self._watcher_thread.start()
        logger.debug("[CodeRetrieval] File watcher started")

    def _watcher_loop(self):
        """Background loop that watches for file changes."""
        # Build initial file state
        file_mtimes: Dict[str, float] = {}
        for row in self.conn.execute("SELECT file_path, indexed_at FROM files").fetchall():
            try:
                file_mtimes[row[0]] = os.path.getmtime(row[0])
            except OSError:
                pass

        while not self._watcher_stop.is_set():
            try:
                changes_found = False

                # Check existing files for modifications
                for file_path, last_mtime in list(file_mtimes.items()):
                    try:
                        current_mtime = os.path.getmtime(file_path)
                        if current_mtime > last_mtime:
                            with self._pending_lock:
                                self._pending_changes[file_path] = "modified"
                            file_mtimes[file_path] = current_mtime
                            changes_found = True
                    except OSError:
                        # File deleted
                        with self._pending_lock:
                            self._pending_changes[file_path] = "deleted"
                        del file_mtimes[file_path]
                        changes_found = True

                # Check for new files (every 5th iteration to save resources)
                # This is handled during periodic scans, not on every tick

                # Process pending changes after debounce
                if changes_found:
                    time.sleep(WATCHER_DEBOUNCE_MS / 1000.0)
                    self._process_pending_changes()

            except Exception as e:
                logger.warning(f"[CodeRetrieval] Watcher error: {e}")

            # Sleep between checks
            self._watcher_stop.wait(timeout=2.0)

    def _process_pending_changes(self):
        """Process accumulated file changes."""
        with self._pending_lock:
            if not self._pending_changes:
                return
            changes = dict(self._pending_changes)
            self._pending_changes.clear()

        logger.info(f"[CodeRetrieval] Processing {len(changes)} file changes")

        repo_row = self.conn.execute(
            "SELECT id FROM repositories WHERE root_path = ?",
            (str(self.root),)
        ).fetchone()
        if not repo_row:
            return
        repo_id = repo_row[0]

        new_chunks = []
        for file_path, change_type in changes.items():
            if change_type == "deleted":
                self._remove_file_from_index(file_path)
            elif change_type in ("modified", "new"):
                chunks = self._index_single_file(Path(file_path), repo_id)
                new_chunks.extend(chunks)

        # Generate annotations for new chunks
        if new_chunks:
            self._generate_annotations(new_chunks)
            self._rebuild_fts5_index()
            self._rebuild_faiss_index()

    def stop_watcher(self):
        """Stop the background file watcher."""
        if self._watcher_thread and self._watcher_thread.is_alive():
            self._watcher_stop.set()
            self._watcher_thread.join(timeout=5.0)
            logger.debug("[CodeRetrieval] File watcher stopped")

    # ====================================================================
    # Utility
    # ====================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get current pipeline statistics."""
        total_files = self.conn.execute("SELECT COUNT(*) FROM files WHERE is_track_only = 0").fetchone()[0]
        total_track = self.conn.execute("SELECT COUNT(*) FROM files WHERE is_track_only = 1").fetchone()[0]
        total_chunks = self.conn.execute("SELECT COUNT(*) FROM code_chunks").fetchone()[0]
        total_annotations = self.conn.execute("SELECT COUNT(*) FROM annotations").fetchone()[0]
        total_deps = self.conn.execute("SELECT COUNT(*) FROM dependencies").fetchone()[0]

        faiss_count = self._faiss_index.ntotal if self._faiss_index else 0

        return {
            "root_path": str(self.root),
            "indexed_files": total_files,
            "tracked_files": total_track,
            "total_chunks": total_chunks,
            "total_annotations": total_annotations,
            "total_dependencies": total_deps,
            "faiss_vectors": faiss_count,
            "llm_provider": self.llm_provider,
            "watcher_active": self._watcher_thread is not None and self._watcher_thread.is_alive(),
        }

    def close(self):
        """Clean up resources."""
        self.stop_watcher()
        if self.conn:
            self.conn.close()

    def __del__(self):
        """Cleanup on garbage collection."""
        try:
            self.close()
        except Exception:
            pass
