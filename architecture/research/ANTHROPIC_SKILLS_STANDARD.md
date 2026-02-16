# Anthropic SKILLs Standard Integration for GAIA

**Date**: February 7, 2026
**Version**: 1.0
**Purpose**: Support Anthropic's SKILLs markdown standard alongside GAIA's native SKILLS system
**Integration**: Extends DYNAMIC_TOOLS_FRAMEWORK.md with markdown-based skill support

---

## Table of Contents

1. [Overview](#overview)
2. [Anthropic SKILLs Standard](#anthropic-skills-standard)
3. [Comparison: Anthropic vs GAIA SKILLS](#comparison-anthropic-vs-gaia-skills)
4. [Integration Architecture](#integration-architecture)
5. [Implementation Specification](#implementation-specification)
6. [Skill Discovery and Loading](#skill-discovery-and-loading)
7. [Skill Execution](#skill-execution)
8. [Bidirectional Sync](#bidirectional-sync)
9. [Migration Path](#migration-path)
10. [Complete Code](#complete-code)

---

## Overview

### The Problem

**Two skill systems, one goal**:

1. **Anthropic SKILLs Standard** (`.md` files in `.claude/skills/`)
   - Human-readable markdown format
   - Designed for Claude Code and Claude Desktop
   - Manual creation and curation
   - Version-controlled in git
   - Great for team-shared patterns

2. **GAIA Native SKILLS** (FAISS vector store from `DYNAMIC_TOOLS_FRAMEWORK.md`)
   - Programmatically created by `ToolBuilderAgent`
   - Stored in FAISS for semantic search
   - Auto-invoked based on context similarity
   - Great for agent-learned patterns

**Current state**: GAIA has its own SKILLS system but doesn't support Anthropic's standard.

**Desired state**: Support **both** standards seamlessly:
- Read Anthropic `.md` skills and make them available to GAIA agents
- Export GAIA-learned skills to Anthropic `.md` format for human review
- Bidirectional sync between markdown and vector store

---

## Anthropic SKILLs Standard

### File Structure

Anthropic SKILLs are markdown files stored in:
```
.claude/
├── skills/
│   ├── python-testing.md
│   ├── api-setup.md
│   ├── docker-compose.md
│   └── error-handling.md
└── SKILLS.md (index/manifest)
```

### Standard Format

Each skill is a markdown file with this structure:

```markdown
# Skill Name

**Category**: Development | Testing | DevOps | Database | etc.
**Complexity**: Simple | Medium | Complex
**Prerequisites**: List of required tools, knowledge, or other skills
**Triggers**: Keywords that suggest this skill (for semantic matching)

## Description

Brief 1-2 sentence description of what this skill does.

## When to Use

- Scenario 1 where this skill applies
- Scenario 2 where this skill applies
- When NOT to use this skill

## Steps

1. First step with explanation
   ```language
   code example
   ```

2. Second step
   ```language
   code example
   ```

3. Continue...

## Validation

How to verify the skill was applied successfully:
- Check 1
- Check 2

## Common Pitfalls

- Pitfall 1 and how to avoid it
- Pitfall 2 and how to avoid it

## Related Skills

- Link to skill-name-1.md
- Link to skill-name-2.md

## Examples

### Example 1: Specific use case
\```language
Full working example
\```

### Example 2: Another use case
\```language
Another example
\```

## Metadata

- **Created**: 2026-02-07
- **Last Updated**: 2026-02-07
- **Version**: 1.0
- **Author**: Human | Agent | Both
- **Success Rate**: 95% (if tracked)
- **Avg Duration**: 5 minutes (if tracked)
```

### Example: Real Anthropic Skill

**File**: `.claude/skills/python-testing.md`

```markdown
# Python Testing with pytest

**Category**: Testing
**Complexity**: Medium
**Prerequisites**: Python project, pytest installed
**Triggers**: test, testing, pytest, unit test, integration test

## Description

Set up comprehensive pytest testing for a Python project with fixtures, parametrization, and coverage reporting.

## When to Use

- Starting a new Python project that needs testing
- Adding tests to an existing project
- Need structured test organization (unit, integration, e2e)
- When NOT to use: For simple scripts that don't need formal testing

## Steps

1. Install pytest with common plugins
   \```bash
   pip install pytest pytest-cov pytest-mock pytest-asyncio
   \```

2. Create test directory structure
   \```
   tests/
   ├── __init__.py
   ├── unit/
   │   ├── __init__.py
   │   └── test_module.py
   ├── integration/
   │   ├── __init__.py
   │   └── test_api.py
   └── conftest.py  # Shared fixtures
   \```

3. Create conftest.py with common fixtures
   \```python
   import pytest

   @pytest.fixture
   def sample_data():
       return {"key": "value"}

   @pytest.fixture
   def mock_client(mocker):
       return mocker.patch("module.Client")
   \```

4. Write tests using fixtures and parametrization
   \```python
   import pytest

   @pytest.mark.parametrize("input,expected", [
       (1, 2),
       (2, 4),
       (3, 6),
   ])
   def test_double(input, expected):
       assert double(input) == expected
   \```

5. Run tests with coverage
   \```bash
   pytest tests/ -v --cov=src --cov-report=html
   \```

## Validation

- All tests pass: `pytest tests/ -v`
- Coverage report generated: `open htmlcov/index.html`
- No warnings in test output

## Common Pitfalls

- Forgetting `__init__.py` in test directories → tests not discovered
- Not using fixtures for setup/teardown → tests not isolated
- Hard-coding values instead of parametrization → less coverage

## Related Skills

- python-project-setup.md
- ci-cd-github-actions.md

## Examples

### Example 1: Testing a REST API client
\```python
def test_api_client_get(mock_client):
    client = APIClient()
    response = client.get("/users/1")
    assert response.status_code == 200
    assert "name" in response.json()
\```

### Example 2: Async test
\```python
@pytest.mark.asyncio
async def test_async_function():
    result = await fetch_data()
    assert result is not None
\```

## Metadata

- **Created**: 2026-01-15
- **Last Updated**: 2026-02-01
- **Version**: 1.2
- **Author**: Human
- **Success Rate**: 98%
- **Avg Duration**: 10 minutes
```

---

## Comparison: Anthropic vs GAIA SKILLS

| Aspect | Anthropic SKILLs (.md) | GAIA SKILLS (FAISS) |
|--------|------------------------|---------------------|
| **Format** | Markdown files | Python code + metadata in vector DB |
| **Creation** | Manual (human-written) | Automatic (agent-generated via ToolBuilderAgent) |
| **Storage** | `.claude/skills/*.md` | FAISS vector store + SQLite metadata |
| **Discoverability** | File listing + keyword search | Semantic similarity search |
| **Version Control** | Git-friendly (text files) | Not directly git-trackable |
| **Human Readability** | Very high (markdown) | Low (binary vector embeddings) |
| **Semantic Search** | Keyword-based | Vector similarity (better for fuzzy matching) |
| **Team Sharing** | Easy (commit .md files) | Hard (need to export) |
| **Agent Learning** | Manual curation | Automatic pattern detection |
| **Invocation** | Manual reference or keyword match | Automatic via semantic similarity |
| **Validation** | Manual checklist | Programmatic success tracking |

### Complementary Strengths

**Use Anthropic SKILLs for**:
- Team-curated best practices
- Company-specific patterns
- Complex multi-step workflows
- Knowledge that needs human review
- Skills that require detailed explanations

**Use GAIA SKILLS for**:
- Automatically learned patterns
- One-off optimizations
- Rapidly evolving workflows
- Statistical validation (success rate tracking)
- Tight LLM integration

**Best of both worlds**: Support both systems, sync automatically.

---

## Integration Architecture

### High-Level Design

```
┌──────────────────────────────────────────────────────────────┐
│                    GAIA Agent Request                         │
│  "Set up Python testing for this project"                    │
└────────────────────┬─────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────────┐
│              Unified Skill Router                             │
│  - Searches both Anthropic .md and GAIA FAISS                │
│  - Ranks by relevance (semantic similarity)                  │
│  - Returns best match from either system                     │
└────────────────────┬─────────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
        ▼                         ▼
┌───────────────────┐     ┌──────────────────┐
│ Anthropic         │     │ GAIA SKILLS      │
│ Skills Loader     │     │ (FAISS)          │
│                   │     │                  │
│ - Parse .md files │     │ - Vector search  │
│ - Extract metadata│     │ - Rank by score  │
│ - Keyword match   │     │ - Return code    │
└────────┬──────────┘     └────────┬─────────┘
         │                         │
         └──────────┬──────────────┘
                    │
                    ▼
┌──────────────────────────────────────────────────────────────┐
│              Skill Execution Engine                           │
│  - Applies selected skill to current context                 │
│  - Tracks success/failure                                    │
│  - Updates statistics                                        │
└────────────────────┬─────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────────┐
│              Bidirectional Sync (Optional)                    │
│  - Export high-success GAIA skills → .md files               │
│  - Import .md skills → FAISS for semantic search             │
└──────────────────────────────────────────────────────────────┘
```

### Directory Structure

```
.claude/
├── skills/              # Anthropic standard location
│   ├── python-testing.md
│   ├── api-setup.md
│   ├── docker-compose.md
│   ├── fastapi-crud.md
│   └── react-component-setup.md
└── SKILLS.md            # Index/manifest (optional)

src/gaia/skills/
├── loader.py            # Loads Anthropic .md skills
├── parser.py            # Parses markdown structure
├── router.py            # Unified skill router (searches both systems)
├── exporter.py          # Exports GAIA skills to .md format
├── store.py             # GAIA FAISS-based skill store (existing)
└── models.py            # Unified skill data model
```

---

## Implementation Specification

### Core Components

#### 1. Skill Data Model (Unified)

```python
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime
from enum import Enum

class SkillSource(Enum):
    """Where the skill came from."""
    ANTHROPIC_MD = "anthropic_md"      # Loaded from .md file
    GAIA_LEARNED = "gaia_learned"      # Generated by ToolBuilderAgent
    HYBRID = "hybrid"                  # Started as learned, exported to .md


class SkillComplexity(Enum):
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"


@dataclass
class Skill:
    """Unified skill representation for both Anthropic and GAIA skills."""

    # Core identity
    name: str
    description: str
    category: str

    # Source tracking
    source: SkillSource
    file_path: Optional[str] = None  # For Anthropic .md skills

    # Metadata
    complexity: SkillComplexity = SkillComplexity.MEDIUM
    prerequisites: List[str] = field(default_factory=list)
    triggers: List[str] = field(default_factory=list)  # Keywords for matching

    # Content
    when_to_use: List[str] = field(default_factory=list)
    steps: List[str] = field(default_factory=list)
    validation: List[str] = field(default_factory=list)
    common_pitfalls: List[str] = field(default_factory=list)
    examples: List[Dict[str, str]] = field(default_factory=list)
    related_skills: List[str] = field(default_factory=list)

    # For GAIA skills (code-based)
    code: Optional[str] = None

    # Statistics (tracked for both types)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    version: str = "1.0"
    author: str = "Unknown"
    invocation_count: int = 0
    success_count: int = 0
    avg_duration_seconds: float = 0.0

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.invocation_count == 0:
            return 0.0
        return (self.success_count / self.invocation_count) * 100

    def to_markdown(self) -> str:
        """Export skill to Anthropic markdown format."""
        md_parts = [
            f"# {self.name}",
            "",
            f"**Category**: {self.category}",
            f"**Complexity**: {self.complexity.value.capitalize()}",
            f"**Prerequisites**: {', '.join(self.prerequisites) if self.prerequisites else 'None'}",
            f"**Triggers**: {', '.join(self.triggers)}",
            "",
            "## Description",
            "",
            self.description,
            "",
        ]

        if self.when_to_use:
            md_parts.extend([
                "## When to Use",
                "",
                *[f"- {item}" for item in self.when_to_use],
                "",
            ])

        if self.steps:
            md_parts.extend([
                "## Steps",
                "",
                *[f"{i+1}. {step}" for i, step in enumerate(self.steps)],
                "",
            ])

        if self.code:
            md_parts.extend([
                "## Code",
                "",
                "```python",
                self.code,
                "```",
                "",
            ])

        if self.validation:
            md_parts.extend([
                "## Validation",
                "",
                *[f"- {item}" for item in self.validation],
                "",
            ])

        if self.common_pitfalls:
            md_parts.extend([
                "## Common Pitfalls",
                "",
                *[f"- {item}" for item in self.common_pitfalls],
                "",
            ])

        if self.related_skills:
            md_parts.extend([
                "## Related Skills",
                "",
                *[f"- {skill}" for skill in self.related_skills],
                "",
            ])

        if self.examples:
            md_parts.extend([
                "## Examples",
                "",
            ])
            for i, example in enumerate(self.examples, 1):
                md_parts.extend([
                    f"### Example {i}: {example.get('title', 'Use case')}",
                    "```" + example.get('language', 'python'),
                    example.get('code', ''),
                    "```",
                    "",
                ])

        md_parts.extend([
            "## Metadata",
            "",
            f"- **Created**: {self.created_at.strftime('%Y-%m-%d')}",
            f"- **Last Updated**: {self.updated_at.strftime('%Y-%m-%d')}",
            f"- **Version**: {self.version}",
            f"- **Author**: {self.author}",
            f"- **Success Rate**: {self.success_rate:.1f}%",
            f"- **Avg Duration**: {self.avg_duration_seconds:.1f} seconds",
        ])

        return "\n".join(md_parts)
```

#### 2. Markdown Parser

```python
import re
from pathlib import Path
from typing import Dict, List, Optional

class SkillMarkdownParser:
    """Parse Anthropic skill markdown files into Skill objects."""

    def parse_file(self, file_path: Path) -> Skill:
        """Parse a .md file into a Skill object."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        return self.parse_content(content, file_path)

    def parse_content(self, content: str, file_path: Optional[Path] = None) -> Skill:
        """Parse markdown content into a Skill object."""

        # Extract title (first H1)
        title_match = re.search(r'^# (.+)$', content, re.MULTILINE)
        name = title_match.group(1) if title_match else "Untitled Skill"

        # Extract metadata fields
        metadata = self._extract_metadata_fields(content)

        # Extract sections
        sections = self._extract_sections(content)

        return Skill(
            name=name,
            description=sections.get('description', '').strip(),
            category=metadata.get('category', 'General'),
            source=SkillSource.ANTHROPIC_MD,
            file_path=str(file_path) if file_path else None,
            complexity=self._parse_complexity(metadata.get('complexity', 'medium')),
            prerequisites=self._parse_list(metadata.get('prerequisites', '')),
            triggers=self._parse_list(metadata.get('triggers', '')),
            when_to_use=self._parse_list_section(sections.get('when to use', '')),
            steps=self._parse_numbered_list(sections.get('steps', '')),
            validation=self._parse_list_section(sections.get('validation', '')),
            common_pitfalls=self._parse_list_section(sections.get('common pitfalls', '')),
            examples=self._extract_examples(sections.get('examples', '')),
            related_skills=self._parse_list_section(sections.get('related skills', '')),
            created_at=self._parse_date(metadata.get('created', '')),
            updated_at=self._parse_date(metadata.get('last updated', '')),
            version=metadata.get('version', '1.0'),
            author=metadata.get('author', 'Unknown'),
            success_count=int(metadata.get('success rate', '0').rstrip('%') or 0),
            invocation_count=100 if metadata.get('success rate') else 0,
        )

    def _extract_metadata_fields(self, content: str) -> Dict[str, str]:
        """Extract **Field**: value metadata from top of file."""
        metadata = {}
        pattern = r'\*\*(\w+(?:\s+\w+)*)\*\*:\s*(.+?)(?:\n|$)'

        for match in re.finditer(pattern, content):
            key = match.group(1).lower()
            value = match.group(2).strip()
            metadata[key] = value

        return metadata

    def _extract_sections(self, content: str) -> Dict[str, str]:
        """Extract markdown sections (## Header)."""
        sections = {}
        pattern = r'## (.+?)\n\n(.*?)(?=\n## |\Z)'

        for match in re.finditer(pattern, content, re.DOTALL):
            section_name = match.group(1).lower()
            section_content = match.group(2).strip()
            sections[section_name] = section_content

        return sections

    def _parse_complexity(self, value: str) -> SkillComplexity:
        """Parse complexity string to enum."""
        value_lower = value.lower()
        if 'simple' in value_lower:
            return SkillComplexity.SIMPLE
        elif 'complex' in value_lower:
            return SkillComplexity.COMPLEX
        else:
            return SkillComplexity.MEDIUM

    def _parse_list(self, value: str) -> List[str]:
        """Parse comma-separated list."""
        if not value or value.lower() == 'none':
            return []
        return [item.strip() for item in value.split(',')]

    def _parse_list_section(self, content: str) -> List[str]:
        """Parse markdown list (- item or * item)."""
        items = []
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith(('- ', '* ')):
                items.append(line[2:].strip())
        return items

    def _parse_numbered_list(self, content: str) -> List[str]:
        """Parse numbered list (1. item)."""
        items = []
        for line in content.split('\n'):
            line = line.strip()
            match = re.match(r'^\d+\.\s+(.+)$', line)
            if match:
                items.append(match.group(1))
        return items

    def _extract_examples(self, content: str) -> List[Dict[str, str]]:
        """Extract code examples from Examples section."""
        examples = []

        # Find ### Example N: Title blocks
        pattern = r'### Example \d+: (.+?)\n```(\w+)\n(.*?)```'

        for match in re.finditer(pattern, content, re.DOTALL):
            examples.append({
                'title': match.group(1).strip(),
                'language': match.group(2),
                'code': match.group(3).strip(),
            })

        return examples

    def _parse_date(self, value: str) -> datetime:
        """Parse date string."""
        if not value:
            return datetime.now()
        try:
            return datetime.strptime(value, '%Y-%m-%d')
        except:
            return datetime.now()
```

#### 3. Skill Loader

```python
from pathlib import Path
from typing import List, Dict

class AnthropicSkillLoader:
    """Load Anthropic skills from .claude/skills/ directory."""

    def __init__(self, skills_dir: Path = Path.home() / ".claude" / "skills"):
        self.skills_dir = skills_dir
        self.parser = SkillMarkdownParser()
        self._skills_cache: Dict[str, Skill] = {}

    def load_all(self) -> List[Skill]:
        """Load all .md skills from the skills directory."""
        if not self.skills_dir.exists():
            print(f"⚠️  Anthropic skills directory not found: {self.skills_dir}")
            print(f"   Create it with: mkdir -p {self.skills_dir}")
            return []

        skills = []
        for md_file in self.skills_dir.glob("*.md"):
            # Skip SKILLS.md manifest
            if md_file.name == "SKILLS.md":
                continue

            try:
                skill = self.parser.parse_file(md_file)
                skills.append(skill)
                self._skills_cache[skill.name] = skill
            except Exception as e:
                print(f"⚠️  Failed to parse {md_file.name}: {e}")

        return skills

    def get_skill(self, name: str) -> Optional[Skill]:
        """Get a specific skill by name."""
        if not self._skills_cache:
            self.load_all()
        return self._skills_cache.get(name)

    def reload(self):
        """Reload all skills from disk."""
        self._skills_cache.clear()
        return self.load_all()
```

#### 4. Unified Skill Router

```python
from typing import List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer

class UnifiedSkillRouter:
    """
    Routes skill queries to either Anthropic .md skills or GAIA FAISS skills.
    Returns the best match from both systems.
    """

    def __init__(self,
                 anthropic_loader: AnthropicSkillLoader,
                 gaia_skill_store: 'SkillStore',  # From DYNAMIC_TOOLS_FRAMEWORK
                 embedding_model: str = "all-MiniLM-L6-v2"):
        self.anthropic_loader = anthropic_loader
        self.gaia_store = gaia_skill_store
        self.encoder = SentenceTransformer(embedding_model)

        # Load Anthropic skills into memory
        self.anthropic_skills = anthropic_loader.load_all()

        # Create embeddings for Anthropic skills
        self._anthropic_embeddings = self._create_anthropic_embeddings()

    def _create_anthropic_embeddings(self) -> np.ndarray:
        """Create embeddings for Anthropic skills."""
        if not self.anthropic_skills:
            return np.array([])

        # Combine name, description, and triggers for better matching
        texts = []
        for skill in self.anthropic_skills:
            text = f"{skill.name}. {skill.description}. "
            text += " ".join(skill.triggers)
            texts.append(text)

        return self.encoder.encode(texts)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Skill, float, SkillSource]]:
        """
        Search both Anthropic and GAIA skills.

        Args:
            query: Natural language query
            top_k: Number of results to return

        Returns:
            List of (skill, similarity_score, source) tuples, sorted by score
        """
        results = []

        # Search Anthropic skills (semantic similarity)
        if len(self.anthropic_skills) > 0:
            query_embedding = self.encoder.encode([query])[0]

            # Calculate cosine similarity
            similarities = np.dot(self._anthropic_embeddings, query_embedding) / (
                np.linalg.norm(self._anthropic_embeddings, axis=1) *
                np.linalg.norm(query_embedding)
            )

            for i, score in enumerate(similarities):
                results.append((
                    self.anthropic_skills[i],
                    float(score),
                    SkillSource.ANTHROPIC_MD
                ))

        # Search GAIA skills (FAISS vector search)
        gaia_results = self.gaia_store.search_skills(query, k=top_k)
        for skill, score in gaia_results:
            results.append((skill, score, SkillSource.GAIA_LEARNED))

        # Sort by score (descending)
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:top_k]

    def get_best_match(self, query: str) -> Optional[Tuple[Skill, float, SkillSource]]:
        """Get the single best matching skill."""
        results = self.search(query, top_k=1)
        return results[0] if results else None
```

#### 5. Skill Exporter

```python
class SkillExporter:
    """Export GAIA skills to Anthropic markdown format."""

    def __init__(self, output_dir: Path = Path.home() / ".claude" / "skills"):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_skill(self, skill: Skill) -> Path:
        """Export a single skill to markdown."""
        # Generate filename from skill name
        filename = skill.name.lower().replace(' ', '-').replace('_', '-')
        filename = re.sub(r'[^a-z0-9-]', '', filename) + '.md'

        file_path = self.output_dir / filename

        # Convert skill to markdown
        markdown = skill.to_markdown()

        # Write to file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(markdown)

        print(f"✓ Exported skill to {file_path}")
        return file_path

    def export_all_gaia_skills(self, skill_store: 'SkillStore',
                                min_success_rate: float = 80.0) -> List[Path]:
        """
        Export all high-quality GAIA skills to markdown.

        Args:
            skill_store: GAIA skill store
            min_success_rate: Only export skills with success rate >= this

        Returns:
            List of exported file paths
        """
        all_skills = skill_store.get_all_skills()
        exported = []

        for skill in all_skills:
            # Only export successful skills
            if skill.success_rate >= min_success_rate:
                # Mark as hybrid (started as GAIA, now exported to .md)
                skill.source = SkillSource.HYBRID
                file_path = self.export_skill(skill)
                exported.append(file_path)

        print(f"\n✓ Exported {len(exported)} high-quality skills to {self.output_dir}")
        return exported
```

---

## Skill Discovery and Loading

### Integration with Agent

```python
from gaia.agents.base import Agent
from gaia.skills import UnifiedSkillRouter, AnthropicSkillLoader, SkillStore

class SkillAwareAgent(Agent):
    """Agent with both Anthropic and GAIA skill support."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Load Anthropic skills
        self.anthropic_loader = AnthropicSkillLoader()

        # GAIA skill store (from DYNAMIC_TOOLS_FRAMEWORK)
        self.gaia_skill_store = SkillStore()

        # Unified router
        self.skill_router = UnifiedSkillRouter(
            self.anthropic_loader,
            self.gaia_skill_store
        )

    def find_skill(self, query: str) -> Optional[Skill]:
        """Find the best matching skill for a query."""
        result = self.skill_router.get_best_match(query)

        if result:
            skill, score, source = result
            print(f"✓ Found skill: {skill.name} (score: {score:.2f}, source: {source.value})")
            return skill

        return None

    def apply_skill(self, query: str) -> bool:
        """Find and apply the best matching skill."""
        skill = self.find_skill(query)

        if not skill:
            print(f"No skill found for: {query}")
            return False

        # Track invocation
        skill.invocation_count += 1

        # Apply the skill
        if skill.source == SkillSource.ANTHROPIC_MD:
            # Execute markdown-based skill (interpret steps)
            success = self._execute_markdown_skill(skill)
        else:
            # Execute GAIA code-based skill
            success = self._execute_code_skill(skill)

        # Update statistics
        if success:
            skill.success_count += 1

        return success

    def _execute_markdown_skill(self, skill: Skill) -> bool:
        """
        Execute an Anthropic markdown skill.
        Interprets the steps and applies them.
        """
        print(f"\nApplying skill: {skill.name}")
        print(f"Description: {skill.description}\n")

        for i, step in enumerate(skill.steps, 1):
            print(f"Step {i}: {step}")

            # Extract code blocks from step
            code_blocks = re.findall(r'```(\w+)\n(.*?)```', step, re.DOTALL)

            for language, code in code_blocks:
                # Execute code based on language
                if language == 'python':
                    try:
                        exec(code)
                    except Exception as e:
                        print(f"  ❌ Error: {e}")
                        return False
                elif language == 'bash':
                    import subprocess
                    try:
                        subprocess.run(code, shell=True, check=True)
                    except subprocess.CalledProcessError as e:
                        print(f"  ❌ Error: {e}")
                        return False

        print(f"\n✓ Skill {skill.name} applied successfully")
        return True

    def _execute_code_skill(self, skill: Skill) -> bool:
        """Execute a GAIA code-based skill."""
        if not skill.code:
            print(f"⚠️  Skill {skill.name} has no executable code")
            return False

        try:
            exec(skill.code)
            return True
        except Exception as e:
            print(f"❌ Error executing skill: {e}")
            return False
```

---

## Bidirectional Sync

### Auto-Export High-Quality GAIA Skills

```python
class SkillSyncManager:
    """Manages bidirectional sync between Anthropic .md and GAIA FAISS skills."""

    def __init__(self,
                 anthropic_loader: AnthropicSkillLoader,
                 gaia_store: SkillStore,
                 exporter: SkillExporter):
        self.anthropic_loader = anthropic_loader
        self.gaia_store = gaia_store
        self.exporter = exporter

    def sync_gaia_to_anthropic(self, min_success_rate: float = 85.0):
        """
        Export high-quality GAIA skills to Anthropic .md format.

        This makes agent-learned patterns available for:
        - Human review and curation
        - Team sharing via git
        - Use in Claude Desktop
        """
        print("=" * 70)
        print("Syncing GAIA skills → Anthropic .md files")
        print("=" * 70)
        print()

        exported = self.exporter.export_all_gaia_skills(
            self.gaia_store,
            min_success_rate=min_success_rate
        )

        if exported:
            print(f"\n✅ Sync complete. Review skills at: {self.exporter.output_dir}")
            print("   Commit to git to share with team!")
        else:
            print("\n⚠️  No skills met success rate threshold")

    def sync_anthropic_to_gaia(self):
        """
        Import Anthropic .md skills into GAIA FAISS store.

        This makes manually curated skills available for:
        - Semantic search
        - Automatic invocation
        - Statistical tracking
        """
        print("=" * 70)
        print("Syncing Anthropic .md files → GAIA FAISS")
        print("=" * 70)
        print()

        skills = self.anthropic_loader.load_all()

        for skill in skills:
            # Check if already in GAIA store
            existing = self.gaia_store.get_skill_by_name(skill.name)

            if existing:
                print(f"⚠️  Skill '{skill.name}' already exists in GAIA store, skipping")
                continue

            # Add to GAIA store
            self.gaia_store.add_skill(skill)
            print(f"✓ Imported: {skill.name}")

        print(f"\n✅ Sync complete. {len(skills)} skills now in GAIA store")

    def full_sync(self):
        """Perform full bidirectional sync."""
        self.sync_anthropic_to_gaia()
        print()
        self.sync_gaia_to_anthropic()
```

### CLI Commands

```python
# src/gaia/cli.py

@click.group()
def skills():
    """Manage GAIA and Anthropic skills."""
    pass

@skills.command()
@click.option('--min-success-rate', default=85.0, help='Minimum success rate to export')
def export(min_success_rate):
    """Export high-quality GAIA skills to Anthropic .md format."""
    from gaia.skills import SkillStore, SkillExporter

    store = SkillStore()
    exporter = SkillExporter()

    exported = exporter.export_all_gaia_skills(store, min_success_rate)

    if exported:
        click.echo(f"✅ Exported {len(exported)} skills to ~/.claude/skills/")
        click.echo("   Commit to git to share with team!")
    else:
        click.echo("⚠️  No skills met success rate threshold")

@skills.command()
def import_anthropic():
    """Import Anthropic .md skills into GAIA."""
    from gaia.skills import AnthropicSkillLoader, SkillStore

    loader = AnthropicSkillLoader()
    store = SkillStore()

    skills = loader.load_all()

    for skill in skills:
        store.add_skill(skill)

    click.echo(f"✅ Imported {len(skills)} Anthropic skills into GAIA")

@skills.command()
def sync():
    """Full bidirectional sync between Anthropic and GAIA skills."""
    from gaia.skills import SkillSyncManager, AnthropicSkillLoader, SkillStore, SkillExporter

    manager = SkillSyncManager(
        AnthropicSkillLoader(),
        SkillStore(),
        SkillExporter()
    )

    manager.full_sync()

@skills.command()
@click.argument('query')
def search(query):
    """Search for skills matching a query."""
    from gaia.skills import UnifiedSkillRouter, AnthropicSkillLoader, SkillStore

    router = UnifiedSkillRouter(
        AnthropicSkillLoader(),
        SkillStore()
    )

    results = router.search(query, top_k=5)

    if not results:
        click.echo("No skills found")
        return

    click.echo(f"Found {len(results)} skills:\n")
    for i, (skill, score, source) in enumerate(results, 1):
        click.echo(f"{i}. {skill.name} (score: {score:.2f}, source: {source.value})")
        click.echo(f"   {skill.description}")
        click.echo()
```

---

## Migration Path

### Phase 1: Read Anthropic Skills (Week 7 in AI_ACCELERATED_TIMELINE)

Implement alongside Dynamic Tools + SKILLS week.

**Deliverables**:
- `AnthropicSkillLoader` - Load .md files
- `SkillMarkdownParser` - Parse markdown structure
- `UnifiedSkillRouter` - Search both systems
- CLI: `gaia skills search <query>`

**Validation**:
```bash
# Create sample Anthropic skill
mkdir -p ~/.claude/skills
cat > ~/.claude/skills/python-testing.md << 'EOF'
# Python Testing with pytest
**Category**: Testing
**Triggers**: pytest, testing, unit test
... (rest of content)
EOF

# Search for it
gaia skills search "how to set up testing"
# Should return the Anthropic skill
```

### Phase 2: Export GAIA Skills (Week 8)

**Deliverables**:
- `SkillExporter` - Export to markdown
- CLI: `gaia skills export`
- Automatic export of high-success skills

**Validation**:
```bash
# Run agent to learn some patterns (trigger pattern detection 3x)
gaia-code "Set up FastAPI project"
gaia-code "Set up Flask project"
gaia-code "Set up Django project"

# Export learned skills
gaia skills export --min-success-rate 80

# Check output
ls ~/.claude/skills/
# Should see: setup-python-web-project.md
```

### Phase 3: Bidirectional Sync (Week 9)

**Deliverables**:
- `SkillSyncManager` - Full sync
- CLI: `gaia skills sync`
- Automatic sync on startup (optional)

**Validation**:
```bash
# Full sync
gaia skills sync

# Should:
# 1. Import all Anthropic .md files into GAIA FAISS
# 2. Export high-quality GAIA skills to .md
# 3. Report stats
```

---

## Benefits

### For Individual Developers

1. **Access team knowledge** - Load company-curated Anthropic skills
2. **Learn from agent** - Export GAIA-learned skills for review
3. **Portable skills** - Use same skills in Claude Desktop and GAIA
4. **Version control** - Track skill evolution in git

### For Teams

1. **Share best practices** - Commit .md skills to repo
2. **Review agent learning** - Human review before accepting learned patterns
3. **Standardization** - Unified skill format across tools
4. **Onboarding** - New team members get curated skills automatically

### For GAIA

1. **Broader ecosystem** - Compatible with Claude Desktop skills
2. **Human + AI curation** - Best of manual and automatic skill creation
3. **Better discoverability** - Semantic search + keyword matching
4. **Hybrid approach** - Code skills for automation, markdown for documentation

---

## Success Metrics

### Week 7 (Anthropic Skill Support)
- ✅ Can load 5+ Anthropic .md skills
- ✅ Unified router searches both systems
- ✅ Search returns relevant skills (>0.7 similarity score)

### Week 8 (GAIA Skill Export)
- ✅ Export learned skills to markdown
- ✅ Exported skills readable by humans
- ✅ Can commit to git and share with team

### Week 9 (Bidirectional Sync)
- ✅ Full sync works in both directions
- ✅ No data loss during sync
- ✅ Statistics preserved across formats

### Long-term
- 📊 80%+ of teams use both Anthropic and GAIA skills
- 📊 50%+ of exported GAIA skills reviewed and kept by humans
- 📊 Skill success rate >85% for both types

---

## File Locations

```
.claude/
├── skills/              # Anthropic skills (user-facing)
│   └── *.md
├── SKILLS.md            # Optional manifest

src/gaia/skills/
├── __init__.py
├── models.py            # Unified Skill dataclass
├── parser.py            # Markdown parser
├── loader.py            # Anthropic loader
├── router.py            # Unified router
├── exporter.py          # Export to markdown
├── store.py             # GAIA FAISS store (existing)
└── sync.py              # Bidirectional sync manager

src/gaia/cli.py          # CLI commands (skills group)
```

---

## Implementation Timeline

**Week 7** (Dynamic Tools + SKILLS):
- Day 1-2: Implement Anthropic skill loader + parser
- Day 3: Implement unified router
- Day 4: CLI commands for search
- Day 5: Testing + integration

**Week 8** (Voice Interface):
- Day 1-2: Implement skill exporter
- Day 3: CLI command for export
- Day 4: Automatic export on shutdown
- Day 5: Testing + validation

**Week 9** (Enhanced TUI):
- Day 1: Implement bidirectional sync
- Day 2: Skills browser panel in TUI
- Day 3: CLI sync command
- Day 4-5: End-to-end testing

---

## Related Documents

- **DYNAMIC_TOOLS_FRAMEWORK.md** - GAIA's native SKILLS system (FAISS-based)
- **AI_ACCELERATED_TIMELINE.md** - Week 7-9 implementation schedule
- **TUI_DESIGN_SPECIFICATION.md** - Skills browser panel in TUI

---

*Supporting Anthropic SKILLs standard - Making agent knowledge shareable, reviewable, and version-controlled.*
