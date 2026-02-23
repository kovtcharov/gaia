# GAIA Code Configuration System Specification (v1)

## Context

GAIA Code has many architectural capabilities (RAC, quality gates, code retrieval, self-introspection, etc.) but no unified way to enable/disable them. Configuration is scattered across:

- `GaiaCodeAgent.__init__` parameters
- `CredentialManager.set_setting()` (JSON file)
- `KnowledgeDB.store_preference()` (SQLite)
- Environment variables
- Hardcoded defaults

This creates problems:
1. **Users can't easily disable features** that don't work well with their LLM
2. **No project-level config** — every session starts from defaults
3. **No capability profiles** — a user with Qwen3-0.6B needs different defaults than one with Claude Opus
4. **No single file to inspect** — understanding what's enabled requires reading code

This spec defines a unified YAML-based configuration system with layered overrides.

---

## Design Principles

1. **YAML first** — Human-readable, widely understood, supports comments
2. **Layered overrides** — Project config overrides user config overrides defaults
3. **Feature flags** — Every capability can be enabled/disabled with a boolean
4. **Profiles** — Pre-built configurations for common scenarios (e.g., "minimal", "standard", "full")
5. **LLM-aware defaults** — Automatically adjust feature flags based on detected LLM capability
6. **Zero-overhead when disabled** — Disabled features must not add latency or memory usage
7. **Graceful degradation** — If a required dependency is missing (e.g., FAISS), warn and disable, don't crash

---

## Configuration File Locations

```
Priority (highest wins):
1. CLI flags:            gaia code --no-self-introspection "task"
2. Environment variables: GAIA_CODE_SELF_INTROSPECTION=false
3. Project config:       ./.gaia/config.yaml (or ./gaia.yaml)
4. User config:          ~/.gaia/config.yaml
5. Profile preset:       gaia code --profile=minimal "task"
6. LLM-detected defaults: Auto-adjust based on model capabilities
7. Built-in defaults:    Defined in code
```

---

## Config Schema

```yaml
# ~/.gaia/config.yaml (or .gaia/config.yaml in project root)

# ============================================================
# LLM Configuration
# ============================================================
llm:
  provider: "claude"          # claude | lemonade | openai
  model: "claude-opus-4-6"    # Model identifier
  api_key_env: "ANTHROPIC_API_KEY"  # Env var containing API key
  temperature: 0.0
  max_tokens: 8192

# ============================================================
# Agent Core
# ============================================================
agent:
  max_steps: 1000             # Max agent steps (0 = unlimited)
  max_plan_iterations: 100
  persona: "pike"             # Personality profile
  silent_mode: false
  tui_mode: "simple"          # full | simple | minimal | off

# ============================================================
# Architectural Capabilities (Feature Flags)
# ============================================================
capabilities:
  # --- RAC (Recursive Agent Composition) ---
  rac:
    enabled: true             # Master switch for RAC
    max_recursion_depth: 10
    enable_shared_state: true
    enable_audit_log: true

  # --- Quality Gates ---
  quality_gates:
    enabled: true
    syntax_check: true
    test_runner: true
    output_verification: true
    max_retries: 3

  # --- Escalation Ladder ---
  escalation:
    enabled: true
    levels: ["retry", "decompose", "cloud", "ask_user"]
    cloud_cost_ceiling: 5.00  # USD per subtask

  # --- Code Retrieval Pipeline ---
  code_retrieval:
    enabled: true
    incremental_indexing: true
    file_watcher: true        # Background auto-update
    llm_annotations: "cloud"  # cloud | local | none
    faiss_search: true        # Requires faiss-cpu
    fts5_search: true
    hybrid_alpha: 0.6         # FAISS weight in hybrid search
    context_budget: 8000      # Token budget for context assembly
    max_chunk_lines: 200      # Split classes larger than this

  # --- Self-Introspection ---
  self_introspection:
    enabled: true             # Index gaia's own codebase
    autonomy_level: "propose_only"  # propose_only | fix_and_ask | auto_pr | self_modify
    max_proposals_per_session: 3
    enable_hot_reload: false
    gaia_repo_url: "https://github.com/amd/gaia"

  # --- Persistent Memory ---
  memory:
    enabled: true
    knowledge_db: true        # Cross-session insights
    skills_db: true           # Learned workflows
    tools_db: true            # Tool registry
    agents_db: true           # Specialist registry
    confidence_decay: true    # Old patterns lose weight
    max_insights: 10000       # Limit to prevent DB bloat

  # --- Checkpoint/Resume ---
  checkpoint:
    enabled: true
    auto_checkpoint: true     # Checkpoint after each task
    checkpoint_interval: 300  # Seconds between auto-checkpoints

  # --- Planning Engine ---
  planning:
    enabled: true
    master_plan: true         # Hierarchical task tree
    project_manifest: true    # Live project state tracking

  # --- Execution & Observation ---
  execution:
    enabled: true
    run_and_observe: true     # Execute code and capture output
    run_until_functional: true # Iterative debug loop
    interactive_cli: true     # Handle interactive prompts

  # --- Vector Search ---
  vector_search:
    enabled: true             # Requires sentence-transformers + faiss-cpu
    embedding_model: "all-MiniLM-L6-v2"
    embedding_cache: true

# ============================================================
# File Handling
# ============================================================
files:
  # Extensions to fully index (AST parse + annotate)
  index_extensions:
    - ".py"
    - ".js"
    - ".ts"
    - ".tsx"
    - ".yaml"
    - ".json"
    - ".md"

  # Extensions to track only (record existence, no annotation)
  track_only_extensions:
    - ".lock"
    - ".png"
    - ".jpg"
    - ".pkl"
    - ".faiss"

  # Filenames to skip entirely
  skip_filenames:
    - "package-lock.json"
    - "yarn.lock"
    - "uv.lock"

  # Directories to exclude
  exclude_dirs:
    - "__pycache__"
    - ".git"
    - "node_modules"
    - "venv"
    - ".venv"

# ============================================================
# Logging
# ============================================================
logging:
  level: "INFO"               # DEBUG | INFO | WARNING | ERROR
  structured: true            # JSON structured logging
  audit_log: true             # Log all agent actions
  log_file: null              # Optional file path (null = stderr only)
```

---

## Profiles

Pre-built configurations for common scenarios. Selected via `--profile=NAME` or `profile: NAME` in config.

### `minimal` — For small LLMs (Qwen3-0.6B, <1B params)

```yaml
profile: minimal
capabilities:
  rac:
    enabled: false            # Too complex for small LLMs
  quality_gates:
    enabled: true
    test_runner: false         # May generate bad test commands
  escalation:
    enabled: false
  code_retrieval:
    enabled: true
    llm_annotations: "none"   # AST-only (no LLM calls)
    faiss_search: false        # Save memory
    fts5_search: true
  self_introspection:
    enabled: false
  memory:
    enabled: true
    skills_db: false
    agents_db: false
  checkpoint:
    enabled: false
  planning:
    enabled: true
    project_manifest: false
  vector_search:
    enabled: false
```

### `standard` — For mid-range LLMs (Qwen3-Coder-30B, Llama 70B)

```yaml
profile: standard
capabilities:
  rac:
    enabled: true
    max_recursion_depth: 5
  quality_gates:
    enabled: true
  code_retrieval:
    enabled: true
    llm_annotations: "local"  # Use local LLM for annotations
  self_introspection:
    enabled: true
    autonomy_level: "propose_only"
  memory:
    enabled: true
  vector_search:
    enabled: true
```

### `full` — For frontier LLMs (Claude Opus, GPT-4)

```yaml
profile: full
capabilities:
  rac:
    enabled: true
    max_recursion_depth: 10
  quality_gates:
    enabled: true
  escalation:
    enabled: true
  code_retrieval:
    enabled: true
    llm_annotations: "cloud"
    faiss_search: true
  self_introspection:
    enabled: true
    autonomy_level: "fix_and_ask"
  memory:
    enabled: true
  checkpoint:
    enabled: true
  planning:
    enabled: true
  execution:
    enabled: true
  vector_search:
    enabled: true
```

---

## LLM-Aware Auto-Detection

When no profile is specified, the system auto-detects the appropriate profile:

```python
def detect_profile(model_id: str) -> str:
    """Auto-detect appropriate profile based on model capabilities."""
    model = model_id.lower()

    # Frontier models → full
    frontier = ["claude-opus", "claude-sonnet", "gpt-4", "gpt-4o"]
    if any(m in model for m in frontier):
        return "full"

    # Mid-range models → standard
    midrange = ["qwen3-coder", "llama-70b", "llama-3", "mixtral", "deepseek"]
    if any(m in model for m in midrange):
        return "standard"

    # Small models → minimal
    return "minimal"
```

---

## Implementation

### `GaiaCodeConfig` Dataclass

```python
@dataclass
class GaiaCodeConfig:
    """Unified configuration for GAIA Code agent."""

    # Nested config sections
    llm: LLMConfig
    agent: AgentConfig
    capabilities: CapabilitiesConfig
    files: FilesConfig
    logging: LoggingConfig

    # Profile (if loaded from preset)
    profile: Optional[str] = None

    @classmethod
    def load(cls, project_dir: Optional[Path] = None) -> "GaiaCodeConfig":
        """Load configuration with layered overrides."""
        config = cls.defaults()

        # 1. User config (~/.gaia/config.yaml)
        user_config = _load_yaml(Path.home() / ".gaia" / "config.yaml")
        if user_config:
            _merge(config, user_config)

        # 2. Project config (.gaia/config.yaml or gaia.yaml)
        if project_dir:
            for name in [".gaia/config.yaml", "gaia.yaml"]:
                project_config = _load_yaml(project_dir / name)
                if project_config:
                    _merge(config, project_config)
                    break

        # 3. Environment variables (GAIA_CODE_*)
        _apply_env_overrides(config)

        # 4. Auto-detect profile if none specified
        if not config.profile:
            config.profile = detect_profile(config.llm.model)
            _apply_profile(config, config.profile)

        # 5. Dependency checks (disable features with missing deps)
        _check_dependencies(config)

        return config

    @classmethod
    def defaults(cls) -> "GaiaCodeConfig":
        """All defaults — safe baseline."""
        ...

    def save(self, path: Path):
        """Save current config to YAML file."""
        ...

    def to_dict(self) -> Dict:
        """Serialize to dict (secrets redacted)."""
        ...
```

### Dependency Checking

```python
def _check_dependencies(config: GaiaCodeConfig):
    """Disable features whose dependencies are missing."""
    # FAISS
    if config.capabilities.code_retrieval.faiss_search:
        try:
            import faiss
        except ImportError:
            config.capabilities.code_retrieval.faiss_search = False
            logger.warning("[Config] FAISS not installed, disabling FAISS search")

    # sentence-transformers
    if config.capabilities.vector_search.enabled:
        try:
            import sentence_transformers
        except ImportError:
            config.capabilities.vector_search.enabled = False
            logger.warning("[Config] sentence-transformers not installed, disabling vector search")

    # gh CLI (for self-improvement PRs)
    if config.capabilities.self_introspection.autonomy_level in ("auto_pr", "self_modify"):
        import shutil
        if not shutil.which("gh"):
            logger.warning(
                "[Config] gh CLI not installed, self-improvement PRs disabled. "
                "Install from https://cli.github.com/ to enable. "
                "Continuing with propose_only mode."
            )
            config.capabilities.self_introspection.autonomy_level = "propose_only"
```

### Environment Variable Mapping

```
GAIA_CODE_LLM_PROVIDER        → llm.provider
GAIA_CODE_LLM_MODEL           → llm.model
GAIA_CODE_PROFILE              → profile
GAIA_CODE_RAC_ENABLED          → capabilities.rac.enabled
GAIA_CODE_QUALITY_GATES        → capabilities.quality_gates.enabled
GAIA_CODE_CODE_RETRIEVAL       → capabilities.code_retrieval.enabled
GAIA_CODE_SELF_INTROSPECTION   → capabilities.self_introspection.enabled
GAIA_CODE_MEMORY               → capabilities.memory.enabled
GAIA_CODE_VECTOR_SEARCH        → capabilities.vector_search.enabled
GAIA_CODE_LOG_LEVEL            → logging.level
```

### CLI Flags

```bash
# Enable/disable via CLI
gaia code --profile=minimal "fix the bug"
gaia code --no-self-introspection "fix the bug"
gaia code --no-rac --no-vector-search "simple fix"
gaia code --autonomy=auto_pr "fix the bug"
```

---

## Integration with GaiaCodeAgent

```python
class GaiaCodeAgent:
    def __init__(self, config: Optional[GaiaCodeConfig] = None, **kwargs):
        # Load config
        self.config = config or GaiaCodeConfig.load()

        # Apply config to agent initialization
        if self.config.capabilities.rac.enabled:
            kwargs["enable_shared_state"] = True
            kwargs["enable_audit_log"] = self.config.capabilities.rac.enable_audit_log

        kwargs["max_steps"] = self.config.agent.max_steps

        super().__init__(**kwargs)

        # Initialize capabilities based on config
        if self.config.capabilities.code_retrieval.enabled:
            self._init_code_retrieval()
        if self.config.capabilities.self_introspection.enabled:
            self._init_self_introspection()
        if self.config.capabilities.vector_search.enabled:
            self._init_vector_search()
```

---

## Key Files

| File | Action |
|------|--------|
| `src/gaia/agents/gaia_code/config.py` | **CREATE** — GaiaCodeConfig, loading, profiles, dependency checks |
| `src/gaia/agents/gaia_code/agent.py` | **MODIFY** — Accept config, wire feature flags |
| `src/gaia/agents/gaia_code/tools.py` | **MODIFY** — Check config before initializing pipeline |
| `tests/unit/test_gaia_code_config.py` | **CREATE** — Config loading, profiles, overrides, dep checks |

---

## Implementation Phases

### Phase 1: Config Dataclass + YAML Loading
- `GaiaCodeConfig` with all sections
- YAML file loading with layered overrides
- Environment variable overrides
- `config.py` created

### Phase 2: Profiles + LLM Detection
- minimal/standard/full profiles
- Auto-detection from model ID
- CLI flag support (`--profile`, `--no-X`)

### Phase 3: Agent Integration
- Wire config into `GaiaCodeAgent.__init__`
- Feature flag checks in tools
- Dependency checking with graceful degradation

### Phase 4: Documentation
- Example config files
- Profile comparison table
- Migration guide from current settings
