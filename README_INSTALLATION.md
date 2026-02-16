# GAIA Code Installation Guide

**Preferred Method**: Use `uv` for faster installation

---

## Quick Install with uv (Recommended)

### 1. Install uv (if not already installed)

```bash
# Linux/macOS/WSL
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Install GAIA with GAIA Code

```bash
cd C:\Users\14255\Work\gaia

# Activate venv
.venv\Scripts\activate  # Windows
# or: source .venv/bin/activate  # Linux/macOS

# Install with uv (much faster than pip)
uv pip install -e ".[dev,gaia_code]"

# Install Playwright browsers
playwright install chromium
```

### 3. Set API Key

```bash
# Windows
set ANTHROPIC_API_KEY=your_key_here

# Linux/macOS
export ANTHROPIC_API_KEY=your_key_here
```

### 4. Test

```bash
gaia code "Create a hello world function" --persona pike
```

---

## What Gets Installed

### Core Dependencies (always)
- openai, pydantic, transformers
- python-dotenv, aiohttp, rich, requests

### GAIA Code Dependencies (with [gaia_code])
- playwright - Web app testing and screenshots
- sentence-transformers - Semantic embeddings
- faiss-cpu - Vector search
- anthropic - Claude API integration

### Dev Dependencies (with [dev])
- pytest, black, pylint, isort
- Plus playwright for testing

---

## Installation Options

### Minimal (Core only)
```bash
uv pip install -e .
```

### With Dev Tools
```bash
uv pip install -e ".[dev]"
```

### With GAIA Code (Recommended)
```bash
uv pip install -e ".[dev,gaia_code]"
```

### Everything
```bash
uv pip install -e ".[dev,gaia_code,rag,api,mcp]"
```

---

## Verify Installation

```bash
# Check GAIA Code is available
python -c "from gaia.agents.gaia_code import GaiaCodeAgent; print('✓ GAIA Code ready')"

# Check Playwright
python -c "from playwright.sync_api import sync_playwright; print('✓ Playwright ready')"

# List available personas
python -c "from gaia.agents.gaia_code.persona import list_personas; print('✓ Personas:', [p['name'] for p in list_personas()])"
```

---

## Troubleshooting

### Issue: "No module named 'playwright'"

**Solution**:
```bash
uv pip install playwright
playwright install chromium
```

### Issue: "No module named 'sentence_transformers'"

**Solution**:
```bash
uv pip install -e ".[gaia_code]"
```

### Issue: "ANTHROPIC_API_KEY not set"

**Solution**: GAIA Code will prompt you for it on first run, or set it:
```bash
export ANTHROPIC_API_KEY=your_key
```

---

## Why uv?

- **10-100x faster** than pip
- **Better dependency resolution**
- **Parallel downloads**
- **Exact reproducibility**

**Use uv for all pip commands**:
```bash
uv pip install ...    # instead of: pip install ...
uv pip list          # instead of: pip list
uv pip uninstall ... # instead of: pip uninstall ...
```

---

**Status**: setup.py updated with gaia_code packages and dependencies

**Next**: Run `gaia code "task" --persona pike` to test!
