# GAIA V2 Deployment Architecture

> **Version:** 0.15.3.2 | **Lemonade Server:** 9.3.0 | **Last Updated:** 2026-02-07
>
> **Scope:** Production deployment patterns for GAIA agents across local, server, cloud, and orchestrated environments.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Local Deployment](#2-local-deployment)
3. [Server Deployment](#3-server-deployment)
4. [Cloud Deployment](#4-cloud-deployment)
5. [Docker / Kubernetes](#5-docker--kubernetes)
6. [Database Migration](#6-database-migration)
7. [Monitoring Setup](#7-monitoring-setup)
8. [Security Hardening](#8-security-hardening)
9. [Scaling Strategy](#9-scaling-strategy)
10. [CI/CD Pipeline](#10-cicd-pipeline)

---

## 1. Executive Summary

### 1.1 Deployment Models

GAIA V2 supports four deployment models, each suited to different operational requirements:

| Model | Users | LLM Backend | Use Case |
|-------|-------|-------------|----------|
| **Local** | Single user | Lemonade Server (NPU/iGPU) | Desktop AI PC, privacy-first |
| **Server** | Multi-user | Lemonade Server or cloud LLM | Team/department deployment |
| **Cloud** | Organization | Cloud LLM providers | Enterprise SaaS, elastic scale |
| **Distributed** | Large-scale | Mixed (local + cloud) | Hybrid edge-cloud pipelines |

### 1.2 Core Components

Every GAIA deployment consists of these runtime components:

```
┌─────────────────────────────────────────────────────────────┐
│                     Client Layer                            │
│  CLI (gaia chat/code/...)  │  GAIA UI (Electron/RAUX)      │
│  VSCode Copilot Extension  │  OpenAI-compatible clients     │
└───────────────┬─────────────────────────┬───────────────────┘
                │                         │
┌───────────────▼─────────────────────────▼───────────────────┐
│                     API Layer                               │
│  FastAPI Server (gaia.api.openai_server)                    │
│  POST /v1/chat/completions  │  GET /v1/models  │  /health  │
│  Agent Registry  │  SSE Streaming  │  CORS Middleware       │
└───────────────┬─────────────────────────┬───────────────────┘
                │                         │
┌───────────────▼─────────────┐ ┌─────────▼───────────────────┐
│       Agent Layer           │ │    Infrastructure Layer      │
│  RoutingAgent               │ │  MCP Bridge (gaia-mcp)       │
│  ChatAgent (RAG)            │ │  DatabaseMixin (SQLite)      │
│  CodeAgent (orchestration)  │ │  RAG SDK (FAISS + PDF)       │
│  BlenderAgent               │ │  Talk SDK (Whisper + Kokoro) │
│  JiraAgent                  │ │  FileWatcher                 │
│  DockerAgent                │ │  Eval Framework              │
└───────────────┬─────────────┘ └─────────────────────────────┘
                │
┌───────────────▼─────────────────────────────────────────────┐
│                     LLM Layer                               │
│  Lemonade Server (AMD NPU/iGPU)  - Default, local          │
│  Claude API (Anthropic)           - Cloud provider          │
│  OpenAI API                       - Cloud provider          │
│  Any OpenAI-compatible endpoint   - Custom backend          │
└─────────────────────────────────────────────────────────────┘
```

### 1.3 Target Environments

| Environment | OS | Python | Hardware |
|-------------|-----|--------|----------|
| AI PC Desktop | Windows 11 24H2, Ubuntu 22.04+ | 3.10 - 3.12 | Ryzen AI 300+ (NPU + iGPU) |
| Linux Server | Ubuntu 22.04 LTS / 24.04 LTS | 3.12 | AMD EPYC, Instinct MI300X |
| Docker Container | Debian 12 / Ubuntu 22.04 | 3.12 | Any x86_64 |
| Kubernetes | Linux nodes | 3.12 | Mixed fleet |
| Cloud VM | Ubuntu 22.04+ | 3.12 | Cloud instances |

### 1.4 Package Coordinates

```
PyPI:       amd-gaia==0.15.3.2
GitHub:     https://github.com/amd/gaia
Installer:  https://github.com/amd/gaia/releases
Docs:       https://amd-gaia.ai
```

---

## 2. Local Deployment

### 2.1 Single-User Desktop Installation

The primary GAIA deployment model is a single-user AI PC installation where the Lemonade Server runs LLMs directly on AMD NPU/iGPU hardware.

#### Architecture

```
┌──────────────────────────────────────────────────┐
│                   AI PC (User Desktop)           │
│                                                  │
│  ┌──────────────┐    ┌────────────────────────┐  │
│  │  GAIA CLI     │───▶│  Lemonade Server       │  │
│  │  gaia chat    │    │  localhost:8000         │  │
│  │  gaia code    │    │  ONNX Runtime GenAI    │  │
│  │  gaia talk    │    │  NPU + iGPU backend    │  │
│  └──────────────┘    └────────────────────────┘  │
│         │                                        │
│  ┌──────▼───────┐    ┌────────────────────────┐  │
│  │  GAIA API     │───▶│  GAIA UI (RAUX)        │  │
│  │  localhost:8080│    │  Electron Desktop App  │  │
│  └──────────────┘    └────────────────────────┘  │
│                                                  │
│  Storage: ~/.gaia/cache/ │ SQLite databases      │
└──────────────────────────────────────────────────┘
```

#### Installation Methods

**Method 1: GAIA UI Installer (Recommended for end users)**

Windows:
```powershell
# Download gaia-ui-setup.exe from https://github.com/amd/gaia/releases
# Run the installer - it handles Lemonade Server, Python, and GAIA setup
```

Ubuntu:
```bash
# Download gaia-ui-setup.deb from https://github.com/amd/gaia/releases
sudo apt update
sudo apt install ./gaia-ui-setup.deb
```

**Method 2: pip install (Developers)**

```bash
# Install uv (fast Python package manager)
# Windows PowerShell:
irm https://astral.sh/uv/install.ps1 | iex
# Linux:
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv .venv --python 3.12

# Activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# Linux:              source .venv/bin/activate

# Install GAIA with desired extras
uv pip install amd-gaia                    # Core only
uv pip install "amd-gaia[api]"             # + FastAPI server
uv pip install "amd-gaia[rag]"             # + Document Q&A
uv pip install "amd-gaia[talk]"            # + Voice (Whisper + Kokoro)
uv pip install "amd-gaia[api,rag,talk]"    # Combined
```

**Method 3: Development install (Contributors)**

```bash
git clone https://github.com/amd/gaia.git
cd gaia
uv venv .venv --python 3.12
source .venv/bin/activate   # or .\.venv\Scripts\Activate.ps1
uv pip install -e ".[dev,api,rag]"
```

### 2.2 Platform Considerations

#### Windows 11

- **NPU Driver:** Minimum version `32.0.203.314`
- **iGPU Driver:** Minimum version `32.0.22029.1019`
- **Lemonade Server:** Native Windows build from https://lemonade-server.ai/
- **Python:** 3.12 via `uv` (auto-downloads if missing)
- **Path length:** Enable long paths via Group Policy or registry (`LongPathsEnabled = 1`)
- **Firewall:** Lemonade binds `localhost:8000`, GAIA API binds `localhost:8080` -- no firewall rules needed for local-only use

```powershell
# Verify NPU driver
Get-WmiObject Win32_PnPSignedDriver | Where-Object { $_.DeviceName -like "*NPU*" } | Select-Object DeviceName, DriverVersion

# Verify iGPU driver
Get-WmiObject Win32_PnPSignedDriver | Where-Object { $_.DeviceName -like "*Radeon*" } | Select-Object DeviceName, DriverVersion
```

#### Ubuntu Linux (22.04 / 24.04)

- **NPU Driver:** XDNA driver from AMD (if using Ryzen AI hardware)
- **Build tools:** `sudo apt install build-essential python3-dev`
- **Audio (Talk):** `sudo apt install portaudio19-dev` (required for `pyaudio`)
- **Systemd:** Available for process management (see Server Deployment)

```bash
# Verify AMD hardware
lspci | grep -i amd

# Check NPU availability (if applicable)
ls /dev/accel* 2>/dev/null && echo "NPU device found" || echo "No NPU device"
```

#### macOS

- **Status:** Supported for development and cloud-provider LLM usage
- **Lemonade Server:** Not available (AMD NPU/iGPU not present)
- **LLM Backend:** Use `--use-claude` or `--use-chatgpt` flags, or point to a remote Lemonade Server via `LEMONADE_BASE_URL`
- **Python:** 3.12 via `uv` or Homebrew

```bash
# macOS development setup
brew install uv
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev,api,rag]"

# Use cloud LLM backend
export ANTHROPIC_API_KEY="sk-ant-..."
gaia chat --use-claude
```

### 2.3 Resource Requirements

| Component | RAM | Disk | GPU/NPU | Notes |
|-----------|-----|------|---------|-------|
| GAIA SDK (core) | 512 MB | 200 MB | None | Python packages |
| Lemonade Server | 2-8 GB | 500 MB | NPU + iGPU | Model-dependent |
| Qwen3-0.6B (default) | 2 GB | 800 MB | NPU | Small tasks |
| Qwen3-Coder-30B-A3B | 6 GB | 5 GB | NPU + iGPU | Code/agent tasks |
| RAG (FAISS + embeddings) | 1-4 GB | 500 MB | CPU | Per-collection |
| Talk (Whisper + Kokoro) | 2 GB | 1.5 GB | CPU | ASR + TTS models |
| GAIA UI (Electron) | 512 MB | 300 MB | None | Desktop app |

**Minimum system:** 16 GB RAM, 20 GB free disk, Ryzen AI 300-series
**Recommended system:** 64 GB RAM, 50 GB free disk, Ryzen AI MAX+ 395

### 2.4 Environment Variables

Create a `.env` file in your project root (GAIA loads it via `python-dotenv`):

```bash
# .env - Local deployment configuration

# LLM Backend
LEMONADE_BASE_URL=http://localhost:8000/api/v1

# Cloud LLM providers (optional, for --use-claude / --use-chatgpt)
# ANTHROPIC_API_KEY=sk-ant-...
# OPENAI_API_KEY=sk-...

# Agent configuration
AGENT_ROUTING_MODEL=Qwen3-Coder-30B-A3B-Instruct-GGUF

# API Server
GAIA_API_DEBUG=0
GAIA_API_STREAMING=0
GAIA_API_SHOW_PROMPTS=0

# Logging
GAIA_LOG_LEVEL=INFO
```

### 2.5 Verification

```bash
# 1. Start Lemonade Server (separate terminal)
lemonade-server serve

# 2. Verify LLM connectivity
gaia llm "Hello, world"

# 3. Start API server
gaia api start --background

# 4. Test API endpoint
curl http://localhost:8080/health

# 5. Test chat completion
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "gaia-code", "messages": [{"role": "user", "content": "Hello"}]}'

# 6. Interactive chat
gaia chat
```

---

## 3. Server Deployment

### 3.1 Multi-User Server Architecture

A server deployment exposes GAIA agents to multiple users via the OpenAI-compatible API, with a reverse proxy handling TLS, authentication, and load balancing.

```
                    ┌─────────────────────┐
                    │    Internet / VPN    │
                    └──────────┬──────────┘
                               │ HTTPS :443
                    ┌──────────▼──────────┐
                    │   nginx / Caddy     │
                    │   TLS termination   │
                    │   Rate limiting     │
                    │   Auth (API keys)   │
                    └──────────┬──────────┘
                               │ HTTP :8080
              ┌────────────────┼────────────────┐
              │                │                │
     ┌────────▼──────┐ ┌──────▼────────┐ ┌─────▼───────┐
     │  GAIA API #1  │ │  GAIA API #2  │ │  GAIA API #3│
     │  :8081        │ │  :8082        │ │  :8083      │
     │  uvicorn      │ │  uvicorn      │ │  uvicorn    │
     └────────┬──────┘ └──────┬────────┘ └─────┬───────┘
              │                │                │
              └────────────────┼────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Lemonade Server    │
                    │  :8000              │
                    │  (shared backend)   │
                    └─────────────────────┘
```

### 3.2 Installation on Server

```bash
# Create dedicated service user
sudo useradd -r -m -s /bin/bash gaia
sudo su - gaia

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# Create application directory
mkdir -p /opt/gaia
cd /opt/gaia

# Create virtual environment
uv venv .venv --python 3.12
source .venv/bin/activate

# Install GAIA with API and RAG support
uv pip install "amd-gaia[api,rag]"

# Create configuration
cat > /opt/gaia/.env << 'EOF'
LEMONADE_BASE_URL=http://localhost:8000/api/v1
GAIA_LOG_LEVEL=INFO
GAIA_API_STREAMING=1
EOF

# Create data directories
mkdir -p /opt/gaia/data /opt/gaia/logs /opt/gaia/rag-collections
```

### 3.3 Reverse Proxy Configuration (nginx)

```nginx
# /etc/nginx/sites-available/gaia-api
upstream gaia_backend {
    # Round-robin across multiple GAIA API instances
    server 127.0.0.1:8081;
    server 127.0.0.1:8082;
    server 127.0.0.1:8083;

    # Sticky sessions (optional, for stateful agent conversations)
    # ip_hash;

    # Health check
    # Note: requires nginx-plus or lua module for active health checks
}

server {
    listen 443 ssl http2;
    server_name gaia-api.example.com;

    # TLS configuration
    ssl_certificate     /etc/letsencrypt/live/gaia-api.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/gaia-api.example.com/privkey.pem;
    ssl_protocols       TLSv1.2 TLSv1.3;
    ssl_ciphers         HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;

    # Security headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Content-Type-Options nosniff always;
    add_header X-Frame-Options DENY always;
    add_header X-XSS-Protection "1; mode=block" always;

    # API key authentication
    # Clients must send: Authorization: Bearer <api-key>
    # Or use a separate auth service (see Security Hardening section)

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=gaia_api:10m rate=30r/m;

    # Request size limit (for RAG document uploads)
    client_max_body_size 100M;

    # Health check endpoint (no auth required)
    location /health {
        proxy_pass http://gaia_backend/health;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # OpenAI-compatible API
    location /v1/ {
        limit_req zone=gaia_api burst=10 nodelay;

        proxy_pass http://gaia_backend/v1/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # SSE streaming support
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_buffering off;
        proxy_cache off;
        chunked_transfer_encoding on;

        # Timeouts for long-running agent operations
        proxy_read_timeout 300s;
        proxy_send_timeout 300s;
        proxy_connect_timeout 10s;
    }

    # Block all other paths
    location / {
        return 404;
    }
}

# Redirect HTTP to HTTPS
server {
    listen 80;
    server_name gaia-api.example.com;
    return 301 https://$host$request_uri;
}
```

```bash
# Enable the site
sudo ln -s /etc/nginx/sites-available/gaia-api /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
```

### 3.4 Process Management (systemd)

#### Lemonade Server Service

```ini
# /etc/systemd/system/lemonade.service
[Unit]
Description=Lemonade Server (AMD LLM Backend)
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=gaia
Group=gaia
WorkingDirectory=/opt/gaia
ExecStart=/opt/gaia/.venv/bin/lemonade-server serve
Restart=always
RestartSec=5
StandardOutput=append:/opt/gaia/logs/lemonade.log
StandardError=append:/opt/gaia/logs/lemonade-error.log

# Resource limits
LimitNOFILE=65536
MemoryMax=16G

# Security hardening
NoNewPrivileges=yes
ProtectSystem=strict
ProtectHome=yes
ReadWritePaths=/opt/gaia

[Install]
WantedBy=multi-user.target
```

#### GAIA API Service (per instance)

```ini
# /etc/systemd/system/gaia-api@.service
# Template unit -- instantiate with: systemctl start gaia-api@8081
[Unit]
Description=GAIA API Server (port %i)
After=network.target lemonade.service
Requires=lemonade.service

[Service]
Type=simple
User=gaia
Group=gaia
WorkingDirectory=/opt/gaia
EnvironmentFile=/opt/gaia/.env
ExecStart=/opt/gaia/.venv/bin/python -m uvicorn \
    gaia.api.openai_server:app \
    --host 127.0.0.1 \
    --port %i \
    --workers 1 \
    --log-level info \
    --access-log
Restart=always
RestartSec=5
StandardOutput=append:/opt/gaia/logs/gaia-api-%i.log
StandardError=append:/opt/gaia/logs/gaia-api-%i-error.log

# Resource limits
LimitNOFILE=65536
MemoryMax=8G

# Security hardening
NoNewPrivileges=yes
ProtectSystem=strict
ProtectHome=yes
ReadWritePaths=/opt/gaia

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start services
sudo systemctl daemon-reload
sudo systemctl enable --now lemonade.service
sudo systemctl enable --now gaia-api@8081.service
sudo systemctl enable --now gaia-api@8082.service
sudo systemctl enable --now gaia-api@8083.service

# Check status
sudo systemctl status lemonade.service
sudo systemctl status gaia-api@8081.service

# View logs
journalctl -u gaia-api@8081.service -f
```

### 3.5 Supervisor Configuration (Alternative)

For environments without systemd (e.g., Docker containers or older init systems):

```ini
; /etc/supervisor/conf.d/gaia.conf

[program:lemonade]
command=/opt/gaia/.venv/bin/lemonade-server serve
directory=/opt/gaia
user=gaia
autostart=true
autorestart=true
stdout_logfile=/opt/gaia/logs/lemonade.log
stderr_logfile=/opt/gaia/logs/lemonade-error.log
environment=HOME="/home/gaia"

[program:gaia-api-8081]
command=/opt/gaia/.venv/bin/python -m uvicorn
    gaia.api.openai_server:app
    --host 127.0.0.1
    --port 8081
    --workers 1
directory=/opt/gaia
user=gaia
autostart=true
autorestart=true
stdout_logfile=/opt/gaia/logs/gaia-api-8081.log
stderr_logfile=/opt/gaia/logs/gaia-api-8081-error.log
environment=HOME="/home/gaia",LEMONADE_BASE_URL="http://localhost:8000/api/v1"

[program:gaia-api-8082]
command=/opt/gaia/.venv/bin/python -m uvicorn
    gaia.api.openai_server:app
    --host 127.0.0.1
    --port 8082
    --workers 1
directory=/opt/gaia
user=gaia
autostart=true
autorestart=true
stdout_logfile=/opt/gaia/logs/gaia-api-8082.log
stderr_logfile=/opt/gaia/logs/gaia-api-8082-error.log
environment=HOME="/home/gaia",LEMONADE_BASE_URL="http://localhost:8000/api/v1"

[group:gaia]
programs=lemonade,gaia-api-8081,gaia-api-8082
priority=999
```

```bash
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl status gaia:*
```

### 3.6 Load Balancing Multiple Agent Instances

Since GAIA agents are stateful during a conversation (they maintain `conversation_history`), load balancing requires either:

1. **Stateless design** (recommended): Each API request contains the full conversation history in the `messages` array (OpenAI-compatible pattern). Any instance can handle any request.

2. **Sticky sessions**: Route all requests from the same client to the same backend. Use `ip_hash` in nginx or a session cookie.

3. **External state store**: Store conversation state in Redis (future enhancement).

The current GAIA API server (`gaia.api.openai_server`) follows the OpenAI pattern where each request includes the full message history, making it naturally stateless at the HTTP level. Each agent instance processes the full context on every request.

```bash
# Test load balancing
for i in $(seq 1 10); do
  curl -s http://localhost/v1/models | jq '.data[0].id'
done
```

---

## 4. Cloud Deployment

### 4.1 Architecture Decision: LLM Backend

In cloud deployments, you typically do NOT run Lemonade Server (it requires AMD NPU/iGPU hardware). Instead, configure GAIA to use cloud LLM providers:

```bash
# Option A: Claude API
export ANTHROPIC_API_KEY="sk-ant-..."
gaia api start --host 0.0.0.0 --port 8080
# Agents use: --use-claude

# Option B: OpenAI API
export OPENAI_API_KEY="sk-..."
gaia api start --host 0.0.0.0 --port 8080
# Agents use: --use-chatgpt

# Option C: Remote Lemonade Server (on AMD hardware elsewhere)
export LEMONADE_BASE_URL="http://lemonade-server.internal:8000/api/v1"
gaia api start --host 0.0.0.0 --port 8080

# Option D: Any OpenAI-compatible endpoint (e.g., vLLM, Ollama, llama.cpp)
export LEMONADE_BASE_URL="http://vllm-server.internal:8000/v1"
gaia api start --host 0.0.0.0 --port 8080
```

### 4.2 AWS Deployment

#### EC2 Instance

```bash
# Launch Ubuntu 22.04 instance
# Recommended: c6a.2xlarge (AMD EPYC, 8 vCPU, 16 GB RAM) for API-only
# Or: g5.xlarge (NVIDIA GPU) if running local LLM via vLLM/Ollama

# User data script (cloud-init)
#!/bin/bash
set -euo pipefail

# System packages
apt-get update && apt-get install -y \
    build-essential python3-dev git nginx certbot python3-certbot-nginx

# Create service user
useradd -r -m -s /bin/bash gaia

# Install uv
su - gaia -c 'curl -LsSf https://astral.sh/uv/install.sh | sh'

# Install GAIA
su - gaia -c '
    source ~/.bashrc
    mkdir -p /opt/gaia && cd /opt/gaia
    uv venv .venv --python 3.12
    source .venv/bin/activate
    uv pip install "amd-gaia[api,rag]"
'

# Configure environment
cat > /opt/gaia/.env << 'ENVEOF'
ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
GAIA_LOG_LEVEL=INFO
GAIA_API_STREAMING=1
ENVEOF
chown gaia:gaia /opt/gaia/.env
chmod 600 /opt/gaia/.env

# Install systemd services (see Section 3.4)
# ...
```

#### ECS (Fargate) Task Definition

```json
{
  "family": "gaia-api",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "4096",
  "executionRoleArn": "arn:aws:iam::ACCOUNT:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::ACCOUNT:role/gaiaTaskRole",
  "containerDefinitions": [
    {
      "name": "gaia-api",
      "image": "ACCOUNT.dkr.ecr.REGION.amazonaws.com/gaia-api:latest",
      "portMappings": [
        {
          "containerPort": 8080,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "GAIA_LOG_LEVEL",
          "value": "INFO"
        },
        {
          "name": "GAIA_API_STREAMING",
          "value": "1"
        }
      ],
      "secrets": [
        {
          "name": "ANTHROPIC_API_KEY",
          "valueFrom": "arn:aws:secretsmanager:REGION:ACCOUNT:secret:gaia/anthropic-key"
        }
      ],
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8080/health || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3,
        "startPeriod": 60
      },
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/gaia-api",
          "awslogs-region": "REGION",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

#### ECS Service with ALB

```json
{
  "serviceName": "gaia-api-service",
  "cluster": "gaia-cluster",
  "taskDefinition": "gaia-api",
  "desiredCount": 3,
  "launchType": "FARGATE",
  "networkConfiguration": {
    "awsvpcConfiguration": {
      "subnets": ["subnet-private-1a", "subnet-private-1b"],
      "securityGroups": ["sg-gaia-api"],
      "assignPublicIp": "DISABLED"
    }
  },
  "loadBalancers": [
    {
      "targetGroupArn": "arn:aws:elasticloadbalancing:REGION:ACCOUNT:targetgroup/gaia-api-tg/...",
      "containerName": "gaia-api",
      "containerPort": 8080
    }
  ],
  "healthCheckGracePeriodSeconds": 120
}
```

#### Lambda (Lightweight Queries Only)

Lambda is suitable only for stateless, short-lived agent invocations (under 15 minutes). Not recommended for complex multi-step agent workflows.

```python
# lambda_handler.py
import json
import os

# Set environment before importing GAIA
os.environ["ANTHROPIC_API_KEY"] = os.environ.get("ANTHROPIC_API_KEY", "")

from gaia.agents.chat.agent import ChatAgent


def handler(event, context):
    """AWS Lambda handler for simple chat queries."""
    body = json.loads(event.get("body", "{}"))
    query = body.get("query", "")

    if not query:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": "Missing 'query' field"})
        }

    agent = ChatAgent(
        use_claude=True,
        silent_mode=True,
        max_steps=5,
        skip_lemonade=True,
    )

    result = agent.process_query(query)

    return {
        "statusCode": 200,
        "body": json.dumps({"response": result})
    }
```

### 4.3 Azure Deployment

#### Azure Container Instances

```bash
# Create resource group
az group create --name gaia-rg --location eastus

# Create container instance
az container create \
    --resource-group gaia-rg \
    --name gaia-api \
    --image ghcr.io/amd/gaia:latest \
    --cpu 2 \
    --memory 4 \
    --ports 8080 \
    --environment-variables \
        GAIA_LOG_LEVEL=INFO \
        GAIA_API_STREAMING=1 \
    --secure-environment-variables \
        ANTHROPIC_API_KEY="sk-ant-..." \
    --dns-name-label gaia-api \
    --restart-policy Always

# Get the FQDN
az container show \
    --resource-group gaia-rg \
    --name gaia-api \
    --query ipAddress.fqdn \
    --output tsv
```

#### Azure VM with cloud-init

```yaml
# cloud-init.yaml
#cloud-config
package_update: true
packages:
  - build-essential
  - python3-dev
  - git
  - nginx

users:
  - name: gaia
    shell: /bin/bash
    groups: [sudo]
    sudo: ['ALL=(ALL) NOPASSWD:ALL']

runcmd:
  - |
    su - gaia -c '
      curl -LsSf https://astral.sh/uv/install.sh | sh
      source ~/.bashrc
      mkdir -p /opt/gaia && cd /opt/gaia
      uv venv .venv --python 3.12
      source .venv/bin/activate
      uv pip install "amd-gaia[api,rag]"
    '
```

### 4.4 GCP Deployment

#### Cloud Run

```yaml
# service.yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: gaia-api
  annotations:
    run.googleapis.com/ingress: all
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/minScale: "1"
        autoscaling.knative.dev/maxScale: "10"
        run.googleapis.com/cpu-throttling: "false"
    spec:
      containerConcurrency: 10
      timeoutSeconds: 300
      containers:
        - image: gcr.io/PROJECT/gaia-api:latest
          ports:
            - containerPort: 8080
          resources:
            limits:
              cpu: "2"
              memory: 4Gi
          env:
            - name: GAIA_LOG_LEVEL
              value: "INFO"
            - name: GAIA_API_STREAMING
              value: "1"
            - name: ANTHROPIC_API_KEY
              valueFrom:
                secretKeyRef:
                  name: anthropic-key
                  key: latest
          startupProbe:
            httpGet:
              path: /health
              port: 8080
            initialDelaySeconds: 10
            periodSeconds: 5
            failureThreshold: 12
          livenessProbe:
            httpGet:
              path: /health
              port: 8080
            periodSeconds: 30
```

```bash
# Deploy to Cloud Run
gcloud run services replace service.yaml --region=us-central1
```

### 4.5 Cost Optimization Strategies

| Strategy | Implementation | Savings |
|----------|---------------|---------|
| **Right-size instances** | Start with 2 vCPU / 4 GB; scale if p99 latency > 5s | 30-50% |
| **Auto-scaling** | Scale to zero when idle (Cloud Run, Fargate) | 60-80% on dev/staging |
| **Spot/Preemptible** | Use for non-critical workloads (batch eval, testing) | 60-90% |
| **Reserved instances** | 1-year commit for steady-state production | 30-40% |
| **LLM cost** | Use local Lemonade Server on AMD hardware instead of cloud LLM APIs | 90%+ on LLM costs |
| **Caching** | Cache repeated LLM responses for identical prompts | 20-40% on LLM costs |
| **Model selection** | Use Qwen3-0.6B for simple tasks, Qwen3-Coder-30B only for complex | 50-70% on local compute |

---

## 5. Docker / Kubernetes

### 5.1 Dockerfile for GAIA API

```dockerfile
# Dockerfile
# GAIA V2 API Server
# Build: docker build -t gaia-api:latest .
# Run:   docker run -p 8080:8080 -e ANTHROPIC_API_KEY=... gaia-api:latest

# ============================================================
# Stage 1: Builder
# ============================================================
FROM python:3.12-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency resolution
RUN pip install --no-cache-dir uv

# Create app directory
WORKDIR /app

# Copy only dependency files first (layer caching)
COPY setup.py pyproject.toml README.md ./
COPY src/gaia/version.py src/gaia/version.py

# Create a minimal package structure for install
RUN mkdir -p src/gaia && touch src/gaia/__init__.py

# Install dependencies (cached unless setup.py changes)
RUN uv pip install --system ".[api,rag]"

# ============================================================
# Stage 2: Runtime
# ============================================================
FROM python:3.12-slim AS runtime

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -r -m -s /bin/bash -u 1000 gaia

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source
WORKDIR /app
COPY --chown=gaia:gaia src/ src/
COPY --chown=gaia:gaia setup.py pyproject.toml README.md ./

# Install GAIA in editable mode (uses pre-installed deps)
RUN pip install --no-cache-dir --no-deps -e "."

# Create directories for data and logs
RUN mkdir -p /app/data /app/logs /app/rag-collections && \
    chown -R gaia:gaia /app

# Switch to non-root user
USER gaia

# Environment defaults
ENV GAIA_LOG_LEVEL=INFO \
    GAIA_API_STREAMING=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Expose API port
EXPOSE 8080

# Start uvicorn server
CMD ["python", "-m", "uvicorn", \
     "gaia.api.openai_server:app", \
     "--host", "0.0.0.0", \
     "--port", "8080", \
     "--workers", "2", \
     "--log-level", "info", \
     "--access-log"]
```

### 5.2 Dockerfile for Lemonade Server

```dockerfile
# Dockerfile.lemonade
# Lemonade Server for AMD NPU/iGPU environments
# Requires AMD hardware passthrough (--device flags)

FROM ubuntu:22.04 AS runtime

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    build-essential curl wget \
    && rm -rf /var/lib/apt/lists/*

# Install Lemonade Server
# Note: actual installation depends on AMD's distribution method
RUN pip3 install --no-cache-dir lemonade-server

# Create non-root user
RUN useradd -r -m -s /bin/bash -u 1000 lemonade

# Create model cache directory
RUN mkdir -p /home/lemonade/.cache/lemonade && \
    chown -R lemonade:lemonade /home/lemonade

USER lemonade
WORKDIR /home/lemonade

# Expose Lemonade API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

CMD ["lemonade-server", "serve", "--host", "0.0.0.0", "--port", "8000"]
```

### 5.3 Docker Compose (Full Stack)

```yaml
# docker-compose.yml
# Full GAIA stack: API server + Lemonade backend + monitoring
#
# Usage:
#   docker compose up -d                    # Start all services
#   docker compose up -d gaia-api           # Start API only (uses cloud LLM)
#   docker compose logs -f gaia-api         # View logs
#   docker compose down                     # Stop all services

version: "3.9"

services:
  # ── Lemonade Server (LLM Backend) ──────────────────────────
  # Only needed when running local models on AMD hardware.
  # Skip this service when using cloud LLM providers.
  lemonade:
    build:
      context: .
      dockerfile: Dockerfile.lemonade
    container_name: gaia-lemonade
    restart: unless-stopped
    ports:
      - "8000:8000"
    volumes:
      - lemonade-cache:/home/lemonade/.cache/lemonade
    # AMD hardware passthrough (uncomment for NPU/iGPU)
    # devices:
    #   - /dev/accel0:/dev/accel0   # NPU device
    #   - /dev/dri:/dev/dri         # iGPU device
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/health"]
      interval: 30s
      timeout: 5s
      retries: 5
      start_period: 120s
    networks:
      - gaia-net
    deploy:
      resources:
        limits:
          memory: 16G
        reservations:
          memory: 4G

  # ── GAIA API Server ────────────────────────────────────────
  gaia-api:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: gaia-api
    restart: unless-stopped
    ports:
      - "8080:8080"
    environment:
      - LEMONADE_BASE_URL=http://lemonade:8000/api/v1
      - GAIA_LOG_LEVEL=INFO
      - GAIA_API_STREAMING=1
      # Cloud LLM providers (uncomment if not using Lemonade)
      # - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      # - OPENAI_API_KEY=${OPENAI_API_KEY}
    env_file:
      - .env
    volumes:
      - gaia-data:/app/data
      - gaia-logs:/app/logs
      - gaia-rag:/app/rag-collections
    depends_on:
      lemonade:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 5s
      retries: 3
      start_period: 60s
    networks:
      - gaia-net
    deploy:
      resources:
        limits:
          memory: 8G
        reservations:
          memory: 2G

  # ── nginx Reverse Proxy ────────────────────────────────────
  nginx:
    image: nginx:1.27-alpine
    container_name: gaia-nginx
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/certs:/etc/nginx/certs:ro
    depends_on:
      - gaia-api
    networks:
      - gaia-net

  # ── Prometheus (Metrics) ───────────────────────────────────
  prometheus:
    image: prom/prometheus:v2.53.0
    container_name: gaia-prometheus
    restart: unless-stopped
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus-data:/prometheus
    networks:
      - gaia-net

  # ── Grafana (Dashboards) ───────────────────────────────────
  grafana:
    image: grafana/grafana:11.1.0
    container_name: gaia-grafana
    restart: unless-stopped
    ports:
      - "3001:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD:-admin}
      - GF_USERS_ALLOW_SIGN_UP=false
    volumes:
      - grafana-data:/var/lib/grafana
      - ./monitoring/grafana/provisioning:/etc/grafana/provisioning:ro
      - ./monitoring/grafana/dashboards:/var/lib/grafana/dashboards:ro
    depends_on:
      - prometheus
    networks:
      - gaia-net

volumes:
  lemonade-cache:
  gaia-data:
  gaia-logs:
  gaia-rag:
  prometheus-data:
  grafana-data:

networks:
  gaia-net:
    driver: bridge
```

### 5.4 Kubernetes Manifests

#### Namespace

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: gaia
  labels:
    app.kubernetes.io/name: gaia
    app.kubernetes.io/part-of: gaia-platform
```

#### ConfigMap

```yaml
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: gaia-config
  namespace: gaia
data:
  GAIA_LOG_LEVEL: "INFO"
  GAIA_API_STREAMING: "1"
  LEMONADE_BASE_URL: "http://lemonade-service:8000/api/v1"
```

#### Secrets

```yaml
# k8s/secrets.yaml
# WARNING: In production, use an external secret manager (Vault, AWS SM, etc.)
# This file is for reference only -- do NOT commit real secrets.
apiVersion: v1
kind: Secret
metadata:
  name: gaia-secrets
  namespace: gaia
type: Opaque
stringData:
  ANTHROPIC_API_KEY: "sk-ant-REPLACE_ME"
  OPENAI_API_KEY: "sk-REPLACE_ME"
```

#### GAIA API Deployment

```yaml
# k8s/gaia-api-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gaia-api
  namespace: gaia
  labels:
    app: gaia-api
    version: v0.15.3
spec:
  replicas: 3
  selector:
    matchLabels:
      app: gaia-api
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  template:
    metadata:
      labels:
        app: gaia-api
        version: v0.15.3
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8080"
        prometheus.io/path: "/metrics"
    spec:
      serviceAccountName: gaia-api
      securityContext:
        runAsUser: 1000
        runAsGroup: 1000
        fsGroup: 1000
        runAsNonRoot: true
      containers:
        - name: gaia-api
          image: ghcr.io/amd/gaia-api:0.15.3
          imagePullPolicy: IfNotPresent
          ports:
            - name: http
              containerPort: 8080
              protocol: TCP
          envFrom:
            - configMapRef:
                name: gaia-config
            - secretRef:
                name: gaia-secrets
          resources:
            requests:
              cpu: "500m"
              memory: "1Gi"
            limits:
              cpu: "2000m"
              memory: "4Gi"
          startupProbe:
            httpGet:
              path: /health
              port: http
            initialDelaySeconds: 10
            periodSeconds: 5
            failureThreshold: 12
          livenessProbe:
            httpGet:
              path: /health
              port: http
            periodSeconds: 30
            timeoutSeconds: 5
            failureThreshold: 3
          readinessProbe:
            httpGet:
              path: /health
              port: http
            periodSeconds: 10
            timeoutSeconds: 3
            failureThreshold: 3
          volumeMounts:
            - name: gaia-data
              mountPath: /app/data
            - name: gaia-rag
              mountPath: /app/rag-collections
      volumes:
        - name: gaia-data
          persistentVolumeClaim:
            claimName: gaia-data-pvc
        - name: gaia-rag
          persistentVolumeClaim:
            claimName: gaia-rag-pvc
      topologySpreadConstraints:
        - maxSkew: 1
          topologyKey: kubernetes.io/hostname
          whenUnsatisfiable: DoNotSchedule
          labelSelector:
            matchLabels:
              app: gaia-api
```

#### Service

```yaml
# k8s/gaia-api-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: gaia-api-service
  namespace: gaia
  labels:
    app: gaia-api
spec:
  type: ClusterIP
  ports:
    - name: http
      port: 8080
      targetPort: http
      protocol: TCP
  selector:
    app: gaia-api
```

#### Ingress

```yaml
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: gaia-api-ingress
  namespace: gaia
  annotations:
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/proxy-body-size: "100m"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "300"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "300"
    nginx.ingress.kubernetes.io/proxy-buffering: "off"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    # Rate limiting
    nginx.ingress.kubernetes.io/limit-rps: "10"
    nginx.ingress.kubernetes.io/limit-burst-multiplier: "3"
spec:
  ingressClassName: nginx
  tls:
    - hosts:
        - gaia-api.example.com
      secretName: gaia-api-tls
  rules:
    - host: gaia-api.example.com
      http:
        paths:
          - path: /v1
            pathType: Prefix
            backend:
              service:
                name: gaia-api-service
                port:
                  number: 8080
          - path: /health
            pathType: Exact
            backend:
              service:
                name: gaia-api-service
                port:
                  number: 8080
```

#### Horizontal Pod Autoscaler

```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: gaia-api-hpa
  namespace: gaia
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: gaia-api
  minReplicas: 2
  maxReplicas: 20
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
        - type: Pods
          value: 2
          periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
        - type: Pods
          value: 1
          periodSeconds: 120
```

#### Lemonade Server StatefulSet (Optional)

```yaml
# k8s/lemonade-statefulset.yaml
# Only deploy this if running local LLM models on AMD hardware nodes.
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: lemonade
  namespace: gaia
spec:
  serviceName: lemonade-service
  replicas: 1
  selector:
    matchLabels:
      app: lemonade
  template:
    metadata:
      labels:
        app: lemonade
    spec:
      nodeSelector:
        # Schedule only on nodes with AMD NPU/GPU
        amd.com/npu: "true"
      containers:
        - name: lemonade
          image: ghcr.io/amd/lemonade-server:9.3.0
          ports:
            - containerPort: 8000
          resources:
            requests:
              cpu: "2000m"
              memory: "8Gi"
              # AMD device plugin resource (if available)
              # amd.com/npu: "1"
            limits:
              cpu: "4000m"
              memory: "16Gi"
          volumeMounts:
            - name: model-cache
              mountPath: /home/lemonade/.cache/lemonade
  volumeClaimTemplates:
    - metadata:
        name: model-cache
      spec:
        accessModes: ["ReadWriteOnce"]
        resources:
          requests:
            storage: 50Gi
---
apiVersion: v1
kind: Service
metadata:
  name: lemonade-service
  namespace: gaia
spec:
  type: ClusterIP
  ports:
    - port: 8000
      targetPort: 8000
  selector:
    app: lemonade
```

#### PersistentVolumeClaims

```yaml
# k8s/pvc.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: gaia-data-pvc
  namespace: gaia
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 10Gi
  storageClassName: standard
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: gaia-rag-pvc
  namespace: gaia
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 50Gi
  storageClassName: standard
```

#### ServiceAccount and RBAC

```yaml
# k8s/rbac.yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: gaia-api
  namespace: gaia
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: gaia-api-role
  namespace: gaia
rules:
  - apiGroups: [""]
    resources: ["configmaps"]
    verbs: ["get", "list"]
  - apiGroups: [""]
    resources: ["secrets"]
    verbs: ["get"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: gaia-api-rolebinding
  namespace: gaia
subjects:
  - kind: ServiceAccount
    name: gaia-api
    namespace: gaia
roleRef:
  kind: Role
  name: gaia-api-role
  apiGroup: rbac.authorization.k8s.io
```

#### NetworkPolicy

```yaml
# k8s/network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: gaia-api-netpol
  namespace: gaia
spec:
  podSelector:
    matchLabels:
      app: gaia-api
  policyTypes:
    - Ingress
    - Egress
  ingress:
    # Allow traffic from ingress controller
    - from:
        - namespaceSelector:
            matchLabels:
              kubernetes.io/metadata.name: ingress-nginx
      ports:
        - protocol: TCP
          port: 8080
    # Allow traffic from Prometheus
    - from:
        - namespaceSelector:
            matchLabels:
              kubernetes.io/metadata.name: monitoring
      ports:
        - protocol: TCP
          port: 8080
  egress:
    # Allow DNS
    - to: []
      ports:
        - protocol: UDP
          port: 53
        - protocol: TCP
          port: 53
    # Allow Lemonade Server
    - to:
        - podSelector:
            matchLabels:
              app: lemonade
      ports:
        - protocol: TCP
          port: 8000
    # Allow cloud LLM APIs (HTTPS)
    - to: []
      ports:
        - protocol: TCP
          port: 443
```

### 5.5 Helm Chart Structure

```
gaia-helm/
├── Chart.yaml
├── values.yaml
├── values-production.yaml
├── values-staging.yaml
├── templates/
│   ├── _helpers.tpl
│   ├── namespace.yaml
│   ├── configmap.yaml
│   ├── secret.yaml
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── ingress.yaml
│   ├── hpa.yaml
│   ├── pvc.yaml
│   ├── serviceaccount.yaml
│   ├── rbac.yaml
│   ├── networkpolicy.yaml
│   ├── lemonade-statefulset.yaml
│   ├── lemonade-service.yaml
│   └── tests/
│       └── test-connection.yaml
└── README.md
```

**Chart.yaml:**

```yaml
apiVersion: v2
name: gaia
description: GAIA V2 - AMD AI Agent Framework
type: application
version: 1.0.0
appVersion: "0.15.3"
keywords:
  - ai
  - agents
  - amd
  - llm
maintainers:
  - name: AMD GAIA Team
    url: https://github.com/amd/gaia
```

**values.yaml:**

```yaml
# Default values for GAIA Helm chart

replicaCount: 2

image:
  repository: ghcr.io/amd/gaia-api
  tag: "0.15.3"
  pullPolicy: IfNotPresent

service:
  type: ClusterIP
  port: 8080

ingress:
  enabled: true
  className: nginx
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/proxy-read-timeout: "300"
    nginx.ingress.kubernetes.io/proxy-buffering: "off"
  hosts:
    - host: gaia-api.example.com
      paths:
        - path: /v1
          pathType: Prefix
        - path: /health
          pathType: Exact
  tls:
    - secretName: gaia-api-tls
      hosts:
        - gaia-api.example.com

resources:
  requests:
    cpu: 500m
    memory: 1Gi
  limits:
    cpu: 2000m
    memory: 4Gi

autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 20
  targetCPUUtilization: 70
  targetMemoryUtilization: 80

config:
  logLevel: INFO
  streaming: "1"
  lemonadeUrl: "http://lemonade-service:8000/api/v1"

secrets:
  # Set via --set or external secret manager
  anthropicApiKey: ""
  openaiApiKey: ""

lemonade:
  enabled: false  # Enable if running local LLM on AMD hardware
  image:
    repository: ghcr.io/amd/lemonade-server
    tag: "9.3.0"
  resources:
    requests:
      cpu: 2000m
      memory: 8Gi
    limits:
      cpu: 4000m
      memory: 16Gi
  storage: 50Gi

persistence:
  data:
    enabled: true
    size: 10Gi
    storageClass: standard
  rag:
    enabled: true
    size: 50Gi
    storageClass: standard

networkPolicy:
  enabled: true

monitoring:
  prometheus:
    enabled: true
  grafana:
    enabled: false
```

```bash
# Install GAIA with Helm
helm install gaia ./gaia-helm \
  --namespace gaia --create-namespace \
  --set secrets.anthropicApiKey="sk-ant-..." \
  --values values-production.yaml

# Upgrade
helm upgrade gaia ./gaia-helm \
  --namespace gaia \
  --set image.tag="0.15.4" \
  --values values-production.yaml

# Rollback
helm rollback gaia 1 --namespace gaia
```

---

## 6. Database Migration

### 6.1 Current State: SQLite

GAIA uses Python's built-in `sqlite3` module via the `DatabaseMixin` class in `src/gaia/database/mixin.py`. This is a zero-dependency, file-based database suitable for single-user and small-team deployments.

```python
# Current DatabaseMixin usage pattern
from gaia.agents.base.agent import Agent
from gaia.database.mixin import DatabaseMixin

class MyAgent(Agent, DatabaseMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.init_db("data/app.db")  # SQLite file

        if not self.table_exists("items"):
            self.execute('''
                CREATE TABLE items (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
```

### 6.2 SQLite to PostgreSQL Migration

For multi-user server deployments, PostgreSQL provides concurrent access, connection pooling, and better durability.

#### Migration Strategy

```
Phase 1: Abstract database layer (DatabaseMixin already provides this)
Phase 2: Add PostgreSQL backend option
Phase 3: Provide migration tooling
```

#### PostgreSQL-Compatible DatabaseMixin Extension

```python
# src/gaia/database/pg_mixin.py (proposed)
"""PostgreSQL database mixin for production GAIA deployments."""

import logging
import os
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import psycopg2
    import psycopg2.pool
    import psycopg2.extras
    PG_AVAILABLE = True
except ImportError:
    PG_AVAILABLE = False


class PostgresMixin:
    """
    PostgreSQL mixin for production deployments.

    Drop-in replacement for DatabaseMixin with connection pooling.

    Environment variables:
        GAIA_DB_URL: PostgreSQL connection URL
        GAIA_DB_POOL_MIN: Minimum pool connections (default: 2)
        GAIA_DB_POOL_MAX: Maximum pool connections (default: 10)

    Example:
        class ProductionAgent(Agent, PostgresMixin):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.init_pg()
    """

    _pool = None

    def init_pg(self, dsn: str = None) -> None:
        """Initialize PostgreSQL connection pool."""
        if not PG_AVAILABLE:
            raise ImportError(
                "psycopg2 is required for PostgreSQL support. "
                "Install with: pip install psycopg2-binary"
            )

        dsn = dsn or os.getenv(
            "GAIA_DB_URL",
            "postgresql://gaia:gaia@localhost:5432/gaia"
        )
        min_conn = int(os.getenv("GAIA_DB_POOL_MIN", "2"))
        max_conn = int(os.getenv("GAIA_DB_POOL_MAX", "10"))

        self._pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=min_conn,
            maxconn=max_conn,
            dsn=dsn,
        )
        logger.info("PostgreSQL pool initialized: min=%d, max=%d", min_conn, max_conn)

    @contextmanager
    def _get_conn(self):
        """Get a connection from the pool."""
        conn = self._pool.getconn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._pool.putconn(conn)

    def execute(self, sql: str, params: tuple = None) -> List[Dict]:
        """Execute SQL and return results as list of dicts."""
        with self._get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, params)
                if cur.description:
                    return [dict(row) for row in cur.fetchall()]
                return []

    def insert(self, table: str, data: Dict[str, Any]) -> int:
        """Insert row and return ID."""
        columns = ", ".join(data.keys())
        placeholders = ", ".join(["%s"] * len(data))
        sql = f"INSERT INTO {table} ({columns}) VALUES ({placeholders}) RETURNING id"
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, tuple(data.values()))
                return cur.fetchone()[0]

    def close_pg(self) -> None:
        """Close all pool connections."""
        if self._pool:
            self._pool.closeall()
            self._pool = None
```

#### Migration Script

```python
#!/usr/bin/env python3
"""
migrate_sqlite_to_pg.py
Migrate GAIA SQLite databases to PostgreSQL.

Usage:
    python migrate_sqlite_to_pg.py \
        --sqlite data/app.db \
        --pg-url postgresql://gaia:gaia@localhost:5432/gaia
"""

import argparse
import sqlite3
import sys

try:
    import psycopg2
    import psycopg2.extras
except ImportError:
    print("ERROR: psycopg2 is required. Install with: pip install psycopg2-binary")
    sys.exit(1)


# SQLite type -> PostgreSQL type mapping
TYPE_MAP = {
    "INTEGER": "INTEGER",
    "TEXT": "TEXT",
    "REAL": "DOUBLE PRECISION",
    "BLOB": "BYTEA",
    "TIMESTAMP": "TIMESTAMP",
    "BOOLEAN": "BOOLEAN",
}


def get_sqlite_tables(sqlite_conn):
    """Get all user tables from SQLite."""
    cursor = sqlite_conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    )
    return [row[0] for row in cursor.fetchall()]


def get_sqlite_schema(sqlite_conn, table_name):
    """Get column info for a SQLite table."""
    cursor = sqlite_conn.execute(f"PRAGMA table_info({table_name})")
    return cursor.fetchall()


def sqlite_to_pg_type(sqlite_type):
    """Convert SQLite type to PostgreSQL type."""
    upper = (sqlite_type or "TEXT").upper()
    for key, value in TYPE_MAP.items():
        if key in upper:
            return value
    return "TEXT"


def migrate_table(sqlite_conn, pg_conn, table_name):
    """Migrate a single table from SQLite to PostgreSQL."""
    schema = get_sqlite_schema(sqlite_conn, table_name)

    # Build CREATE TABLE statement
    columns = []
    for col in schema:
        cid, name, col_type, notnull, default, pk = col
        pg_type = sqlite_to_pg_type(col_type)
        parts = [f'"{name}"']

        if pk:
            parts.append("SERIAL PRIMARY KEY")
        else:
            parts.append(pg_type)
            if notnull:
                parts.append("NOT NULL")
            if default is not None:
                parts.append(f"DEFAULT {default}")

        columns.append(" ".join(parts))

    create_sql = f'CREATE TABLE IF NOT EXISTS "{table_name}" ({", ".join(columns)})'

    with pg_conn.cursor() as cur:
        cur.execute(create_sql)
        pg_conn.commit()

    # Migrate data
    rows = sqlite_conn.execute(f"SELECT * FROM {table_name}").fetchall()
    if not rows:
        print(f"  {table_name}: 0 rows (empty table)")
        return

    col_names = [col[1] for col in schema if not col[5]]  # Skip PK for SERIAL
    if not col_names:
        col_names = [col[1] for col in schema]

    placeholders = ", ".join(["%s"] * len(col_names))
    insert_sql = f'INSERT INTO "{table_name}" ({", ".join(col_names)}) VALUES ({placeholders})'

    with pg_conn.cursor() as cur:
        for row in rows:
            # Skip PK column if using SERIAL
            values = list(row[1:]) if schema[0][5] else list(row)
            cur.execute(insert_sql, values)
        pg_conn.commit()

    print(f"  {table_name}: {len(rows)} rows migrated")


def main():
    parser = argparse.ArgumentParser(description="Migrate GAIA SQLite to PostgreSQL")
    parser.add_argument("--sqlite", required=True, help="Path to SQLite database")
    parser.add_argument("--pg-url", required=True, help="PostgreSQL connection URL")
    parser.add_argument("--dry-run", action="store_true", help="Print SQL without executing")
    args = parser.parse_args()

    sqlite_conn = sqlite3.connect(args.sqlite)
    pg_conn = psycopg2.connect(args.pg_url)

    tables = get_sqlite_tables(sqlite_conn)
    print(f"Found {len(tables)} tables: {', '.join(tables)}")

    for table in tables:
        migrate_table(sqlite_conn, pg_conn, table)

    sqlite_conn.close()
    pg_conn.close()
    print("Migration complete.")


if __name__ == "__main__":
    main()
```

### 6.3 Connection Pooling Setup

For server deployments with PostgreSQL, use PgBouncer as a connection pooler:

```ini
# /etc/pgbouncer/pgbouncer.ini
[databases]
gaia = host=localhost port=5432 dbname=gaia

[pgbouncer]
listen_port = 6432
listen_addr = 127.0.0.1
auth_type = scram-sha-256
auth_file = /etc/pgbouncer/userlist.txt
pool_mode = transaction
default_pool_size = 20
max_client_conn = 200
min_pool_size = 5
reserve_pool_size = 5
reserve_pool_timeout = 3
server_lifetime = 3600
server_idle_timeout = 600
log_connections = 1
log_disconnections = 1
```

```bash
# /etc/pgbouncer/userlist.txt
"gaia" "SCRAM-SHA-256$4096:salt$stored_key:server_key"
```

Application connects to PgBouncer instead of PostgreSQL directly:

```bash
# .env
GAIA_DB_URL=postgresql://gaia:password@localhost:6432/gaia
GAIA_DB_POOL_MIN=2
GAIA_DB_POOL_MAX=10
```

### 6.4 Backup and Recovery

#### SQLite Backup

```bash
#!/bin/bash
# backup_sqlite.sh -- Daily SQLite backup
BACKUP_DIR="/opt/gaia/backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "$BACKUP_DIR"

# Online backup using SQLite .backup command
for db in /opt/gaia/data/*.db; do
    BASENAME=$(basename "$db" .db)
    sqlite3 "$db" ".backup '${BACKUP_DIR}/${BASENAME}_${TIMESTAMP}.db'"
done

# Compress and rotate (keep 30 days)
find "$BACKUP_DIR" -name "*.db" -mtime +30 -delete
gzip "${BACKUP_DIR}"/*_${TIMESTAMP}.db

echo "Backup complete: ${BACKUP_DIR}"
```

#### PostgreSQL Backup

```bash
#!/bin/bash
# backup_postgres.sh -- Daily PostgreSQL backup
BACKUP_DIR="/opt/gaia/backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PG_URL="${GAIA_DB_URL:-postgresql://gaia:gaia@localhost:5432/gaia}"

mkdir -p "$BACKUP_DIR"

# Full dump
pg_dump "$PG_URL" \
    --format=custom \
    --compress=9 \
    --file="${BACKUP_DIR}/gaia_${TIMESTAMP}.dump"

# Rotate (keep 30 days)
find "$BACKUP_DIR" -name "gaia_*.dump" -mtime +30 -delete

echo "Backup complete: ${BACKUP_DIR}/gaia_${TIMESTAMP}.dump"
```

```bash
# Restore from backup
pg_restore \
    --dbname=postgresql://gaia:gaia@localhost:5432/gaia_restored \
    --clean --if-exists \
    /opt/gaia/backups/gaia_20260207_120000.dump
```

Add to cron:

```bash
# /etc/cron.d/gaia-backup
0 2 * * * gaia /opt/gaia/scripts/backup_postgres.sh >> /opt/gaia/logs/backup.log 2>&1
```

---

## 7. Monitoring Setup

### 7.1 Prometheus Configuration

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
            - alertmanager:9093

scrape_configs:
  # GAIA API server metrics
  - job_name: "gaia-api"
    metrics_path: /metrics
    static_configs:
      - targets:
          - "gaia-api:8080"
        labels:
          service: gaia-api
    # For Kubernetes service discovery:
    # kubernetes_sd_configs:
    #   - role: pod
    #     namespaces:
    #       names: [gaia]
    # relabel_configs:
    #   - source_labels: [__meta_kubernetes_pod_label_app]
    #     regex: gaia-api
    #     action: keep

  # Lemonade Server metrics
  - job_name: "lemonade"
    static_configs:
      - targets:
          - "lemonade:8000"
        labels:
          service: lemonade

  # Node exporter (system metrics)
  - job_name: "node"
    static_configs:
      - targets:
          - "node-exporter:9100"

  # nginx metrics (requires nginx-prometheus-exporter)
  - job_name: "nginx"
    static_configs:
      - targets:
          - "nginx-exporter:9113"
```

### 7.2 Alert Rules

```yaml
# monitoring/alert_rules.yml
groups:
  - name: gaia-api-alerts
    rules:
      # API server is down
      - alert: GaiaAPIDown
        expr: up{job="gaia-api"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "GAIA API server is down"
          description: "{{ $labels.instance }} has been down for more than 1 minute."

      # High error rate
      - alert: GaiaHighErrorRate
        expr: |
          rate(http_requests_total{job="gaia-api",status=~"5.."}[5m])
          / rate(http_requests_total{job="gaia-api"}[5m]) > 0.05
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "GAIA API high error rate"
          description: "Error rate is {{ $value | humanizePercentage }} over the last 5 minutes."

      # High latency
      - alert: GaiaHighLatency
        expr: |
          histogram_quantile(0.99,
            rate(http_request_duration_seconds_bucket{job="gaia-api"}[5m])
          ) > 30
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "GAIA API high latency (p99)"
          description: "p99 latency is {{ $value | humanizeDuration }}."

      # Memory usage
      - alert: GaiaHighMemoryUsage
        expr: |
          process_resident_memory_bytes{job="gaia-api"}
          / (4 * 1024 * 1024 * 1024) > 0.85
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "GAIA API high memory usage"
          description: "Memory usage is above 85% of limit."

  - name: lemonade-alerts
    rules:
      # Lemonade server is down
      - alert: LemonadeDown
        expr: up{job="lemonade"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Lemonade Server is down"
          description: "LLM backend {{ $labels.instance }} has been down for more than 1 minute."

      # Lemonade high inference latency
      - alert: LemonadeHighLatency
        expr: |
          histogram_quantile(0.95,
            rate(inference_duration_seconds_bucket{job="lemonade"}[5m])
          ) > 60
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Lemonade inference latency is high"
          description: "p95 inference latency is {{ $value | humanizeDuration }}."
```

### 7.3 Custom GAIA Metrics Middleware

To expose Prometheus metrics from the GAIA API server, add a metrics middleware:

```python
# src/gaia/api/metrics.py (proposed)
"""Prometheus metrics for GAIA API server."""

import time
from typing import Callable

from fastapi import FastAPI, Request, Response
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

# Metrics
REQUEST_COUNT = Counter(
    "gaia_http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)

REQUEST_DURATION = Histogram(
    "gaia_http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
    buckets=[0.1, 0.5, 1, 2, 5, 10, 30, 60, 120, 300],
)

ACTIVE_REQUESTS = Gauge(
    "gaia_active_requests",
    "Number of active requests",
)

AGENT_INVOCATIONS = Counter(
    "gaia_agent_invocations_total",
    "Total agent invocations",
    ["agent_type", "status"],
)

AGENT_STEPS = Histogram(
    "gaia_agent_steps",
    "Number of steps per agent invocation",
    ["agent_type"],
    buckets=[1, 2, 3, 5, 10, 15, 20, 50],
)

LLM_TOKENS = Counter(
    "gaia_llm_tokens_total",
    "Total LLM tokens consumed",
    ["direction"],  # "prompt" or "completion"
)


def add_metrics(app: FastAPI) -> None:
    """Add Prometheus metrics middleware to FastAPI app."""

    @app.middleware("http")
    async def metrics_middleware(request: Request, call_next: Callable) -> Response:
        ACTIVE_REQUESTS.inc()
        start_time = time.time()

        response = await call_next(request)

        duration = time.time() - start_time
        endpoint = request.url.path
        method = request.method
        status = response.status_code

        REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=status).inc()
        REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(duration)
        ACTIVE_REQUESTS.dec()

        return response

    @app.get("/metrics")
    async def metrics():
        """Prometheus metrics endpoint."""
        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST,
        )
```

### 7.4 Grafana Dashboard Template

```json
{
  "dashboard": {
    "title": "GAIA API Server",
    "uid": "gaia-api-overview",
    "timezone": "browser",
    "refresh": "30s",
    "panels": [
      {
        "title": "Request Rate",
        "type": "timeseries",
        "gridPos": { "h": 8, "w": 12, "x": 0, "y": 0 },
        "targets": [
          {
            "expr": "rate(gaia_http_requests_total{job=\"gaia-api\"}[5m])",
            "legendFormat": "{{method}} {{endpoint}} {{status}}"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "stat",
        "gridPos": { "h": 4, "w": 6, "x": 12, "y": 0 },
        "targets": [
          {
            "expr": "rate(gaia_http_requests_total{job=\"gaia-api\",status=~\"5..\"}[5m]) / rate(gaia_http_requests_total{job=\"gaia-api\"}[5m])",
            "legendFormat": "Error Rate"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "percentunit",
            "thresholds": {
              "steps": [
                { "value": 0, "color": "green" },
                { "value": 0.01, "color": "yellow" },
                { "value": 0.05, "color": "red" }
              ]
            }
          }
        }
      },
      {
        "title": "Request Latency (p50 / p95 / p99)",
        "type": "timeseries",
        "gridPos": { "h": 8, "w": 12, "x": 0, "y": 8 },
        "targets": [
          {
            "expr": "histogram_quantile(0.50, rate(gaia_http_request_duration_seconds_bucket{job=\"gaia-api\"}[5m]))",
            "legendFormat": "p50"
          },
          {
            "expr": "histogram_quantile(0.95, rate(gaia_http_request_duration_seconds_bucket{job=\"gaia-api\"}[5m]))",
            "legendFormat": "p95"
          },
          {
            "expr": "histogram_quantile(0.99, rate(gaia_http_request_duration_seconds_bucket{job=\"gaia-api\"}[5m]))",
            "legendFormat": "p99"
          }
        ]
      },
      {
        "title": "Active Requests",
        "type": "gauge",
        "gridPos": { "h": 4, "w": 6, "x": 12, "y": 4 },
        "targets": [
          {
            "expr": "gaia_active_requests{job=\"gaia-api\"}",
            "legendFormat": "Active"
          }
        ]
      },
      {
        "title": "Agent Invocations by Type",
        "type": "timeseries",
        "gridPos": { "h": 8, "w": 12, "x": 12, "y": 8 },
        "targets": [
          {
            "expr": "rate(gaia_agent_invocations_total{job=\"gaia-api\"}[5m])",
            "legendFormat": "{{agent_type}} ({{status}})"
          }
        ]
      },
      {
        "title": "LLM Token Usage",
        "type": "timeseries",
        "gridPos": { "h": 8, "w": 12, "x": 0, "y": 16 },
        "targets": [
          {
            "expr": "rate(gaia_llm_tokens_total{job=\"gaia-api\"}[5m])",
            "legendFormat": "{{direction}} tokens/sec"
          }
        ]
      },
      {
        "title": "Memory Usage",
        "type": "timeseries",
        "gridPos": { "h": 8, "w": 12, "x": 12, "y": 16 },
        "targets": [
          {
            "expr": "process_resident_memory_bytes{job=\"gaia-api\"} / 1024 / 1024",
            "legendFormat": "{{instance}} RSS (MB)"
          }
        ]
      }
    ]
  }
}
```

### 7.5 Log Aggregation (ELK Stack)

#### Filebeat Configuration

```yaml
# /etc/filebeat/filebeat.yml
filebeat.inputs:
  # GAIA API logs
  - type: log
    enabled: true
    paths:
      - /opt/gaia/logs/gaia-api-*.log
    fields:
      service: gaia-api
    fields_under_root: true
    multiline:
      pattern: '^[0-9]{4}-[0-9]{2}-[0-9]{2}'
      negate: true
      match: after

  # Lemonade Server logs
  - type: log
    enabled: true
    paths:
      - /opt/gaia/logs/lemonade*.log
    fields:
      service: lemonade
    fields_under_root: true

  # nginx access logs
  - type: log
    enabled: true
    paths:
      - /var/log/nginx/access.log
    fields:
      service: nginx
    fields_under_root: true

output.elasticsearch:
  hosts: ["elasticsearch:9200"]
  index: "gaia-%{+yyyy.MM.dd}"

setup.kibana:
  host: "kibana:5601"

processors:
  - add_host_metadata: ~
  - add_docker_metadata: ~
```

#### Structured Logging Configuration

Configure GAIA to emit JSON logs for easier parsing:

```python
# Proposed: src/gaia/logging_config.py
import json
import logging
import sys


class JSONFormatter(logging.Formatter):
    """JSON log formatter for production deployments."""

    def format(self, record):
        log_entry = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry)


def configure_production_logging():
    """Configure JSON logging for production."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())

    root_logger = logging.getLogger("gaia")
    root_logger.handlers = [handler]
    root_logger.setLevel(logging.INFO)
```

---

## 8. Security Hardening

### 8.1 TLS/SSL Configuration

#### Self-Signed Certificates (Development/Testing)

```bash
# Generate self-signed certificates for development
mkdir -p /opt/gaia/certs

openssl req -x509 -nodes -days 365 \
    -newkey rsa:2048 \
    -keyout /opt/gaia/certs/gaia.key \
    -out /opt/gaia/certs/gaia.crt \
    -subj "/C=US/ST=California/O=AMD/CN=gaia-api.local"

chmod 600 /opt/gaia/certs/gaia.key
```

#### Let's Encrypt (Production)

```bash
# Install certbot
sudo apt install certbot python3-certbot-nginx

# Obtain certificate
sudo certbot --nginx -d gaia-api.example.com

# Auto-renewal (certbot adds cron automatically)
sudo certbot renew --dry-run
```

#### Uvicorn Direct TLS (No Reverse Proxy)

```bash
# Run GAIA API with TLS directly
python -m uvicorn gaia.api.openai_server:app \
    --host 0.0.0.0 \
    --port 8443 \
    --ssl-keyfile /opt/gaia/certs/gaia.key \
    --ssl-certfile /opt/gaia/certs/gaia.crt
```

### 8.2 Network Policies

#### Host-Level Firewall (iptables)

```bash
# Allow SSH
sudo iptables -A INPUT -p tcp --dport 22 -j ACCEPT

# Allow HTTPS (nginx)
sudo iptables -A INPUT -p tcp --dport 443 -j ACCEPT

# Allow HTTP (redirect to HTTPS)
sudo iptables -A INPUT -p tcp --dport 80 -j ACCEPT

# Allow Prometheus scraping from monitoring subnet
sudo iptables -A INPUT -s 10.0.10.0/24 -p tcp --dport 8080 -j ACCEPT

# Block direct access to GAIA API and Lemonade from outside
sudo iptables -A INPUT -p tcp --dport 8080 -j DROP
sudo iptables -A INPUT -p tcp --dport 8081 -j DROP
sudo iptables -A INPUT -p tcp --dport 8082 -j DROP
sudo iptables -A INPUT -p tcp --dport 8083 -j DROP
sudo iptables -A INPUT -p tcp --dport 8000 -j DROP

# Block everything else
sudo iptables -A INPUT -j DROP

# Save rules
sudo iptables-save > /etc/iptables/rules.v4
```

#### Kubernetes NetworkPolicy

See Section 5.4 for the complete Kubernetes NetworkPolicy manifest.

### 8.3 Secret Management

#### Local Deployment

Use `.env` files with restricted permissions:

```bash
# Create .env with restricted permissions
touch /opt/gaia/.env
chmod 600 /opt/gaia/.env
chown gaia:gaia /opt/gaia/.env

# Never commit .env to version control
echo ".env" >> .gitignore
```

#### Server / Cloud Deployment

| Platform | Secret Store | Integration |
|----------|-------------|-------------|
| AWS | AWS Secrets Manager | ECS task secrets, Lambda env |
| Azure | Azure Key Vault | Container Instance env, VM env |
| GCP | Secret Manager | Cloud Run secrets, GCE env |
| Kubernetes | K8s Secrets + External Secrets Operator | Pod env injection |
| HashiCorp | Vault | Agent sidecar injection |

**Kubernetes External Secrets Operator Example:**

```yaml
# k8s/external-secret.yaml
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: gaia-secrets
  namespace: gaia
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: aws-secretsmanager
    kind: ClusterSecretStore
  target:
    name: gaia-secrets
    creationPolicy: Owner
  data:
    - secretKey: ANTHROPIC_API_KEY
      remoteRef:
        key: gaia/production/anthropic
        property: api_key
    - secretKey: OPENAI_API_KEY
      remoteRef:
        key: gaia/production/openai
        property: api_key
```

### 8.4 API Authentication

The GAIA API server currently accepts all requests (CORS allows all origins). For production, add authentication at the reverse proxy or middleware level.

#### nginx API Key Authentication

```nginx
# API key validation in nginx
# Store valid keys in a file
# /etc/nginx/gaia_api_keys.conf
# map format: "Bearer KEY" 1;

map $http_authorization $is_valid_key {
    default 0;
    "Bearer gaia-prod-key-abc123" 1;
    "Bearer gaia-prod-key-def456" 1;
}

server {
    # ... (TLS config from Section 3.3)

    location /v1/ {
        # Require valid API key
        if ($is_valid_key = 0) {
            return 401 '{"error": "Invalid or missing API key"}';
        }

        proxy_pass http://gaia_backend/v1/;
        # ... (rest of proxy config)
    }

    # Health check (no auth)
    location /health {
        proxy_pass http://gaia_backend/health;
    }
}
```

#### FastAPI Middleware Authentication (Application Level)

```python
# src/gaia/api/auth.py (proposed)
"""API authentication middleware for production GAIA deployments."""

import hashlib
import hmac
import logging
import os
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

logger = logging.getLogger(__name__)
security = HTTPBearer(auto_error=False)

# Load valid API keys from environment
# Format: comma-separated keys
VALID_KEYS = set(
    k.strip()
    for k in os.getenv("GAIA_API_KEYS", "").split(",")
    if k.strip()
)

# Paths that do not require authentication
PUBLIC_PATHS = {"/health", "/metrics", "/docs", "/openapi.json"}


def verify_api_key(key: str) -> bool:
    """Constant-time API key verification."""
    for valid_key in VALID_KEYS:
        if hmac.compare_digest(key, valid_key):
            return True
    return False


def add_auth_middleware(app: FastAPI) -> None:
    """Add API key authentication middleware."""

    if not VALID_KEYS:
        logger.warning(
            "GAIA_API_KEYS not set -- API authentication DISABLED. "
            "Set GAIA_API_KEYS environment variable for production."
        )
        return

    @app.middleware("http")
    async def auth_middleware(request: Request, call_next):
        # Skip auth for public paths
        if request.url.path in PUBLIC_PATHS:
            return await call_next(request)

        # Extract Bearer token
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing API key")

        token = auth_header[7:]  # Strip "Bearer "
        if not verify_api_key(token):
            raise HTTPException(status_code=401, detail="Invalid API key")

        return await call_next(request)

    logger.info("API authentication enabled with %d valid keys", len(VALID_KEYS))
```

### 8.5 Security Checklist

| Category | Item | Local | Server | Cloud |
|----------|------|-------|--------|-------|
| **Network** | TLS/HTTPS | Optional | Required | Required |
| **Network** | Firewall rules | OS firewall | iptables/nftables | Security Groups |
| **Network** | API only on localhost | Yes | Behind proxy | Behind LB |
| **Auth** | API key validation | N/A | Required | Required |
| **Auth** | Rate limiting | N/A | Required | Required |
| **Secrets** | .env file permissions | 600 | 600 | Secret Manager |
| **Secrets** | No secrets in images | N/A | Yes | Yes |
| **Runtime** | Non-root user | Recommended | Required | Required |
| **Runtime** | Read-only filesystem | N/A | Recommended | Recommended |
| **Runtime** | Resource limits | N/A | systemd limits | K8s limits |
| **Data** | Database encryption | N/A | At-rest + in-transit | Cloud KMS |
| **Data** | Backup encryption | N/A | GPG/AES | Cloud KMS |
| **Audit** | Access logs | N/A | nginx logs | Cloud audit |
| **Audit** | Request logging | Debug mode | Structured JSON | Cloud Logging |
| **Updates** | Dependency scanning | `pip-audit` | CI pipeline | CI pipeline |
| **Updates** | Image scanning | N/A | Trivy/Snyk | ECR/ACR scanning |

---

## 9. Scaling Strategy

### 9.1 Horizontal Scaling (Multiple Agent Instances)

GAIA's OpenAI-compatible API follows the stateless request pattern: each POST `/v1/chat/completions` includes the full conversation history in the `messages` array. This means any API instance can handle any request.

```
                         ┌─────────────────┐
                         │  Load Balancer   │
                         │  (nginx / ALB)   │
                         └────┬───┬───┬─────┘
                              │   │   │
                    ┌─────────┘   │   └─────────┐
                    │             │             │
              ┌─────▼──┐   ┌─────▼──┐   ┌──────▼─┐
              │ API #1 │   │ API #2 │   │ API #3 │
              │ 2 CPU  │   │ 2 CPU  │   │ 2 CPU  │
              │ 4 GB   │   │ 4 GB   │   │ 4 GB   │
              └────┬───┘   └────┬───┘   └────┬───┘
                   │            │            │
                   └────────────┼────────────┘
                                │
                    ┌───────────▼───────────┐
                    │   LLM Backend         │
                    │   (Lemonade / Claude)  │
                    └───────────────────────┘
```

**Scaling triggers:**

| Metric | Scale Up | Scale Down |
|--------|----------|------------|
| CPU utilization | > 70% for 60s | < 30% for 300s |
| Memory utilization | > 80% for 60s | < 40% for 300s |
| Request queue depth | > 10 pending | 0 pending for 300s |
| Response latency (p95) | > 10s for 120s | < 2s for 300s |

### 9.2 Vertical Scaling (GPU/NPU Allocation)

For deployments using Lemonade Server with AMD hardware:

| Workload | Recommended Hardware | Model | Throughput |
|----------|---------------------|-------|------------|
| Light (chat, simple Q&A) | Ryzen AI 300 (NPU only) | Qwen3-0.6B | ~50 tokens/sec |
| Medium (code gen, RAG) | Ryzen AI MAX (NPU + iGPU) | Qwen3-Coder-30B-A3B | ~30 tokens/sec |
| Heavy (batch eval, multi-agent) | Instinct MI300X | Large models | ~200 tokens/sec |

For cloud LLM backends, vertical scaling means selecting higher-tier API plans or higher concurrency limits.

### 9.3 Queue-Based Work Distribution

For long-running agent tasks (code generation, multi-step workflows), use a message queue to decouple the API layer from agent execution:

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  Client   │───▶│  API     │───▶│  Queue   │───▶│  Worker  │
│           │◀───│  Server  │◀───│  (Redis) │◀───│  Pool    │
└──────────┘    └──────────┘    └──────────┘    └──────────┘
                                                      │
                                                ┌─────▼─────┐
                                                │  LLM      │
                                                │  Backend   │
                                                └───────────┘
```

#### Redis Queue Implementation

```python
# worker.py (proposed)
"""Queue-based worker for async agent execution."""

import json
import os
import redis
import time

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
QUEUE_NAME = "gaia:tasks"
RESULT_PREFIX = "gaia:result:"
RESULT_TTL = 3600  # 1 hour

r = redis.from_url(REDIS_URL)


def process_task(task_data: dict) -> dict:
    """Process a single agent task."""
    from gaia.agents.chat.agent import ChatAgent

    agent = ChatAgent(
        use_claude=task_data.get("use_claude", False),
        silent_mode=True,
        max_steps=task_data.get("max_steps", 20),
        skip_lemonade=task_data.get("skip_lemonade", False),
    )

    result = agent.process_query(task_data["query"])
    return {"status": "completed", "result": result}


def worker_loop():
    """Main worker loop -- blocks on queue."""
    print(f"Worker started, listening on {QUEUE_NAME}")
    while True:
        # BRPOP blocks until a task is available
        _, raw = r.brpop(QUEUE_NAME)
        task = json.loads(raw)
        task_id = task["task_id"]

        print(f"Processing task {task_id}")
        try:
            result = process_task(task)
            r.setex(
                f"{RESULT_PREFIX}{task_id}",
                RESULT_TTL,
                json.dumps(result),
            )
        except Exception as e:
            r.setex(
                f"{RESULT_PREFIX}{task_id}",
                RESULT_TTL,
                json.dumps({"status": "error", "error": str(e)}),
            )


if __name__ == "__main__":
    worker_loop()
```

#### API Endpoint for Async Tasks

```python
# Proposed: async task submission endpoint
import uuid

from fastapi import APIRouter

router = APIRouter()


@router.post("/v1/tasks")
async def submit_task(request: dict):
    """Submit an async agent task."""
    task_id = str(uuid.uuid4())
    task = {
        "task_id": task_id,
        "query": request["query"],
        "model": request.get("model", "gaia-code"),
        "use_claude": request.get("use_claude", False),
        "max_steps": request.get("max_steps", 20),
    }
    r.lpush("gaia:tasks", json.dumps(task))
    return {"task_id": task_id, "status": "queued"}


@router.get("/v1/tasks/{task_id}")
async def get_task_result(task_id: str):
    """Poll for task result."""
    result = r.get(f"gaia:result:{task_id}")
    if result is None:
        return {"task_id": task_id, "status": "pending"}
    return json.loads(result)
```

### 9.4 Stateless Agent Design

GAIA agents maintain state during a single `process_query()` invocation but are stateless across HTTP requests. This is because:

1. The OpenAI API pattern sends full `messages` history with each request
2. Agent `conversation_history` is rebuilt from the request each time
3. Tool results and intermediate state live within a single request lifecycle

**Recommendations for stateless operation:**

- **Do not** rely on in-memory state between requests
- **Do** send full conversation history in each request (OpenAI pattern)
- **Do** use external storage (database, object store) for persistent data
- **Do** use RAG collections stored on shared volumes for multi-instance access
- **Do** use Redis for session state if sticky sessions are undesirable

---

## 10. CI/CD Pipeline

### 10.1 Current GAIA CI/CD

GAIA uses GitHub Actions with the following existing workflows:

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| `test_unit.yml` | push, PR to main | Unit tests, packaging validation, CLI dry-run |
| `lint.yml` | push, PR | Black, isort, flake8, mypy, bandit |
| `test_api.yml` | push, PR | API server integration tests |
| `test_chat_agent.yml` | push, PR | ChatAgent tests |
| `test_code_agent.yml` | push, PR | CodeAgent tests |
| `test_rag.yml` | push, PR | RAG SDK tests |
| `test_mcp.yml` | push, PR | MCP protocol tests |
| `test_security.yml` | push, PR | Security scanning |
| `pypi.yml` | tag push | Build, verify, publish to PyPI |
| `build-electron-apps.yml` | release | Build Electron desktop apps |
| `publish_installer.yml` | release | Publish Windows/Linux installers |
| `docs.yml` | push | Build and deploy Mintlify docs |

### 10.2 Production Build Pipeline

```yaml
# .github/workflows/build-docker.yml (proposed)
name: Build Docker Image

on:
  push:
    branches: [main]
    tags: ["v*"]
  pull_request:
    branches: [main]

permissions:
  contents: read
  packages: write

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: amd/gaia-api

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v6

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Log in to GitHub Container Registry
        if: github.event_name != 'pull_request'
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: |
            type=ref,event=branch
            type=ref,event=pr
            type=semver,pattern={{version}}
            type=semver,pattern={{major}}.{{minor}}
            type=sha

      - name: Build and push
        uses: docker/build-push-action@v6
        with:
          context: .
          push: ${{ github.event_name != 'pull_request' }}
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
          platforms: linux/amd64

      - name: Scan image for vulnerabilities
        if: github.event_name != 'pull_request'
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:sha-${{ github.sha }}
          format: 'sarif'
          output: 'trivy-results.sarif'
          severity: 'CRITICAL,HIGH'

      - name: Upload scan results
        if: github.event_name != 'pull_request'
        uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: 'trivy-results.sarif'
```

### 10.3 Testing Pipeline

```yaml
# .github/workflows/test-integration.yml (proposed)
name: Integration Tests

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  api-integration:
    runs-on: ubuntu-latest
    services:
      redis:
        image: redis:7-alpine
        ports:
          - 6379:6379
    steps:
      - uses: actions/checkout@v6

      - name: Set up Python
        uses: actions/setup-python@v6
        with:
          python-version: '3.12'

      - name: Install uv
        run: curl -LsSf https://astral.sh/uv/install.sh | sh

      - name: Install dependencies
        run: uv pip install --system -e ".[dev,api,rag]"

      - name: Start GAIA API server
        run: |
          python -m uvicorn gaia.api.openai_server:app \
            --host 127.0.0.1 --port 8080 &
          sleep 5

      - name: Test health endpoint
        run: curl -f http://localhost:8080/health

      - name: Test models endpoint
        run: |
          curl -s http://localhost:8080/v1/models | python -m json.tool

      - name: Run API tests
        run: pytest tests/integration/ -v --tb=short

  docker-smoke:
    runs-on: ubuntu-latest
    needs: [api-integration]
    steps:
      - uses: actions/checkout@v6

      - name: Build Docker image
        run: docker build -t gaia-api:test .

      - name: Start container
        run: |
          docker run -d --name gaia-test \
            -p 8080:8080 \
            -e GAIA_LOG_LEVEL=DEBUG \
            gaia-api:test
          sleep 10

      - name: Test health
        run: curl -f http://localhost:8080/health

      - name: Check logs
        if: failure()
        run: docker logs gaia-test

      - name: Cleanup
        if: always()
        run: docker rm -f gaia-test
```

### 10.4 Deployment Pipeline

```yaml
# .github/workflows/deploy.yml (proposed)
name: Deploy to Production

on:
  workflow_dispatch:
    inputs:
      environment:
        description: 'Target environment'
        required: true
        type: choice
        options:
          - staging
          - production
      version:
        description: 'Image tag to deploy'
        required: true
        type: string

jobs:
  deploy:
    runs-on: ubuntu-latest
    environment: ${{ inputs.environment }}
    steps:
      - uses: actions/checkout@v6

      - name: Configure kubectl
        uses: azure/setup-kubectl@v4

      - name: Set kubeconfig
        run: |
          echo "${{ secrets.KUBECONFIG }}" | base64 -d > /tmp/kubeconfig
          export KUBECONFIG=/tmp/kubeconfig

      - name: Deploy to Kubernetes
        run: |
          # Update image tag in deployment
          kubectl set image deployment/gaia-api \
            gaia-api=ghcr.io/amd/gaia-api:${{ inputs.version }} \
            --namespace gaia \
            --record

          # Wait for rollout
          kubectl rollout status deployment/gaia-api \
            --namespace gaia \
            --timeout=300s

      - name: Verify deployment
        run: |
          # Wait for pods to be ready
          kubectl wait --for=condition=Ready pod \
            -l app=gaia-api \
            --namespace gaia \
            --timeout=120s

          # Test health endpoint via port-forward
          kubectl port-forward svc/gaia-api-service 8080:8080 \
            --namespace gaia &
          sleep 5
          curl -f http://localhost:8080/health

      - name: Notify on failure
        if: failure()
        run: |
          echo "Deployment of ${{ inputs.version }} to ${{ inputs.environment }} FAILED"
          # Add Slack/Teams notification here
```

### 10.5 Rollback Strategy

```bash
# Kubernetes rollback
kubectl rollout undo deployment/gaia-api --namespace gaia

# Rollback to specific revision
kubectl rollout history deployment/gaia-api --namespace gaia
kubectl rollout undo deployment/gaia-api --namespace gaia --to-revision=3

# Helm rollback
helm rollback gaia 1 --namespace gaia

# Docker Compose rollback
# Update docker-compose.yml image tag to previous version
docker compose pull gaia-api
docker compose up -d gaia-api

# Blue-green rollback (switch traffic back to old deployment)
kubectl patch service gaia-api-service \
    -p '{"spec":{"selector":{"version":"v0.15.2"}}}' \
    --namespace gaia
```

### 10.6 Release Checklist

```
Pre-release:
  [ ] All CI checks pass (unit, integration, lint, security)
  [ ] version.py updated (__version__ = "X.Y.Z")
  [ ] CHANGELOG updated
  [ ] Documentation updated (docs/*.mdx)
  [ ] Docker image builds and passes smoke test
  [ ] Helm chart values updated (appVersion)

Staging deployment:
  [ ] Deploy to staging environment
  [ ] Run integration test suite against staging
  [ ] Verify health endpoint, /v1/models, /v1/chat/completions
  [ ] Test with VSCode Copilot extension
  [ ] Monitor for 1 hour (error rate, latency)

Production deployment:
  [ ] Create GitHub release with tag (triggers PyPI publish)
  [ ] Deploy Docker image to production Kubernetes
  [ ] Verify health and smoke tests
  [ ] Monitor for 4 hours
  [ ] Announce release

Post-release:
  [ ] Verify PyPI package installs correctly
  [ ] Verify Electron installers (Windows + Ubuntu)
  [ ] Update documentation site (auto-deployed via docs.yml)
  [ ] Close related GitHub issues
```

---

## Appendix A: Quick Reference Commands

```bash
# ── Local Development ──────────────────────────
gaia llm "Hello"                              # Test LLM
gaia chat                                     # Interactive chat
gaia api start --background                   # Start API server
gaia api status                               # Check API status
gaia api stop                                 # Stop API server
gaia mcp start --background                   # Start MCP bridge

# ── Docker ─────────────────────────────────────
docker build -t gaia-api:latest .             # Build image
docker run -p 8080:8080 gaia-api:latest       # Run container
docker compose up -d                          # Start full stack
docker compose logs -f gaia-api               # View logs
docker compose down                           # Stop all

# ── Kubernetes ─────────────────────────────────
kubectl apply -f k8s/ --namespace gaia        # Apply all manifests
kubectl get pods -n gaia                      # List pods
kubectl logs -f deploy/gaia-api -n gaia       # View logs
kubectl rollout restart deploy/gaia-api -n gaia # Rolling restart
kubectl rollout undo deploy/gaia-api -n gaia  # Rollback

# ── Helm ───────────────────────────────────────
helm install gaia ./gaia-helm -n gaia         # Install
helm upgrade gaia ./gaia-helm -n gaia         # Upgrade
helm rollback gaia 1 -n gaia                  # Rollback
helm uninstall gaia -n gaia                   # Remove

# ── Monitoring ─────────────────────────────────
curl http://localhost:8080/health             # Health check
curl http://localhost:8080/metrics            # Prometheus metrics
curl http://localhost:9090/api/v1/alerts      # Prometheus alerts

# ── Database ───────────────────────────────────
sqlite3 data/app.db ".tables"                 # List SQLite tables
sqlite3 data/app.db ".backup backup.db"       # Backup SQLite
pg_dump $GAIA_DB_URL > backup.sql             # Backup PostgreSQL
```

## Appendix B: Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `LEMONADE_BASE_URL` | `http://localhost:8000/api/v1` | Lemonade Server URL |
| `ANTHROPIC_API_KEY` | (none) | Claude API key |
| `OPENAI_API_KEY` | (none) | OpenAI API key |
| `AGENT_ROUTING_MODEL` | `Qwen3-Coder-30B-A3B-Instruct-GGUF` | Model for routing agent |
| `GAIA_LOG_LEVEL` | `INFO` | Logging level |
| `GAIA_API_DEBUG` | `0` | Enable API debug logging |
| `GAIA_API_STREAMING` | `0` | Enable SSE streaming |
| `GAIA_API_SHOW_PROMPTS` | `0` | Display LLM prompts |
| `GAIA_API_STEP_THROUGH` | `0` | Step-through debugging |
| `GAIA_API_KEYS` | (none) | Comma-separated valid API keys |
| `GAIA_DB_URL` | `sqlite:///data/app.db` | Database connection URL |
| `GAIA_DB_POOL_MIN` | `2` | Min PostgreSQL pool connections |
| `GAIA_DB_POOL_MAX` | `10` | Max PostgreSQL pool connections |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis URL for task queue |

## Appendix C: Port Reference

| Port | Service | Protocol |
|------|---------|----------|
| 80 | nginx (HTTP redirect) | TCP |
| 443 | nginx (HTTPS) | TCP |
| 3000 | GAIA UI / Next.js dev server | TCP |
| 5173 | Vite dev server | TCP |
| 8000 | Lemonade Server | TCP |
| 8080 | GAIA API Server | TCP |
| 8081-8083 | GAIA API Server (multi-instance) | TCP |
| 9090 | Prometheus | TCP |
| 3001 | Grafana | TCP |
| 9200 | Elasticsearch | TCP |
| 5601 | Kibana | TCP |
| 6379 | Redis | TCP |
| 5432 | PostgreSQL | TCP |
| 6432 | PgBouncer | TCP |

---

*Copyright (C) 2024-2026 Advanced Micro Devices, Inc. All rights reserved. SPDX-License-Identifier: MIT*
