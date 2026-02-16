# Requirements Gathering State Specification

**Date**: February 7, 2026
**Version**: 1.0
**Purpose**: Initial state for complex tasks that gathers clarifying information before planning/implementation
**Integration**: Part of State Machine framework (Week 3 in AI_ACCELERATED_TIMELINE.md)

---

## Problem Statement

**Current behavior** (without Requirements Gathering):
```
User: "Build an e-commerce platform"

Agent: [Assumes React + FastAPI + PostgreSQL + Stripe]
Agent: [Builds for 2 hours]
Agent: "Done!"

User: "Wait, I wanted Vue.js and PayPal integration"
Agent: [Must rebuild everything]
```

**With Requirements Gathering**:
```
User: "Build an e-commerce platform"

Agent: [Enters Requirements Gathering state]
Agent: "I need some clarification before starting:
  1. Frontend framework preference? (React/Vue/Svelte/Angular)
  2. Backend framework? (FastAPI/Flask/Django/Express)
  3. Database? (PostgreSQL/MySQL/MongoDB)
  4. Payment provider? (Stripe/PayPal/Square)
  5. Authentication method? (JWT/OAuth/Session-based)
  6. Do you need an admin dashboard? (Yes/No)
  7. Any specific features or constraints?"

User: "Vue.js, FastAPI, PostgreSQL, PayPal, JWT, yes to admin, keep it simple"

Agent: [Transitions to Planning state with clear requirements]
Agent: [Builds exactly what was requested]
```

---

## When to Enter Requirements Gathering State

### Auto-enter (default for complex tasks):
- Task is vague or high-level ("build an app", "create a system", "implement a feature")
- Task has multiple valid interpretations
- Task requires architectural decisions (framework, database, auth, etc.)
- Task scope is large (estimated 10+ files, 2+ hours)

### Auto-skip (for simple/clear tasks):
- Task is specific ("fix bug in auth.py line 42")
- Task has all necessary details ("build a REST API with FastAPI, PostgreSQL, JWT auth, CRUD for users and posts")
- Task is small (1-3 files, < 30 minutes)
- User explicitly says "skip requirements" or "proceed with defaults"

---

## Requirements Gathering Flow

```
┌─────────────────────────────────────────────────────────────┐
│                        User Task                             │
│  "Build an e-commerce platform"                             │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│           Task Complexity Analyzer                          │
│  - Parse task description                                   │
│  - Estimate scope (files, time)                             │
│  - Identify ambiguities                                     │
│  - Count decision points                                    │
└────────────────────┬────────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
        ▼ Complex/Vague           ▼ Simple/Clear
┌───────────────────┐     ┌──────────────────┐
│  ENTER            │     │  SKIP             │
│  Requirements     │     │  Requirements     │
│  Gathering        │     │  Gathering        │
└────────┬──────────┘     └────────┬─────────┘
         │                         │
         ▼                         │
┌────────────────────────┐         │
│  Ask Clarifying Qs     │         │
│  (3-7 questions)       │         │
└────────┬───────────────┘         │
         │                         │
         ▼                         │
┌────────────────────────┐         │
│  User Answers          │         │
│  (or skips)            │         │
└────────┬───────────────┘         │
         │                         │
         └─────────┬───────────────┘
                   │
                   ▼
         ┌─────────────────┐
         │  PLANNING State  │
         │  (with req's)    │
         └─────────────────┘
```

---

## Question Generation Strategy

The agent should ask questions in these categories:

### 1. **Technology Stack** (for new projects)
- Frontend framework? (React/Vue/Svelte/Angular)
- Backend framework? (FastAPI/Flask/Django/Express)
- Database? (PostgreSQL/MySQL/MongoDB/SQLite)
- Language preference? (Python/TypeScript/Go/Rust)

### 2. **Architecture Decisions**
- Monolith or microservices?
- REST API or GraphQL?
- Authentication method? (JWT/OAuth/Session)
- Caching layer? (Redis/Memcached/None)
- Message queue? (RabbitMQ/Kafka/None)

### 3. **Features & Scope**
- Required features? (list core features)
- Optional features? (nice-to-have)
- Admin interface needed?
- Mobile support?
- Real-time features? (WebSockets/SSE)

### 4. **Constraints & Preferences**
- Performance requirements?
- Security constraints?
- Deployment target? (Docker/AWS/Local)
- Test coverage preference?
- Code style preferences?

### 5. **Data & Integrations**
- External APIs to integrate? (Stripe, SendGrid, etc.)
- Data models/schema?
- File storage? (Local/S3/Cloud)

---

## Example Requirements Gathering Sessions

### Example 1: E-commerce Platform (Complex)

```
User: "Build an e-commerce platform"

Agent: [Enters Requirements Gathering state]

Agent: "I need clarification on several aspects:

**Technology Stack:**
1. Frontend framework? (React/Vue/Svelte - I recommend React for e-commerce)
2. Backend framework? (FastAPI/Django - I recommend FastAPI for performance)

**Core Features:**
3. Product catalog features? (categories, search, filters, reviews?)
4. Payment integration? (Stripe/PayPal/Square?)
5. User authentication? (Email/password, social login, or both?)
6. Shipping integration? (ShipStation/manual entry?)

**Additional Features:**
7. Admin dashboard needed? (product management, orders, analytics?)
8. Inventory tracking?
9. Email notifications? (order confirmation, shipping updates?)

**Deployment:**
10. Deployment target? (Docker/AWS/Heroku/Local dev only?)

You can answer with specific choices, or say 'use defaults' for sensible defaults."

User: "React, FastAPI, PostgreSQL, Stripe, email/password only, no shipping integration yet,
yes to admin dashboard, basic inventory, yes to emails, Docker deployment"

Agent: [Stores requirements in task metadata]
Agent: [Transitions to Planning state]
Agent: "Got it! Building:
  - React frontend with product catalog (categories, search, filters)
  - FastAPI backend with JWT authentication
  - PostgreSQL database
  - Stripe payment integration
  - Admin dashboard (products, orders, analytics)
  - Basic inventory tracking
  - Email notifications via SendGrid
  - Docker containerization

Proceeding to planning phase..."
```

### Example 2: Bug Fix (Simple - Auto-skip)

```
User: "Fix the authentication bug in auth.py line 42 where JWT tokens expire too quickly"

Agent: [Task Complexity Analyzer determines: simple, specific, clear]
Agent: [Auto-skips Requirements Gathering]
Agent: [Goes directly to Planning state]
Agent: "Analyzing auth.py to fix JWT token expiration issue..."
```

### Example 3: User Skips Requirements

```
User: "Build a blog platform"

Agent: [Enters Requirements Gathering state]
Agent: "I need clarification:
1. Frontend framework?
2. Backend framework?
3. ..."

User: "Just use sensible defaults and get started"

Agent: [Skips remaining questions]
Agent: [Uses defaults: React, FastAPI, PostgreSQL, JWT]
Agent: [Transitions to Planning]
Agent: "Using defaults: React + FastAPI + PostgreSQL. Proceeding to planning..."
```

---

## Implementation Details

### State Definition

```python
from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class RequirementsGatheringState:
    """Requirements Gathering state for the State Machine."""

    state_name: str = "requirements_gathering"
    description: str = "Gather clarifying information before planning"

    # Questions to ask
    questions: List[str] = None

    # User responses
    responses: Dict[str, str] = None

    # Auto-skip conditions
    skip_if_simple: bool = True
    skip_if_detailed: bool = True

    # Complexity threshold (0-1, where 1 = very complex)
    complexity_threshold: float = 0.3

    def should_enter(self, task_description: str) -> bool:
        """Determine if we should enter requirements gathering."""
        complexity = self._analyze_complexity(task_description)

        # Skip if task is simple/clear
        if complexity < self.complexity_threshold:
            return False

        # Skip if task already has detailed requirements
        if self._has_sufficient_detail(task_description):
            return False

        return True

    def _analyze_complexity(self, task_description: str) -> float:
        """Analyze task complexity (0-1 scale)."""
        indicators = {
            'vague_keywords': ['build', 'create', 'make', 'implement', 'develop'],
            'decision_points': ['app', 'platform', 'system', 'service'],
            'ambiguous': ['feature', 'functionality', 'capability'],
        }

        score = 0.0
        words = task_description.lower().split()

        # Check for vague keywords
        if any(keyword in words for keyword in indicators['vague_keywords']):
            score += 0.3

        # Check for decision points
        if any(keyword in words for keyword in indicators['decision_points']):
            score += 0.3

        # Check for missing details (short description)
        if len(words) < 10:
            score += 0.2

        # Check for ambiguity
        if any(keyword in words for keyword in indicators['ambiguous']):
            score += 0.2

        return min(score, 1.0)

    def _has_sufficient_detail(self, task_description: str) -> bool:
        """Check if task has enough detail to skip requirements gathering."""
        # Look for specific technology mentions
        tech_keywords = [
            'react', 'vue', 'fastapi', 'django', 'flask',
            'postgresql', 'mysql', 'mongodb', 'stripe', 'jwt'
        ]

        words = task_description.lower().split()
        tech_mentions = sum(1 for keyword in tech_keywords if keyword in words)

        # If 3+ technologies are specified, probably detailed enough
        return tech_mentions >= 3

    def generate_questions(self, task_description: str) -> List[str]:
        """Generate relevant questions based on task."""
        questions = []

        # Detect project type
        if any(keyword in task_description.lower() for keyword in ['app', 'platform', 'system']):
            questions.extend([
                "What frontend framework do you prefer? (React/Vue/Svelte/Angular)",
                "What backend framework? (FastAPI/Flask/Django/Express)",
                "What database? (PostgreSQL/MySQL/MongoDB)",
            ])

        # Detect e-commerce
        if any(keyword in task_description.lower() for keyword in ['ecommerce', 'e-commerce', 'shop', 'store']):
            questions.extend([
                "Payment provider? (Stripe/PayPal/Square)",
                "Do you need an admin dashboard? (Yes/No)",
                "Shipping integration needed? (Yes/No)",
            ])

        # Detect auth needs
        if any(keyword in task_description.lower() for keyword in ['auth', 'login', 'user']):
            questions.extend([
                "Authentication method? (JWT/OAuth/Session-based)",
                "Social login? (Google/GitHub/None)",
            ])

        # Always ask about deployment
        questions.append("Deployment target? (Docker/AWS/Heroku/Local dev)")

        # Limit to 3-7 questions
        return questions[:7]
```

### Integration with State Machine

```python
class StateMachine:
    def __init__(self):
        self.states = {
            'requirements_gathering': RequirementsGatheringState(),
            'planning': PlanningState(),
            'implementation': ImplementationState(),
            'testing': TestingState(),
            'debug': DebugState(),
            'review': ReviewState(),
        }
        self.current_state = None

    def start_task(self, task_description: str):
        """Start a new task, entering requirements gathering if needed."""
        req_state = self.states['requirements_gathering']

        if req_state.should_enter(task_description):
            # Enter requirements gathering
            self.enter_state('requirements_gathering')
            questions = req_state.generate_questions(task_description)
            return {
                'state': 'requirements_gathering',
                'questions': questions,
                'message': 'I need some clarification before starting:'
            }
        else:
            # Skip to planning
            self.enter_state('planning')
            return {
                'state': 'planning',
                'message': 'Task is clear, proceeding to planning...'
            }

    def answer_requirements(self, responses: Dict[str, str]):
        """User answered requirements questions."""
        req_state = self.states['requirements_gathering']
        req_state.responses = responses

        # Store requirements in task metadata
        self.task_metadata['requirements'] = responses

        # Transition to planning
        self.enter_state('planning')
        return {
            'state': 'planning',
            'message': f'Got it! Proceeding with: {self._format_requirements(responses)}'
        }

    def skip_requirements(self):
        """User wants to skip requirements and use defaults."""
        # Use default responses
        defaults = self._get_default_requirements()
        return self.answer_requirements(defaults)
```

---

## TUI Integration

The Requirements Gathering state should be clearly visible in the TUI:

```
┌──────────────────────────────────────────────────────────────┐
│  GAIA Agent - Requirements Gathering                         │
├──────────────────────────────────────────────────────────────┤
│  Task: "Build an e-commerce platform"                        │
│                                                               │
│  📋 Gathering Requirements (3/7 answered)                    │
│                                                               │
│  ✓ 1. Frontend framework?                                    │
│      → React                                                  │
│                                                               │
│  ✓ 2. Backend framework?                                     │
│      → FastAPI                                                │
│                                                               │
│  ✓ 3. Database?                                              │
│      → PostgreSQL                                             │
│                                                               │
│  ⏳ 4. Payment provider? (Stripe/PayPal/Square)              │
│      _                                                        │
│                                                               │
│  Commands:                                                    │
│  - Type answer or option                                     │
│  - 'skip' - Use defaults for remaining questions             │
│  - 'back' - Change previous answer                           │
└──────────────────────────────────────────────────────────────┘
```

---

## Voice Integration

Requirements gathering should work seamlessly with voice:

```
User (voice): "Hey Gaia, build an e-commerce platform"

Agent (TTS): "I need some clarification. What frontend framework do you prefer?
Options are React, Vue, Svelte, or Angular."

User (voice): "React"

Agent (TTS): "Got it, React. What backend framework? FastAPI, Flask, or Django?"

User (voice): "FastAPI"

Agent (TTS): "FastAPI selected. What database? PostgreSQL, MySQL, or MongoDB?"

User (voice): "Use your best judgment for the rest"

Agent (TTS): "Understood. Using defaults for remaining questions: Stripe for payments,
JWT authentication, admin dashboard included. Proceeding to planning phase."
```

---

## Benefits

1. **Builds the right thing first time** - No wasted effort on wrong assumptions
2. **Educational** - User learns what decisions are needed for their project type
3. **Faster overall** - 2 minutes of questions saves 2 hours of rebuilding
4. **Better documentation** - Requirements are captured in task metadata
5. **Reusable** - Similar tasks can reference previous requirements
6. **Flexible** - Can skip for simple tasks or when user wants defaults

---

## Implementation Timeline

- **Week 3, Day 2**: Implement `RequirementsGatheringState` class
- **Week 3, Day 3**: Add auto-skip logic and question generation
- **Week 3, Day 4**: Integrate with TUI (requirements panel)
- **Week 5**: Add to full TUI with proper visualization
- **Week 8**: Integrate with voice interface

---

## Success Metrics

- ✅ Complex tasks trigger requirements gathering 90%+ of the time
- ✅ Simple tasks auto-skip 95%+ of the time
- ✅ Average 3-7 questions per complex task
- ✅ Users can skip with "use defaults" < 5 seconds
- ✅ Requirements stored in task metadata for reference
- ✅ 80%+ reduction in "I wanted X not Y" feedback

---

*Requirements Gathering State - Ensuring the agent builds what you actually want.*
