# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
Planning Engine: Thorough, Interactive, Adaptive Planning

Critical Principle: Planning Quality = Output Quality

The agent MUST:
1. Spend significant time planning before implementation
2. Ask clarifying questions (don't assume)
3. Decompose tasks thoroughly
4. Get user approval before starting
5. Adapt plan as development progresses
6. Allow plan revision at any time

Planning Modes:
- INTERACTIVE: Conversational planning with user (default)
- AUTONOMOUS: Agent plans independently (for simple tasks)
- COLLABORATIVE: Mix of agent proposals + user feedback

Planning Phases:
1. UNDERSTANDING: Clarify requirements, ask questions
2. DECOMPOSITION: Break into detailed subtasks
3. VALIDATION: Check assumptions, get user approval
4. EXECUTION: Implement with plan as guide
5. ADAPTATION: Revise plan as needed during execution
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class PlanningMode(Enum):
    """Planning mode determines how interactive planning is."""

    INTERACTIVE = "interactive"  # Full conversation with user
    COLLABORATIVE = "collaborative"  # Agent proposes, user refines
    AUTONOMOUS = "autonomous"  # Agent plans independently


class PlanningPhase(Enum):
    """Phases of the planning process."""

    UNDERSTANDING = "understanding"  # Clarify requirements
    DECOMPOSITION = "decomposition"  # Break down tasks
    VALIDATION = "validation"  # Verify plan with user
    APPROVED = "approved"  # Ready to execute
    EXECUTING = "executing"  # Currently executing
    ADAPTING = "adapting"  # Revising plan during execution


@dataclass
class PlanningQuestion:
    """A question the agent needs answered before planning."""

    id: str
    question: str
    category: str  # "requirements", "technical", "design", "scope"
    importance: str  # "critical", "important", "optional"
    options: Optional[List[str]] = None  # Multiple choice options
    answer: Optional[str] = None
    asked_at: Optional[datetime] = None
    answered_at: Optional[datetime] = None


@dataclass
class TaskDecomposition:
    """A task broken down into subtasks."""

    id: str
    description: str
    subtasks: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)  # Task IDs this depends on
    estimated_complexity: str = "medium"  # low, medium, high
    requires_specialist: Optional[str] = None  # Which specialist
    validation_criteria: List[str] = field(default_factory=list)  # How to verify done


@dataclass
class DetailedPlan:
    """A comprehensive, user-approved plan."""

    goal: str
    mode: PlanningMode
    phase: PlanningPhase
    tasks: List[TaskDecomposition] = field(default_factory=list)
    questions: List[PlanningQuestion] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    alternatives_considered: List[str] = field(default_factory=list)
    user_approved: bool = False
    created_at: Optional[datetime] = None
    approved_at: Optional[datetime] = None
    revision_count: int = 0

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "goal": self.goal,
            "mode": self.mode.value,
            "phase": self.phase.value,
            "tasks": [
                {
                    "id": t.id,
                    "description": t.description,
                    "subtasks": t.subtasks,
                    "dependencies": t.dependencies,
                    "complexity": t.estimated_complexity,
                    "specialist": t.requires_specialist,
                    "validation": t.validation_criteria,
                }
                for t in self.tasks
            ],
            "questions": [
                {
                    "id": q.id,
                    "question": q.question,
                    "category": q.category,
                    "importance": q.importance,
                    "answer": q.answer,
                }
                for q in self.questions
            ],
            "assumptions": self.assumptions,
            "risks": self.risks,
            "alternatives": self.alternatives_considered,
            "approved": self.user_approved,
            "revision_count": self.revision_count,
        }


class PlanningEngine:
    """
    Interactive planning engine for thorough task decomposition.

    Philosophy:
    - "Measure twice, cut once"
    - Better to spend 10 minutes planning than 1 hour fixing
    - Questions are cheaper than mistakes
    - User knows their requirements better than agent
    """

    def __init__(self, mode: PlanningMode = PlanningMode.INTERACTIVE):
        self.mode = mode
        self.current_plan: Optional[DetailedPlan] = None
        self.conversation_history: List[Dict] = []

    def start_planning(self, user_request: str) -> DetailedPlan:
        """
        Start the planning process.

        Args:
            user_request: User's initial request

        Returns:
            DetailedPlan (may be incomplete, needs user input)
        """
        self.current_plan = DetailedPlan(
            goal=user_request,
            mode=self.mode,
            phase=PlanningPhase.UNDERSTANDING,
            created_at=datetime.now(),
        )

        logger.info(f"Planning started: {user_request}")
        logger.info(f"Planning mode: {self.mode.value}")

        return self.current_plan

    def generate_clarifying_questions(self, request: str) -> List[PlanningQuestion]:
        """
        Generate questions to clarify requirements.

        Based on request complexity and ambiguity.

        Args:
            request: User's request

        Returns:
            List of questions to ask
        """
        questions = []

        # Analyze request for ambiguity
        request_lower = request.lower()

        # Question 1: Technology stack (if not specified)
        if "rest api" in request_lower or "api" in request_lower:
            if "fastapi" not in request_lower and "flask" not in request_lower:
                questions.append(
                    PlanningQuestion(
                        id="tech_stack_api",
                        question="Which framework for the API?",
                        category="technical",
                        importance="important",
                        options=["FastAPI (modern, fast)", "Flask (simple)", "Django REST (full-featured)"],
                    )
                )

        # Question 2: Database (if data involved)
        if any(word in request_lower for word in ["store", "save", "database", "users", "data"]):
            if "postgresql" not in request_lower and "sqlite" not in request_lower:
                questions.append(
                    PlanningQuestion(
                        id="database",
                        question="What database should we use?",
                        category="technical",
                        importance="important",
                        options=["SQLite (simple, file-based)", "PostgreSQL (production-ready)", "In-memory (temporary)"],
                    )
                )

        # Question 3: Authentication (for APIs/web apps)
        if any(word in request_lower for word in ["api", "web", "users", "login"]):
            if "auth" not in request_lower:
                questions.append(
                    PlanningQuestion(
                        id="authentication",
                        question="Do you need user authentication?",
                        category="requirements",
                        importance="critical",
                        options=["Yes - JWT tokens", "Yes - Session-based", "No authentication needed"],
                    )
                )

        # Question 4: Testing requirements
        if "test" not in request_lower:
            questions.append(
                PlanningQuestion(
                    id="testing",
                    question="What level of test coverage do you want?",
                    category="requirements",
                    importance="important",
                    options=["Comprehensive (80%+ coverage)", "Basic (happy path only)", "None (skip tests)"],
                )
            )

        # Question 5: Frontend (if web app)
        if "web app" in request_lower or "frontend" in request_lower:
            if "react" not in request_lower and "vue" not in request_lower:
                questions.append(
                    PlanningQuestion(
                        id="frontend_framework",
                        question="Which frontend framework?",
                        category="technical",
                        importance="important",
                        options=["React (most popular)", "Vue (simpler)", "Vanilla JS (no framework)"],
                    )
                )

        # Question 6: Deployment
        questions.append(
            PlanningQuestion(
                id="deployment",
                question="How will this be deployed?",
                category="requirements",
                importance="optional",
                options=["Docker (containerized)", "Cloud (AWS/Azure/GCP)", "Local only"],
            )
        )

        # Question 7: Scope clarification
        questions.append(
            PlanningQuestion(
                id="scope",
                question="Any specific features or constraints I should know about?",
                category="requirements",
                importance="critical",
            )
        )

        logger.info(f"Generated {len(questions)} clarifying questions")

        return questions

    def decompose_task(
        self, goal: str, answers: Dict[str, str]
    ) -> List[TaskDecomposition]:
        """
        Decompose task into detailed subtasks based on answers.

        Args:
            goal: Main goal
            answers: Answers to clarifying questions

        Returns:
            List of detailed task decompositions
        """
        tasks = []

        # Example decomposition for REST API
        if "api" in goal.lower():
            # Phase 1: Project setup
            tasks.append(
                TaskDecomposition(
                    id="task_1",
                    description="Set up project structure",
                    subtasks=[
                        "Create directory structure",
                        "Initialize git repository",
                        "Create requirements.txt or package.json",
                        "Set up virtual environment",
                    ],
                    dependencies=[],
                    estimated_complexity="low",
                    validation_criteria=["Project initializes", "Dependencies install"],
                )
            )

            # Phase 2: Data models
            tasks.append(
                TaskDecomposition(
                    id="task_2",
                    description="Define data models and schemas",
                    subtasks=[
                        "Design database schema",
                        "Create model classes",
                        "Add validation",
                        "Write model tests",
                    ],
                    dependencies=["task_1"],
                    estimated_complexity="medium",
                    validation_criteria=["Models validate input", "Model tests pass"],
                )
            )

            # Phase 3: API endpoints (if API)
            if answers.get("tech_stack_api") == "FastAPI (modern, fast)":
                tasks.append(
                    TaskDecomposition(
                        id="task_3",
                        description="Implement API endpoints",
                        subtasks=[
                            "Create CRUD endpoints",
                            "Add request/response models",
                            "Implement error handling",
                            "Add API documentation (OpenAPI)",
                        ],
                        dependencies=["task_2"],
                        estimated_complexity="high",
                        validation_criteria=["All endpoints respond", "Returns correct data"],
                    )
                )

            # Phase 4: Authentication (if needed)
            if "Yes" in answers.get("authentication", ""):
                tasks.append(
                    TaskDecomposition(
                        id="task_4",
                        description="Implement authentication",
                        subtasks=[
                            "Create user authentication endpoints",
                            "Implement JWT token generation",
                            "Add middleware for protected routes",
                            "Write auth tests",
                        ],
                        dependencies=["task_2", "task_3"],
                        estimated_complexity="high",
                        requires_specialist="SecurityAgent",
                        validation_criteria=["Login works", "Protected routes secured", "Tokens validate"],
                    )
                )

            # Phase 5: Testing
            if "Comprehensive" in answers.get("testing", ""):
                tasks.append(
                    TaskDecomposition(
                        id="task_5",
                        description="Write comprehensive test suite",
                        subtasks=[
                            "Unit tests for models",
                            "Integration tests for endpoints",
                            "Edge case tests",
                            "Security tests (if auth enabled)",
                        ],
                        dependencies=["task_3"],
                        estimated_complexity="high",
                        requires_specialist="TestingAgent",
                        validation_criteria=["80%+ code coverage", "All tests pass"],
                    )
                )

            # Phase 6: Documentation
            tasks.append(
                TaskDecomposition(
                    id="task_6",
                    description="Create documentation",
                    subtasks=[
                        "README with setup instructions",
                        "API documentation",
                        "Code docstrings",
                    ],
                    dependencies=["task_3"],
                    estimated_complexity="medium",
                    requires_specialist="DocumentationAgent",
                    validation_criteria=["README complete", "All public APIs documented"],
                )
            )

        logger.info(f"Decomposed into {len(tasks)} main tasks")

        return tasks

    def identify_assumptions(
        self, goal: str, answers: Dict[str, str]
    ) -> List[str]:
        """
        Identify assumptions being made in the plan.

        Args:
            goal: Goal
            answers: User answers

        Returns:
            List of assumptions
        """
        assumptions = []

        # From goal
        if "api" in goal.lower():
            assumptions.append("Assuming REST API (not GraphQL, gRPC, or WebSocket)")

        if "web" in goal.lower():
            assumptions.append("Assuming web-based application (not desktop or mobile)")

        # From answers
        if answers.get("database") == "SQLite (simple, file-based)":
            assumptions.append("Assuming single-user or low concurrency (SQLite limitation)")

        if answers.get("testing", "").startswith("Basic"):
            assumptions.append("Assuming edge cases and error paths less critical")

        if not answers.get("deployment"):
            assumptions.append("Assuming deployment strategy can be decided later")

        # General
        assumptions.append("Assuming modern Python 3.10+ environment")
        assumptions.append("Assuming code follows PEP 8 style guide")
        assumptions.append("Assuming English language for code and comments")

        return assumptions

    def identify_risks(
        self, goal: str, tasks: List[TaskDecomposition]
    ) -> List[str]:
        """
        Identify potential risks in the plan.

        Args:
            goal: Goal
            tasks: Planned tasks

        Returns:
            List of risks
        """
        risks = []

        # Complexity risk
        if len(tasks) > 10:
            risks.append(f"Large number of tasks ({len(tasks)}) - may take longer than expected")

        # Dependency risk
        task_deps = [len(t.dependencies) for t in tasks]
        if max(task_deps) > 3:
            risks.append("Deep dependency chain - delays cascade if early tasks blocked")

        # Technical risk
        if any("authentication" in t.description.lower() for t in tasks):
            risks.append("Authentication is security-critical - requires extra validation")

        if any("database" in t.description.lower() for t in tasks):
            risks.append("Database schema changes are hard to fix later - validate early")

        # Scope risk
        if "unclear" in goal.lower() or "tbd" in goal.lower():
            risks.append("Requirements not fully specified - may need plan revision")

        return risks

    def format_plan_for_user_review(self, plan: DetailedPlan) -> str:
        """
        Format plan as readable text for user review.

        Args:
            plan: The plan to format

        Returns:
            Formatted plan text
        """
        lines = [
            "=" * 70,
            "PROPOSED EXECUTION PLAN",
            "=" * 70,
            "",
            f"Goal: {plan.goal}",
            "",
            "TASKS:",
            "",
        ]

        for i, task in enumerate(plan.tasks, 1):
            lines.append(f"{i}. {task.description} [{task.estimated_complexity} complexity]")

            if task.requires_specialist:
                lines.append(f"   Specialist: {task.requires_specialist}")

            if task.subtasks:
                for subtask in task.subtasks:
                    lines.append(f"   - {subtask}")

            if task.dependencies:
                dep_numbers = [int(d.replace("task_", "")) for d in task.dependencies]
                lines.append(f"   Depends on: Task {', '.join(map(str, dep_numbers))}")

            if task.validation_criteria:
                lines.append(f"   Validation: {', '.join(task.validation_criteria)}")

            lines.append("")

        if plan.assumptions:
            lines.extend([
                "ASSUMPTIONS:",
                "",
            ])
            for assumption in plan.assumptions:
                lines.append(f"  • {assumption}")
            lines.append("")

        if plan.risks:
            lines.extend([
                "RISKS:",
                "",
            ])
            for risk in plan.risks:
                lines.append(f"  ⚠ {risk}")
            lines.append("")

        if plan.alternatives_considered:
            lines.extend([
                "ALTERNATIVES CONSIDERED:",
                "",
            ])
            for alt in plan.alternatives_considered:
                lines.append(f"  • {alt}")
            lines.append("")

        lines.extend([
            "=" * 70,
            "",
            "Does this plan look good? Any changes needed?",
            "",
        ])

        return "\n".join(lines)

    def revise_plan(
        self,
        plan: DetailedPlan,
        feedback: str,
        changes: Optional[Dict] = None,
    ) -> DetailedPlan:
        """
        Revise plan based on user feedback.

        Args:
            plan: Current plan
            feedback: User feedback
            changes: Specific changes to make

        Returns:
            Revised plan
        """
        plan.revision_count += 1
        plan.phase = PlanningPhase.DECOMPOSITION  # Back to decomposition

        if changes:
            # Apply specific changes
            if "add_task" in changes:
                new_task = changes["add_task"]
                plan.tasks.append(new_task)

            if "remove_task" in changes:
                task_id = changes["remove_task"]
                plan.tasks = [t for t in plan.tasks if t.id != task_id]

            if "modify_task" in changes:
                task_id = changes["modify_task"]["id"]
                modifications = changes["modify_task"]["changes"]
                for task in plan.tasks:
                    if task.id == task_id:
                        for key, value in modifications.items():
                            setattr(task, key, value)

        logger.info(f"Plan revised (revision #{plan.revision_count})")

        return plan

    def adaptive_replan(
        self,
        current_plan: DetailedPlan,
        execution_feedback: Dict,
    ) -> DetailedPlan:
        """
        Adapt plan based on execution feedback.

        During execution, if we discover new requirements or issues,
        revise the plan accordingly.

        Args:
            current_plan: Current plan
            execution_feedback: What we learned during execution

        Returns:
            Adapted plan
        """
        current_plan.phase = PlanningPhase.ADAPTING

        # Example adaptations based on feedback
        if "missing_dependency" in execution_feedback:
            # Add task for dependency
            new_task = TaskDecomposition(
                id=f"task_{len(current_plan.tasks) + 1}",
                description=f"Add missing dependency: {execution_feedback['missing_dependency']}",
                subtasks=["Install package", "Update requirements.txt"],
                estimated_complexity="low",
            )
            current_plan.tasks.insert(0, new_task)  # Add at beginning

        if "performance_issue" in execution_feedback:
            # Add optimization task
            new_task = TaskDecomposition(
                id=f"task_{len(current_plan.tasks) + 1}",
                description="Optimize performance",
                subtasks=["Profile code", "Identify bottleneck", "Apply optimization"],
                estimated_complexity="medium",
                requires_specialist="PerformanceAgent",
            )
            current_plan.tasks.append(new_task)

        current_plan.revision_count += 1

        logger.info(f"Plan adapted based on execution feedback (revision #{current_plan.revision_count})")

        return current_plan

    def get_planning_status(self, plan: DetailedPlan) -> Dict[str, Any]:
        """
        Get current planning status.

        Args:
            plan: Current plan

        Returns:
            Status dict
        """
        unanswered_questions = [q for q in plan.questions if not q.answer]

        return {
            "phase": plan.phase.value,
            "total_tasks": len(plan.tasks),
            "unanswered_questions": len(unanswered_questions),
            "assumptions": len(plan.assumptions),
            "risks": len(plan.risks),
            "revision_count": plan.revision_count,
            "user_approved": plan.user_approved,
            "ready_to_execute": plan.user_approved and plan.phase == PlanningPhase.APPROVED,
        }


def create_interactive_plan(
    user_request: str,
    mode: PlanningMode = PlanningMode.INTERACTIVE,
) -> DetailedPlan:
    """
    Create an interactive plan with user input.

    Args:
        user_request: User's request
        mode: Planning mode

    Returns:
        DetailedPlan ready for execution
    """
    engine = PlanningEngine(mode)
    plan = engine.start_planning(user_request)

    # Phase 1: Ask clarifying questions
    questions = engine.generate_clarifying_questions(user_request)
    plan.questions = questions

    # In interactive mode, would pause here for user to answer questions
    # Then continue with decomposition

    return plan


def format_questions_for_user(questions: List[PlanningQuestion]) -> str:
    """
    Format questions for user to answer.

    Args:
        questions: List of questions

    Returns:
        Formatted text
    """
    lines = [
        "=" * 70,
        "CLARIFYING QUESTIONS",
        "=" * 70,
        "",
        "Before I start planning, I need to clarify a few things:",
        "",
    ]

    for i, q in enumerate(questions, 1):
        importance_marker = {
            "critical": "❗",
            "important": "⚠️",
            "optional": "ℹ️",
        }.get(q.importance, "")

        lines.append(f"{importance_marker} Question {i} ({q.category}):")
        lines.append(f"  {q.question}")

        if q.options:
            for j, option in enumerate(q.options, 1):
                lines.append(f"    {j}. {option}")

        lines.append("")

    lines.extend([
        "=" * 70,
        "",
        "Please answer these questions so I can create an accurate plan.",
        "",
    ])

    return "\n".join(lines)
