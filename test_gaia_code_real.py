#!/usr/bin/env python3
"""
REAL GAIA Code Validation - Test Actual Functionality

Tests the agent's actual logic paths, not just syntax.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, 'src')

print("=" * 70)
print("GAIA CODE - REAL FUNCTIONALITY VALIDATION")
print("=" * 70)
print()

# Test 1: SharedAgentState actually works
print("[1] Testing SharedAgentState functionality...")
try:
    from gaia.agents.gaia_code.shared_state import get_shared_state

    with tempfile.TemporaryDirectory() as tmpdir:
        state = get_shared_state(Path(tmpdir))

        # Test memory cache
        state.memory.cache_file("test.py", "print('hello')")
        content = state.memory.get_file("test.py")
        assert content == "print('hello')", "Memory cache broken!"
        print("  ✓ Memory cache: WORKS")

        # Test knowledge storage
        insight_id = state.knowledge.store_insight(
            category="test",
            content="Test insight",
            triggers=["test"]
        )
        assert insight_id is not None, "Knowledge storage broken!"
        results = state.knowledge.recall("test")
        assert len(results) > 0, "Knowledge recall broken!"
        print("  ✓ Knowledge DB: WORKS")

        # Test plan creation
        task = state.plan.create_task("Test task")
        assert task.id is not None, "Plan creation broken!"
        retrieved = state.plan.get_task(task.id)
        assert retrieved.description == "Test task", "Plan retrieval broken!"
        print("  ✓ Master plan: WORKS")

        print("✓ SharedAgentState: FULLY FUNCTIONAL")

except Exception as e:
    print(f"✗ SharedAgentState FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Quality gates actually work
print("\n[2] Testing Quality Gates...")
try:
    from gaia.agents.gaia_code.quality_gates import QualityGateRunner, SyntaxGate

    runner = QualityGateRunner()

    # Create test files
    with tempfile.TemporaryDirectory() as tmpdir:
        # Valid file
        valid_file = Path(tmpdir) / "valid.py"
        valid_file.write_text("def hello():\n    print('hello')\n")

        # Invalid file
        invalid_file = Path(tmpdir) / "invalid.py"
        invalid_file.write_text("def broken(\n    print('missing colon')\n")

        # Test on valid file
        results_valid = runner.run_all([str(valid_file)])
        assert runner.all_passed(results_valid), "Should pass on valid file!"
        print("  ✓ Valid file: PASS")

        # Test on invalid file
        results_invalid = runner.run_all([str(invalid_file)])
        assert not runner.all_passed(results_invalid), "Should fail on invalid file!"
        print("  ✓ Invalid file: FAIL (as expected)")

        print("✓ Quality Gates: FULLY FUNCTIONAL")

except Exception as e:
    print(f"✗ Quality Gates FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Escalation ladder logic
print("\n[3] Testing Escalation Ladder...")
try:
    from gaia.agents.gaia_code.quality_gates import EscalationLadder

    ladder = EscalationLadder()

    # Initial: should retry
    assert ladder.get_action() == "retry", "Initial action should be retry!"
    print("  ✓ Initial: retry")

    # After 1 increment: still retry
    ladder.increment()
    assert ladder.get_action() == "retry", "After 1: should still retry!"
    print("  ✓ After 1: retry")

    # After 2 increments: decompose
    ladder.increment()
    action = ladder.get_action()
    assert action in ("decompose", "alternative"), f"After 2: should decompose, got {action}!"
    print(f"  ✓ After 2: {action}")

    # Reset works
    ladder.reset()
    assert ladder.get_action() == "retry", "Reset should go back to retry!"
    print("  ✓ Reset: back to retry")

    print("✓ Escalation Ladder: FULLY FUNCTIONAL")

except Exception as e:
    print(f"✗ Escalation Ladder FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Persona system
print("\n[4] Testing Persona System...")
try:
    from gaia.agents.gaia_code.persona import PERSONALITY_PROFILES, create_persona

    # Check all 8 profiles exist
    expected = ["torvalds", "knuth", "pike", "carmack", "hickey", "kay", "thompson", "hopper"]
    for name in expected:
        assert name in PERSONALITY_PROFILES, f"Missing persona: {name}!"
    print(f"  ✓ All 8 personas defined")

    # Test persona creation
    with tempfile.TemporaryDirectory() as tmpdir:
        persona = create_persona("torvalds", Path(tmpdir))
        assert persona.profile.name == "Torvalds", "Persona name mismatch!"
        print("  ✓ Persona creation: WORKS")

        # Test pushback logic
        assert persona.should_push_back(0.9), "Should push back on critical!"
        print("  ✓ Pushback logic: WORKS")

        # Test message generation
        msg = persona.get_pushback_message("issue", "alternative", 0.8)
        assert len(msg) > 10, "Message too short!"
        assert "issue" in msg or "alternative" in msg, "Message missing content!"
        print("  ✓ Message generation: WORKS")

    print("✓ Persona System: FULLY FUNCTIONAL")

except Exception as e:
    print(f"✗ Persona System FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Planning engine
print("\n[5] Testing Planning Engine...")
try:
    from gaia.agents.gaia_code.planning_engine import PlanningEngine, PlanningMode

    engine = PlanningEngine(PlanningMode.INTERACTIVE)

    # Start planning
    plan = engine.start_planning("Build a REST API")
    assert plan.goal == "Build a REST API", "Goal not set!"
    print("  ✓ Planning start: WORKS")

    # Generate questions
    questions = engine.generate_clarifying_questions("Build a REST API")
    assert len(questions) > 0, "Should generate questions!"
    print(f"  ✓ Question generation: {len(questions)} questions")

    # Test decomposition
    answers = {"tech_stack_api": "FastAPI", "testing": "Comprehensive"}
    tasks = engine.decompose_task("Build REST API", answers)
    assert len(tasks) > 0, "Should decompose into tasks!"
    print(f"  ✓ Task decomposition: {len(tasks)} tasks")

    print("✓ Planning Engine: FULLY FUNCTIONAL")

except Exception as e:
    print(f"✗ Planning Engine FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("=" * 70)
print("✅ REAL VALIDATION COMPLETE - ALL CORE SYSTEMS FUNCTIONAL")
print("=" * 70)
print()
print("Validated:")
print("  ✓ SharedAgentState (memory, knowledge, plan)")
print("  ✓ Quality Gates (syntax checking, escalation)")
print("  ✓ Escalation Ladder (retry logic)")
print("  ✓ Persona System (8 profiles, pushback)")
print("  ✓ Planning Engine (questions, decomposition)")
print()
print("Next: Install dependencies and test with real LLM")
print("  pip install -e '.[dev]'")
print("  gaia code 'task' --persona pike")
print()
print("=" * 70)
