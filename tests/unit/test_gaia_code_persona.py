# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
Unit tests for PersonaEngine and personality profiles.
"""

import tempfile
from pathlib import Path

import pytest


class TestPersonalityProfiles:
    """Test personality profile definitions."""

    def test_all_profiles_exist(self):
        """Test that all 8 profiles are defined."""
        from gaia.agents.gaia_code.persona import PERSONALITY_PROFILES

        expected_personas = [
            "torvalds",
            "knuth",
            "pike",
            "carmack",
            "hickey",
            "kay",
            "thompson",
            "hopper",
        ]

        for persona_name in expected_personas:
            assert persona_name in PERSONALITY_PROFILES
            profile = PERSONALITY_PROFILES[persona_name]
            assert profile.name is not None
            assert profile.description is not None
            assert len(profile.examples) > 0

    def test_profile_traits(self):
        """Test that profiles have all required traits."""
        from gaia.agents.gaia_code.persona import PERSONALITY_PROFILES, PersonalityTrait

        for profile in PERSONALITY_PROFILES.values():
            # Each profile should have trait values
            assert len(profile.traits) > 0

            # Trait values should be between 0 and 1
            for trait, value in profile.traits.items():
                assert 0.0 <= value <= 1.0

    def test_torvalds_is_direct(self):
        """Test Torvalds persona is highly direct."""
        from gaia.agents.gaia_code.persona import PERSONALITY_PROFILES, PersonalityTrait

        torvalds = PERSONALITY_PROFILES["torvalds"]

        assert torvalds.get_trait(PersonalityTrait.DIRECTNESS) >= 0.9
        assert torvalds.get_trait(PersonalityTrait.ASSERTIVENESS) >= 0.9
        assert torvalds.pushback_threshold < 0.3  # Pushes back often


class TestPersonaEngine:
    """Test PersonaEngine functionality."""

    def test_create_persona(self):
        """Test creating a persona engine."""
        from gaia.agents.gaia_code.persona import create_persona

        with tempfile.TemporaryDirectory() as tmpdir:
            persona = create_persona("pike", Path(tmpdir))

            assert persona is not None
            assert persona.profile.name == "Pike"

    def test_system_prompt_generation(self):
        """Test system prompt generation."""
        from gaia.agents.gaia_code.persona import create_persona

        with tempfile.TemporaryDirectory() as tmpdir:
            persona = create_persona("torvalds", Path(tmpdir))

            prompt = persona.get_system_prompt_addition()

            assert len(prompt) > 100
            assert "Torvalds" in prompt
            assert "honest" in prompt.lower() or "direct" in prompt.lower()

    def test_pushback_threshold(self):
        """Test pushback decision logic."""
        from gaia.agents.gaia_code.persona import create_persona

        with tempfile.TemporaryDirectory() as tmpdir:
            # Torvalds pushes back more
            torvalds = create_persona("torvalds", Path(tmpdir))
            assert torvalds.should_push_back(0.4) is True  # Low severity, still pushes back

            # Hopper pushes back less
            hopper = create_persona("hopper", Path(tmpdir))
            assert hopper.should_push_back(0.3) is False  # Low severity, doesn't push back

            # Everyone pushes back on critical issues
            assert torvalds.should_push_back(0.9) is True
            assert hopper.should_push_back(0.9) is True

    def test_pushback_message_generation(self):
        """Test pushback message generation in different voices."""
        from gaia.agents.gaia_code.persona import create_persona

        with tempfile.TemporaryDirectory() as tmpdir:
            # Test different personas generate different messages
            torvalds = create_persona("torvalds", Path(tmpdir))
            pike = create_persona("pike", Path(tmpdir))

            issue = "This creates a circular dependency"
            alternative = "Use dependency injection"

            msg_torvalds = torvalds.get_pushback_message(issue, alternative, 0.7)
            msg_pike = pike.get_pushback_message(issue, alternative, 0.7)

            # Messages should be different (different voices)
            assert msg_torvalds != msg_pike

            # Both should contain the issue and alternative
            assert issue in msg_torvalds or "circular" in msg_torvalds
            assert issue in msg_pike or "circular" in msg_pike

    def test_adaptation_from_feedback(self):
        """Test personality adaptation based on user feedback."""
        from gaia.agents.gaia_code.persona import PersonalityTrait, create_persona

        with tempfile.TemporaryDirectory() as tmpdir:
            persona = create_persona("pike", Path(tmpdir))

            initial_verbosity = persona.profile.get_trait(PersonalityTrait.VERBOSITY)

            # User wants more detail
            persona.adapt_from_feedback("wants_more_detail", "User asked for explanation")

            new_verbosity = persona.profile.get_trait(PersonalityTrait.VERBOSITY)

            assert new_verbosity > initial_verbosity

    def test_pushback_tracking(self):
        """Test pushback acceptance tracking."""
        from gaia.agents.gaia_code.persona import create_persona

        with tempfile.TemporaryDirectory() as tmpdir:
            persona = create_persona("carmack", Path(tmpdir))

            initial_threshold = persona.profile.pushback_threshold

            # Simulate user accepting pushback multiple times
            for i in range(8):
                persona.record_pushback(accepted=True, context=f"Optimization {i}")

            # Threshold should decrease (be more assertive)
            # Since user accepts pushback
            # This happens after 10 interactions, so not yet changed
            assert persona.pushback_accepted_count == 8


class TestPersonaIntegration:
    """Test persona integration with agent."""

    def test_list_personas(self):
        """Test listing available personas."""
        from gaia.agents.gaia_code.persona import list_personas

        personas = list_personas()

        assert len(personas) == 8
        assert all("name" in p for p in personas)
        assert all("description" in p for p in personas)
