"""Framework adapters for agent_energy_budget.

Each adapter tracks token usage and USD cost for a specific agent framework
without requiring the framework package to be installed.
"""
from __future__ import annotations

from agent_energy_budget.adapters.anthropic_sdk import AnthropicCostTracker
from agent_energy_budget.adapters.crewai import CrewAICostTracker
from agent_energy_budget.adapters.langchain import LangChainCostTracker
from agent_energy_budget.adapters.microsoft_agents import MicrosoftCostTracker
from agent_energy_budget.adapters.openai_agents import OpenAICostTracker

__all__ = [
    "AnthropicCostTracker",
    "CrewAICostTracker",
    "LangChainCostTracker",
    "MicrosoftCostTracker",
    "OpenAICostTracker",
]
