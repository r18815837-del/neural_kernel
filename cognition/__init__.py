"""Cognition — the thinking layer of Neural Kernel.

Receives a user query, decides what it knows vs. what it needs to find out,
reasons through the answer, persists the result, and returns a response.
"""

from .orchestrator import Orchestrator

__all__ = ["Orchestrator"]
