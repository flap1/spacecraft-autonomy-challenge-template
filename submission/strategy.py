"""Submission entry point; safety settings live in autonomy.controller."""

from typing import TYPE_CHECKING

# Core loads this file by path and puts its directory on sys.path. Tests import
# it as a package instead; select explicitly without hiding dependency errors.
if TYPE_CHECKING or __package__:
    from .autonomy.controller import ConservativeStrategy
else:
    from autonomy.controller import ConservativeStrategy


class MyStrategy(ConservativeStrategy):
    """Conservative adaptive MPC with thermal and momentum safety reserves."""
