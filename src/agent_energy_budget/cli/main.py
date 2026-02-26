"""CLI entry point for agent-energy-budget.

Invoked as::

    agent-energy-budget [OPTIONS] COMMAND [ARGS]...

or, during development::

    python -m agent_energy_budget.cli.main
"""
from __future__ import annotations

import click
from rich.console import Console

console = Console()


@click.group()
@click.version_option()
def cli() -> None:
    """Agent cost control, energy budget management, and token tracking"""


@cli.command(name="version")
def version_command() -> None:
    """Show detailed version information."""
    from agent_energy_budget import __version__

    console.print(f"[bold]agent-energy-budget[/bold] v{__version__}")


@cli.command(name="plugins")
def plugins_command() -> None:
    """List all registered plugins loaded from entry-points."""
    console.print("[bold]Registered plugins:[/bold]")
    console.print("  (No plugins registered. Install a plugin package to see entries here.)")


if __name__ == "__main__":
    cli()
