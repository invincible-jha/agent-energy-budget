"""CLI entry point for agent-energy-budget.

Invoked as::

    agent-energy-budget [OPTIONS] COMMAND [ARGS]...

or, during development::

    python -m agent_energy_budget.cli.main
"""
from __future__ import annotations

import sys

import click
from rich.console import Console
from rich.table import Table

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


@cli.command(name="route")
@click.option(
    "--prompt",
    required=True,
    help="Prompt text to route to an LLM model.",
)
@click.option(
    "--budget",
    default=10.0,
    show_default=True,
    type=float,
    help="Total budget in USD for this routing session.",
)
@click.option(
    "--strategy",
    default="balanced",
    show_default=True,
    type=click.Choice(
        ["cheapest_first", "quality_first", "balanced", "budget_aware"],
        case_sensitive=False,
    ),
    help="Routing strategy to apply.",
)
@click.option(
    "--max-cost",
    default=None,
    type=float,
    help="Hard cap on estimated cost per request in USD.",
)
@click.option(
    "--min-quality",
    default=0.0,
    show_default=True,
    type=float,
    help="Minimum acceptable quality score [0.0–1.0].",
)
@click.option(
    "--json-output",
    is_flag=True,
    default=False,
    help="Emit the routing decision as JSON (for scripting).",
)
def route_command(
    prompt: str,
    budget: float,
    strategy: str,
    max_cost: float | None,
    min_quality: float,
    json_output: bool,
) -> None:
    """Route a prompt to the most cost-effective LLM model.

    Examples:

    \b
        agent-budget route --prompt "Summarise this article" --budget 10.0 --strategy balanced
        agent-budget route --prompt "Hello" --strategy cheapest_first --json-output
    """
    from agent_energy_budget.router.cost_router import CostAwareRouter
    from agent_energy_budget.router.models import RouterBudgetConfig
    from agent_energy_budget.router.strategies import NoAffordableModelError

    # Validate min_quality range before constructing config so we can emit a
    # friendly error rather than a raw ValueError traceback.
    if not 0.0 <= min_quality <= 1.0:
        console.print(
            f"[red]Error:[/red] --min-quality must be in [0.0, 1.0]; got {min_quality}.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    if budget < 0.0:
        console.print(
            f"[red]Error:[/red] --budget must be >= 0; got {budget}.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    budget_config = RouterBudgetConfig(
        total_budget_usd=budget,
        min_quality_score=min_quality,
    )

    # strategy comes from click.Choice so it is already validated.
    from typing import Literal, get_args
    from agent_energy_budget.router.models import RoutingStrategy as RoutingStrategyLiteral

    router = CostAwareRouter(
        budget=budget_config,
        strategy=strategy,  # type: ignore[arg-type]
    )

    try:
        decision = router.route(prompt, max_cost=max_cost)
    except NoAffordableModelError as exc:
        if json_output:
            import json as _json

            click.echo(
                _json.dumps(
                    {
                        "error": "NoAffordableModelError",
                        "message": str(exc),
                        "remaining_budget": exc.remaining_budget,
                        "min_quality_score": exc.min_quality_score,
                    },
                    indent=2,
                )
            )
        else:
            console.print(f"[red]No affordable model found:[/red] {exc}")
        raise SystemExit(1)

    if json_output:
        import json as _json

        output = {
            "selected_model": decision.selected_model.name,
            "provider": decision.selected_model.provider,
            "quality_score": decision.selected_model.quality_score,
            "cost_per_1k_input": decision.selected_model.cost_per_1k_input,
            "cost_per_1k_output": decision.selected_model.cost_per_1k_output,
            "estimated_cost_usd": decision.estimated_cost,
            "remaining_budget_usd": decision.remaining_budget,
            "reason": decision.reason,
            "strategy": strategy,
        }
        # Use click.echo instead of console.print to prevent Rich from
        # injecting ANSI codes or word-wrap newlines into the JSON payload.
        click.echo(_json.dumps(output, indent=2))
        return

    # Rich table output
    table = Table(title="Routing Decision", show_header=True, header_style="bold cyan")
    table.add_column("Field", style="bold")
    table.add_column("Value")

    model = decision.selected_model
    table.add_row("Selected Model", model.name)
    table.add_row("Provider", model.provider)
    table.add_row("Strategy", strategy)
    table.add_row("Quality Score", f"{model.quality_score:.2f}")
    table.add_row("Cost / 1K Input", f"${model.cost_per_1k_input:.6f}")
    table.add_row("Cost / 1K Output", f"${model.cost_per_1k_output:.6f}")
    table.add_row("Estimated Cost", f"${decision.estimated_cost:.6f}")
    table.add_row("Remaining Budget", f"${decision.remaining_budget:.6f}")
    table.add_row("Reason", decision.reason)

    console.print(table)


if __name__ == "__main__":
    cli()
