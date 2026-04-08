"""Build and load program.md files for autoresearch experiments."""

from pathlib import Path


_TEMPLATES_DIR = Path(__file__).parent / "templates"


def get_default_template() -> str:
    """Read and return the default program.md template."""
    return (_TEMPLATES_DIR / "default.md").read_text()


def load_program(path: str) -> str:
    """Read and return a program.md file from disk."""
    return Path(path).read_text()


def build_from_description(
    goal: str,
    constraints: list[str] | None = None,
    directions: list[str] | None = None,
    setup: str | None = None,
) -> str:
    """Fill in the default template with provided values.

    Args:
        goal: The research goal.
        constraints: List of constraint strings. Defaults to ["Only edit train.py"].
        directions: List of research direction strings. Defaults to
            ["Explore hyperparameter changes"].
        setup: Optional setup override (currently unused; reserved for future use).

    Returns:
        The filled-in program.md content.
    """
    if constraints is None:
        constraints = ["Only edit train.py"]
    if directions is None:
        directions = ["Explore hyperparameter changes"]

    constraints_text = "\n".join(f"- {item}" for item in constraints)
    directions_text = "\n".join(
        f"{i}. {item}" for i, item in enumerate(directions, 1)
    )

    template = get_default_template()
    return template.format(
        goal=goal,
        constraints=constraints_text,
        directions=directions_text,
    )
