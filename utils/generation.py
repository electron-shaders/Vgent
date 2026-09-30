"""Generation budgets shared by the native CLI and lmms-eval adapter."""

import os

DEFAULT_TOKEN_LIMIT = 16384


def resolve_final_token_limit():
    """Hard output-token cap for raw final-answer retries in lmms-eval."""
    value = os.environ.get("VGENT_TRUNCATION_MAX_TOKENS", DEFAULT_TOKEN_LIMIT)
    if not str(value).isdigit() or int(value) < 1:
        raise ValueError(f"VGENT_TRUNCATION_MAX_TOKENS must be a positive integer, got {value!r}")
    return int(value)


class VgentResponseTruncatedError(RuntimeError):
    """The server stopped generation at the requested output-token limit."""

    def __init__(self, max_new_tokens):
        self.max_new_tokens = max_new_tokens
        super().__init__(f"Vgent response exhausted max_new_tokens={max_new_tokens}")


def resolve_graph_token_budgets(graph_max_new_tokens=None, graph_max_new_tokens_limit=None):
    """Explicit arguments override environment variables, then defaults."""
    values = []
    for name, value, default in (
        ("graph_max_new_tokens", graph_max_new_tokens, 2048),
        ("graph_max_new_tokens_limit", graph_max_new_tokens_limit, DEFAULT_TOKEN_LIMIT),
    ):
        if value is None:
            value = os.environ.get(f"VGENT_{name.upper()}", default)
        if isinstance(value, bool) or not str(value).isdigit() or int(value) < 1:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
        values.append(int(value))
    if values[1] < values[0]:
        raise ValueError("graph_max_new_tokens_limit must be greater than or equal to graph_max_new_tokens")
    return tuple(values)


def add_graph_generation_args(parser):
    parser.add_argument(
        "--graph_max_new_tokens",
        type=int,
        default=None,
        help="Initial output-token budget per graph chunk (VGENT_GRAPH_MAX_NEW_TOKENS; default: 2048).",
    )
    parser.add_argument(
        "--graph_max_new_tokens_limit",
        type=int,
        default=None,
        help="Maximum output-token budget when retrying truncated graph chunks (VGENT_GRAPH_MAX_NEW_TOKENS_LIMIT; default: 16384).",
    )


def validate_graph_generation_args(args, parser):
    try:
        args.graph_max_new_tokens, args.graph_max_new_tokens_limit = resolve_graph_token_budgets(args.graph_max_new_tokens, args.graph_max_new_tokens_limit)
    except ValueError as exc:
        parser.error(str(exc))
