"""Small, dependency-free helpers for displaying RAG results safely."""

from __future__ import annotations

from collections.abc import Mapping


def extract_answer(response: str, prompt: str) -> str:
    """Return generated text without an accidentally echoed input prompt."""
    if not isinstance(response, str):
        raise TypeError("response must be a string")

    answer = response
    if prompt and answer.startswith(prompt):
        answer = answer[len(prompt) :]
    return answer.strip()


def format_source_label(metadata: Mapping[str, object] | None) -> str:
    """Format only a source basename and scalar page number for display."""
    metadata = metadata or {}
    source = metadata.get("source")
    if isinstance(source, str) and source.strip():
        # Normalize both POSIX and Windows paths so machine-specific paths are
        # not rendered in the user-facing source list.
        filename = source.strip().replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    else:
        filename = "Unknown source"
    if not filename:
        filename = "Unknown source"

    page = metadata.get("page")
    if isinstance(page, (str, int)) and not isinstance(page, bool) and str(page).strip():
        return f"{filename} (page {page})"
    return filename
