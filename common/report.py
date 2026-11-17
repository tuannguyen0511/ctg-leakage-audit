"""Aggregate-only P21 reporting helpers."""

from collections.abc import Mapping


def safe_count(count: int) -> str:
    """Suppress nonzero counts smaller than 10 in user-facing reports."""
    count = int(count)
    return "<10" if 1 <= count <= 9 else str(count)


def render_report(
    task: str,
    summary: Mapping[str, str],
    acceptance: Mapping[str, bool],
    warnings: str = "none",
) -> str:
    lines = [f"=== REPORT {task} ==="]
    lines.extend(f"{name}: {value}" for name, value in summary.items())
    lines.append(
        "acceptance: "
        + "; ".join(f"{name}={'PASS' if result else 'FAIL'}" for name, result in acceptance.items())
    )
    lines.append(f"warnings: {warnings}")
    lines.append("=== END ===")
    if len(lines) > 30:
        raise ValueError("REPORT exceeds 30 lines")
    return "\n".join(lines)
