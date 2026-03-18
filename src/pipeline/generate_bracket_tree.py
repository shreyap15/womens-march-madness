from __future__ import annotations

import pandas as pd


def _slot_row(df: pd.DataFrame, slot: str) -> pd.Series | None:
    rows = df[df["Slot"] == slot]
    if rows.empty:
        return None
    return rows.iloc[0]


def _fmt_match(row: pd.Series) -> str:
    a = row.get("StrongTeamName")
    b = row.get("WeakTeamName")
    w = row.get("WinnerTeamName")
    p = row.get("WinnerProb")
    if pd.isna(a) or pd.isna(b) or pd.isna(w):
        return f"{row['Slot']}: TBD"
    return f"{row['Slot']}: {a} vs {b} -> {w} (p={p:.4f})"


def generate_bracket_tree(
    results_path: str = "submissions/women/2026/bracket_2026_results.csv",
    out_path: str = "submissions/women/2026/bracket_2026_tree.txt",
) -> None:
    df = pd.read_csv(results_path)

    regions = ["W", "X", "Y", "Z"]
    lines: list[str] = []
    lines.append("2026 NCAA Women's Bracket Tree (Model Picks)")
    lines.append("")

    for region in regions:
        lines.append(f"Region {region}")
        # Round 1
        lines.append("  Round 1")
        for i in range(1, 9):
            row = _slot_row(df, f"R1{region}{i}")
            if row is not None:
                lines.append(f"    {_fmt_match(row)}")
        # Round 2
        lines.append("  Round 2")
        for i in range(1, 5):
            row = _slot_row(df, f"R2{region}{i}")
            if row is not None:
                lines.append(f"    {_fmt_match(row)}")
        # Sweet 16
        lines.append("  Sweet 16")
        for i in range(1, 3):
            row = _slot_row(df, f"R3{region}{i}")
            if row is not None:
                lines.append(f"    {_fmt_match(row)}")
        # Elite 8
        lines.append("  Elite 8")
        row = _slot_row(df, f"R4{region}1")
        if row is not None:
            lines.append(f"    {_fmt_match(row)}")
        lines.append("")

    lines.append("Final Four")
    row = _slot_row(df, "R5WX")
    if row is not None:
        lines.append(f"  {_fmt_match(row)}")
    row = _slot_row(df, "R5YZ")
    if row is not None:
        lines.append(f"  {_fmt_match(row)}")
    lines.append("")

    lines.append("Championship")
    row = _slot_row(df, "R6CH")
    if row is not None:
        lines.append(f"  {_fmt_match(row)}")

    with open(out_path, "w", encoding="ascii") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    generate_bracket_tree()
