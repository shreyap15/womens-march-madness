from __future__ import annotations

import pandas as pd


def _slot_row(df: pd.DataFrame, slot: str):
    rows = df[df["Slot"] == slot]
    return rows.iloc[0] if not rows.empty else None


def generate_bracket_tree(
    results_path: str = "submissions/men/2026/bracket_2026_results.csv",
    out_path: str = "submissions/men/2026/bracket_2026_tree.txt",
) -> None:
    df = pd.read_csv(results_path)

    lines = []
    lines.append("2026 NCAA Men's Bracket Tree (Model Picks)")
    lines.append("")

    for rnd, label in [(1, "Round 1"), (2, "Round 2"), (3, "Sweet 16"), (4, "Elite 8"), (5, "Final Four"), (6, "Championship")]:
        lines.append(label)
        subset = df[df["Slot"].str.startswith(f"R{rnd}")]
        for _, r in subset.iterrows():
            a = r.get("StrongTeamName")
            b = r.get("WeakTeamName")
            w = r.get("WinnerTeamName")
            p = r.get("WinnerProb")
            if pd.isna(a) or pd.isna(b) or pd.isna(w):
                continue
            lines.append(f"  {r['Slot']}: {a} vs {b} -> {w} (p={p:.4f})")
        lines.append("")

    with open(out_path, "w", encoding="ascii") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    generate_bracket_tree()
