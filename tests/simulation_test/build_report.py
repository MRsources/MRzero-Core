"""
Assemble the PR comment from the summaries written by plot_comparison.py.

Usage: python build_report.py <report_folder> <image_base_url> <run_url>
Prints the markdown comment to stdout.
"""
import glob
import json
import os
import sys

MARKER = "<!-- simulation-test-report -->"
ICONS = {"passed": "✅", "failed": "❌", "error": "⚠️"}


def main():
    report_folder, image_base_url, run_url = sys.argv[1:4]
    summaries = []
    for path in sorted(glob.glob(os.path.join(report_folder, "*.json"))):
        with open(path) as f:
            summaries.append(json.load(f))

    lines = [
        MARKER,
        "## Simulation test: main (before) vs. this PR (after)",
        "",
        "| Sequence | Status | max mag NRMSE | max phase RMSE | max runtime Δ |",
        "|---|---|---|---|---|",
    ]
    for s in summaries:
        if s["status"] == "error":
            lines.append(f"| {s['name']} | {ICONS['error']} | – | – | – |")
        else:
            lines.append(
                f"| {s['name']} | {ICONS[s['status']]} | {s['max_mag_nrmse']:.2e} "
                f"| {s['max_phase_rmse']:.2e} | {s['max_dt_percent']:+.1f}% |"
            )
    lines.append("")

    for s in summaries:
        # Failing sequences are expanded, passing ones collapsed
        open_attr = " open" if s["status"] != "passed" else ""
        lines.append(f"<details{open_attr}><summary>{ICONS[s['status']]} {s['name']}: {s['message']}</summary>")
        lines.append("")
        if "image" in s:
            lines.append(f"![{s['name']}]({image_base_url}/{s['image']})")
        lines.append("")
        lines.append("</details>")
        lines.append("")

    lines.append(f"[Workflow run]({run_url}) — raw `.npz` data is attached as artifacts.")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
