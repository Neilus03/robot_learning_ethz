"""Regenerate PDFs from logs/: run from repo root with `python results_pdf/build_pdfs.py`."""

from __future__ import annotations

from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
OUT_DIR = Path(__file__).resolve().parent

DQN_EVAL_SUMMARY = """Using device: cuda
GPU name: NVIDIA GeForce GTX 1650 Ti
Loaded checkpoint from: logs/dqn/models/dqn_cartpole.pth

===== Evaluation Summary =====
Number of episodes : 20
Mean return        : 493.60
Std return         : 18.77
Min return         : 416.00
Max return         : 500.00
Median return      : 500.00
Mean length        : 493.60
Std length         : 18.77
Success threshold  : 475.0
Success rate       : 95.0%
"""


def _add_image_page(pdf: PdfPages, path: Path, title: str) -> None:
    if not path.is_file():
        return
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")
    ax.set_title(title, fontsize=11, pad=12)
    ax.imshow(mpimg.imread(path))
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _add_text_page(pdf: PdfPages, title: str, body: str) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", loc="left", pad=16)
    ax.text(
        0.02,
        0.98,
        body,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        family="monospace",
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    slip = [("0", "slip = 0"), ("0.01", "slip = 0.01"), ("0.2", "slip = 0.2")]
    with PdfPages(OUT_DIR / "hw4_ex1_mdp_results.pdf") as pdf:
        for s, lab in slip:
            _add_image_page(pdf, LOGS / "mdp" / f"policy_iteration_values_slip_{s}.png", f"PI — values ({lab})")
        for s, lab in slip:
            _add_image_page(pdf, LOGS / "mdp" / f"policy_iteration_policy_slip_{s}.png", f"PI — policy ({lab})")
        for s, lab in slip:
            _add_image_page(pdf, LOGS / "mdp" / f"value_iteration_values_slip_{s}.png", f"VI — values ({lab})")
        for s, lab in slip:
            _add_image_page(pdf, LOGS / "mdp" / f"value_iteration_policy_slip_{s}.png", f"VI — policy ({lab})")

    with PdfPages(OUT_DIR / "hw4_ex2_dqn_results.pdf") as pdf:
        _add_image_page(pdf, LOGS / "dqn" / "results" / "dqn_training_curve.png", "DQN training curve")
        _add_text_page(pdf, "DQN evaluation summary", DQN_EVAL_SUMMARY)

    ppo = sorted((LOGS / "ppo").rglob("*.png")) if (LOGS / "ppo").exists() else []
    sac = sorted((LOGS / "sac").rglob("*.png")) if (LOGS / "sac").exists() else []
    with PdfPages(OUT_DIR / "hw4_ex3_ex4_ppo_sac_results.pdf") as pdf:
        if ppo:
            for p in ppo:
                _add_image_page(pdf, p, f"PPO — {p.name}")
        else:
            _add_text_page(
                pdf,
                "Exercise 3 (PPO)",
                "No PNGs under logs/ppo/. Add TensorBoard screenshots or plots, then re-run this script.\n"
                "Include eval output from: python scripts/eval_ppo.py",
            )
        if sac:
            for p in sac:
                _add_image_page(pdf, p, f"SAC — {p.name}")
        else:
            _add_text_page(
                pdf,
                "Exercise 4 (SAC)",
                "No PNGs under logs/sac/. Add TensorBoard screenshots or plots, then re-run this script.\n"
                "Include eval output from: python scripts/eval_sac.py",
            )

    print("Wrote", OUT_DIR / "hw4_ex1_mdp_results.pdf")
    print("Wrote", OUT_DIR / "hw4_ex2_dqn_results.pdf")
    print("Wrote", OUT_DIR / "hw4_ex3_ex4_ppo_sac_results.pdf")


if __name__ == "__main__":
    main()
