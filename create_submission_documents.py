"""Generate the five-page report and presentation PDFs from repository results."""

from __future__ import annotations

import csv
import textwrap
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def add_text_page(pdf: PdfPages, title: str, sections: list[tuple[str, str]]) -> None:
    figure = plt.figure(figsize=(8.27, 11.69))
    figure.text(0.08, 0.96, title, fontsize=18, weight="bold", va="top")
    y = 0.91
    for heading, body in sections:
        figure.text(0.08, y, heading, fontsize=12, weight="bold", va="top")
        y -= 0.027
        for paragraph in body.split("\n"):
            lines = textwrap.wrap(paragraph, width=104) or [""]
            for line in lines:
                figure.text(0.08, y, line, fontsize=9.2, va="top")
                y -= 0.018
            y -= 0.008
    pdf.savefig(figure)
    plt.close(figure)


def add_image_page(pdf: PdfPages, title: str, image_paths: list[Path], caption: str) -> None:
    figure = plt.figure(figsize=(8.27, 11.69))
    figure.text(0.08, 0.96, title, fontsize=18, weight="bold", va="top")
    positions = [(0.08, 0.52, 0.84, 0.35), (0.08, 0.10, 0.84, 0.35)]
    for image_path, position in zip(image_paths, positions):
        axis = figure.add_axes(position)
        axis.imshow(mpimg.imread(image_path))
        axis.axis("off")
    figure.text(0.08, 0.06, caption, fontsize=8.5, va="top")
    pdf.savefig(figure)
    plt.close(figure)


def add_results_page(pdf: PdfPages, table: str, pvalues: str) -> None:
    figure = plt.figure(figsize=(8.27, 11.69))
    figure.text(0.08, 0.96, "Results and Convergence Analysis", fontsize=18, weight="bold", va="top")
    figure.text(0.08, 0.91, "30-run summary (mean gap is relative to published BKS)", fontsize=12, weight="bold", va="top")
    y = 0.88
    for line in table.splitlines():
        figure.text(0.08, y, line, fontsize=8.2, family="monospace", va="top")
        y -= 0.018
    figure.text(0.08, 0.75, "Statistical comparison", fontsize=12, weight="bold", va="top")
    figure.text(0.08, 0.72, f"Two-sided Mann–Whitney U at α=0.05: {pvalues}. SA medians are lower for all instances.",
                fontsize=8.8, va="top")
    figure.text(0.08, 0.68, "Convergence and final-cost plots", fontsize=12, weight="bold", va="top")
    image_paths = [
        RESULTS / "plots" / f"{instance}_convergence.png"
        for instance in ("att48", "ch130", "a280")
    ] + [
        RESULTS / "plots" / f"{instance}_final_costs.png"
        for instance in ("att48", "ch130", "a280")
    ]
    positions = [
        (0.08, 0.49, 0.27, 0.16), (0.365, 0.49, 0.27, 0.16), (0.65, 0.49, 0.27, 0.16),
        (0.08, 0.20, 0.27, 0.16), (0.365, 0.20, 0.27, 0.16), (0.65, 0.20, 0.27, 0.16),
    ]
    for image_path, position in zip(image_paths, positions):
        axis = figure.add_axes(position)
        axis.imshow(mpimg.imread(image_path))
        axis.axis("off")
    pdf.savefig(figure)
    plt.close(figure)


def make_report() -> None:
    sa = read_csv(RESULTS / "summary" / "sa.csv")
    baseline = read_csv(RESULTS / "summary" / "baseline.csv")
    tests = read_csv(RESULTS / "summary" / "mann_whitney.csv")
    rows = {("SA", row["instance"]): row for row in sa}
    rows.update({("Baseline", row["instance"]): row for row in baseline})
    table = "\n".join(
        f"{label:8} {instance:6} best={rows[(label, instance)]['best']} "
        f"mean={float(rows[(label, instance)]['mean']):.2f} "
        f"median={rows[(label, instance)]['median']} "
        f"std={float(rows[(label, instance)]['std_dev']):.2f} "
        f"gap={float(rows[(label, instance)]['mean_gap_percent']):.2f}%"
        for label in ("SA", "Baseline")
        for instance in ("att48", "ch130", "a280")
    )
    pvalues = ", ".join(f"{r['instance']} p={float(r['p_value']):.3g}" for r in tests)

    with PdfPages(ROOT / "Group5_SA_Report.pdf") as pdf:
        add_text_page(pdf, "Solving NP-Hard Problems Using Simulated Annealing",
            [("ADDIS ABABA UNIVERSITY | Assignment 1 Submission",
              "Registered algorithm: Simulated Annealing (SA)\n"
              "Problem: Traveling Salesperson Problem (TSP) — att48, ch130, a280\n"
              "Group: 5 | Repository URL: [INSERT PRIVATE GITHUB URL]\n"
              "Member roster: smn | [INSERT STUDENT ID] | [INSERT TASK] | [INSERT %]\n"
              "Add remaining members and ensure contributions sum to 100%."),
             ("Executive summary",
              "This study implements simulated annealing from scratch for permutation-encoded TSP tours. "
              "It compares SA with an equal-budget random-restart 2-opt baseline over 30 independent runs "
              "per instance. The objective is tour length minimization."),
             ("Submission checklist",
              "Code, data scripts, experiments, raw CSV traces, plots, and reproducibility commands are in the repository. "
              "Complete the bracketed cover-page fields before submission.")])

        add_text_page(pdf, "Natural Inspiration and Mathematical Formulation",
            [("Annealing analogy",
              "Metallurgical annealing heats a material so particles move freely, then cools it slowly so the material settles "
              "into a low-energy structure. In SA, high temperature enables exploration by accepting some worse tours; low "
              "temperature emphasizes exploitation and improvement."),
             ("Metropolis rule and cooling",
              "For cost change Δ=f(candidate)-f(current), accept every improving move. For a worsening move, accept with "
              "P(accept)=exp(-Δ/T). Temperature is updated once per block: T_new=T_old*r, where "
              "r=(T_final/T_initial)^(1/number_of_blocks). Initial temperature is estimated from the median worsening "
              "2-opt change d using T0=-d/log(target_acceptance), then multiplied by the tuned temperature scale."),
             ("Encoding and 2-opt",
              "A solution is a permutation of city indices, so every city appears exactly once. A 2-opt move reverses a "
              "contiguous segment and therefore preserves feasibility. With edges (a,b) and (c,d) replaced by (a,c) and "
              "(b,d), Δ = dist(a,c)+dist(b,d)-dist(a,b)-dist(c,d)."),
             ("Numerical example",
              "Suppose current cost is 120, candidate cost is 130, so Δ=10. At T=20, P=exp(-10/20)=0.6065. "
              "If the random draw is 0.40, accept the worse candidate because 0.40<0.6065. If the draw is 0.80, reject it. "
              "This temporary worsening can escape a local optimum.")])

        add_text_page(pdf, "Problem and Experimental Setup",
            [("TSP formulation and benchmarks",
              "Given cities and distances d(i,j), minimize the closed-tour length "
              "Σ d(t_i,t_(i+1) mod n), subject to t being a permutation. "
              "att48 has 48 cities and ATT distances with BKS 10628; ch130 has 130 cities and EUC_2D distances with "
              "BKS 6110; a280 has 280 cities and EUC_2D distances with BKS 2579. BKS values are from the TSPLIB solutions "
              "file and Reinelt (1991)."),
             ("Baseline and evaluation budget",
              "The baseline is random-restart improving 2-opt. It accepts only improving moves and restarts after a fixed "
              "number of non-improving attempts. Both methods use 50,000 objective evaluations per run; the initial tour "
              "cost counts as evaluation 1. Each method uses 30 runs with seeds 2026–2055."),
             ("Parameter tuning",
              "The tuning grid uses five runs per configuration and seeds beginning at 9000. It tests target acceptance, "
              "initial-temperature scale, final temperature, and moves per temperature block. The selected settings are "
              "att48 (0.7, 0.75, 0.1, 200), ch130 (0.3, 0.25, 0.5, 50), and a280 (0.1, 0.25, 0.5, 200)."),
             ("Environment",
              "Python 3.12.3, matplotlib 3.6.3, scipy 1.11.4, Linux x86_64. Record the exact CPU and RAM before submission.")])

        add_results_page(pdf, table, pvalues)

        add_text_page(pdf, "Discussion, Limitations and References",
            [("Discussion",
              "SA balances exploration and exploitation through the temperature schedule. The heat maps document sensitivity "
              "to acceptance targets, initial temperature, final temperature, and block length. The a280 mean gap of 46.65% "
              "shows a scalability limitation: 280-city tours have a larger and more difficult search space, while the fixed "
              "50,000-evaluation budget and simple 2-opt neighborhood limit refinement."),
             ("Limitations",
              "The tuning study uses five runs per configuration, so its estimates are noisy. Only one neighborhood and one "
              "baseline are used. Hardware details must be completed from the actual machine, and the cover-page identities "
              "must be completed by the group."),
             ("References",
              "[1] S. Kirkpatrick, C. D. Gelatt Jr., and M. P. Vecchi, “Optimization by Simulated Annealing,” Science, "
              "vol. 220, no. 4598, pp. 671–680, 1983.\n"
              "[2] N. Metropolis et al., “Equation of State Calculations by Fast Computing Machines,” Journal of Chemical Physics, "
              "vol. 21, no. 6, pp. 1087–1092, 1953.\n"
              "[3] G. Reinelt, “TSPLIB—A Traveling Salesman Problem Library,” ORSA Journal on Computing, vol. 3, no. 4, "
              "pp. 376–384, 1991.\n"
              "[4] G. A. Croes, “A Method for Solving Traveling-Salesman Problems,” Operations Research, vol. 6, no. 6, "
              "pp. 791–812, 1958.")])


def make_slides() -> None:
    slides = [
        ("Group 5 — Simulated Annealing for TSP",
         "Cover fields to complete: member names, IDs, contributions, and private repository URL.\n"
         "Instances: att48, ch130, a280"),
        ("1. Natural inspiration | 2 minutes",
         "Annealing heats metal to allow particle movement, then cools slowly to settle into a stable structure.\n\n"
         "SA mapping:\n• High T: exploration; accept some worse tours.\n• Low T: exploitation; focus on improvements.\n• Cooling: gradually reduces disruption."),
        ("2. Mathematical formulation | 3 minutes",
         "Tour = permutation of city indices.\n\n"
         "Δ = f(candidate) − f(current)\n"
         "Accept if Δ≤0; otherwise accept with P=exp(−Δ/T).\n"
         "T_new=T_old·r, r=(T_final/T_initial)^(1/blocks).\n"
         "2-opt replaces (a,b),(c,d) with (a,c),(b,d)."),
        ("Numerical example",
         "Current cost = 120; candidate cost = 130; Δ=10; T=20.\n"
         "P=exp(−10/20)=0.6065.\n"
         "Random draw 0.40 → accept.\n"
         "Random draw 0.80 → reject.\n\n"
         "Initial T is estimated from typical worsening 2-opt moves and the target acceptance."),
        ("3. Benchmark protocol | 4 minutes",
         "TSPLIB: att48 (48, ATT, BKS 10628), ch130 (130, EUC_2D, BKS 6110), "
         "a280 (280, EUC_2D, BKS 2579).\n\n"
         "30 runs per method, seeds 2026–2055, 50,000 objective evaluations per run.\n"
         "Baseline: random-restart improving 2-opt."),
        ("Results",
         "SA mean gaps: att48 2.59%, ch130 9.04%, a280 46.65%.\n"
         "Baseline mean gaps: att48 14.26%, ch130 42.52%, a280 76.00%.\n\n"
         "Mann–Whitney p-values: approximately 3.0×10⁻¹¹ for every instance; SA medians are significantly lower."),
        ("Convergence and sensitivity",
         "Show the convergence and final-cost plots from results/plots/.\n"
         "Show the six tuning heat maps from results/plots/tuning/.\n\n"
         "Selected settings: att48 (0.7, 0.75, 0.1, 200), ch130 (0.3, 0.25, 0.5, 50), "
         "a280 (0.1, 0.25, 0.5, 200)."),
        ("4. Synthesis | 1 minute + Q&A",
         "Strength: accepts temporary worsening and escapes local optima.\n"
         "Sensitivity: initial temperature and cooling schedule matter.\n"
         "Limitation: fixed budget and simple 2-opt neighborhood scale poorly; a280 retains a 46.65% mean gap.\n\n"
         "References: Kirkpatrick et al. (1983), Metropolis et al. (1953), Reinelt (1991), Croes (1958)."),
    ]
    with PdfPages(ROOT / "Group5_SA_Slides.pdf") as pdf:
        for title, body in slides:
            figure = plt.figure(figsize=(13.33, 7.5))
            figure.text(0.08, 0.84, title, fontsize=25, weight="bold", va="top")
            figure.text(0.08, 0.70, body, fontsize=18, va="top", linespacing=1.5)
            pdf.savefig(figure)
            plt.close(figure)


if __name__ == "__main__":
    make_report()
    make_slides()
    print("Created Group5_SA_Report.pdf and Group5_SA_Slides.pdf")
