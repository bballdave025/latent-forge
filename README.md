# Latent Forge

Lean experimental repository for a recommender-system project.

## Trunk
Improve holdout MSE with small, interpretable experiments built from the rank-1 / matrix-factorization baseline.

## Dave Rule
Before implementing an idea:
1. What assumption changes?
2. Why should it improve?
3. How will success be measured?
4. Is the added complexity justified?

Interesting extensions that do not earn implementation now are captured in `docs/branches.md`.

## Layout
- `notebooks/` — experimental narrative and submissions
- `src/latent_forge/` — reusable model/evaluation code
- `docs/` — experiment plan and captured branches
- `tests/` — lightweight correctness checks
- `data/` — local contest data (gitignored)
- `submissions/` — generated leaderboard CSVs (gitignored)
