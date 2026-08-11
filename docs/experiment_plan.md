# Experiment plan

The leaderboard scores plain MSE on unseen ratings, so validation MSE is the decision metric.

For every candidate, write down the Dave Rule answers *before* implementation.

## First rung
1. Reproduce the course item-mean/rank-1 baseline.
2. Add simple bias structure (global + user + item), with regularization if useful.
3. Try low-rank latent factors and tune rank/regularization using the same validation protocol.
4. Only then consider uncertainty/weighting ideas whose assumptions match the data.

Each rung should produce one comparable validation result and, when warranted, a submission CSV.
