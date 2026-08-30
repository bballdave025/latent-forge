*BEGIN: Context Document Header*

# CONTEXT DOCUMENT --- Continuation

## Project

**Name:**\
Latent Forge

**Description:**\
A synthetic-data playground for learning matrix factorization,
collaborative filtering, latent representations, and recommender-system
experimentation.

------------------------------------------------------------------------

## Continuation Metadata

**Prepared at:**\
1788094667_2026-08-30T13:57:47-0400

Generated via:

``` bash
date +'%s_%Y-%m-%dT%H:%M:%S%z'
```

(Boston, MA time)

**Continued from chat:**\
Latent Forge Implementation

**Also involving:** - reconstruction of the prior MLU/course
matrix-factorization assignment - design of a negative-control
synthetic-data generator - a separate MMJL CSV MIME-type investigation,
intentionally moved out of this project thread

------------------------------------------------------------------------

## Author / Source

**User (GitHub):**\
@bballdave025

**User (ChatGPT):**\
Dave / D Black

------------------------------------------------------------------------

## Intent for This Context

Preserve the exact Latent Forge engineering and mathematical state
immediately after reconstructing the course's learned
matrix-factorization and early-stopping implementation, so work can
resume without repeating that archaeology. The immediate continuation is
to finish the fixed baselines/invariants, preserve the course
implementation faithfully, and then introduce learned rank-1
factorization and Generator 0.

------------------------------------------------------------------------

## Usage Instructions

-   Treat this document as **authoritative project state**.
-   Continue with **minimal re-derivation**.
-   Reinterpret only when explicitly requested.

*ENDOF: Context Document Header*

# Current State

The project is at the boundary between the fixed-baseline phase and
learned matrix factorization.

The previous course/MLU implementation has now been recovered
sufficiently from the printed assignment pages. The important missing
material is no longer missing: the learned `k=2` factorization, masked
MSE objective, PyTorch autograd training loop, validation behavior,
early-stopping logic, best-parameter restoration, and test-prediction
path have all been captured.

**The physical-course-paper capture task is complete.** Future work
should not assume that another missing backpropagation page still needs
to be found.

The current implementation philosophy remains:

> **Capture the branch. Stay on the trunk.**

and:

> **Lean-to, not cathedral.**

The notebook should continue to tell the mathematical/experimental
story. Reusable code can move into `src/latent_forge` when repetition or
clarity justifies it.

# Mathematical and Naming Conventions

## Matrix-factorization notation

Use lower-case mathematical `k` for latent dimensionality/rank. Use
`\ell` as the Einstein summation/contraction index so that `k` is not
simultaneously used as a free hyperparameter and a dummy index:

$$
\hat R^i{}_j = A^i{}_\ell F^\ell{}_j,
\qquad \ell=1,\ldots,k.
$$

Matrix form:

$$
\hat{\mathbf R} = \mathbf A\mathbf F.
$$

For `n` users and `m` items,

$$
\mathbf A\in\mathbb R^{n\times k},
\qquad
\mathbf F\in\mathbb R^{k\times m}.
$$

Python names should generally use the explicit `num_` prefix rather than
`n_`, while ordinary compact mathematical symbols such as `n`, `m`, and
`k` remain appropriate in equations.

## Truth and generator notation

`T` is reserved for dense synthetic **truth**, rather than treating
`R = ratings` as sacrosanct.

Generator versions may be denoted by `G^(v)` in prose/math when version
identity matters. Generator 0 is the negative-control generator,
`G^(0)`, and its dense truth may be written:

$$
T^{(0)i}{}_j.
$$

Do not force generator-version notation into every equation when the
version is already obvious from context.

## Existing observed-matrix convention

The reconstructed notebook currently uses:

-   `S_train`, `S_validate`: rating matrices with unobserved entries
    filled with zero only after masks are created.
-   `R_train`, `R_validate`: observation masks.
-   `S_train_df`, `S_validate_df`: independently constructed pivot
    DataFrames, explicitly reindexed to canonical user/item coordinates
    before conversion to arrays.

The zero in an unobserved entry of `S_*` is a storage placeholder, **not
a zero-star rating**.

<!--
# Original MLU Assignment → Latent Forge Variable Equivalences

The right arrow means "the old assignment variable is now represented in
Latent Forge as":

``` text
training -> train_data
test_features -> test_data
unique_users -> users_list
unique_asins -> asins_list
n_users -> num_users
n_asins -> num_asins

train -> train
val -> validate
n_train -> num_train
n_val -> num_validate

counts_reviews_users -> reviews_per_user
counts_rating_asin -> ratings_book_counts
average_rating_asin -> ratings_book_mean

S_train -> S_train
R_train -> R_train
S_val -> S_validate
R_val -> R_validate

n_null -> num_train_null        # when specifically counting training-matrix nulls
n_non_null -> num_train_observed

n_features -> num_features      # mathematical latent dimensionality/rank k

baseline_A -> baseline_A_asin_mean
baseline_F -> baseline_F_asin_mean
baseline_P -> baseline_P_asin_mean
avg_rating -> avg_rating_by_asin

{{stuff with submission}} -> baseline_k1_asin_mean_result
```

For PyTorch tensors, prefer explicit names such as:

``` text
s_train -> S_train_tensor
r_train -> R_train_tensor
s_val -> S_validate_tensor
r_val -> R_validate_tensor
```

rather than making capitalization alone carry the NumPy-versus-PyTorch
distinction.

This mapping is intentionally a lean working dictionary, not a
requirement to catalog every historical variable before continuing.

-->

# Current Data-to-Matrix Construction

The canonical user/item lists come from the full labeled data. Train and
validation pivots are created independently, then explicitly reindexed
onto those canonical coordinates:

``` python
S_train_df = pd.pivot_table(
    train, values="Rating", index="User", columns="ASIN"
).reindex(index=users_list, columns=asins_list)

S_validate_df = pd.pivot_table(
    validate, values="Rating", index="User", columns="ASIN"
).reindex(index=users_list, columns=asins_list)

R_train = S_train_df.notna().to_numpy(dtype=np.int8)
R_validate = S_validate_df.notna().to_numpy(dtype=np.int8)

S_train = S_train_df.fillna(0).to_numpy()
S_validate = S_validate_df.fillna(0).to_numpy()
```

Important invariant: masks are created **before** missing values are
filled with zero.

Current correctness checks include, or should include before moving on:

``` python
expected_shape = (num_users, num_asins)

assert S_train.shape == expected_shape
assert R_train.shape == expected_shape
assert S_validate.shape == expected_shape
assert R_validate.shape == expected_shape

assert S_train.shape == S_validate.shape
assert R_train.shape == R_validate.shape

assert int(R_train.sum()) == num_train
assert int(R_validate.sum()) == num_validate
```

These assertions should be upgraded with informative failure messages.

Also add an explicit uniqueness assertion/check for `(User, ASIN)` pairs
before relying on `pivot_table`, because `pivot_table` can silently
aggregate duplicates.

For sparsity:

$$
N_{\rm null} + N_{\rm observed} = nm,
$$

and:

$$
\text{sparsity} + \text{density} = 1.
$$

# Fixed Baselines

The intended fixed-baseline progression is:

$$
\text{global mean}
\rightarrow
\{\text{item mean},\text{user mean}\}
\rightarrow
\text{learned }k=1
\rightarrow
k>1.
$$

## Item-mean constrained rank-1 baseline

$$
\hat R^i{}_j = \mu_j
$$

or:

$$
\hat{\mathbf R} = \mathbf 1_n\boldsymbol\mu_{\rm item}^\mathsf T.
$$

This is a constrained `k=1` factorization:

$$
A^i{}_1=1,
\qquad
F^1{}_j=\mu_j.
$$

Current implementation names:

``` python
baseline_A_asin_mean
baseline_F_asin_mean
baseline_P_asin_mean
avg_rating_by_asin
```

## User-mean constrained rank-1 baseline

$$
\hat R^i{}_j = \mu^i
$$

or:

$$
\hat{\mathbf R} = \boldsymbol\mu_{\rm user}\mathbf 1_m^\mathsf T.
$$

This is the complementary constrained `k=1` factorization:

$$
A^i{}_1=\mu^i,
\qquad
F^1{}_j=1.
$$

## Global mean baseline

This is selected as the next missing fixed baseline and should be added
before learned factorization:

$$
\hat R^i{}_j = \mu.
$$

All baseline statistics must be estimated from **training data only**.

# Evaluation Objective

The reconstructed course derives squared error from an independent
Gaussian observation model. The implemented masked mean-squared error
is:

$$
\mathcal L(A,F) = \frac{1}{N_{\rm obs}}
\sum_{i,j}
R^i{}_j
\left(
S^i{}_j-A^i{}_\ell F^\ell{}_j
\right)^2.
$$

Equivalent matrix-style implementation:

``` python
def loss_function(A, F, R, S, num_observed):
    return (
        1 / num_observed
    ) * (torch.linalg.norm(R * (S - model(A, F))) ** 2)
```

For evaluation, use **MSE** and **RMSE** terminology. The course
material sometimes says RMS; Latent Forge should use RMSE when referring
to root mean squared prediction error.

# Reconstructed Course Learned-Factorization Implementation

The recovered course model uses:

``` python
num_features = 2
A = torch.normal(
    0, 1,
    size=(num_users, num_features),
    requires_grad=True,
)
F = torch.normal(
    0, 1,
    size=(num_features, num_asins),
    requires_grad=True,
)

def model(A, F):
    return torch.matmul(A, F)
```

Thus the course implementation is learned `k=2`.

The course does **not** manually implement the derivatives. PyTorch
autograd performs backpropagation:

``` python
train_loss.backward()
```

and plain gradient descent updates the factors:

``` python
with torch.no_grad():
    A -= lr * A.grad
    F -= lr * F.grad

A.grad.zero_()
F.grad.zero_()
```

Recovered course hyperparameters:

``` python
num_iter = 500
lr = 25
torch.manual_seed(1)
```

`lr = 25` should be preserved when documenting/reproducing the
historical course implementation, but should **not** automatically
become a general Latent Forge default. The loss is normalized by the
number of observed ratings, which affects the gradient scale.

One subtlety in the recovered loop: the recorded training loss is
computed immediately **before** the parameter update, while validation
loss is computed **after** the update. This means nominally
corresponding train/validation points do not evaluate exactly the same
`A,F`.

This historical behavior should be preserved when documenting the course
implementation, but Latent Forge will deliberately correct it rather than
begin the project experiments with the inconsistency.

# Deliberate Latent Forge Corrections to the Recovered Course Loop

The recovered course implementation is useful historical/reference behavior,
but Latent Forge will not intentionally reproduce known bookkeeping problems
merely for fidelity.

## Train/validation evaluation timing

In the recovered course loop, the stored training loss is evaluated before
the gradient update, while the validation loss is evaluated after the update.
Thus nominally corresponding train and validation points refer to different
parameter states:

$$
\mathcal L_{\mathrm{train}}(A_t,F_t)
\qquad\text{vs.}\qquad
\mathcal L_{\mathrm{validate}}(A_{t+1},F_{t+1}).
$$

This is an evaluation/bookkeeping error, **not** an error in PyTorch
backpropagation or in the gradient update itself.

Latent Forge will intentionally correct it by evaluating both reported
training and validation losses from the **same parameter state after each
update**. The historical mismatch should be mentioned when reproducing or
comparing against the course implementation.

A suitable structure is:

```python
for epoch in range(num_iter):
    train_loss = loss_function(
        A, F,
        R_train_tensor,
        S_train_tensor,
        num_train,
    )

    train_loss.backward()

    with torch.no_grad():
        A -= learning_rate * A.grad
        F -= learning_rate * F.grad

    A.grad.zero_()
    F.grad.zero_()

    # Report both losses from the same updated parameter state.
    with torch.no_grad():
        train_loss = loss_function(
            A, F,
            R_train_tensor,
            S_train_tensor,
            num_train,
        )
        validate_loss = loss_function(
            A, F,
            R_validate_tensor,
            S_validate_tensor,
            num_validate,
        )

    train_losses.append(train_loss.detach())
    validate_losses.append(validate_loss.detach())
```

## PyTorch tensor naming

Use explicit tensor names:

```text
S_train_tensor
R_train_tensor
S_validate_tensor
R_validate_tensor
```

rather than using lowercase variable names to distinguish PyTorch tensors
from the corresponding NumPy arrays. Capitalization should not carry the
semantic burden of NumPy-versus-PyTorch representation.

## Learning rate

The recovered course value `lr = 25` is reference behavior, not a Latent
Forge default.

Do **not** assume that increasing latent dimension $k$ mechanically requires
decreasing the learning rate. There is no general rule that increasing model
complexity alone requires a lower learning rate, either for this matrix
factorization or for machine-learning models in general.

A useful learning rate depends more directly on optimization properties such
as gradient scale, loss curvature, parameterization, initialization, loss
normalization, optimizer choice, and training dynamics. Increasing $k$ can
change those properties, so the useful learning rate may change with $k$;
that should be observed or tested rather than encoded as a monotonic rule.

For this model, normalization of the loss by the number of observed ratings
is especially relevant to gradient scale and helps explain why the recovered
course value `lr = 25` can be numerically reasonable despite looking large.

# Reconstructed Validation and Early Stopping

The original `k=2` run showed monotonically improving training loss
while validation loss bottomed out and then began to rise slightly.

Approximate recovered values:

``` text
epoch  50: train MSE 0.5205, validation MSE 0.6383
epoch 100: train MSE 0.4054, validation MSE 0.4884
epoch 150: train MSE 0.3914, validation MSE 0.4691
epoch 200: train MSE 0.3849, validation MSE 0.4624
epoch 250: train MSE 0.3806, validation MSE 0.4598
epoch 300: train MSE 0.3772, validation MSE 0.4592
epoch 350: train MSE 0.3741, validation MSE 0.4595
epoch 400: train MSE 0.3714, validation MSE 0.4602
epoch 450: train MSE 0.3689, validation MSE 0.4612
epoch 500: train MSE 0.3667, validation MSE 0.4623
```

The full run ended around validation MSE `0.4623`, RMSE `0.6799`.

The course then inspected a selected plotting range:

``` python
start, end = 100, 500
plot_epochs = range(start, end)

plt.plot(plot_epochs, train_losses[start:end], label="Train Loss")
plt.plot(plot_epochs, val_losses[start:end], label="Validation Loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.legend()

best_val_mse = min(val_losses)
best_val_rms = np.sqrt(best_val_mse)
min_index = np.argmin(val_losses)

plt.title(
    f"Best Validation Loss: {best_val_mse:.4f}; "
    f"Best Validation RMS: {best_val_rms:.4f}\n"
    f"at iteration {min_index}"
)
plt.show()
```

Recovered best validation behavior was approximately:

$$
\operatorname{MSE}_{val}\approx0.4592,
\qquad
\operatorname{RMSE}_{val}\approx0.6776,
$$

around iteration 304.

The retraining/early-stopping path resets the seed and factors, then
tracks:

``` python
best_val_loss = float("inf")
best_epoch = 0
patience_counter = 0
saved_A = None
saved_F = None
```

On improvement:

``` python
if val_loss < best_val_loss:
    best_val_loss = val_loss
    best_epoch = epoch
    patience_counter = 0
    saved_A = A.clone()
    saved_F = F.clone()
else:
    patience_counter += 1
```

The recovered implementation uses a patience criterion (apparently
`patience = 1` in the printout) and restores the best saved factors
before final prediction.

Latent Forge should retain the pedagogically important behavior of saving the
best-known `A` and `F` whenever validation loss improves and restoring those
parameters for final predictions. Early stopping therefore means:

1. monitor validation loss;
2. save a checkpoint on improvement;
3. increment a patience counter when improvement fails;
4. stop when the selected patience rule is satisfied; and
5. restore the best saved checkpoint.

Do **not** replace this with merely stopping at the first observed increase.

The exact epoch/index semantics of the recovered course implementation,
including its apparent `patience = 1`, should still be reconstructed before
choosing Latent Forge's final patience convention.

# Learned Rank-1 Bridge

The pedagogical bridge selected for Latent Forge is to insert a
genuinely learned `k=1` model between the constrained rank-1 baselines
and the course's learned `k=2` model.

Item-mean baseline:

$$
A^i{}_1=1,
\qquad
F^1{}_j=\mu_j.
$$

User-mean baseline:

$$
A^i{}_1=\mu^i,
\qquad
F^1{}_j=1.
$$

Learned rank-1:

$$
A^i{}_1
\quad\text{and}\quad
F^1{}_j
$$

are both trainable.

Then generalize to:

$$
\hat R^i{}_j=A^i{}_\ell F^\ell{}_j,
\qquad \ell=1,\ldots,k.
$$

Before merely copying PyTorch autograd, derive the learned-`k=1`
gradients explicitly in the project's Einstein notation so the
mathematical step is understood.

# Generator 0 --- Negative Control

The first synthetic generator is deliberately a **negative-control
generator**: it should contain no genuine collaborative preference
structure for matrix factorization to discover.

Target scale remains approximately:

``` text
num_users = 1800
num_items = 1500
```

Dense truth is cheap enough at this scale and should be generated
directly:

$$
T^{(0)i}{}_j
\sim
\operatorname{DiscreteUniform}\{1,2,3,4,5\}.
$$

Conceptual parameters currently selected:

``` python
NUM_USERS = 1800
NUM_ITEMS = 1500

MIN_RATINGS_PER_USER = 40
MIN_RATINGS_PER_ITEM = 40

OBSERVATION_PROBABILITY = 0.04

RATING_MIN = 1
RATING_MAX = 5

RANDOM_SEED = 137
```

Use `np.random.default_rng(RANDOM_SEED)` for the generator rather than
relying on legacy global NumPy RNG state.

## Observation process

The dense truth and observation process are conceptually separate.

A simple observation process:

1.  Draw a Boolean observation mask independently with probability
    approximately `0.04`.
2.  For any user with fewer than 40 observed entries, randomly add
    currently-unobserved items until the floor is reached.
3.  For any item with fewer than 40 observed entries, randomly add
    currently-unobserved users until the floor is reached.

Repairs only **add** edges, so repairing item support cannot invalidate
the already-established user floor.

A Boolean coordinate naturally guarantees at most one rating per
user-item pair.

At `p = 0.04`:

$$
E[\text{ratings/user}]=1500(0.04)=60,
$$

and:

$$
E[\text{ratings/item}]=1800(0.04)=72.
$$

Thus 40 is a support floor, not a forced constant degree.

Do **not** force each user/item to contain exactly 20% of each star
rating. The truth entries should be IID uniform; finite-sample
fluctuations are desirable.

## Scientific expectation

For Generator 0:

$$
P(T^{(0)i}{}_j=r)=\frac15,
\qquad
E[T^{(0)i}{}_j]=3.
$$

There is no true user/item preference signal.

Therefore:

-   the global mean should be close to the population-optimal predictor;
-   item/user means may fit sampling fluctuations and can perform
    slightly worse on validation;
-   learned `k=1` or larger `k` should have no genuine held-out
    advantage;
-   a dramatic held-out improvement from factorization is evidence to
    investigate leakage, masking, coordinate alignment, or evaluation
    bugs.

Generator 0's experimental question is:

> **Can Latent Forge correctly discover that there is no latent
> structure to discover?**

This is why `G^(0)` is a negative control rather than merely "the first
crude generator."

# Synthetic Train / Validation / Hidden-Test Strategy --- Still Open

The synthetic path should ultimately expose a course-compatible
long-form interface while retaining the complete dense truth privately.

Likely structure:

-   dense `truth_matrix` generated first;
-   sparse observation coordinates generated separately;
-   observed coordinates converted to long-form data;
-   observed data divided into training/validation/hidden-test roles;
-   hidden labels remain available internally for evaluation even when
    the model-facing table omits them.

The existing file-input workflow uses labeled `train_data` plus
unlabeled `test_data`. The synthetic path should probably create
compatible tables without pretending the dense truth is observed.

**Not yet settled:** the exact splitting procedure and whether minimum
support constraints apply to total observations only or must also be
guaranteed after the training split. Baseline user/item means require
every evaluated user/item to have appropriate training support, so this
must be asserted or deliberately enforced.

Do not solve this by adding generator complexity prematurely.

# Experiment Progression

The current selected progression is:

1.  Global mean baseline.
2.  Item-mean and user-mean constrained rank-1 baselines.
3.  Learned `k=1`.
4.  Learned `k=2` as the recovered course/reference case.
5.  General learned `k`.
6.  Later fuller bias-plus-latent model:

$$
\hat R^i{}_jb = \mu+b^i+c_j+U^i{}_\ell V^\ell{}_j.
$$

Generator complexity should independently increase one assumption at a
time after Generator 0 works. Candidate future additions include biases,
planted low-rank structure, clustered preferences, popularity effects,
non-random missingness, noise, and eventually richer interpretable/lore
layers.

These are **deferred experiment families**, not requirements for the
current checkpoint.

# Effective Rank --- Planned, Not Yet the Trunk

Later investigations may compare:

-   known generative dimensionality,
-   singular-value spectrum of dense truth,
-   energy-threshold rank,
-   spectral elbow,
-   stable rank,
-   entropy/effective rank,
-   validation-optimal `k`.

This is deliberately downstream of the negative control and simple
learned-factorization progression.

# MMJL Side Branch

During capture of the printed course work, a separate MMJL issue was
rediscovered:

``` text
%%jupy_file --mime text/csv ...
```

fails with an error resembling:

``` text
ValueError: Unsupported or undetected MIME type: None
```

while `text/plain` works as a workaround.

That investigation has been intentionally moved to a separate chat so it
does not muddy Latent Forge context. It should be investigated from MMJL
source before filing an issue, including whether JSON/XML are supported.

No MMJL implementation work belongs on the Latent Forge trunk.

# Current Eightfold Way / Execution Plan

The current plan, ordered for execution rather than historical
conception:

1.  **Course-paper capture --- COMPLETE.** The
    factorization/backprop/early-stopping material needed for
    continuation has been recovered.
2.  **Finish fixed-baseline checkpoint.** Add global mean; add
    informative assertion messages; add explicit duplicate
    `(User, ASIN)` checks.
3.  **Preserve/reconstruct the course `k=2` implementation faithfully.**
    Keep historical behavior distinguishable from deliberate LF
    improvements; in the Latent Forge loop, intentionally fix the recovered
    pre-update-training/post-update-validation timing mismatch so both
    reported losses use the same parameter state.
4.  **Derive and implement learned `k=1`.** Work through the gradients
    in Einstein notation, then connect them to PyTorch autograd.
5.  **Establish the first notebook checkpoint as Markdown.** Separate
    notebook cells into separate code blocks and include the agreed
    corrections before continuing.
6.  **Implement `G^(0)`, the negative-control generator.** Dense IID
    1--5 truth plus a separately defined sparse observation process.
7.  **Use generator complexity as an experimental axis.** Add one known
    source of structure at a time and test whether the model recovers
    it.
8.  **Generalize learned factorization beyond `k=1`.** Compare `k`
    values and later approach the fuller bias-plus-latent model.

# Immediate Next Steps

When resuming, do **not** begin by redesigning the generator or
rereading the printed packet.

Use a small executable batch:

1.  Add the **global-mean baseline** to the current checkpoint.
2.  Add **informative assertion messages** and the explicit **duplicate
    `(User, ASIN)` invariant**.
3.  Reconstruct the recovered **course `k=2` training cell** in Latent
    Forge naming while clearly marking any historical behavior that has
    not yet been intentionally changed.
4.  Inspect results/state before proceeding to the learned-`k=1`
    derivation.

The first two steps are the preferred immediate re-entry task if only a
short work session is available. Do not skip the global-mean baseline merely because the item-mean and user-mean baselines are already implemented; it is the intentionally simpler reference point for the entire baseline hierarchy.

After those first two steps are understood and stable, prepare/review
`first_checkpoint_notebook_as_markdown.md` rather than trying to polish
it prematurely.

# Active Questions

1. What exact synthetic train/validation/hidden-test split best preserves
   the course-compatible interface while guaranteeing sufficient training
   support?
2. What learning-rate value or selection procedure works robustly for learned
   $k=1$ and subsequent $k>1$ experiments?
3. What exact early-stopping semantics should Latent Forge use after the
   recovered `patience = 1` behavior is reproduced and understood?
4. At what point should learned-factorization code move from the notebook
   into `src/latent_forge`?
5. After Generator 0, what is the smallest useful **positive-control**
   generator---e.g. a single planted low-rank structure---without jumping
   into generator cathedral-building?

# Deliberately Deferred

-   rich Dwarf Fortress-style lore/agent simulation;
-   neural collaborative filtering;
-   temporal preference models;
-   implicit-feedback models;
-   multimodal extensions;
-   sophisticated missingness;
-   elaborate generator architecture;
-   premature semantic naming of latent factors.

These remain captured branches, not current implementation commitments.

# Restart Sentence

If resuming after a long interruption:

> **The course archaeology is finished. Add the global-mean baseline and
> stronger invariants, faithfully reconstruct the recovered `k=2`
> PyTorch training path, then derive learned `k=1`; Generator 0 is the
> IID-uniform negative control waiting immediately after that
> checkpoint.**

*END: Context Document*
