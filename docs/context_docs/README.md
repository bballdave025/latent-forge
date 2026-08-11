# Context Documents

This directory contains **continuation-ready context documents** (Lab
Notebook / LN files) for the **Latent Forge** project.

> **Repository customization**
>
> This README has been customized for Latent Forge. The project marker used
> for context-document filenames is `ltfg`.
>
> Double-brace placeholders (`{{...}}`) appearing in the templates below are
> intentionally retained. They are filled when creating a future context
> document or Pre-Context-Document Prompt (PCDP).

These are **not** polished documentation. They are:

- structured snapshots of project state,
- continuation-ready checkpoints,
- architecture and design records,
- "lab notebook" entries for engineering progress.

------------------------------------------------------------------------

# Purpose

Each document is designed to answer:

> **If I had to resume this work in a fresh environment, what would I
> need to know?**

They preserve:

- current architecture
- terminology
- naming decisions
- design constraints
- implementation status
- next-step execution plans
- important non-obvious insights
- active questions

Think of these as **checkpoint files for thinking**, not summaries.

------------------------------------------------------------------------

# Documentation Hierarchy

```text
README.md
    ↓
Project Charter
    ↓
Lab Notebook / Context Documents
    ↓
Experiment Notes
```

Each layer answers a different question.

For Latent Forge, context documents may bridge work involving:

- synthetic recommender-system data
- collaborative filtering
- matrix factorization
- sparse user-item matrices
- latent representations
- baseline and improved models
- experiment design and evaluation
- Dwarf Fortress-inspired interpretability and synthetic-world concepts
- portfolio-project development
- branches intentionally deferred to later work

The context document should preserve enough of the mathematical and
engineering reasoning that future work can continue without having to
reconstruct why an experiment exists.

------------------------------------------------------------------------

# Context Document Header Template

Use **double-brace placeholders** (`{{...}}`) rather than angle
brackets.

```text
*BEGIN: Context Document Header*

# CONTEXT DOCUMENT — Continuation

## Project

**Name:**
Latent Forge

**Description:**
A synthetic-data playground for learning matrix factorization,
collaborative filtering, latent representations, and recommender-system
experimentation.

---

## Continuation Metadata

**Prepared at:**
{{ssssssssss_YYYY-mm-ddTHH:MM:SS±ZZZZ}}

Generated via:

date +'%s_%Y-%m-%dT%H:%M:%S%z'

(Boston, MA time)

**Continued from chat:**
{{Exact chat title}}

**Also involving:**
- {{Related topic}}
- {{Related topic}}
- *(or: no other subjects specified)*

---

## Author / Source

**User (GitHub):**
@bballdave025

**User (ChatGPT):**
{{optional}}

---

## Intent for This Context

{{1–2 sentences describing what this continuation should enable}}

---

## Usage Instructions

- Treat this document as **authoritative project state**.
- Continue with **minimal re-derivation**.
- Reinterpret only when explicitly requested.

*ENDOF: Context Document Header*
```

------------------------------------------------------------------------

# Pre-Context-Document Prompt (PCDP)

Before pasting a context document into a fresh chat, you may send a
short PCDP to establish context and immediate goals.

```text
## Current Work

Project:
Latent Forge — synthetic recommender-system and matrix-factorization
learning/portfolio project.

Starting with:
- {{Step}}
- {{Step}}
- {{Step}}

---

## Upcoming Context Document

The next message will be a CONTEXT DOCUMENT for:

Latent Forge

This continues discussion begun in:

"{{Previous chat title}}"

---

## Timing

Preparation:
{{YYYY-MM-DDTHH:MM:SS±ZZ:ZZ}}

(Optional) New chat:
{{YYYY-MM-DDTHH:MM:SS±ZZ:ZZ}}

---

## Instructions for Next Message

Instructions
- Do not summarize.
- Do not reformat.
- Do not analyze.
- Do not critique.
- Do not extract bullet points.
- Do not optimize language.
- Treat the context document as authoritative state.
- Your response should only:
  1. Confirm receipt.
  2. Confirm readiness to continue.

---

## Immediate Focus

Help me:

{{Concrete, task-oriented, ADHD-friendly next task}}

*End of PCDP*
```

------------------------------------------------------------------------

## Practical Note

In normal use, the **Instructions for Next Message** section is often
**omitted** _for the chat creating the context document_.

Experience has shown that including it sometimes causes models to interpret
the transferred context as operational instructions rather than as
information to pass on. In most cases, the PCDP establishes the project,
immediate focus, and previous chat.

However, giving the directions to the _new_ chat (unlike giving them to the
old chat) actually reduces confusion and extra explanations where Dave would
prefer to continue cognitive momentum. After it is clear that the forthcoming
Context Document is not a usual prompt, the Context Document can more easily
supply the authoritative project state.

Typical workflow:

```text
PCDP (optional)
      ↓
Context Document (authoritative)
      ↓
Continue work
      ↓
Update / create new LN document before stopping
```

------------------------------------------------------------------------

# Naming Convention

```text
LN_ltfg_YYYY-MM-DD_{{optional-tag}}_-_{{short-slug}}.md
```

Examples:

```text
LN_ltfg_2026-08-11_ctx01_-_initial-recommender-architecture.md
LN_ltfg_2026-08-15_-_matrix-factorization-experiments.md
LN_ltfg_2026-08-20_parked_-_agent-simulation-generator.md
```

`LN` = Lab Notebook. This is Dave's longstanding name for these files,
including from undergrad and industry work.

The Latent Forge project marker is:

```text
ltfg
```

The optional tag is usually omitted unless it adds useful context. Examples
of useful optional tags:

- `ctx01`
- `parked`
- `submitted-addenda`
- `pr-for-{{collaborator}}`

When in doubt, omit the optional tag.

Choose the slug to describe the primary engineering topic rather than the
implementation detail.

------------------------------------------------------------------------

# What Belongs in a Context Document?

A context document should capture the project's **current engineering
state**, not merely what changed.

Typical sections include:

- Current architecture
- Major modules and responsibilities
- Recent implementation work
- Current design rationale
- Constraints and assumptions
- Active questions
- Immediate next steps

For Latent Forge, useful state may additionally include:

- current synthetic-data assumptions
- dimensions and sparsity constraints
- train/test/holdout strategy
- implemented recommendation baselines
- current matrix-factorization formulation
- hyperparameters already explored
- validation results
- effective-rank investigations
- experiment hypotheses
- rejected or unsuccessful approaches
- deferred experiment families
- interpretability/lore-layer decisions

When useful, include small code snippets, directory layouts, equations, or
command examples that reduce future re-derivation.

Do **not** turn a Context Document into a requirement that every interesting
idea be implemented.

Latent Forge follows the project-work rule:

> **Capture the branch. Stay on the trunk.**

Interesting extensions can be preserved as future work without expanding the
current implementation scope.

------------------------------------------------------------------------

# Experiment Continuation

Latent Forge is intended to support rapid experimentation while preserving
understanding.

When recording an experiment, capture enough information to answer:

1. What assumption changes?
2. Why should it improve?
3. How will success be measured?
4. Is the added complexity justified?

Context documents should make it possible to distinguish:

- experiments already run,
- experiments currently being implemented,
- experiments selected as near-term next steps,
- and interesting branches deliberately deferred.

This distinction is especially important for larger extensions such as
neural collaborative filtering, temporal models, implicit-feedback models,
or richer agent simulation. Such ideas may be worth preserving without
belonging in the current lean implementation.

------------------------------------------------------------------------

# Scope

These documents are intended to bridge work across:

- multiple ChatGPT conversations,
- multiple development sessions,
- different machines,
- local development and MLU AWS SageMaker environments,
- and interruptions lasting days or months.

They are written primarily for the future maintainer of the project,
which is usually the author.

For interactive MLU/SageMaker work, continuation state should favor
**small executable next-step batches** rather than long procedural plans.
The goal is to preserve both:

- deep mathematical understanding,
- and fast experimental momentum.

------------------------------------------------------------------------

# Retrieval Tip

1. Sort by filename.
2. Open the newest `LN_ltfg_*`.
3. Resume from **Immediate Next Steps** (or equivalent).

------------------------------------------------------------------------

*End of README*
