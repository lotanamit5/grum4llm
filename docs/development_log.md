## Step 0 - Package Foundation and Boundaries

Implemented the initial provider-agnostic package scaffold with explicit separation between GRUM core modules and external provider adapters, plus deterministic seed utilities and baseline contract/smoke tests. This maps to the project proposal requirement to keep GRUM independent from the elicitation source (human, simulation, or LLM) and prepares the codebase structure for the paper's Algorithm 1 (adaptive elicitation orchestration) and Algorithm 3 (MC-EM inference) to be added in later steps without cross-layer coupling.

## Step 1 - Mathematical Core and Data Contracts

Implemented typed GRUM parameter containers, deterministic utility computation $\mu_{ij}=\delta_j + x_i B z_j^T$, explicit ranking representations (full and partial), and pre-inference checks for Condition 1 connectivity and interaction identifiability. This directly maps to Section 2.1 definitions in the paper and to the proposal's theorem/condition framing that requires identifiable, well-posed estimation before inference and elicitation.

## Step 2 - MC-EM Inference Baseline (Normal Family)

Implemented a first MC-EM inference engine for the Normal-family GRUM path with Gibbs-based E-step under ranking constraints, ridge-regularized closed-form M-step updates for $\delta$ and $B$, and objective/convergence diagnostics. This corresponds to the paper's Section 4 Monte Carlo E-step and M-step flow (Equations (8) and (9)) and supports the proposal requirement to estimate intrinsic and interaction preferences before adaptive elicitation and model comparison.

## Step 3 - Observed Fisher Information and Elicitation Criteria

Implemented Normal-family observed/candidate Fisher information utilities and a criterion layer containing D-optimality, E-optimality, and the social-choice certainty criterion aligned with Equation (5). This maps to the paper's Section 4.3 and Table 1 (information-based elicitation scoring) and supports the proposal's Experiment 1 framing by making intrinsic preference uncertainty explicitly measurable and optimizable.

## Step 4 - Adaptive Elicitation Engine (Provider-Agnostic)

Implemented an Algorithm-1 style adaptive elicitation engine that iterates MAP fitting, precision update, criterion-based candidate scoring, and provider query/update steps while keeping all data acquisition behind the `PreferenceProvider` contract. This maps to paper Section 3's iterative elicitation loop and to the proposal requirement that GRUM logic remain decoupled from the elicitation source (human/simulation/LLM) while enabling persona-targeted querying.

## Step 5 - Social-Choice Reproduction Harness

Implemented synthetic Dataset 1 and Dataset 2 generators, social-choice Kendall-tau evaluation, and reproducible benchmark runners for asymptotic recovery and criterion comparison against random selection. This maps to paper Section 6.1 (synthetic social-choice experiments and criterion comparisons) and enforces the proposal requirement to validate correctness by replication-style behavior before using GRUM on LLM preference elicitation.

## Step 5.1 - Quick Experiment Entrypoint

Added a lightweight script entrypoint to run the current social-choice synthetic experiments from one command (asymptotic, criteria comparison, or both) with JSON output for downstream plotting/reporting. This is an intentional bridge before a broader experiment CLI refactor and keeps reproducibility aligned with the Step 5 harness while making execution practical for iterative research runs.

## Step 5.3 - Runtime Progress Bars

Added optional tqdm-backed progress bars for both asymptotic and criteria phases, wired through benchmark progress callbacks so we can track unit completion during long runs and better estimate resource requirements. This complements Step 5 timing instrumentation by providing live visibility into run advancement and helps decide whether to stay on local CPU or move to larger compute resources.

## Step 6 - Personalized Choice Path

Implemented a personalized-choice criterion (Eq. 6 style certainty approximation), deterministic per-agent ranking prediction APIs, and personalized evaluation/asymptotic routines with seed-stability tests. This maps to paper Sections 3.2 and 6.2 by adding agent-specific preference quality evaluation and prepares the codebase for proposal Experiment 2 analysis of persona-dependent rank reversals.

## Stage A - Reproduction Config Presets and Guardrails

Implemented config-first experiment execution via `--config` YAML with strict key/type validation, CLI override precedence, reproducibility config files under `configs/repro/`, and regression tests for config loading/validation/output path behavior. This establishes an auditable, professor-friendly guardrail for paper-reproduction runs and provides a deterministic execution matrix that we will use in subsequent reproduction stages.

## Stage B - Social-Choice Asymptotic (Figure 2 Style) Foundation

Implemented dataset-selectable social-choice runners (`dataset1`/`dataset2`), CLI/config dataset routing, and a smoothing utility (moving average) for report curves, with additional test coverage for selector validation and smoothing behavior. Initial asymptotic runs completed for both synthetic datasets, but trend similarity to the paper is not yet sufficient; further iteration on Stage B settings/implementation is required before advancing.

## Stage B.1 - CPU Parallelization for Reproduction Sweeps

Added process-based parallel execution for repeat-level workloads in both asymptotic and criteria social-choice benchmarks, exposed via `n_jobs` in CLI and YAML configs with strict validation (`n_jobs > 0`). Added deterministic equivalence tests showing serial and parallel runs match for fixed seeds, enabling high-core-count servers (for example 256/512 CPUs) to accelerate full reproduction sweeps without changing statistical outputs.

## Stage B.2 - Orchestration Runs with Per-Subconfig Traceability

Implemented a separate orchestration pipeline driven by a single YAML file that creates timestamped run folders, generates one subconfig and one output JSON per seed/subrun, executes each subrun through isolated subprocess calls to the single-run worker CLI, and records full config-to-output mappings plus per-subrun logs in metadata for debugging. Added modular aggregation functions (`asymptotic`, `criteria`, `timing`) that can be called independently on any set of existing JSON outputs, enabling re-aggregation of historical runs without rerunning experiments.
