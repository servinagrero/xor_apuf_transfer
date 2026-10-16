#!/usr/bin/env bash
set -euo pipefail

PHASE="${1:-all}"
PROFILE="${2:-smoke}"
case "$PHASE" in
    all|alpha-beta|baseline|theory|theory-calibrated|prepare|empirical|pooled|fixed-cohort|validation|control|figures) ;;
    *) echo "Usage: bash run_publication_study.sh [all|alpha-beta|baseline|theory|theory-calibrated|prepare|empirical|pooled|fixed-cohort|validation|control|figures] [smoke|publication]" >&2; exit 2 ;;
esac
case "$PROFILE" in
    smoke|publication) ;;
    *) echo "Profile must be smoke or publication." >&2; exit 2 ;;
esac

JOBS="${PUF_JOBS:-8}"
EMPIRICAL_CSV="${EMPIRICAL_CSV:-data/raw/crps.csv}"
CHALLENGES="${CHALLENGES:-data/raw/all_challenges.npy}"
CACHE="${EMPIRICAL_CACHE:-data/empirical_cache}"
STUDY_REVISION="${PUF_STUDY_REVISION:-ber_cost_v2}"
FIGURE_REVISION="${PUF_FIGURE_REVISION:-analysis_v2}"

if [[ "$PROFILE" == "publication" ]]; then
    K_THEORY="1,2,3,4,5,6,7,8,9"
    K_EMPIRICAL="1,2,3,4"
    ALPHA_GRID="0:1:11"
    BETA_GRID="0:1:11"
    METRIC_DEVICES=30
    SIMULATION_METRIC_BUDGET=5000
    EMPIRICAL_METRIC_BUDGET=20000
    METRIC_TRIALS=5
    BASELINE_BUDGETS="100,500,1000,5000,10000,20000,50000,100000"
    TARGET_BUDGETS="100,500,1000,5000,10000,20000"
    SOURCE_COUNTS="1,5,10,20"
    SOURCE_BUDGETS="500,1000,5000"
    FIGURE_SOURCE_BUDGET=1000
    TARGET_BERS="0.0,0.03,0.05"
    THEORY_TRIALS=5
    EMPIRICAL_TRIALS=5
    NULL_TRIALS=100
    VALIDATION_SOURCE_COUNT=10
    NEAREST_EXCLUSIONS=3
else
    K_THEORY="1,2,3"
    K_EMPIRICAL="1,2"
    ALPHA_GRID="0:1:3"
    BETA_GRID="0:1:3"
    METRIC_DEVICES=8
    SIMULATION_METRIC_BUDGET=200
    EMPIRICAL_METRIC_BUDGET=200
    METRIC_TRIALS=2
    BASELINE_BUDGETS="100,500"
    TARGET_BUDGETS="100,500"
    SOURCE_COUNTS="1,2"
    SOURCE_BUDGETS="100"
    FIGURE_SOURCE_BUDGET=100
    TARGET_BERS="0.0,0.03"
    THEORY_TRIALS=2
    EMPIRICAL_TRIALS=2
    NULL_TRIALS=5
    VALIDATION_SOURCE_COUNT=2
    NEAREST_EXCLUSIONS=1
fi

run_alpha_beta() {
    uv run python scripts/00-study_alpha_beta.py --run \
        --k-values "$K_THEORY" --alpha "$ALPHA_GRID" --beta "$BETA_GRID" \
        --n-devices "$METRIC_DEVICES" --budget "$SIMULATION_METRIC_BUDGET" \
        --n-trials "$METRIC_TRIALS" --n-jobs "$JOBS" \
        --stealth-mode baseline-relative \
        --output alpha_beta_metrics_trials.csv \
        --summary-output alpha_beta_metrics_summary.csv \
        --stealth-output alpha_beta_stealth_regions.csv \
        --study-grid-output alpha_beta_study_grid.csv
}

run_baseline() {
    uv run python scripts/01-study_baseline_modeling.py --run \
        --k-values "$K_THEORY" --budgets "$BASELINE_BUDGETS" \
        --models logreg,target_nn --n-trials "$THEORY_TRIALS" \
        --target-ber "$TARGET_BERS" \
        --n-jobs "$JOBS" --output baseline_modeling.csv
}

run_theory() {
    uv run python scripts/02-study_theoretical_transfer.py --run \
        --grid-csv alpha_beta_study_grid.csv --k-xor "$K_THEORY" \
        --k-pretrain "$SOURCE_COUNTS" --n-base-budget "$SOURCE_BUDGETS" \
        --m-target-budget "$TARGET_BUDGETS" --target-ber "$TARGET_BERS" \
        --n-trials "$THEORY_TRIALS" --n-jobs "$JOBS" \
        --output theoretical_transfer.csv
}

run_theory_calibrated() {
    uv run python scripts/02-study_theoretical_transfer.py --run --append \
        --calibrated-alpha 0.95 --calibrated-beta "0.5,0.75,0.95" \
        --k-xor "1,2,3,4" --k-pretrain 20 --n-base-budget 5000 \
        --m-target-budget "$TARGET_BUDGETS" --target-ber 0.0 \
        --n-trials "$THEORY_TRIALS" --n-jobs "$JOBS" \
        --output theoretical_transfer.csv
}

prepare_empirical() {
    if [[ -f "$CACHE/manifest.json" ]]; then
        echo "Using existing empirical cache: $CACHE"
        return
    fi
    [[ -f "$EMPIRICAL_CSV" ]] || { echo "Missing $EMPIRICAL_CSV" >&2; exit 2; }
    uv run python scripts/03-study_empirical_transfer.py --prepare \
        --csv "$EMPIRICAL_CSV" --challenges "$CHALLENGES" --cache-dir "$CACHE"
}

run_empirical() {
    local output="data/empirical_transfer_${PROFILE}_${STUDY_REVISION}"
    local resume=()
    [[ -f "$output/transfer_results.csv" ]] && resume=(--resume)
    uv run python scripts/03-study_empirical_transfer.py --estimate --run "${resume[@]}" \
        --challenges "$CHALLENGES" --cache-dir "$CACHE" --output-dir "$output" \
        --k-values "$K_EMPIRICAL" --k-pretrain "$SOURCE_COUNTS" \
        --n-base-budget "$SOURCE_BUDGETS" --m-target-budget "$TARGET_BUDGETS" \
        --metric-budget "$EMPIRICAL_METRIC_BUDGET" --metric-null-trials "$NULL_TRIALS" \
        --n-trials "$EMPIRICAL_TRIALS" --n-jobs "$JOBS"
}

run_pooled() {
    local random="data/empirical_transfer_${PROFILE}_${STUDY_REVISION}"
    local output="data/empirical_transfer_pooled_${PROFILE}_${STUDY_REVISION}"
    local resume=()
    [[ -f "$output/transfer_results.csv" ]] && resume=(--resume)
    uv run python scripts/03-study_empirical_transfer.py --run "${resume[@]}" \
        --challenges "$CHALLENGES" --cache-dir "$CACHE" --output-dir "$output" \
        --metric-distributions "$random/population_metric_distributions.npy" \
        --diagnostic-baselines --k-values "$K_EMPIRICAL" \
        --k-pretrain "$VALIDATION_SOURCE_COUNT" \
        --n-base-budget "$FIGURE_SOURCE_BUDGET" --m-target-budget 100 \
        --n-trials "$EMPIRICAL_TRIALS" --n-jobs "$JOBS"
}

run_fixed_cohort() {
    local output="data/empirical_transfer_fixed_cohort_${PROFILE}_${STUDY_REVISION}"
    local resume=()
    [[ -f "$output/fixed_cohort_results.csv" ]] && resume=(--resume)
    uv run python scripts/03-study_empirical_fixed_cohort.py "${resume[@]}" \
        --challenges "$CHALLENGES" --cache-dir "$CACHE" --output-dir "$output" \
        --k-values "$K_EMPIRICAL" --k-pretrain "$VALIDATION_SOURCE_COUNT" \
        --n-base-budget "$FIGURE_SOURCE_BUDGET" --m-target-budget 100 \
        --n-trials "$EMPIRICAL_TRIALS" --n-jobs "$JOBS"
}

run_control() {
    local output="data/empirical_transfer_permuted_${PROFILE}_${STUDY_REVISION}"
    local resume=()
    [[ -f "$output/transfer_results.csv" ]] && resume=(--resume)
    uv run python scripts/03-study_empirical_transfer.py --run "${resume[@]}" \
        --challenges "$CHALLENGES" --cache-dir "$CACHE" --output-dir "$output" \
        --source-label-control challenge-permuted \
        --k-values "$K_EMPIRICAL" --k-pretrain "$VALIDATION_SOURCE_COUNT" \
        --n-base-budget "$FIGURE_SOURCE_BUDGET" --m-target-budget "$TARGET_BUDGETS" \
        --n-trials "$EMPIRICAL_TRIALS" --n-jobs "$JOBS"
}

run_validation() {
    local random="data/empirical_transfer_${PROFILE}_${STUDY_REVISION}"
    local output="data/empirical_transfer_exclude_nearest_${PROFILE}_${STUDY_REVISION}"
    local resume=()
    [[ -f "$output/transfer_results.csv" ]] && resume=(--resume)
    uv run python scripts/03-study_empirical_transfer.py --run "${resume[@]}" \
        --challenges "$CHALLENGES" --cache-dir "$CACHE" --output-dir "$output" \
        --metric-distributions "$random/population_metric_distributions.npy" \
        --source-policy exclude-nearest \
        --exclude-nearest-count "$NEAREST_EXCLUSIONS" \
        --cluster-threshold 0.15 \
        --diagnostic-baselines --k-values "$K_EMPIRICAL" --k-pretrain "$VALIDATION_SOURCE_COUNT" \
        --n-base-budget "$FIGURE_SOURCE_BUDGET" --m-target-budget "$TARGET_BUDGETS" \
        --n-trials "$EMPIRICAL_TRIALS" --n-jobs "$JOBS"
}

run_figures() {
    local random="data/empirical_transfer_${PROFILE}_${STUDY_REVISION}"
    local pooled="data/empirical_transfer_pooled_${PROFILE}_${STUDY_REVISION}"
    local fixed="data/empirical_transfer_fixed_cohort_${PROFILE}_${STUDY_REVISION}"
    local excluded="data/empirical_transfer_exclude_nearest_${PROFILE}_${STUDY_REVISION}"
    local control="data/empirical_transfer_permuted_${PROFILE}_${STUDY_REVISION}"
    local figure_output="figures/empirical_transfer_${PROFILE}_${STUDY_REVISION}_${FIGURE_REVISION}"
    uv run python scripts/00-figures_alpha_beta.py \
        --input data/alpha_beta_metrics_summary.csv --k-values "$K_THEORY"
    uv run python scripts/01-figures_baseline_modeling.py \
        --input data/baseline_modeling.csv
    uv run python scripts/02-figures_theoretical_transfer.py \
        --input data/theoretical_transfer.csv
    uv run python scripts/03-figures_empirical_transfer.py \
        --input "$random/transfer_results.csv" \
        --output-dir "$figure_output" \
        --estimates "$random/population_estimates.csv" \
        --theory data/theoretical_transfer.csv \
        --similarity-control-results "$excluded/transfer_results.csv" \
        --metric-distributions "$random/population_metric_distributions.npy" \
        --metrics-summary "$random/population_metrics_summary.csv" \
        --metrics-null-summary "$random/population_metrics_null_summary.csv" \
        --population-parameter-pairs "$random/population_parameter_pairs.csv" \
        --population-model-fit "$random/population_model_fit.csv" \
        --negative-control-results "$control/transfer_results.csv" \
        --pooled-control-results "$pooled/transfer_results.csv" \
        --fixed-cohort-results "$fixed/fixed_cohort_results.csv" \
        --coverage "$random/coverage.csv" \
        --skipped-configurations "$random/skipped_source_configurations.csv" \
        --k-pretrain "$VALIDATION_SOURCE_COUNT" \
        --n-base-budget "$FIGURE_SOURCE_BUDGET" \
        --source-scaling-source-budgets "$SOURCE_BUDGETS" \
        --source-scaling-target-budget 100 --formats png,pdf
    uv run python scripts/03-table_empirical_transfer.py \
        --break-even "$figure_output/transfer_break_even.csv" \
        --frontier "$figure_output/transfer_frontier.csv" \
        --source-count "$VALIDATION_SOURCE_COUNT" \
        --source-budget "$FIGURE_SOURCE_BUDGET" --target-budget 100 \
        --output "$figure_output/transfer_break_even_selected.tex" \
        --csv-output "$figure_output/transfer_break_even_selected.csv"
}

case "$PHASE" in
    alpha-beta) run_alpha_beta ;;
    baseline) run_baseline ;;
    theory) run_theory ;;
    theory-calibrated) run_theory_calibrated ;;
    prepare) prepare_empirical ;;
    empirical) run_empirical ;;
    pooled) run_pooled ;;
    fixed-cohort) run_fixed_cohort ;;
    validation) run_validation ;;
    control) run_control ;;
    figures) run_figures ;;
    all)
        run_alpha_beta
        run_baseline
        run_theory
        run_theory_calibrated
        prepare_empirical
        run_empirical
        run_pooled
        run_fixed_cohort
        run_validation
        run_control
        run_figures
        ;;
esac
