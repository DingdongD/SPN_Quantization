# Reports

Generated CSV, JSON, PNG, and profiler outputs belong in local experiment
directories and are ignored by Git. Keep only compact, reviewed summaries in
this directory.

The joint hardware study writes the following files below
`reports/generated/spn_joint_dse_bf16_lpddr4/analysis`:

- `joint_architecture_catalog.csv`: physical and bandwidth assumptions for all
  array/memory points;
- `joint_layer_results.csv`: selected mapping, traffic, cycles, and energy for
  every dense GEMM;
- `joint_model_summary.csv`: latency and energy aggregated per SPN model;
- `joint_candidate_summary.csv`: all latency-area-energy candidates, hard
  constraint status, fairness, and Pareto membership;
- `joint_pareto_frontier.csv`: non-dominated feasible candidates;
- `joint_pareto.png`: latency-area projection with mean energy as color;
- `metadata.json`: hashes, tool revisions, objective definitions, assumptions,
  exclusions, and confidence.

The LLMCompass--TCM--FactorFlow validation writes the following files below
`reports/generated/spn_three_mapper_validation_bf16_lpddr4/analysis`:

- `architecture_manifest.csv`: array, SRAM, LPDDR4, CACTI, and area-proxy
  configuration for the four anchors;
- `backend_status.csv`: pinned revision, coverage, and wall time per mapper;
- `workload_coverage.csv`: 486 included dense layers and four explicitly
  excluded propagation operations;
- `unique_gemm_results.csv`: 1,380 backend/architecture/unique-GEMM results;
- `layer_results.csv`: 5,832 results expanded to original layer identities;
- `model_summary.csv`: 48 backend/architecture/model aggregates;
- `architecture_ranking.csv`: 12 within-backend normalized EDP rankings;
- `validation.json`: comparison policy, mapper methodology, revisions, and
  the machine-checkable validation marker;
- `architecture_rank_comparison.png`: native normalized-EDP comparison.
