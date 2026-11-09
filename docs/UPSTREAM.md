# Source provenance and preparation changes

Upstream: [jingyaogong/minimind](https://github.com/jingyaogong/minimind), commit `f659b55761b754d306bd140573493a6543cafd7f`. The recovered UPSTREAM_COMMIT.txt and 27 byte-identical source/document files agree with that pinned reference. This is a fresh source snapshot with no recovered commit history. Hashes in RECOVERY_MANIFEST.json describe original recovered files; PREPARED_MANIFEST.json describes the final non-Git tree, excluding that manifest itself.

## Ownership boundary

The model architecture, tokenizer, training algorithms, dataset loader, utilities and original READMEs are upstream work by jingyaogong and contributors. All original notices and LICENSE are retained. Files under experiments/ are personal project additions relative to the pinned upstream source; file timestamps and the supplied archive do not independently prove authorship. No personal ownership claim is made over upstream implementation.

## Exact preparation changes

- LICENSE: unchanged from recovery and pinned upstream. No upstream NOTICE exists in the pinned snapshot.
- model/, trainer/, scripts/, eval_llm.py, requirements.txt: recovered upstream bytes unchanged.
- README.md and README_en.md: original bytes relocated to docs/upstream/; new portfolio README.md added.
- dataset/__init__.py, dataset/lm_dataset.py, dataset/dataset.md and CODE_OF_CONDUCT.md: restored unchanged from pinned upstream; not personal implementation and not recovered training data.
- .gitignore: replaced upstream rules with portfolio exclusions, including explicit dataset-source exceptions.
- experiments/compare_pretrain.py, plot_pretrain_loss.py, test_sft_chat.py, compare_sft_dpo_grpo_cpu.py and benchmark_sft_grpo.py: replaced AutoDL absolute artifact roots with MINIMIND_ARTIFACT_ROOT (default repository working directory); added change notices and os import. No model, generation, or scoring logic changed.
- experiments/first_training_step.py and every experiments/results/ file: unchanged recovery bytes.
- UPSTREAM_COMMIT.txt: copied unchanged from archive root.
- docs/ audit, plan, provenance, defaults, validation and manifests: portfolio preparation additions.

Apache-2.0 applies to the distributed source; original authors retain their copyrights. No copyright years or personal legal name were fabricated. Dataset/model licensing must be reviewed separately before adding those assets.

## Primary evidence update

The larger minimind-evidence-backup.tar.gz now takes precedence. ARCHIVE_MANIFEST.json records both packages in full; RECOVERY_MANIFEST.json is retained as the earlier smaller-archive inventory, not the complete primary inventory. Common files in both packages match. Additional recovered upstream files (CODE_OF_CONDUCT.md and images) also match the pinned source; promotional/example images were not added to the public repository.

Personal launch commands and reviewed training logs are now included. Raw auto-shutdown/monitor scripts remain private and were never executed. The configs JSON separates exact historical flags (with path placeholders) from derived local flags. tools/run_stage.py is a new preparation adapter, not a recovered historical command; it changes filesystem layout to flat repository/out and starts fresh unless --resume is explicit. No core trainer code changed.

The benchmark now supports MINIMIND_SFT_CHECKPOINT, MINIMIND_DPO_CHECKPOINT and MINIMIND_GRPO_CHECKPOINT overrides, plus MINIMIND_RESULTS_DIR for isolated output. Generation/scoring logic remains unchanged. Other personal path adaptations remain as described above. Training log derivatives, SVG figures, audit/plan/README updates and verification scripts are new portfolio preparation work. The recovered 10K PNG/CSV are original personal evidence. Reviewed public logs redact any matching private absolute artifact/reward roots; REDACTIONS.json records whether each training log changed. No generated answer was edited.

## Presentation revision

The root README is titled “MiniMind: End-to-End LLM Training & Post-Training” and focuses on architecture, training, evidence, evaluation and reproduction. Attribution is consolidated into one acknowledgment; the full ownership/change record above remains authoritative. The original READMEs are preserved unchanged under docs/upstream/. Their relative image links assume the full original layout; complete rendered versions are available at the pinned upstream [Chinese README](https://github.com/jingyaogong/minimind/blob/f659b55761b754d306bd140573493a6543cafd7f/README.md) and [English README](https://github.com/jingyaogong/minimind/blob/f659b55761b754d306bd140573493a6543cafd7f/README_en.md). Quick Start installation instructions are prospective and have not been runtime-validated. Source, experiments, logs, figures and generated answers are unchanged by this presentation revision.
