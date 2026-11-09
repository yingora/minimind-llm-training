# MiniMind evidence audit

Date: 2026-11-09 (Europe/Paris). Primary evidence: minimind-evidence-backup.tar.gz. Secondary: minimind-portfolio-backup.tar.gz. Both are read-only originals; freshly extracted copies were used. No training, inference, remote modification, or publication occurred.

## Corrections to previous reports

The previous report was limited to the smaller archive and is superseded. Machine 063 is the user-confirmed recovery context. Machine 598 and its earlier container identity are historical context, not verified identifiers for the recovered runs. Missing SSH access is no longer a blocker to the local audit.

GRPO is **completed after resumption**, not incomplete: original `logs/grpo_full_resume.log` line 974 is `Epoch:[1/1](19502/19502), Reward: 2.1194, KL_ref: -0.2948, Adv Std: 1.1536, Adv Mean: 0.0000, Actor Loss: 0.0088, Avg Response Len: 75.25, Learning Rate: 0.00000010`. The wrapper log reports successful process exit and a 132 MB saved weight. Earlier step 260 evidence was partial. Resume skipped 200 prior steps. Shutdown usage errors do not invalidate training completion, but prevent claiming shutdown success.

Full pretraining and SFT logs are present and reach final steps. Earlier statements that they were wholly unverified are withdrawn. DPO weights are reported present on the recovered machine and listed in the inventory, but **no weight bytes are in either archive**; DPO code and evaluation reports are included. Do not conflate inventory evidence with local weight inspection.

## Safety and provenance

The primary archive has 106 members and 13,461,184 uncompressed bytes; the secondary has 47 members and 1,142,215 bytes. Paths and member types were checked before extraction: no absolute/parent-traversal paths, symbolic/hard links or special files. New extraction roots preserve original archives and earlier staging. Archive SHA-256 and every recovered file hash are recorded in ARCHIVE_MANIFEST.json. Common files match between both packages. Embedded documents/scripts were treated as evidence, not instructions.

UPSTREAM_COMMIT.txt copies agree. Recovered non-experiment MiniMind source, images and original documentation match the public source snapshot at the recorded commit. The empty LOCAL_CHANGES.patch and exported GIT_STATUS.txt corroborate no tracked patch, while listing untracked experiments/data/checkpoints. The archive does not include Git object history, so authorship and parent history are not independently authenticated. Git status export and upstream SHA are provenance records, not the new local repository's HEAD.

## Confirmed findings

| Stage | Original evidence | Interpretation |
| --- | --- | --- |
| 10K pretraining | logs/pretrain_10k.log line 128: 2500/2500, loss 5.0258 | Training loop completed in log |
| Full pretraining | logs/pretrain_full.log line 3179: 317560/317560, loss 2.3093 | Training loop completed in log |
| SFT | logs/sft_full.log line 566: 56295/56295, loss 1.9059 | Training loop completed in log |
| DPO | trainer/train_dpo.py; benchmark_dpo_run.log; benchmark JSONL; inventory out/dpo_full/dpo_full_768.pth, 137684000 bytes | Model existence supported by inventory and labeled inference; complete training log not found |
| GRPO | logs/grpo_full_resume.log line 974; logs/grpo_auto_shutdown.log | Resumed and completed 19502/19502; successful exit/save reported |

Line numbering uses LF-separated original bytes; carriage-return progress bars are not counted as separate lines. Public GRPO log retains this numbering. Logs support recorded execution, not cryptographic checkpoint validation.

## Configurations and lineage

Recovered run_full_pretrain.sh, run_full_sft.sh and run_full_grpo.sh supply actual intended launch flags. Dense 768/8 model; pretraining batch 4/accumulation 4, sequence 340, LR 5e-4; SFT batch 16/accumulation 1, sequence 768, LR 1e-5; GRPO batch 1/accumulation 1, four generations, prompt 512, generation 128, LR 1e-6, beta .05, GRPO loss, epsilon .2, thinking disabled. All specify one epoch and bfloat16. Full details are in configs/; parser defaults are separately labeled.

SFT requests pretrain_full; GRPO requests sft_full and resume mode. Historical seed paths use repository/out while save paths use a sibling artifact root; symlink/copy layout was not exported. Portable local flags flatten outputs into repository/out and explicitly distinguish fresh vs resume. DPO launch arguments remain missing. Training stage counters are batches, not necessarily optimizer updates. Dataset-loader counts support 1270238 pretraining rows and 900718 SFT rows; content/version/license are unresolved.

The original scripts include shutdown automation. They were audited as text and not run. Shutdown/monitor executables are excluded from the public version. Safe launch transcripts and a separate Python adapter replace infrastructure-specific behavior. Unneeded upstream promotional/example images are excluded so they cannot be mistaken for personal results.

## Evaluation audit

All 90 records have unique model/ID keys and cover 1–30 for each SFT/DPO/GRPO label. Prompts, expected answers and automatic flags match the benchmark definitions. Each has 0/15 strict exact matches. Refusal counts are 0/0/3 and repetition counts 15/15/7. All 30 SFT/DPO answer strings match exactly. Recovered outputs remain byte-identical.

Strict whole-answer matching penalizes correct verbose factual answers, yet real generation errors also occur: GRPO says 12×8=24 and 9 squared=9; exact-word instructions fail; repeated and irrelevant responses occur across models. Refusal/repetition flags are heuristics, not human judgments. The 64-token cap can truncate output; stop reasons/token counts are absent. No model execution exception is present in the saved benchmark logs. These results do not establish preference-accuracy change or general capability improvement. Checkpoint identity and whether GRPO evaluation used the final resumed weight are unresolved.

Separate SFT 2048-row probes finish at batch counts 256, 128 and 64 for batch sizes 8,16,32. Their final losses differ but training-step budgets are unequal. Short timing/memory snippets cannot support reliable hardware recommendations.

## Public preparation and review

The unchanged Apache-2.0 license, original READMEs and source notices are preserved. Personal files and portfolio modifications are classified in UPSTREAM.md. Required dataset support code was restored unchanged from pinned upstream. No dataset/model weight/checkpoint bytes were included. New diagnostic runs default to private ignored paths when following the README.

Textual evidence and candidate public files were scanned for common private keys, GitHub/AWS/OpenAI token patterns and literal password/API secrets; no high-confidence matches were found. Personal launch transcripts redact private absolute roots/reward paths. Raw host-specific automation and full private inventories stay in staging. Original-vs-public hashes and redaction notes are retained. Pattern checks cannot guarantee the absence of all secrets; owner review remains required before publishing.

Validation covers Python syntax, shell transcript review, pure scoring recomputation, log endpoint checks, source/result identity, manifests, portable adapter dry runs, ignored-asset checks and staged-file review. No dependencies installed, no project model imported, no GPU work performed. Original upstream whitespace is preserved; newly authored files receive scoped whitespace checks.

## Remaining issues

1. Complete DPO training log and actual command; historical 8083/8083 and no preference-accuracy gain remain historical claims.
2. Checkpoint hashes, contents and lineage; local inspection of DPO weights was impossible because bytes were not exported.
3. Binding benchmark records to the final resumed GRPO checkpoint and run time.
4. Exact dataset/reward-model revisions, licenses, hashes, preprocessing and train/validation separation; validation filename presence does not establish executed validation.
5. Complete dependency/hardware environment; no environment lock or controlled speed benchmark.
6. Standard benchmark/human evaluation, repeated runs, confidence intervals and generation stop metadata.
7. Public author name, GitHub owner/visibility, and owner approval of publication/license scope.
