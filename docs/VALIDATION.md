# Validation of primary evidence preparation

2026-11-09. Only static/standard-library evidence processing and launcher dry runs were executed. No model modules imported, inference/training run, dependencies installed, or remote files modified.

- Both archives checked for safe paths/types before separate fresh extraction. Common files match; all recovered upstream source/docs/assets match the pinned public reference.
- Recomputed prompts, targets, exact-match/refusal/repetition flags and summary totals for 90 unique records. All agree; all 30 SFT/DPO answers are identical. Evaluation files remain byte-identical to primary recovery.
- Confirmed original LF-delimited log endpoints: pretrain_full line 3179, sft_full line 566, grpo_full_resume line 974. The original CR progress bars are not counted as new lines.
- Parsed personal training curves into CSV and regenerated SVGs; axes show actual batch-step ranges and logged-value ranges. SVG XML parses and all point coordinates are finite. No upstream example curve is represented as a personal result.
- Python AST parsing and pretrain/SFT/GRPO local-launcher dry runs passed. Local config flags were checked against upstream parser arguments. The local path/resume changes are explicit, not a claim of identical runtime reproduction.
- Common credential/private-key patterns scanned in original evidence text and prepared public text; no high-confidence matches. Private host-specific scripts and full raw inventory are excluded; launch transcript roots are redacted. Scanning cannot certify all possible secrets absent.
- Model, trainer, utilities and restored dataset source remain pinned-upstream bytes. License is unchanged. Results remain recovery bytes. Candidate archive/weight/data/cache/secret extensions and staged sizes reviewed; ignore rules checked.
- A fresh local Git index stages only the prepared package. No remote or commit is configured. Original archives and earlier prepared repository remain outside this tree.
- Original upstream trailing spaces/CRLF are intentionally retained. Scoped whitespace checking applies to new documentation/config/tool files.

Run `python tools/verify_evidence.py` for saved-record, endpoint, syntax and manifest checks; `python tools/render_evidence_figures.py` regenerates the log figures. These commands do not run models. Hashes cover the prepared non-Git tree except PREPARED_MANIFEST.json itself.

Runtime compatibility, numerical training/inference reproduction, dataset licensing, checkpoint integrity, and final-checkpoint-to-evaluation identity remain unresolved.

## Final presentation review

README title, project narrative, Quick Start installation/prerequisites, and upstream image-reference links were revised. docs/UPSTREAM.md records the presentation change. All previously staged implementation, experiment, evidence, figure and result bytes remain unchanged. The prepared manifest was refreshed.

Re-ran tools/verify_evidence.py and pretrain, SFT and resumed-GRPO launcher dry runs. Reviewed final staged blobs for credential patterns, excluded assets and file sizes; no high-confidence secret candidates or large excluded artifacts were found. No installation, model runtime verification, commit, remote or publication was performed. The previous minimind-final-review.zip is an earlier review snapshot and was not regenerated in this revision.
