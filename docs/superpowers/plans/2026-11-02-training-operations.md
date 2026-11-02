# Training Operations Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans. User approved direct implementation; execute inline without another approval round.

**Goal:** Data inspection, truthful live training feedback, and safe model selection in the existing desktop workbench.
**Architecture:** learn.py owns axis validation; training_info.py inspects NPZ without torch. gui_training.py owns training widgets; gui_window.py and gui_support.Task retain shared single-task orchestration.
**Tech Stack:** Python 3.12, NumPy, PySide6, Matplotlib, pytest; optional PyTorch.
**Spec:** ../specs/2026-11-02-training-operations-design.md

## Global Constraints

- Keep train/predict signatures, 1D-CNN, split, learning rate and early stopping unchanged.
- Inspections work without torch and never load .pt weights to display metadata.
- Background work shares busy/close guards; only main thread updates widgets.
- Legacy model names, data schema, CLI and physical formula remain compatible.
- Already isolated ZIP workspace; no Git history exists, no worktree or fabricated commits.

## Review Focus

- NaN/Inf, zero/negative axes must fail both dataset inspection and training validation.
- Changed paths cannot retain successful summaries for an earlier selection.
- Failed inspection cannot destroy existing measurement results or leave controls locked.
- Early stopping must show actual epochs, not the configured maximum.
- Model selection cannot deserialize untrusted checkpoints or imply weight validity before inference.

### Task 1: Safe inspection

Files: learn.py, new training_info.py, tests/test_training_info.py.
Interfaces: validate_axes(times, depths) -> None; inspect_dataset(path) -> dict (samples,n_times,n_depths,time_range_s,depth_range_m,rho_range); inspect_model(path_or_prefix) -> dict (prefix,n_times,n_depths,time_range_s,depth_range_m).
- [x] Add failing tests for illegal axes, sample metadata, missing paired files, invalid scaler dimensions/values and safe model metadata inspection.
- [x] Implement shared validation and inspection; run focused tests without torch.

### Task 2: Training controls and monitoring

Files: new gui_training.py, gui_window.py, gui_support.py, tests/test_training_gui.py.
Interfaces: TrainingPanel provides dataset/modelpath/model_input, inspection and browsing signals; begin_training(epochs), update_progress(epoch,total,train_mse,val_mse), complete_training(summary). Task emits epoch_progress(int,int,float,float), adds dataset_info/model_info operations; train returns original summary plus worker-read metadata for GUI only.
- [x] Add failing GUI tests for data inspection and path invalidation, model browsing/inspection without torch, output path selection, structured epoch feedback, early completion, measurement-result preservation and failure recovery.
- [x] Implement cards and live loss curves with axis labels identifying normalized log-target MSE; preserve real progress on completion.
- [x] Run focused tests then full suite; if CPU torch can be installed via trusted configured routes, run actual short training and inference, preserving optional behavior tests without torch.

### Task 3: Verify and deliver

Files: README.md, docs/TRAINING_OPERATIONS_CHANGELOG.md, external ZIP and screenshots.
- [x] Fresh read-only code review; fix important findings with regression tests.
- [x] Run full suite, CLI smoke/baseline array comparison, real Qt captures at default/minimum size, pip check.
- [x] Save installation/startup configuration only if dependency setup changes; package source without caches or reference PDF.

## Ledger

- Baseline: 27 passed, 1 skipped; torch absent; existing sample: 80 samples, 28 gates, 18 depths.
- Ruling: User approved direct implementation. Keep work inline in imported source folder; preserve all original upload archives.

- Task 1: complete — 13 tests observed failing then passing; shared axis validator rejects invalid physical axes; inspection never imports torch or deserializes checkpoints.
- Task 2: complete — original 6 new GUI operations tests first failed; independent CPU torch installation then actual training/inference passed. Added stale-model-result guard with a failing regression before fix.
- Ruling: Training widgets live in gui_training.py; Canvas moved to gui_support.py so both result and training canvases share export suspension. No algorithm or model naming change.
- Final review: fresh independent reviewer found two Important issues, no Critical/Minor findings. Fixes: literal .pt-ending prefix keeps existing naming; every GUI inference revalidates metadata before calling predict_line. All three added regression cases observed RED then GREEN.
- Final suite: 52 passed, 24 upstream Matplotlib Qt deprecation warnings, no skips; actual CPU training and inference included. CLI arrays match original baseline exactly; pip check passes.
- Ruling: Native Windows/high DPI/accessibility, GPU, solver physics and model predictive quality remain outside this operation task; Linux synthetic checks cannot establish those capabilities.
- Task 3: complete — actual 30-epoch teaching-data curve captured; README/changelog updated; clean source package prepared. CPU install/start instructions saved to environment draft, requiring environment settings review/save/publish for reuse.
