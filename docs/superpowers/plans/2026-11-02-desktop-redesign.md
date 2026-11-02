# Desktop Redesign Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task. User explicitly approved direct implementation.

**Goal:** Deliver the approved Chinese desktop workbench and prevent stale results and overlapping tasks.
**Architecture:** Keep CLI and numerical modules unchanged. Put widgets and workflow orchestration in gui.py; reusable styling, read-only copyable table and worker execution in gui_support.py. GUI-owned task completion is serialized through QThread.finished.
**Tech Stack:** Python 3.12, PySide6, Matplotlib, pytest; no additional dependencies.
**Spec:** ../specs/2026-11-02-desktop-redesign-design.md

## Global Constraints

- Keep create_window(), main(), CLI and existing exported file schemas.
- All UI text is Chinese; retain the half-space / candidate interpretation boundaries.
- One task at a time, no terminate, no invented progress; finished precedes control restoration.
- Imported ZIP is the isolated workspace. No Git metadata exists here; preserve original upload and record changes via file hashes, not invented commits/worktrees.

## Review Focus

- Import failure must preserve previously usable data and results.
- Changing selected line, parameters or engine invalidates both visible and exportable results.
- Failed and repeated jobs cannot destroy a running QThread or leave controls locked.
- Export path must remain below the selected folder even for untrusted line IDs.
- At 1024×720 all pages must remain usable via scroll areas and splitters.

### Task 1: Behavioral regression tests

Files: tests/test_gui.py. Existing fixtures use real demo data and processing; mock only dialogs and deliberately slow/erroring operations.
- [x] Add offscreen QApplication/window fixtures and event-driven wait helper, with thread cleanup in fixture teardown.
- [x] Add readonly table/copy, navigation, invalidation, busy/close, error recovery, async import/export and optional-dependency tests.
- [x] Run tests against existing GUI; confirm failures correspond to absent behavior.

### Task 2: Implement workbench and lifecycle

Files: mine_tem/gui.py, mine_tem/gui_support.py.
Interfaces: create_window() -> QMainWindow retains raw/result/worker/fields/lines/pl(q)ots state; Window.import_files(paths), Window.export_to(folder), Window._start_task(kind, **kwargs) return a bool indicating accepted launch. Task(kind, **kwargs) emits progress(str), returns payload/error via attributes, consumed only after finished. Tables copy selected values as TSV.
- [x] Create sidebar navigation + stacked pages, grouped scrollable forms, metric cards, real chart toolbars and Chinese result headings.
- [x] Add atomic GUI state transitions, contextual result invalidation, one-task guard and running-close guard.
- [x] Move import/export into workers; disable chart input during export; handle render failures without retaining exportable invalid results.
- [x] Test selected-folder containment using a traversal-like line ID; keep CLI output semantics unchanged.
- [x] Run GUI tests and full suite; inspect actual window screenshot and minimum-size layout.

### Task 3: Delivery and review

Files: README.md, docs/UI_REDESIGN_CHANGELOG.md; external screenshot and ZIP deliverables.
- [x] Run CLI demo and compare output arrays to baseline; run all tests and saved gui_smoke.py.
- [x] Run fresh reviewer against approved spec, changed files and tests; fix important findings with regression tests.
- [x] Record verification and Windows/optional-dependency limitations. Create clean downloadable project ZIP (exclude caches, virtualenvs, paper, local outputs).

## Execution ledger

- Baseline: 8 passed, 1 skipped (torch absent).
- Ruling: Direct approval covers implementation; execute inline without another plan-approval round. No Git commits or worktree because input is an uploaded ZIP in an already isolated cloud task.

- Task 1: complete — 12 GUI cases initially failed against the old interface; follow-up small-window, empty-import and export-path regressions also observed failing before fixes.
- Task 2: complete — 19 GUI cases pass; retained 8 core passes, 1 real torch training skip. Saved onboarding GUI smoke passes; pip check reports no broken requirements.
- Ruling: Window composition moved into gui_window.py so gui.py remains a small lazy facade and support.py owns worker execution and presentation helpers. Public CLI/create_window/main stay unchanged; cost is one extra internal module.
- Ruling: Results page scrolls at smaller sizes instead of compressing graph labels. Minimum chart size is covered by a renderer-bounds test; cost is scrolling to lower details on small screens.
- Final review: fresh independent reviewer, 1 Important export-directory collision, no Critical or Minor items.
- Final: fixed unsafe/truncated ID collisions and foreign-line directory overwrite — 4 regression cases RED→GREEN; full suite 27 passed / 1 skipped, with 24 upstream Matplotlib Qt deprecation warnings.
- Final: Ruling: Numerical physical validity, Windows native/high-DPI/accessibility/EXE and actual optional ML/SimPEG operation are outside this UI task and remain unverified. Cost is that Linux smoke tests cannot establish those capabilities. CLI numerical arrays match the prior baseline exactly.
- Task 3: complete — readme/changelog, actual desktop screenshots and clean source ZIP supplied. Original uploaded archive remains intact; source folder retained, no merge or push.
