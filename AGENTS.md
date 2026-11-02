# Codex repository instructions — Mine TEM Water Risk V1.0

## Project and boundaries

This is a **Chinese-language research prototype** for transient electromagnetic (TEM) data processing and *candidate* low-resistivity anomaly interpretation in mining. It is **not** an instrument validated for mine-water hazard decisions. Read `README.md`, `docs/CODEX_HANDOFF.md`, `docs/DEVELOPMENT_ROADMAP.md`, and `docs/PAPER_ADAPTATION.md` before substantial changes.

Keep the explicit methodological distinction: the reference paper MTDLPy concerns **magnetotellurics (MT)**, *not* TEM. Its modular API/UI/deep-learning workflow may inspire architecture, but MT equations, phase channels, data, and pretrained models must not be reused as TEM physics.

## Development principles

- Preserve the existing Chinese GUI and CLI; never silently break the CSV schemas or previously exported project artifacts.
- Physics first: model source waveform, current, loop/receiver geometry, time-zero, calibration, sensor response, uncertainty and observable units explicitly. Don't pretend a late-time half-space transform or a stack of 1D inversions is 2D/3D full-space TEM inversion.
- Keep source and processing provenance, QC flags and limitations in export. Avoid silently interpolating through strong interference or replacing rejected data with fabricated values.
- Keep low-resistivity flags named **candidate**; do not label them confirmed water bodies or claim classification accuracy without independent borehole/geological ground truth. Never turn a classifier output into an automatic excavation authorization.
- Add deterministic synthetic tests or authoritative benchmark comparisons for numerical/physical changes, including dimensional/unit tests, noise robustness, waveform sign, timing, forward-predicted data residuals and uncertainty.
- All third-party code must carry license acknowledgement and provenance. Do not copy/paste GPL code into MIT code without an explicit licensing decision. Prefer SimPEG's published interfaces, only under tested model assumptions.
- Prefer small, reviewable changes, keep optional `torch` and `simpeg` imports lazy and don't break basic CLI with them absent.
- GUI tasks should not block the Qt event loop; use worker threads and test error states. Ensure export never mislabels schematic 2.5D as real 3D reconstruction.

## Workflow

1. Read `docs/CODEX_HANDOFF.md` and `docs/DEVELOPMENT_ROADMAP.md` to select the next incomplete milestone.
2. Inspect relevant code and run `python -m pytest -q` before edits; report environment issues explicitly.
3. Implement one coherent milestone at a time, with tests and a short change log. Do not rewrite the project from scratch without a demonstrable need.
4. Run unit tests and at least one CLI smoke test; if dependencies unavailable, document which checks remain unverified.
5. At finish, summarize changed files, tests, physical assumptions, and remaining limitations in Chinese. No unsupported performance claims.

## Setup (Windows / Python 3.12 recommended)

`py -3.12 -m venv .venv`

`.venv\Scripts\activate`

`python -m pip install -e .`

`python -m pytest -q`

`python -m mine_tem.cli --demo --output output_demo`

`python -m mine_tem.gui`

Optional dependencies: `pip install torch` and `pip install 'simpeg>=0.25,<0.27'`; test their installed versions before changing numerical APIs.
