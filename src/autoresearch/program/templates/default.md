# Research Program

## Goal
{goal}

## Setup
- Training script: `train.py`
- Evaluation metric: val_bpb (bits per byte, lower is better)
- Time budget: 5 minutes per experiment (wall clock training time)
- Data: FineWeb-Edu dataset (pre-downloaded via prepare.py)

## Constraints
{constraints}

## Experimentation
- Only modify `train.py` — all other files are read-only
- Each experiment: modify code -> commit -> train -> evaluate -> keep/discard
- Keep if val_bpb improves (lower is better); discard if equal or worse
- Log all results to results.tsv
- Simplicity criterion: prefer simpler code at equal val_bpb

## Research Directions
{directions}

## Output Format
After each training run, the script prints:
````
---
val_bpb:          <value>
training_seconds: <value>
peak_vram_mb:     <value>
````

Log to results.tsv (tab-separated):
````
commit	val_bpb	memory_gb	status	description
````
