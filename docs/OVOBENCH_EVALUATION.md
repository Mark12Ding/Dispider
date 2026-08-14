# Example Evaluation on OVO-Bench

This example evaluates the released Dispider checkpoint on the current
OVO-Bench edition. The helper runs one deterministic, call-balanced shard per
GPU, resumes completed shards, strictly merges every prediction, and computes
the official three-category macro score.

## Prepare the evaluator and data

Clone the OVO-Bench evaluator and use the revision validated with this
repository:

```bash
git clone https://github.com/JoeLeelyf/OVO-Bench.git ../OVO-Bench
git -C ../OVO-Bench checkout c34093f
pip install -r ../OVO-Bench/requirements.txt
```

Download the current annotation and pre-chunked videos from the
[OVO-Bench dataset](https://huggingface.co/datasets/JoeLeelyf/OVO-Bench). The
evaluation directory should contain clips named by annotation ID:

```text
/path/to/ovo-data/
└── chunked_videos/
    ├── 0.mp4
    ├── 1468_0.mp4
    └── ...
```

The source videos are not required when the pre-chunked archive is used. The
script reads `data/ovo_bench_new.json` from the OVO-Bench checkout by default.

## Run the full evaluation

From the Dispider repository, run:

```bash
python scripts/evaluate_ovobench.py \
  --ovo-root ../OVO-Bench \
  --model-path /path/to/Dispider \
  --data-root /path/to/ovo-data \
  --output-dir outputs/ovobench \
  --gpus 0,1,2,3 \
  --resume
```

`--gpus` accepts any comma-separated CUDA device list. Each GPU gets one
weighted shard, so forward tasks with many clips are distributed by inference
call count rather than only by annotation count. Use `--dry-run` to validate
the inputs and inspect the shard plan without loading the model or writing
files.

The command records an immutable run manifest and stores each worker in a
separate directory. Re-running the same command with `--resume` skips a shard
only when its IDs, preserved annotation fields, call count, and every response
pass validation. Interrupted or null-containing shards are run again. A
different annotation, checkpoint identity, evaluator/Dispider source, clip
metadata, or shard count fails closed; use a new output directory for a
different run.

After all workers finish, the same command writes:

```text
outputs/ovobench/
├── logs/                         # one log per GPU worker
├── predictions.json              # canonical annotation-order predictions
├── score_report.json             # score report for this local run
├── shards/                       # deterministic worker annotations
└── run_manifest.json             # resume and provenance identity
```

These files are generated locally under `--output-dir`; they are not bundled
with the repository or model checkpoint.

To validate and score existing outputs without launching CUDA workers, append
`--score-only` and keep the same paths and GPU count.

## Scoring

The current edition contains 1,640 annotation rows and 3,035 independent
inference calls. The OVO-Bench average is not a sample-level micro average: it
first averages tasks within each of the three categories, then averages the
three category scores. The evaluator prints those metrics for the current run
without comparing them with a hard-coded reference result.

The upstream OVO-Bench `score.py` currently does not dispatch the model name
`Dispider`. This helper therefore implements the same task scoring rules while
also checking exact current-edition coverage and rejecting duplicate, missing,
or null predictions.
