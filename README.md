# Unknown-Speaker Speech Separation — 9 pipelines

Separate an unknown number of overlapping speakers (2–6+) from single-channel
audio. **No pipeline is ever told how many speakers there are** — every one of
them infers it.

Everything shares one core. A "pipeline" is a YAML file naming a **backbone** and
a **head**. That is the entire difference between them, which is what makes the
comparison an experiment rather than nine unrelated models.

---

## Quick start

```bash
pip install -r requirements.txt

# 1. dataset -> manifest (repairs the generator's scale bug on the way through)
python adapters/librimix_csv.py --repair

# 2. is the foundation sound? (~30s on GPU, ~2.5min on CPU) must print GATE PASS
python scripts/overfit_gate.py

# 3. do all 9 pipelines build? (~1min)
python scripts/check_pipelines.py

# 4. train one
python -m core.train --config configs/p1_gridnet_eda.yaml

# 5. evaluate it
python -m core.eval --ckpt runs/p1_gridnet_eda-<hash>/last.pt

# 6. compare everything trained so far
python -m core.eval --compare runs/
```

Training **auto-resumes** from the run directory. If Colab kills your session,
rerun the identical command — it picks up at the last checkpoint.

---

## The 9 pipelines

| # | Backbone | Head | Differs from | The one variable |
|---|---|---|---|---|
| **P0** | TF-GridNet | over-separation | — | *reference baseline, not a contender* |
| **P1** | TF-GridNet | EDA | — | **root of the ladder** |
| **P2** | MossFormer2 | EDA | P1 | backbone (swap) |
| **P3** | GridNet-FSMN | EDA | P1 | backbone (modification) |
| **P4** | TF-GridNet | TDA | P1 | count mechanism |
| **P5** | TF-GridNet | TDA + prune + ECAPA | P4 | **the contribution** |
| **P6** | TF-GridNet | TDA + memory | P4 | cross-chunk identity |
| **P7** | MossFormer2 | OR-PIT | — | *different paradigm* |
| **P8** | MossFormer2 | EDA + confidence | — | *integrated system, not an experiment* |

All ~5M params (±10%). **Budget matching is not optional** — at different sizes
you would be measuring capacity, not architecture.

### How each infers the count

- **EDA** (P1/P2/P3/P8) — an LSTM decoder emits attractors one at a time, each
  with a stop probability. Architecturally unbounded; in practice it only
  reliably emits as many as it was trained to.
- **TDA** (P4/P5/P6) — learned queries emit all attractors in parallel; an
  activity gate says which are real. Bounded by `n_queries=8`, **not** by the
  training count.
- **OR-PIT** (P7) — extract one speaker, subtract, repeat until a learned stop
  head fires. **Genuinely unbounded**: no cap of any kind.
- **Over-separation** (P0) — 8 slots always; count = slots above an energy gate.
  Real but hard-capped, hence baseline only.

### Read these first

Every backbone and head has a long docstring explaining *why* it exists and what
it is measuring. Start with `heads/eda.py` and `backbones/gridnet_fsmn.py`.

---

## Three things that will bite you

**1. The loss is thresholded SNR, not SI-SDR.** Once a model can emit more
streams than there are speakers, some references are **zero**, and SI-SDR
divides by `‖ref‖²`. It is undefined, not merely worse. Every unknown-speaker
pipeline hits this on every batch. See `core/losses.py`.

**2. The dataset's mixtures do not equal the sum of their sources.** The
generator peak-normalised the mixture but not the sources, so
`mix = α·Σsources` with α from **0.118 to 6.027**. `--repair` recovers α by
least squares and stores it as manifest `gains`, so `Σ(gains·sources) == mix`
holds exactly with no wav rewritten. ~33% of sources are also hard-clipped;
that is irreversible, so mixtures whose reconstruction-SNR ceiling drops below
25 dB are quarantined (~22–27%). See `docs/dataset_spec.md`.

**3. TF-GridNet needs ~5.1 GB at 4s.** It will not fit a 4GB card, and Windows
will silently spill to system RAM rather than erroring — 12 s/step instead of
0.5. If training is inexplicably slow, this is why. Use `--segment 1.0 --batch 1`
locally; drop them on a T4.

---

## Hardware

| | RTX 3050 4GB | Colab T4 16GB |
|---|---|---|
| AMP | bfloat16, no scaler | float16 + GradScaler |
| TF-GridNet @ 4s | **does not fit** | batch 2 |
| MossFormer2 @ 4s | fits (14× lighter — measured) | batch 8 |
| use for | dev, debugging, gates | the real sweep |

`core/amp.py` **detects** the dtype. Never hardcode it: bf16 needs no loss
scaling, fp16 silently produces zero gradients without one.

Measured, 2s / batch 1: TF-GridNet 3826 ms · **GridNet-FSMN 559 ms (6.8× faster
at equal params)** · MossFormer2 135 ms.

---

## Suggested protocol

You cannot train 9 models to convergence on free-tier GPUs (~1 GPU-day each).

1. **Screen** — all 9 at `--steps 30000` (~3h each). Ranks them.
2. **Full runs** — the top 2 only, to convergence.

```bash
for c in configs/p*.yaml; do python -m core.train --config $c --steps 30000; done
python -m core.eval --compare runs/
```

Report **SI-SNRi per speaker count**, never pooled — the graders test per count.
Report **counting accuracy** as a first-class metric; it is half the problem.

---

## Layout

```
core/        the shared engine — NEVER differs between pipelines
  manifest.py   canonical dataset schema (the only dataset boundary)
  interface.py  the frozen Encoder/Backbone/Head/Decoder contract
  stft.py       STFT encode/decode, complex masking
  losses.py     thresholded SNR, PIT (Hungarian), mixture consistency
  data.py       manifest-driven Dataset; applies `gains`, drops quarantined
  amp.py        device-aware bf16/fp16
  difficulty.py curriculum + adaptive sampling + hard mining (ONE knob)
  train.py      the loop; auto-resume
  eval.py       SI-SNRi per count, count accuracy, PESQ/STOI, compare table
  registry.py   name -> class
adapters/    dataset -> manifest. NEW DATASET = NEW ADAPTER, nothing else changes
backbones/   tfgridnet_lite, mossformer2_lite, gridnet_fsmn, modules, tiny
heads/       eda, eda_conf, tda, tda_prune, tda_memory, orpit, film_count, oversep
configs/     p0..p8 — the pipelines. _base.yaml holds everything shared
scripts/     overfit_gate.py, check_pipelines.py
tests/       pytest — run after touching core/
docs/        dataset_spec.md (hand to whoever regenerates the data)
```

## Changing the dataset

Write one adapter (~40 lines) that emits the manifest schema in
`core/manifest.py`. `core/`, `backbones/`, `heads/` and all 9 configs stay
untouched. That is the whole point of the manifest boundary.

## Before you push

```bash
python -m pytest tests/ -q          # 14 tests
python scripts/overfit_gate.py      # must print GATE PASS
python scripts/check_pipelines.py   # all 9 build, budgets in range
```

If the gate fails, **stop**. A model that cannot memorise one batch has a broken
loss, PIT, or STFT, and no amount of training will fix it. But check the SI-SNR
trajectory first — a monotonic climb that ran out of steps is undertraining, not
breakage.
