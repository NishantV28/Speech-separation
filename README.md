# Unknown-Speaker Speech Separation

Take audio with several people talking over each other. Return each voice
separately.

**The model is never told how many people are talking.** It works that out
itself. That is the hard part, and it is the point of the project.

---

## ⭐ Run these three

Everything else is optional. These three, in this order:

| | Command | What it's for |
|---|---|---|
| **1️⃣** | `configs/p2_mossformer_eda.yaml` | **The baseline.** Everything else is compared against it. Run this first or the other numbers mean nothing. |
| **2️⃣** | `configs/p4_mossformer_tda.yaml` | **The comparison.** Same model, different way of counting speakers. P2 vs P4 is the main experiment. |
| **3️⃣** | `configs/p5_tda_prune_ecapa.yaml` | **Your contribution.** P4 plus three new ideas. Its score only means something next to P4's. |

```bash
python -m core.train --config configs/p2_mossformer_eda.yaml  --steps 10000 --warmup 1000
python -m core.train --config configs/p4_mossformer_tda.yaml  --steps 10000 --warmup 1000
python -m core.train --config configs/p5_tda_prune_ecapa.yaml --steps 10000 --warmup 1000
```

**~2 hours each** on an RTX 3050. Training auto-resumes — if it crashes, run the
same command again.

---

## Setup — once

You need the `mixtures/` folder at the repo root. It is **not** in git (2.3 GB):

```
Speech-separation/
  mixtures/
    train/{2,3,4,5,6}spk/{mixed,sources}/...
    validation/...
    test/...
  core/  configs/  heads/  ...
```

```bash
pip install -r requirements.txt

# turn your wavs into manifests AND fix the data bug. ~1 min. NOT optional.
python adapters/librimix_csv.py --repair

# sanity check -- ~30s, must print "GATE PASS"
python scripts/overfit_gate.py
```

If the gate fails, **stop**. Something is broken and training won't fix it.

If `mixtures/` lives elsewhere:
`python adapters/librimix_csv.py --repair --root /path/to/parent`

---

## The workflow — repeat for each pipeline

### Step 1 — train (~2h)

```bash
python -m core.train --config configs/p2_mossformer_eda.yaml --steps 10000 --warmup 1000
```

Watch the `sep` number. It starts around +25 and must **go negative**. Negative
means it is actually separating voices.

```
      1000  exist 0.69  sep  0.15   0.72s/step
      3000  exist 0.55  sep -4.20   0.72s/step    <- healthy
```

If `sep` is still positive at step 3000, stop and investigate.

**A checkpoint is saved every 500 steps** to `runs/<name>-<hash>/last.pt`. If the
run dies — crash, power cut, closed laptop — **run the exact same command again**
and it resumes where it stopped. Nothing is lost.

⚠️ `--steps` is part of the folder name. Resume with the *same* `--steps 10000`,
or it starts a fresh run in a new folder.

### Step 2 — get the numbers

```bash
python -m core.eval --ckpt runs/p2_mossformer_eda-<hash>/last.pt
```

```
  3spk (n= 82): SI-SNRi  6.41 dB   count acc 71.2%  (mean n_est 3.2)
  4spk (n= 74): SI-SNRi  4.88 dB   count acc 58.1%  (mean n_est 4.4)
  6spk (n= 64): SI-SNRi  2.10 dB   count acc 31.2%  <- EXTRAPOLATION (never trained)
```

**SI-SNRi** — how much cleaner the voices got, in dB. Higher is better.
**count acc** — how often it guessed the right number of speakers. This is half
your project; report it.

Saves `results.jsonl` beside the checkpoint. Don't copy these by hand — step 4
collects them.

### Step 3 — listen to it

```bash
python separate.py --ckpt runs/p2_mossformer_eda-<hash>/last.pt \
    --wav mixtures/test/4spk/mixed/test_4spk_00000000.wav \
    --out demo/ --figure
```

```
==> DETECTED 4 SPEAKERS   (never told; inferred by the model)
  demo/speaker1.wav    +0.0 dB rel. loudest
  demo/speaker2.wav    -2.1 dB rel. loudest
```

Writes each voice as a wav plus `demo/spectrograms.png`. **This is your demo
video**: play the mixture (chaos), show the detected count, play each voice.

Any length works — 10s files run at 11× realtime.

### Step 4 — next pipeline, then compare

Repeat 1–3 for `p4_mossformer_tda.yaml`, then `p5_tda_prune_ecapa.yaml`. Then:

```bash
python -m core.eval --compare runs/
```

```
pipeline                    3spk      4spk      5spk   count acc
----------------------------------------------------------------
p2_mossformer_eda           6.41      4.88      3.12       64.3%
p4_mossformer_tda           7.05      5.60      3.71       71.8%
p5_tda_prune_ecapa          7.44      6.02      4.15       79.1%
```

**That table is your result.** It's built from the `results.jsonl` files, so it's
reproducible and you never transcribe a number by hand.

### Keeping checkpoints

Each run folder holds `last.pt` (~60 MB), `config.json`, `train_log.jsonl`,
`results.jsonl`. **Checkpoints are gitignored** — too big for GitHub. Copy them
to Drive or a USB stick yourself if you want them kept.

---

## Your data, and how the code handles it

**Clips are 10 seconds, 2–6 speakers, 16 kHz.**

**Training uses a random 4-second window from each clip.** Not the whole 10s —
deliberately:

- MossFormer2's attention cost grows with the *square* of length. 4s → 10s is
  about **6× more expensive**, and your batch size would collapse from 16 to ~4.
- The TDA heads (P4/P5) cap context at 512 frames. 10s is 1251 frames → crash.

Cropping avoids both, and the 10s clips still help — they're a bigger pool to
draw windows from.

**Evaluation also uses 4s windows**, for a less obvious reason: a model trained
on 4s has never seen position 502 or beyond, so testing it on a full 10s clip
would run it on parts it never learned and report nonsense. `--full` overrides
this if you want it.

**The demo (`separate.py`) handles any length** — it splits long audio into
chunks, separates each, and stitches them back. A 10s file runs at 11× realtime.

**6-speaker mixtures are excluded from training on purpose.** They're the test
for "can it handle a number of speakers it has never seen?" — which is your
headline claim. If you train on them, you lose the ability to make it.

---

## The seven pipelines

⭐ = run these

| | Model | How it counts speakers | |
|---|---|---|---|
| P0 | MossFormer2 | energy threshold | a dumb yardstick, not a real entry |
| ⭐ **P2** | MossFormer2 | **EDA** | **baseline** |
| P3 | GridNet-FSMN | EDA | tests a different model shape |
| ⭐ **P4** | MossFormer2 | **TDA** | **the count-mechanism comparison** |
| ⭐ **P5** | MossFormer2 | **TDA + 3 new ideas** | **your contribution** |
| P7 | MossFormer2 | OR-PIT | the only one with no speaker limit |
| P8 | MossFormer2 | EDA + confidence | everything stacked; proves nothing on its own |

All are ~5M parameters, within 10% of each other. That matters: at different
sizes you'd be measuring *model size*, not *ideas*.

### EDA vs TDA (the P2 vs P4 question)

**EDA** guesses speakers **one at a time**, each guess built on the last. If
guess 1 is wrong, guesses 2 and 3 inherit the mistake.

**TDA** guesses them **all at once**, and the guesses can see each other and
divide up the speakers. No chain, so no compounding errors.

That's the bet. P2 vs P4 tests it on your data.

---

## P5: the contribution

Three ideas, all attacking **one** problem: two "attractors" grabbing the same
speaker while a real speaker gets missed.

**1. Confidence pruning** — a small head listens to each separated voice and
predicts how clean it is. Two attractors stuck on one speaker both sound muddy,
which the existence check can't see (it decides *before* separating).

**2. Attractor repulsion** — if two attractors grabbed the same person, they're
two *similar vectors*. So penalise them for being similar. EDA and SepTDA both
just hope attractors spread out; neither adds anything to make them. One matrix
multiply, no extra libraries.

**3. Overlap-weighted loss** — your mixtures average 0.67 overlap, so about a
third of every clip has only *one* person talking, which is trivially easy. The
normal loss spends a third of its effort there. This weights the loss toward the
moments where people actually talk over each other.

All three are **off by default everywhere else**, so P4 is a clean comparison
and P5's improvement is attributable.

`configs/ablations/` turns on one idea at a time, if you want to know which one
did the work. Three more runs; skip if you're short on time.

**ECAPA is switched off and does nothing** — it needs `speechbrain`, which isn't
installed, and it fails silently. Repulsion replaces it.

---

## Batch size: don't go above 16

Measured on your RTX 3050 4GB:

| batch | time/step | VRAM | speed |
|---|---|---|---|
| 8 | 0.48s | 1.27G | 16.8 samples/s |
| **16** | **0.72s** | **2.31G** | **22.2 samples/s ← best** |
| 24 | 1.87s | 3.35G | 12.8 ← **3× slower** |
| 32 | 3.88s | 4.41G | 8.2 |

**Above 16 you run out of GPU memory and Windows doesn't tell you.** It quietly
moves data to normal RAM and everything crawls, with no error. If training feels
weirdly slow, this is why. 16 is already the default.

---

## Steps and epochs

**2,343 usable training mixtures** (3,000 minus ~22% thrown out for damaged
audio — see below).

| steps @ batch 16 | epochs | time | result |
|---|---|---|---|
| 6,000 | 41 | 1.2h | rough, but the demo works |
| **10,000** | **68** | **2.0h** | **what to run** |
| 30,000 | 205 | 6h | barely better — it starts memorising |

⚠️ **`warmup` must be about 10% of your steps.** The default in `_base.yaml` is
4000, meant for a 100k run. On a 10k run use `--warmup 1000`, or the learning
rate never warms up properly and the model barely trains. This already cost one
wasted run.

---

## Two things that will confuse you

**1. Your dataset's mixtures don't equal the sum of their parts.** The generator
scaled the mixture but forgot to scale the individual voices, so
`mixture = α × (sum of voices)` with α anywhere from **0.118 to 6.027**. Also
~33% of the voice files are clipped.

`--repair` fixes the scaling exactly (no files rewritten) and throws out the
~22% that are too damaged. **Always run it.** It's not optional.

**2. The loss is "thresholded SNR", not SI-SDR.** When the model can output more
voices than exist, some targets are *silence* — and SI-SDR divides by the target,
so silence makes it divide by zero. It doesn't just do badly; it's undefined.
This happens on literally every batch.

---

## Files

```
configs/     the pipelines. Change these, not the code.
  _base.yaml   settings shared by all
  p*.yaml      one file per pipeline
  ablations/   P5's ideas, one at a time
core/        the engine. Same for every pipeline.
  train.py     training
  eval.py      scoring
  losses.py    thresholded SNR, PIT, repulsion, overlap weighting
  data.py      loads the manifest
adapters/    dataset -> manifest. New dataset = new adapter, nothing else.
backbones/   the "listening" part
heads/       the "how many speakers + split them" part
separate.py  THE DEMO: audio in -> separate voices out
scripts/     overfit_gate.py, check_pipelines.py, generate_dataset.py
tests/       21 tests. Run after touching core/.
docs/        dataset_spec.md, dataset_generation.md
```

## Before pushing changes

```bash
python -m pytest tests/ -q          # 21 tests
python scripts/overfit_gate.py      # must say GATE PASS
python scripts/check_pipelines.py   # all 8 build
```

If the gate fails, **stop and fix it**. A model that can't memorise a single
batch has something broken in the loss or the audio handling, and training
longer won't help.

## Removed on purpose

- **TF-GridNet** — needs ~5.1 GB and 3826 ms/step. Won't fit a 4 GB laptop.
- **P6 (attractor memory)** — its idea is remembering speakers across chunks of
  long audio, but training feeds one 4s window at a time, so the memory never
  activates. It would train identically to P4 with dead parameters. Needs a
  chunked training loop that doesn't exist.
- **MixIT / difficulty scheduler** — were referenced by P8's config but never
  implemented, so they silently did nothing. Better gone than pretending.
