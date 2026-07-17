# Unknown-Speaker Speech Separation

Take audio with several people talking over each other. Return each voice
separately.

**The model is never told how many people are talking.** It works that out
itself. That is the hard part, and it is the point of the project.

---

## ⭐ Every command, in order

```bash
# ---- ONCE, after cloning or whenever mixtures/ changes -------------------
pip install -r requirements.txt
python adapters/librimix_csv.py --repair        # ~4 min. NOT optional. See below.
python scripts/overfit_gate.py                  # ~30s, must print "GATE PASS"

# ---- TRAIN (~4h) --------------------------------------------------------
python -m core.train --config configs/p5_tda_prune_ecapa.yaml --steps 20000 --warmup 2000

# ---- CHECK AT ~STEP 3000, do not wait 4h --------------------------------
python -m core.eval --ckpt runs/p5_tda_prune_ecapa-<hash>/last.pt --split manifests/train.jsonl --limit 30

# ---- SCORE IT (run train first, test LAST) ------------------------------
python -m core.eval --ckpt runs/p5_tda_prune_ecapa-<hash>/last.pt --split manifests/train.jsonl --limit 30
python -m core.eval --ckpt runs/p5_tda_prune_ecapa-<hash>/last.pt

# ---- DEMO ---------------------------------------------------------------
python separate.py --ckpt runs/p5_tda_prune_ecapa-<hash>/last.pt     --wav mixtures/test/3spk/mixed/test_3spk_00000000.wav --out demo/ --figure

# ---- COMPARE EVERYTHING TRAINED SO FAR ----------------------------------
python -m core.eval --compare runs/
```

The run prints its own `<hash>` at startup — copy it from there.

**Auto-resume**: if training dies, rerun the *identical* command. `--steps` is
part of the folder name, so keep it the same or you start over.

**`--fresh`**: only when the model shape changed (e.g. `n_max`). It ignores the
existing checkpoint and starts from zero.

### What to watch while training

```
      3000  exist 0.55  conf 0.24  repel 0.15  sep -4.20   0.72s/step
```

- **`sep` must go negative.** Positive at step 3000 = something is wrong.
- **`s/step` should be ~0.7.** If it is 4+, something else is using the GPU.
- **`repel` should fall** — attractors spreading apart.

### The mid-run check that matters

At ~step 3000, run the train-split eval above and look at **3spk SI-SNRi**:

| | |
|---|---|
| above ~2 dB and climbing | it is learning — let it finish |
| still ~1.7 dB | more steps will not help — stop, something else is wrong |

The previous run sat at 1.78 dB on 3spk train and never moved for 20,000 steps.
That one number saves you four hours.

---

## Your data, and how the code handles it

**Clips are 10 seconds, 2–6 speakers, 16 kHz.**

**Training takes a random 4-second window from each clip — a DIFFERENT one every
epoch.** That randomness matters: it was previously frozen (same window every
time), and the model memorised 3,091 fixed clips instead of learning. 2spk hit
7.08 dB on train and 0.96 dB on test — a 6 dB overfitting gap. Fresh windows give
roughly 2.5× the effective data from files you already have.

Not the whole 10s, deliberately: MossFormer2's attention cost grows with the
*square* of length (4s → 10s is ~6× on attention alone, batch collapses 16 → 4),
and the TDA heads cap context at 512 frames while 10s is 1251 → crash.

**Eval uses a FIXED window** so scores don't wobble between runs, and the same 4s
length — a model trained at 4s has never seen frame 502+, so scoring it on a full
clip would report noise. `--full` overrides.

**The demo (`separate.py`) handles any length** — chunks and stitches. 10s runs at
11× realtime.

**Training is on 2 and 3 speakers only.** This is SepTDA's protocol — the state of
the art trains on exactly 2+3 and tests beyond. Spreading a 5M model and 3k
mixtures across 2–5 speakers gave ~1 dB on *all* of them. The model is still never
*told* the count; it infers 2 or 3 itself. **Unknown ≠ unbounded**, and this
project is about unknown.

**4spk is the extrapolation test** — never trained, so it evidences the scaling
claim.

**`n_max: 4`** = training max (3) + 1 headroom slot. Slots whose target is silence
are free points, so 8 slots against 2–3 speakers meant most of the gradient
rewarded staying quiet.

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

**1,649 usable training mixtures** (2spk + 3spk, minus ~25% quarantined for
clipped audio).

| steps @ batch 16 | epochs | time |
|---|---|---|
| 6,000 | 58 | 1.2h |
| **20,000** | **194** | **4.0h** |

Because crops are now random, "epochs" overstates repetition — each pass sees
different windows, which is the point.

⚠️ **`warmup` must be ~10% of your steps.** The default in `_base.yaml` is 4000,
sized for a 100k run. On 20k use `--warmup 2000`. Get this wrong and the learning
rate never warms up — it already cost one wasted run.

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
