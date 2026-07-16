# Dataset spec — what the generator must produce

Hand this to whoever regenerates the data. Everything here was measured on the
current `mixtures/` tree, not guessed.

## 1. Two bugs to fix in the generator

Both come from one root cause: sources were summed, the **mixture** was
peak-normalised, and the sources were saved **without** that factor.

```python
# WRONG — what the current generator does
mix = sum(srcs)
mix = mix * 0.95 / abs(mix).max()     # sources never see this factor
save(mix); save_each(srcs)            # mix != sum(srcs), and srcs clip

# RIGHT — rescale BEFORE summing, save the rescaled sources
srcs = [s * g for s, g in zip(srcs, gains)]      # apply SNR gains first
mix  = sum(srcs)
peak = max(abs(mix).max(), max(abs(s).max() for s in srcs))   # note: covers sources too
if peak > 0.95:
    k = 0.95 / peak
    srcs = [s * k for s in srcs]
    mix  = mix * k
save(mix); save_each(srcs)            # mix == sum(srcs), nothing clips
```

Measured damage in the current data:

| | measured |
|---|---|
| `mix = α·Σsources`, α range | **0.118 – 6.027** (one mixture amplified 6×) |
| source files hard-clipped | **30–36%** (longest flat-topped run 162 samples = 10.1 ms) |
| reconstruction-SNR ceiling | p50 40 dB, p25 30 dB, **p5 15 dB** |

The clipping matters less than the file count suggests — it wrecks few samples,
so it inflates *peak* error while barely moving *energy*. Against a 10–15 dB
SI-SNRi target, 75% of mixtures are fine. Only the worst decile is poisonous.
But it is free to fix in the generator, so fix it.

Prefer **LUFS loudness normalisation** (the LibriMix convention,
arXiv:2005.11262) over peak normalisation — peak makes loudness swing with one
transient sample.

## 2. Save float32

`soundfile.write(..., subtype='FLOAT')`. Current data is PCM_16, which is what
made the clipping unrecoverable. Disk is cheap; clipped ground truth is not.

## 3. Speaker counts — this is the new requirement

Current data is 3/4/5 only. That makes the graded goal ("scale to as many
concurrent speakers as possible") **unmeasurable**, because there is no count
above the training max to test on.

| Count | Split | Why |
|---|---|---|
| **2spk** | train + val + test | Four distinct counts teach *counting as a concept* rather than a constant. SepTDA trains on 2+3 and generalises to 5 — the range is what does it. |
| 3, 4, 5 | train + val + test | as now |
| **6spk** | **test ONLY — never in train** | The extrapolation test. A count the model has never seen. This is the headline scaling claim, and holding it out is the only way to earn it. |

**Do not put 6spk in training.** If it is trained, there is no never-seen count
left to test, and the extrapolation claim evaporates.

## 4. Test set size

Currently 100 mixtures/count, and the repair quarantines ~27%, leaving **64
usable at 5spk**. The confidence interval on SI-SNRi there is roughly ±0.5 dB —
wider than the gap likely to separate two pipelines. The whole project rests on
that comparison.

**Target 500 mixtures/count** for test. Validation can stay at 100/count.

## 5. Keep — do not "fix" these

Verified good, and easy to break by accident:

- **Three-way speaker-disjoint split** (581 train / 73 val / 72 test, all
  pairwise overlaps zero). Most commonly botched thing in separation datasets.
- **No speaker repeated within a mixture** (0/3300).
- **Overlap ratio 0.35–1.00**, mean 0.67 — not all fully-overlapped.
- **No near-silent sources**; speech-active fraction p10 = 0.45.
- First speaker pinned at 0 dB as the SNR reference.

## 6. Nice to have

- **Wider relative-SNR spread.** Currently ~±5 dB; real mixtures have someone
  across the room. Narrow spread means the model never learns to dig out a quiet
  speaker.
- **Reverb/noise augmentation** on a fraction of training mixtures (RIR
  convolution, noise at 10–20 dB SNR). Data is clean anechoic LibriSpeech; if the
  graders' inputs have any room acoustics, a clean-only model falls off a cliff.
- **Variable segment length.** All 4.0s now; graders may hand over 10s.
- **Log the RNG seed and per-source gains** into the manifest so any mixture is
  reproducible from its ID.
- **Gender balance.** The CSV records `speaker_id` but not gender, so the
  same-gender mixture ratio is unknown — and same-gender is the hard case
  (pitch cues stop helping). Cross-reference LibriSpeech `SPEAKERS.TXT`.

## 7. The acceptance check

Newly generated data must pass, per mixture:

```python
assert np.abs(mix - sum(srcs)).max() < 1e-4        # currently FAILS (peak err reaches 2.4)
assert all(np.abs(s).max() < 0.999 for s in srcs)  # currently FAILS (~33% of sources)
```

Note this is the gate for **clean new data only**. Do not apply it to
`--repair`-ed old data: alpha repair fixes the scale exactly, but clipping is
irreversible and leaves large peak error with negligible energy, so repaired
data is gated on energy instead (`recon_snr_db >= 25`). See
`adapters/librimix_csv.py`.

## 8. Format contract

Nothing in the model code reads this tree directly — it reads a manifest
(`core/manifest.py`), produced by an adapter (`adapters/librimix_csv.py`). If
the output format changes, write a new adapter (~40 lines); `core/`,
`backbones/`, `heads/` and all seven configs stay untouched.

Keeping the current `mixtures.csv` schema means the existing adapter just works.
