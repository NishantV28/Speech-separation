# Building the dataset from scratch — LibriSpeech + WHAM!

Complete guide, assuming nothing exists yet. End state: mixtures of 2–6
concurrent speakers, optionally noisy, in the manifest format the training code
reads.

Budget roughly **40 GB disk** and **2–4 hours**, most of it downloading.

---

## 0. Why this is worth doing carefully

The generator determines the ceiling on everything downstream. Two mistakes are
easy to make, silent, and expensive — both were found in a previous version of
this dataset by auditing the wavs:

1. **Normalising the mixture after summing.** Gives you `mix = α·Σsources` with
   α anywhere in 0.118–6.027. Every loss that assumes `mix == Σsources` is then
   quietly wrong.
2. **Writing sources as int16 after applying gains.** ~33% of source files were
   hard-clipped (10 ms of flat-topped waveform in the worst case). Clipping is
   not invertible — the ground truth is permanently damaged.

Both are avoided by two rules: **scale before summing**, and **save float32**.
The generator below does.

---

## 1. Download LibriSpeech

From <https://www.openslr.org/12/>.

| Subset | Size | Speakers | Use |
|---|---|---|---|
| `train-clean-100` | 6.3 GB | 251 | training (minimum) |
| `train-clean-360` | 23 GB | 921 | training (**recommended**) |
| `dev-clean` | 337 MB | 40 | validation |
| `test-clean` | 346 MB | 40 | test |

```bash
mkdir -p raw && cd raw
for s in train-clean-100 train-clean-360 dev-clean test-clean; do
  wget https://www.openslr.org/resources/12/$s.tar.gz
  tar -xzf $s.tar.gz
done
```

Take `train-clean-360` if you can. More speakers is the cheapest generalisation
win available — separation quality on unseen speakers scales with how many you
trained on.

Already 16 kHz FLAC, mono. No resampling needed.

**Also grab `SPEAKERS.TXT`** (in the LibriSpeech root). It carries each speaker's
gender, which you need for §6.

## 2. Download WHAM! noise

From <https://wham.whisper.ai/>.

Get **`wham_noise.zip`** (16 kHz) — not `high_res_wham.zip`, which is 48 kHz and
~74 GB in 74 chunks. You are working at 16 kHz; the high-res version is a waste
of a download. Check the site for the current URL and MD5s.

```bash
cd raw
wget <wham_noise.zip URL from the site>
unzip wham_noise.zip
```

WHAM! is real ambient noise — restaurants, cafes, bars, parks around the SF Bay
Area, recorded on a binaural mic. That realism is the point: LibriSpeech is clean
anechoic audiobook narration, and a model trained only on that falls apart on
anything with a room in it. If the graders' test inputs have any background at
all, noise augmentation is what saves you.

## 3. Generate

```bash
python scripts/generate_dataset.py \
    --librispeech raw/LibriSpeech \
    --wham raw/wham_noise \
    --out mixtures \
    --noise-prob 0.5 \
    --seed 0
```

~1–2 hours. Then build the manifests:

```bash
python adapters/librimix_csv.py --repair
```

(`--repair` is a no-op on correctly generated data — α comes out at 1.0 and
nothing is quarantined. That is itself a useful check.)

---

## 4. What the generator does, and why

### Speaker-disjoint splits

Speakers are partitioned **before** any mixture is built, so no speaker appears
in more than one split. This is the single most commonly botched thing in
separation datasets: if a speaker leaks from train into test, your test score
measures memorisation, and it will not survive the graders' set.

LibriSpeech makes this easy — `train-clean-*` / `dev-clean` / `test-clean` are
already disjoint. The generator asserts it anyway.

### Speaker counts — 2/3/4/5 train, 6 test-only

| Count | train | val | test |
|---|---|---|---|
| 2, 3, 4, 5 | yes | yes | yes |
| **6** | **no** | **no** | **yes** |

**6spk is held out of training on purpose.** It is the extrapolation test: a
count the model has never seen. That is the evidence for "scales to as many
speakers as possible", and holding it out is the only way to earn the claim.
Train on it and you have nothing left to prove the point with.

2spk matters more than it looks: four distinct counts teach the model *counting
as a concept* rather than a constant. SepTDA trains on 2+3 and generalises to 5 —
the range is what does it.

### Normalisation order — the whole ballgame

```python
# 1. loudness-normalise each source to a target LUFS, plus a per-source offset
srcs = [loudness_normalise(s, target_lufs + offset_i) for i, s in enumerate(srcs)]

# 2. sum
mix = sum(srcs)

# 3. if ANYTHING clips, scale EVERYTHING by the same factor -- sources included
peak = max(abs(mix).max(), max(abs(s).max() for s in srcs))
if peak > 0.95:
    k = 0.95 / peak
    srcs = [s * k for s in srcs]
    mix  = mix * k

# 4. save float32
```

Step 3 is the fix. Note it takes the max over the **sources too**, not just the
mixture — that is what stops individual sources clipping. And the same `k` hits
everything, so `mix == Σsources` survives exactly.

LUFS (perceptual loudness) rather than peak: peak normalisation makes a file's
loudness depend on one transient sample, so two sources at the same peak can
differ wildly in how loud they actually sound.

### Overlap control

Each source gets a random start offset within the segment. Overlap ratio =
fraction of frames where ≥2 speakers are simultaneously active (energy-based
VAD). The generator samples offsets until overlap lands in `--overlap-range`
(default 0.3–1.0).

Fully-overlapped-only data is unrealistic; zero-overlap data is not the task.

### VAD check

Any source whose speech-active fraction inside the segment is below
`--min-active` (default 0.15) causes a resample. Otherwise you get near-silent
references, and SI-SNR is undefined on those — they poison training.

### Noise

With `--noise-prob 0.5`, half the mixtures get a WHAM! segment at a random SNR
in `--noise-snr` (default 10–20 dB).

**Important consequence:** when noise is added, `mix = Σsources + noise`, so
`mix ≠ Σsources`. That is correct — the model should separate speakers, not
reproduce the noise — but it means **mixture consistency must be disabled for
noisy mixtures**. The generator flags them with `meta.noisy: true` and also
writes `mix_clean` so you can check consistency on the clean part.

---

## 5. Verify before you trust it

```bash
python scripts/generate_dataset.py --verify mixtures
```

Per mixture, the acceptance check:

```python
assert np.abs(mix_clean - sum(srcs)).max() < 1e-4      # consistency
assert all(np.abs(s).max() < 0.999 for s in srcs)      # no clipping
assert sr == 16000 and dtype == float32
```

And across the dataset:

- speaker overlap between every pair of splits == 0
- no speaker repeated within a mixture
- overlap ratio inside `--overlap-range`
- every source above `--min-active`

If the consistency assert fails, **stop** — every downstream loss is wrong.

---

## 6. Choices worth making deliberately

**Test set size.** Default 500/count. 100/count gives roughly ±0.5 dB confidence
on SI-SNRi — wider than the gap likely to separate two pipelines, which makes the
comparison the whole project rests on unable to resolve anything. Validation at
100/count is fine.

**Gender balance.** Same-gender mixtures are the hard case — pitch stops being a
usable cue. Random sampling gives you ~50% mixed-gender at 2 speakers, and your
score will be flattered by it. `--same-gender-prob 0.5` forces a harder,
better-controlled distribution. Requires `SPEAKERS.TXT`.

**SNR spread.** `--snr-range -5 5` by default. Widen it (e.g. `-10 10`) and the
model learns to dig out quiet speakers — someone across the room. Narrow spread
means it never has to.

**Segment length.** 4.0s default. Generate a longer test set too (`--segment 8`)
if the graders may hand over longer clips — none of the models are length-locked,
but you want to know before they do.

**Dynamic mixing beats all of this.** The generator writes fixed files, which is
what the current code reads. But 10,861 utterances make ~10¹² possible 3-speaker
mixtures, and freezing 3,000 of them throws away essentially all of it.
Regenerating mixtures every epoch is worth +1–2 dB on its own — more than any
architecture choice in this project. The generator's `sample_mixture()` is
written to be callable from a `Dataset.__getitem__` for exactly this.

---

## 7. Output layout

```
mixtures/
  train/{2,3,4,5}spk/
    mixed/<id>.wav              float32, what the model sees (noisy if noised)
    clean/<id>.wav              float32, sum of sources, no noise
    sources/<id>/speaker{1..n}.wav
    noise/<id>.wav              only if noised
    mixtures.csv
  validation/{2,3,4,5}spk/...
  test/{2,3,4,5,6}spk/...       <- 6spk here ONLY
```

`mixtures.csv` keeps the existing schema, so `adapters/librimix_csv.py` reads it
unchanged. If you change the schema, write a new adapter (~40 lines) — `core/`,
`backbones/`, `heads/` and all 9 configs stay untouched. That is what the
manifest boundary is for.

---

## Sources

- LibriSpeech — <https://www.openslr.org/12/>
- WHAM! — <https://wham.whisper.ai/>, [README](https://wham.whisper.ai/WHAM_README.html)
- LibriMix (the recipe this follows, incl. LUFS convention) — [arXiv:2005.11262](https://arxiv.org/abs/2005.11262), [github](https://github.com/JorisCos/LibriMix)
- WHAMR! (if you later add reverb) — [arXiv:1910.10279](https://arxiv.org/abs/1910.10279)
