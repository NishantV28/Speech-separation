# Unknown Speaker Speech Separation

> **Automatic speech separation for an unknown number of overlapping speakers using MossFormer2 and Transformer Decoder Attractors (SepTDA).**

This project addresses one of the most challenging speech separation problems: given a single audio recording containing multiple people speaking simultaneously, the model automatically estimates **how many speakers are present** and separates each speaker into an individual audio stream.

Unlike conventional speech separation systems, **the number of speakers is never provided** during inference. The model learns to infer the speaker count directly from the mixture.

---

# Features

- 🎙️ Unknown speaker count estimation
- 🧠 MossFormer2-based separation backbone
- 🔥 Transformer Decoder Attractors (SepTDA)
- ✨ Confidence-guided attractor pruning
- ✨ Attractor repulsion loss 
- ✨ Overlap-weighted separation loss 
- ⚡ End-to-end speech separation
- 📈 Hungarian matching for permutation-invariant training

---

## P5: architecture

Shapes are for one 4-second window at 16 kHz. **5.14M parameters.**

```
                        10 s mixture wav
                               |
                    random 4 s window   (a different one every epoch)
                               |
                          (B, 64000)
                               |
   +---------------------------v---------------------------+
   | STFT    n_fft=256   hop=128                            |
   | real+imag stacked as 2 channels                        |
   +---------------------------+---------------------------+
                               |
                      X  (B, 2, 129, 501)
                          |  |    |    |
                          |  |    |    +-- 501 time frames
                          |  |    +------- 129 frequency bins
                          |  +------------ 2 = real / imaginary
                               |
   +---------------------------v---------------------------+
   | MossFormer2-lite            "the listening"            |
   |                                                        |
   |   fold freq into features  ->  (B, 258, 501)           |
   |   Conv1d                   ->  (B, 192, 501)           |
   |                                one sequence over time  |
   |                                                        |
   |   +-- x7 blocks -------------------------------------+ |
   |   |   GatedAttention    every frame sees every frame | |
   |   |         |           (global: who belongs to whom)| |
   |   |   GatedFSMN         conv over time               | |
   |   |                     (local, parallel, no RNN)    | |
   |   +--------------------------------------------------+ |
   |                                                        |
   |   Conv1d  ->  unfold back to the freq x time grid      |
   +---------------------------+---------------------------+
                               |
                      H  (B, 48, 129, 501)
             a 48-dim description of every time-freq point
                               |
   +---------------------------v---------------------------+
   | TDA head          "who is talking, and how many"       |
   |                                                        |
   |   H --> summary Conv2d --> mean over frequency         |
   |              |                                         |
   |         memory (B, 501, 128)  + positional encoding    |
   |              |                                         |
   |              |        4 LEARNED QUERIES  (4, 128)      |
   |              |        = 4 empty "seats" for speakers   |
   |              |               |                         |
   |              +-------+-------+                         |
   |                      v                                 |
   |         Transformer decoder   (2 layers, 4 heads)      |
   |           - queries CROSS-ATTEND to the mixture        |
   |           - queries SELF-ATTEND to each other          |
   |             `- they negotiate: "you take that voice,   |
   |                I take this one". All at once, so no    |
   |                chain and no compounding errors --      |
   |                this is the whole difference from EDA.  |
   |                      |                                 |
   |              attractors (B, 4, 128)                    |
   |              one 128-d vector per speaker              |
   |                      |                                 |
   |        +-------------+-------------+                   |
   |        v             v             v                   |
   |    exists()      to_mask()      conf()        (1)      |
   |     (B,4)       (B,4,48,2)      (B,4)                  |
   |  "am I real?"   mask weights   "is my output clean?"   |
   +--------+-------------+-------------+-------------------+
            |             |             |
            |     dot with H at every time-freq point
            |             v             |
            |    masks (B, 4, 2, 129, 501)
            |             |             |
            |    complex multiply into X, then iSTFT
            |             v             |
            |      est (B, 4, 64000)    |
            |             |             |
            |    mixture consistency:   |
            |    est += (mix - sum(est)) / 4
            |    (outputs must account for the whole input; free)
            |             |             |
            v             v             v
   +--------------------------------------------------------+
   | LOSS                                                   |
   |                                                        |
   |   outputs come out in arbitrary order, refs too        |
   |     -> HUNGARIAN matching finds the best pairing       |
   |                                                        |
   |   thresholded SNR   (NOT SI-SDR: with 4 slots and 2    |
   |     speakers some refs are SILENCE, and SI-SDR divides |
   |     by the ref -> undefined, on every single batch)    |
   |                                                        |
   |   + exist BCE       (0.1)      was this slot a speaker?|
   |   + conf  BCE       (0.1) (1)  regress its real SI-SNR |
   |   + repulsion       (0.1) (2)  attractors must differ  |
   |   x overlap weight  (1.0) (3)  weight by speaker count |
   +--------------------------------------------------------+

                    ---- AT INFERENCE ----
        activity = sigmoid(exists) > 0.5
        keep     = activity AND conf > 0.3 x best conf
        n_est    = keep.sum()          <-- THE ANSWER
                   nobody ever passed in the count
```

# Pipeline

```
                Mixed Audio
                     │
                 STFT Encoder
                     │
             MossFormer2 Backbone
                     │
        Transformer Decoder Attractors
                     │
          Speaker Activity Estimation
                     │
          Speaker Mask Generation
                     │
                 iSTFT Decoder
                     │
      ┌─────────┬─────────┬─────────┐
      │Speaker 1│Speaker 2│Speaker N│
      └─────────┴─────────┴─────────┘
```

---

# Repository Structure

```
.
├── adapters/           # Dataset adapters
├── backbones/          # MossFormer2 and other backbones
├── configs/            # Training configurations
│   ├── ablations/
│   ├── p2.yaml
│   ├── p4.yaml
│   └── p5.yaml
├── core/               # Training and evaluation engine
├── docs/               # Documentation
├── heads/              # EDA, TDA, OR-PIT heads
├── scripts/            # Utility scripts
├── tests/              # Unit tests
├── manifests/          # Dataset manifests
├── separate.py         # Inference demo
└── README.md
```

---

# Installation

Clone the repository

```bash
git clone <repository_url>

cd <repository_name>
```

Install dependencies

```bash
pip install -r requirements.txt
```

Repair the LibriMix metadata (required)

```bash
python adapters/librimix_csv.py --repair
```

Run the sanity check

```bash
python scripts/overfit_gate.py
```

The script should print

```
GATE PASS
```

---

# Training

Train the proposed **P5** model:

```bash
python -m core.train \
    --config configs/p5_tda_prune_ecapa.yaml \
    --steps 20000 \
    --warmup 2000
```

If training stops unexpectedly, rerun the exact same command to automatically resume from the latest checkpoint.

---

# Evaluation

Quick evaluation

```bash
python -m core.eval \
    --ckpt runs/p5_tda_prune_ecapa-<hash>/last.pt \
    --split manifests/train.jsonl \
    --limit 30
```

Full evaluation

```bash
python -m core.eval \
    --ckpt runs/p5_tda_prune_ecapa-<hash>/last.pt
```

Compare all trained models

```bash
python -m core.eval --compare runs/
```

---

# Inference

Separate an unseen mixture

```bash
python separate.py \
    --ckpt runs/p5_tda_prune_ecapa-<hash>/last.pt \
    --wav mixtures/test/3spk/mixed/test_3spk_00000000.wav \
    --out demo/ \
    --figure
```

---

# Implemented Pipelines

| Pipeline | Backbone | Speaker Counting |
|-----------|----------|------------------|
| **P0** | MossFormer2 | Energy Threshold |
| **P2** | MossFormer2 | EDA |
| **P3** | GridNet-FSMN | EDA |
| **P4** | MossFormer2 | TDA |
| **P5** | MossFormer2 | TDA + Proposed Improvements |
| **P7** | MossFormer2 | OR-PIT |
| **P8** | MossFormer2 | EDA + Confidence |

All models contain approximately **5 million parameters**, ensuring fair comparisons across different speaker-count estimation methods.

---

# Proposed Improvements (P5)

This work extends the original SepTDA framework with three additional contributions.

## 1. Confidence-Guided Attractor Pruning

Each attractor predicts a confidence score corresponding to the quality of its separated speech. Low-confidence attractors are discarded during inference, reducing duplicate speaker predictions.

---

## 2. Attractor Repulsion Loss

A cosine-similarity penalty is introduced between active attractors to discourage multiple attractors from converging onto the same speaker representation.

---

## 3. Overlap-Weighted Separation Loss

Instead of treating every frame equally, the separation loss assigns larger weights to highly-overlapped regions where multiple speakers are simultaneously active.

---

# Training Protocol

| Property | Value |
|----------|-------|
| Dataset | LibriMix |
| Sampling Rate | 16 kHz |
| Training Speakers | 2 & 3 |
| Evaluation | 2, 3 & 4 Speakers |
| Audio Crop | Random 4 seconds |
| Maximum Speaker Slots | 4 |

The model is **never given the number of speakers** during training or inference. Speaker count is inferred directly from the mixture.

---

# Example Workflow

```text
Clone Repository
        │
Install Dependencies
        │
Repair Dataset
        │
Run Sanity Check
        │
Train Model
        │
Evaluate
        │
Separate New Audio
```

---

# Documentation

Detailed documentation is available in the `docs/` directory.

- Architecture
- Dataset Preparation
- Training Guide
- Ablation Studies
- Results
- Troubleshooting

---

# Future Work

- Support for more than four simultaneous speakers
- Multi-channel speech separation
- Streaming real-time inference
- Self-supervised pretraining
- Domain adaptation for noisy real-world conversations

---

# Acknowledgements

This project builds upon ideas from:

- **MossFormer2**
- **SepTDA**
- **Transformer Decoder Attractors**
- **OR-PIT**
- **LibriMix**
- **SpeechBrain**

---


