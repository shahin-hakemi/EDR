# Faithful Image Editing via Degraded Representations

**Shahin Hakemi, Naveed Akhtar, Ghulam Mubashar Hassan, Ajmal Mian**

*Transactions on Machine Learning Research (TMLR), 2026* · [OpenReview](https://openreview.net/forum?id=U2fY7u10QY)

<p align="center"><img src="assets/fig1.jpg" width="95%"></p>

## Abstract

Rectified flow and diffusion-based models currently represent the state-of-the-art in image editing. Despite their impressive capabilities, maintaining faithfulness to the source image — preserving structure and photometric characteristics while satisfying a target prompt — remains a persistent challenge. We propose an optimization- and inversion-free editing framework for rectified flow models (SD3, FLUX.1-dev) that constrains editing trajectories by projecting them onto a degraded representation, suppressing unfaithful trajectory deviations while preserving the flexibility required to satisfy the target prompt.

## Method

<p align="center"><img src="assets/fig2.jpg" width="85%"></p>

The editing trajectory is projected onto a degraded representation of its own recent motion, obtained by Gaussian structural smoothing and dynamic range reduction. Directions suppressed by this representation are attenuated with a weight that decays along the trajectory. See Section 3 of the paper.

## Installation

```bash
git clone https://github.com/shahin-hakemi/EDR.git
cd EDR
pip install -r requirements.txt
huggingface-cli login   # required for SD3 / FLUX.1-dev weights
```

## Reproducing Figure 1

This repository releases the EDR method and the per-edit configurations of Figure 1.

```bash
python edit.py --config configs/fig1/bear_moose.yaml
```

| Edit | Backbone | Config |
|---|---|---|
| Woman and baby goat → puppy | FLUX | `configs/fig1/woman_puppy.yaml` |
| Statue → photorealistic | FLUX | `configs/fig1/statue_photoreal.yaml` |
| Bear → moose | FLUX | `configs/fig1/bear_moose.yaml` |
| Empire State → Eiffel Tower | FLUX | `configs/fig1/empire_eiffel.yaml` |
| OPEN → HOME | SD3 | `configs/fig1/open_home.yaml` |
| Cheetahs → tigers | SD3 | `configs/fig1/cheetahs_tigers.yaml` |
| Heirloom tomatoes → golf balls | SD3 | `configs/fig1/tomatoes_golf.yaml` |
| Yellow van → army jeep | SD3 | `configs/fig1/van_jeep.yaml` |

Each config specifies the source image, prompts, and the hyperparameters used for that edit.

## Hyperparameters

All values can be set in a config or overridden on the command line:

```bash
python edit.py --config configs/fig1/bear_moose.yaml --t0 0.85 --gamma 4
```

| Argument | Meaning | Paper default |
|---|---|---|
| `--t0` | editing strength (start of trajectory) | 0.9 (FLUX) / [ ] (SD3) |
| `--gamma` | decay rate of the projection weight | 5 (FLUX) / 2 (SD3) |
| `--sigma` | Gaussian smoothing std (kernel 6σ) | 5 |
| `--i_min --i_max` | intensity bounds of the degradation | 0.25, 0.75 |
| `--steps` | total timesteps | [ ] |
| `--seed` | random seed | [ ] |

## Demo

```bash
python demo.py
```

Launches a local Gradio interface: upload an image, enter source and target prompts, adjust hyperparameters, edit.

## Watermarking and use

Every output is embedded with an invisible blind watermark by default (Zhang et al., 2019; evaluated in Appendix F of the paper). Disable with `--no-watermark`. Use of this software must comply with the acceptable-use policies of the underlying models (Stability AI Community License; FLUX.1-dev Non-Commercial License), which prohibit deceptive manipulation of real individuals. Do not use it to fabricate events, misrepresent identifiable people, or produce non-consensual edits.

## License

Code: MIT. Model weights are not distributed here and are governed by their own licenses.

## Citation

```bibtex
@article{hakemi2026faithful,
  title   = {Faithful Image Editing via Degraded Representations},
  author  = {Hakemi, Shahin and Akhtar, Naveed and Hassan, Ghulam Mubashar and Mian, Ajmal},
  journal = {Transactions on Machine Learning Research},
  issn    = {2835-8856},
  year    = {2026},
  url     = {https://openreview.net/forum?id=U2fY7u10QY}
}
