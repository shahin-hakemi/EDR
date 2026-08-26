# Faithful Image Editing via Degraded Representations

**Shahin Hakemi, Naveed Akhtar, Ghulam Mubashar Hassan, Ajmal Mian**

*Transactions on Machine Learning Research (TMLR), 2026* · [OpenReview](https://openreview.net/forum?id=U2fY7u10QY)

<p align="center"><img src="assets/fig1.jpg" width="95%"></p>

## Abstract

Rectified flow and diffusion-based models currently represent the state-of-the-art in image editing. Despite their impressive capabilities, maintaining faithfulness to the source image — preserving structure and photometric characteristics while satisfying a target prompt — remains a persistent challenge. We propose an optimization- and inversion-free editing framework for rectified flow models (SD3, FLUX.1-dev) that constrains editing trajectories by projecting them onto a degraded representation, suppressing unfaithful trajectory deviations while preserving the flexibility required to satisfy the target prompt.

## Key insight

<p align="center"><img src="assets/fig2.jpg" width="85%"></p>

The editing trajectory is projected onto a degraded representation of its own recent motion, obtained by Gaussian structural smoothing and dynamic range reduction. Directions suppressed by this representation are attenuated with a weight that decays along the trajectory. See Section 3 of the paper.

## Installation

```bash
git clone https://github.com/shahin-hakemi/EDR.git
cd EDR
pip install -r requirements.txt
huggingface-cli login   # required to download SD3 and FLUX.1-dev weights
```

## Example usage

The eight edits shown in Figure 1 are predefined. Pick one and run it by name:

```bash
python edit.py --edit bear_moose --output bear_moose.png
```

Available edits: `woman_puppy`, `statue_photoreal`, `bear_moose`, `empire_eiffel` (FLUX.1-dev) and `open_home`, `cheetahs_tigers`, `tomatoes_golf`, `van_jeep` (SD3).

### Editing your own image

```bash
python edit.py \
    --model sd3 \
    --image path/to/image.png \
    --source "a photo of a cat on a sofa" \
    --target "a photo of a dog on a sofa" \
    --output edited.png
```

Use `--model flux` for FLUX.1-dev. Controls can be set per run: `--t0` (editing strength), `--steps`, `--seed`, `--res`.

By default every output carries an invisible watermark (see below); add `--no-watermark` to disable it.

## Demo

```bash
python demo.py
```

Launches a local Gradio interface.

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
```
