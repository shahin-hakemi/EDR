import argparse
from pathlib import Path
from PIL import Image

import edr
from edr import presets, watermark


def main():
    p = argparse.ArgumentParser(description="EDR — Faithful Image Editing via Degraded Representations")
    p.add_argument("--edit", help="named Figure 1 edit: " + ", ".join(presets.EDITS))
    p.add_argument("--model", choices=["sd3", "flux"])
    p.add_argument("--image")
    p.add_argument("--source")
    p.add_argument("--target")
    p.add_argument("--t0", type=float)
    p.add_argument("--steps", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--res", type=int)
    p.add_argument("--output", default="out.png")
    p.add_argument("--no-watermark", action="store_true")
    a = p.parse_args()

    if a.edit:
        cfg = presets.resolve(a.edit)
    else:
        if not (a.model and a.image and a.source and a.target):
            p.error("use --edit NAME, or --model --image --source --target")
        bb = presets.BACKBONES[a.model]
        cfg = {"model": a.model, "model_id": bb["model_id"], "image": a.image,
               "source": a.source, "target": a.target,
               "gamma": bb["gamma"], "sigma": bb["sigma"],
               "i_min": bb["i_min"], "i_max": bb["i_max"],
               "src_guidance": bb["src_guidance"], "tar_guidance": bb["tar_guidance"],
               "t0": bb["t0"], "steps": bb["steps"], "res": bb["res"], "seed": None}
    for k in ("t0", "steps", "seed", "res"):
        if getattr(a, k) is not None:
            cfg[k] = getattr(a, k)

    img = Image.open(cfg["image"]).convert("RGB")
    out = edr.edit(cfg["model"], cfg["model_id"], img, cfg["source"], cfg["target"],
                   cfg["t0"], cfg["gamma"], cfg["sigma"], cfg["i_min"], cfg["i_max"],
                   cfg["steps"], cfg.get("seed"),
                   cfg["src_guidance"], cfg["tar_guidance"], cfg["res"])
    if not a.no_watermark:
        out = watermark.embed(out)
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    out.save(a.output)
    print(a.output)


if __name__ == "__main__":
    main()
