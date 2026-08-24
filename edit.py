import argparse
from pathlib import Path
from PIL import Image

from edr import edit as run_edit
from edr import presets, watermark


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--edit", help="one of: " + ", ".join(presets.EDITS))
    p.add_argument("--model", choices=list(presets.BACKBONES))
    p.add_argument("--image")
    p.add_argument("--source")
    p.add_argument("--target")
    p.add_argument("--t0", type=float)
    p.add_argument("--gamma", type=float)
    p.add_argument("--sigma", type=float)
    p.add_argument("--steps", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--output", default="out.png")
    p.add_argument("--no-watermark", action="store_true")
    a = p.parse_args()

    if a.edit:
        cfg = presets.resolve(a.edit)
    else:
        if not (a.model and a.image and a.source and a.target):
            p.error("either --edit NAME, or --model --image --source --target")
        bb = presets.BACKBONES[a.model]
        cfg = {"model": a.model, "model_id": bb["model_id"], "image": a.image,
               "source": a.source, "target": a.target,
               "t0": bb["t0"], "gamma": bb["gamma"], "sigma": bb["sigma"],
               "i_min": bb["i_min"], "i_max": bb["i_max"],
               "steps": bb["steps"], "seed": None}
    for k in ("t0", "gamma", "sigma", "steps", "seed"):
        v = getattr(a, k)
        if v is not None:
            cfg[k] = v

    img = Image.open(cfg["image"]).convert("RGB")
    out = run_edit(cfg["model"], cfg["model_id"], img, cfg["source"], cfg["target"],
                   cfg["t0"], cfg["gamma"], cfg["sigma"], cfg["i_min"], cfg["i_max"],
                   cfg["steps"], cfg.get("seed"))
    if not a.no_watermark:
        out = watermark.embed(out)
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    out.save(a.output)
    print(a.output)


if __name__ == "__main__":
    main()
