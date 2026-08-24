# Per-backbone constants (fixed across all edits) and the Figure 1 edit entries.
# gamma / sigma / bounds / guidance are backbone-level and never vary per edit.

BACKBONES = {
    "sd3": {
        "model_id": "stabilityai/stable-diffusion-3-medium-diffusers",
        "gamma": 2.0, "sigma": 5.0, "i_min": 0.25, "i_max": 0.75,
        "src_guidance": 3.5, "tar_guidance": 13.5,
        "steps": 50, "t0": 0.76, "res": 1024,
    },
    "flux": {
        "model_id": "black-forest-labs/FLUX.1-dev",
        "gamma": 5.0, "sigma": 5.0, "i_min": 0.25, "i_max": 0.75,
        "src_guidance": 1.5, "tar_guidance": 5.5,
        "steps": 28, "t0": 0.9, "res": 1024,
    },
}

# Per edit: backbone, source image, prompts, and optional per-edit t0/steps/seed.
EDITS = {
    "woman_puppy":    {"model": "flux", "image": "examples/woman.png",
                       "source": "", "target": ""},
    "statue_photoreal": {"model": "flux", "image": "examples/statue.png",
                       "source": "", "target": ""},
    "bear_moose":     {"model": "flux", "image": "examples/bear.png",
                       "source": "", "target": ""},
    "empire_eiffel":  {"model": "flux", "image": "examples/empire_state.png",
                       "source": "", "target": ""},
    "open_home":      {"model": "sd3", "image": "examples/open_sign.png",
                       "source": "", "target": ""},
    "cheetahs_tigers": {"model": "sd3", "image": "examples/cheetahs.png",
                       "source": "", "target": ""},
    "tomatoes_golf":  {"model": "sd3", "image": "examples/tomatoes.png",
                       "source": "", "target": ""},
    "van_jeep":       {"model": "sd3", "image": "examples/van.png",
                       "source": "", "target": ""},
}


def resolve(name):
    if name not in EDITS:
        raise KeyError(f"unknown edit '{name}'. Available: {', '.join(EDITS)}")
    e = dict(EDITS[name])
    bb = BACKBONES[e["model"]]
    e["model_id"] = bb["model_id"]
    for k in ("gamma", "sigma", "i_min", "i_max", "src_guidance", "tar_guidance", "res"):
        e[k] = bb[k]
    for k in ("t0", "steps", "seed"):
        e.setdefault(k, bb.get(k))
    if not e["source"] or not e["target"]:
        raise ValueError(f"fill source/target prompts for edit '{name}' in presets.py")
    return e
