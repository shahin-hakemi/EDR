# Per-backbone settings (fixed across all edits) and the Figure 1 edit entries.

BACKBONES = {
    "sd3": {
        "model_id": "stabilityai/stable-diffusion-3-medium-diffusers",
        "gamma": 2,
        "sigma": 5,
        "i_min": 0.25,
        "i_max": 0.75,
        "t0": None,      # FILL: SD3 default editing strength (Section 4.1)
        "steps": None,   # FILL
    },
    "flux": {
        "model_id": "black-forest-labs/FLUX.1-dev",
        "gamma": 5,
        "sigma": 5,
        "i_min": 0.25,
        "i_max": 0.75,
        "t0": 0.9,
        "steps": None,   # FILL
    },
}

# Each entry: backbone, source image, prompts, and per-edit values
# (t0 / steps / seed override the backbone defaults when set).
EDITS = {
    "woman_puppy": {
        "model": "flux",
        "image": "examples/woman.png",
        "source": "",  # FILL
        "target": "",  # FILL
        "t0": None, "steps": None, "seed": None,
    },
    "statue_photoreal": {
        "model": "flux",
        "image": "examples/statue.png",
        "source": "",  # FILL
        "target": "",  # FILL
        "t0": None, "steps": None, "seed": None,
    },
    "bear_moose": {
        "model": "flux",
        "image": "examples/bear.png",
        "source": "",  # FILL
        "target": "",  # FILL
        "t0": None, "steps": None, "seed": None,
    },
    "empire_eiffel": {
        "model": "flux",
        "image": "examples/empire_state.png",
        "source": "",  # FILL
        "target": "",  # FILL
        "t0": None, "steps": None, "seed": None,
    },
    "open_home": {
        "model": "sd3",
        "image": "examples/open_sign.png",
        "source": "",  # FILL
        "target": "",  # FILL
        "t0": None, "steps": None, "seed": None,
    },
    "cheetahs_tigers": {
        "model": "sd3",
        "image": "examples/cheetahs.png",
        "source": "",  # FILL
        "target": "",  # FILL
        "t0": None, "steps": None, "seed": None,
    },
    "tomatoes_golf": {
        "model": "sd3",
        "image": "examples/tomatoes.png",
        "source": "",  # FILL
        "target": "",  # FILL
        "t0": None, "steps": None, "seed": None,
    },
    "van_jeep": {
        "model": "sd3",
        "image": "examples/van.png",
        "source": "",  # FILL
        "target": "",  # FILL
        "t0": None, "steps": None, "seed": None,
    },
}


def resolve(name):
    e = dict(EDITS[name])
    bb = dict(BACKBONES[e["model"]])
    for k in ("t0", "steps"):
        if e.get(k) is None:
            e[k] = bb[k]
    for k in ("gamma", "sigma", "i_min", "i_max", "model_id"):
        e[k] = bb[k]
    return e
