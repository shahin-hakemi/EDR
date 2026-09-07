# Per-backbone defaults, and the Figure 1 edit entries.
#
# gamma / i_min / i_max / guidance are backbone-level.
# sigma, t0, steps, seed, res are backbone defaults that an edit MAY override
# by adding the key to its entry; if the key is absent, the backbone value is used.
#
# NOTE: the eight paper edits below intentionally leave sigma at the backbone
# default (sigma = 5), matching the single-configuration setup reported in the
# paper. Per-edit sigma is available for your own images / experimentation.

BACKBONES = {
    "sd3": {
        "model_id": "stabilityai/stable-diffusion-3-medium-diffusers",
        "gamma": 2.0, "sigma": 5.0, "i_min": 0.25, "i_max": 0.75, "seed": 0,
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

# Per edit: backbone, source image, prompts.
# Optional per-edit overrides: t0, sigma, steps, seed, i_min, i_max, res.
#   e.g.  "bear_moose": {..., "t0": 0.85, "sigma": 7.0},
EDITS = {
    "goat_puppy":    {"model": "flux", "image": "examples/woman.png",
                       "source": "A young woman tenderly cradles a baby goat in a pastoral setting.",
                       "target": "A young woman tenderly cradles a puppy in a pastoral setting."},
    "statue_photoreal": {"model": "flux", "image": "examples/statue.png",
                       "source": "A bronze sculpture depicts a young boy with his dog on his lap, sharing a moment with a curious cat.",
                       "target": "A photograph depicts a real young boy with his dog on his lap, sharing a moment with a curious cat.",
                       "gamma": 4.5},
    "bear_moose":     {"model": "flux", "image": "examples/bear.png",
                       "source": "A large brown bear stands near a pond, seemingly sniffing the air.",
                       "target": "A large moose stands near a pond, seemingly sniffing the air.",
                       "gamma": 4.5, "sigma": .5, "t0": 1},
    "empire_eiffel":  {"model": "flux", "image": "examples/empire_state.png",
                       "source": "The iconic Empire State Building stands tall amidst the vibrant cityscape of New York City at sunset.",
                       "target": "The iconic eiffel tower stands tall emitting yellow light amidst the vibrant cityscape of New York City at sunset.",
                       "gamma": 8,"sigma": .5, "seed": 42, "i_min": .25, "i_max": .75, "t0": 1},
    "open_home":      {"model": "sd3", "image": "examples/open_sign.png",
                       "source": "A solitary young woman looks at distant out of the window next to the vibrant 'OPEN' sign written in bright red. Her reflection can be seen in the glass.",
                       "target": "A solitary young woman looks at distant out of the window next to the vibrant 'HOME' sign written in bright red. Her reflection can be seen in the glass.", "gamma": 5, "seed": 0},
    "cheetahs_tigers": {"model": "flux", "image": "examples/cheetahs.png",
                       "source": "Two cheetahs relax in a grassy enclosure beneath the shade of a pine tree.",
                       "target": "Two tigers relax in a grassy enclosure beneath the shade of a pine tree."},
    "tomatoes_golf":  {"model": "sd3", "image": "examples/tomatoes.png",
                       "source": "A vibrant assortment of heirloom tomatoes in a rainbow of colors, artfully arranged on a rustic wooden surface.",
                       "target": "A vibrant assortment of golf balls in a rainbow of colors, artfully arranged on a rustic wooden surface.",
                       "gamma":3},
    "van_jeep":       {"model": "sd3", "image": "examples/van.png",
                       "source": "A van parked in front of a house, outside a garage. The van is of a classic model. The house has a brown exterior, and there is a tree nearby.",
                       "target": "A military jeep parked in front of a house, outside a garage. The van is of a classic model. The house has a brown exterior, and there is a tree nearby."},
}

# keys taken from the backbone unless the edit overrides them
_INHERIT = ("gamma", "sigma", "i_min", "i_max", "src_guidance", "tar_guidance",
            "t0", "steps", "seed", "res")


def resolve(name):
    if name not in EDITS:
        raise KeyError(f"unknown edit '{name}'. Available: {', '.join(EDITS)}")
    e = dict(EDITS[name])
    bb = BACKBONES[e["model"]]
    e["model_id"] = bb["model_id"]
    for k in _INHERIT:
        e.setdefault(k, bb.get(k))   # edit's own value wins; else backbone default
    if not e["source"] or not e["target"]:
        raise ValueError(f"fill source/target prompts for edit '{name}' in presets.py")
    return e
