import gradio as gr
from PIL import Image

from edr import edit as run_edit
from edr import presets, watermark


def run(image, source, target, model, t0, steps, seed, keep_watermark):
    bb = presets.BACKBONES[model]
    out = run_edit(model, bb["model_id"], image.convert("RGB"), source, target,
                   t0, bb["gamma"], bb["sigma"], bb["i_min"], bb["i_max"],
                   int(steps), int(seed) if seed else None)
    if keep_watermark:
        out = watermark.embed(out)
    return out


with gr.Blocks(title="EDR — Faithful Image Editing") as demo:
    gr.Markdown("## EDR — Faithful Image Editing via Degraded Representations (TMLR 2026)")
    with gr.Row():
        with gr.Column():
            image = gr.Image(type="pil", label="Source image")
            source = gr.Textbox(label="Source prompt")
            target = gr.Textbox(label="Target prompt")
            model = gr.Dropdown(list(presets.BACKBONES), value="sd3", label="Backbone")
            t0 = gr.Slider(0.3, 1.0, value=0.9, step=0.01, label="Editing strength t0")
            steps = gr.Slider(10, 100, value=50, step=1, label="Steps")
            seed = gr.Number(value=None, label="Seed (optional)", precision=0)
            keep_watermark = gr.Checkbox(value=True, label="Embed watermark (default)")
            btn = gr.Button("Edit")
        with gr.Column():
            out = gr.Image(label="Edited image")
    btn.click(run, [image, source, target, model, t0, steps, seed, keep_watermark], out)

if __name__ == "__main__":
    demo.launch()
