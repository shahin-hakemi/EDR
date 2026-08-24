import gradio as gr
import edr
from edr import presets, watermark


def run(image, source, target, model, t0, steps, seed, keep_wm):
    bb = presets.BACKBONES[model]
    out = edr.edit(model, bb["model_id"], image.convert("RGB"), source, target,
                   float(t0), bb["gamma"], bb["sigma"], bb["i_min"], bb["i_max"],
                   int(steps), int(seed) if seed else None,
                   bb["src_guidance"], bb["tar_guidance"], bb["res"])
    return watermark.embed(out) if keep_wm else out


with gr.Blocks(title="EDR — Faithful Image Editing") as demo:
    gr.Markdown("## EDR — Faithful Image Editing via Degraded Representations (TMLR 2026)")
    with gr.Row():
        with gr.Column():
            image = gr.Image(type="pil", label="Source image")
            source = gr.Textbox(label="Source prompt")
            target = gr.Textbox(label="Target prompt")
            model = gr.Dropdown(["sd3", "flux"], value="sd3", label="Backbone")
            t0 = gr.Slider(0.3, 1.0, value=0.76, step=0.01, label="Editing strength t0")
            steps = gr.Slider(10, 60, value=50, step=1, label="Steps")
            seed = gr.Number(value=None, label="Seed (optional)", precision=0)
            keep_wm = gr.Checkbox(value=True, label="Embed watermark (default)")
            btn = gr.Button("Edit")
        with gr.Column():
            out = gr.Image(label="Edited image")
    btn.click(run, [image, source, target, model, t0, steps, seed, keep_wm], out)

if __name__ == "__main__":
    demo.launch()
