"""Gradio front end.

Layout is deliberate: the verified assessment and the model judgement sit in
separate panes, so a reviewer can always see which sentence came from a tool
and which came from a model.

Mobile notes:
  * Columns use min_width so Gradio's flex row wraps instead of squeezing.
  * A second run button lives in a sticky bar that only exists under 640px.
  * Markdown tables scroll inside their own container, never the page.
  * gr.Progress + queue() cover the 30-60s a model run can take.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import gradio as gr
import pandas as pd

import agent_core
import data_layer as dl

DATASET = dl.load()
CHOICES = [("Portfolio — all parts", "__ALL__")] + [
    (f"{p['part_id']} · {p['name']} (crit {p['criticality']})", p["part_id"])
    for p in sorted(DATASET["parts"], key=lambda x: (-x["criticality"], x["part_id"]))
]
MODEL_CHOICES = [("Auto — discover a tool-capable model", "")] + [
    (mid, mid) for mid in agent_core.hf_candidates()
]

# Only 6 of the 10 columns survive on a phone. The rest live in the memo.
TABLE_COLUMNS = ["Part", "Score", "Band", "DOS (d)", "Lead (d)", "Exposed"]

HEAD_HTML = """
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="theme-color" content="#0f172a">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
"""

CSS = """
:root { --tap: 46px; }

.gradio-container { max-width: 1180px !important; padding: 0 12px !important; }

.hero h1 { font-size: 1.55rem; line-height: 1.2; margin: 0 0 .2rem 0; }
.hero p  { font-size: .9rem; opacity: .8; margin: 0 0 .6rem 0; }

#run-btn { min-height: var(--tap); font-size: 1rem; font-weight: 600; width: 100%; }

/* Gradio defaults inputs to 14px; anything under 16px triggers iOS zoom. */
.gradio-container input[type="text"],
.gradio-container input[type="number"],
.gradio-container textarea,
.gradio-container select { font-size: 16px !important; }

/* Keep wide tables inside their own scroll box instead of widening the page. */
.prose table, .md table, .markdown table {
  display: inline-block;
  max-width: 100%;
  overflow-x: auto;
  white-space: nowrap;
}

.mobile-bar { display: none !important; }

@media (max-width: 640px) {
  .gradio-container { padding: 0 8px calc(8px + env(safe-area-inset-bottom)) 8px !important; }
  .hero h1 { font-size: 1.25rem; }
  .hero p  { font-size: .8rem; }
  .gradio-container .block { padding: 8px !important; }
  .gradio-container .prose { font-size: .92rem; }
  .gradio-container .prose h2 { font-size: 1.05rem; margin-top: 1rem; }

  .mobile-bar {
    display: flex !important;
    position: sticky;
    bottom: 0;
    z-index: 60;
    padding: 8px 0 calc(8px + env(safe-area-inset-bottom)) 0;
    background: var(--background-fill-primary, #ffffff);
    border-top: 1px solid var(--border-color-primary, #e5e7eb);
  }
  .mobile-bar button { min-height: var(--tap); width: 100%; font-weight: 600; }
}

@media (prefers-color-scheme: dark) {
  .mobile-bar { background: #0b1220; }
}
"""

ABOUT = """
**What it does.** Programs a portfolio of parts against active hazard exposure, source
coverage and days of supply, then drafts courses of action for human approval. It cannot
write to any system of record — there is no write path in the codebase.

**Not an accredited environment.** Hugging Face Spaces has no FedRAMP or DoW Impact Level
authorization. Every record is fabricated. For pipeline evaluation only, never CUI or
classified data.

**On a phone.** A single part returns in a few seconds. The full portfolio plus a model run
can take a minute; the progress line under the controls shows what is happening.
"""


def on_run(selection: str, min_criticality: int, mode_label: str, probe: bool,
           model_choice: str = "", progress=gr.Progress()):
    mode = "agent" if mode_label.startswith("Agent") else "deterministic"
    progress(0.05, desc="Loading dataset")
    try:
        progress(0.2, desc="Assessing parts" if mode == "deterministic"
                 else "Assessing parts, then calling the model (this can take a minute)")
        result = agent_core.run(selection, int(min_criticality), mode, bool(probe),
                                model_id=model_choice or None)
        progress(0.95, desc="Verifying audit chain")
    except Exception as exc:  # noqa: BLE001
        message = f"**Run failed:** `{type(exc).__name__}: {exc}`"
        return message, message, pd.DataFrame(), {}, "", None, message

    audit_path = Path(tempfile.gettempdir()) / "audit_log.jsonl"
    audit_path.write_text(result["audit_jsonl"], encoding="utf-8")

    frame = pd.DataFrame(result["risk_table"])
    if not frame.empty:
        frame = frame[[c for c in TABLE_COLUMNS if c in frame.columns]]

    skipped = len(result.get("attempts") or [])
    status = (
        f"**Engine:** {result['engine']}  \n"
        f"**Parts assessed:** {len(result['parts'])} · "
        f"**Models skipped:** {skipped}  \n"
        f"**Audit chain:** {'verified' if result['audit_ok'] else 'BROKEN'} "
        f"({result['audit_count']} entries)"
    )
    model_pane = result["memo_model"] or (
        "_Deterministic mode. Switch the engine to an agent to see model judgement here._"
    )
    progress(1.0, desc="Done")
    return (model_pane, result["memo_verified"], frame,
            result["trace"], result["audit_jsonl"], str(audit_path), status)


with gr.Blocks(title="Supply chain risk agent", theme=gr.themes.Soft(),
               css=CSS, head=HEAD_HTML) as demo:
    gr.HTML(
        '<div class="hero"><h1>Supply chain risk agent</h1>'
        '<p>Synthetic data · read-only tools · hash-chained audit log</p></div>'
    )

    with gr.Row():
        # Controls. min_width makes this column wrap above the results on narrow screens.
        with gr.Column(scale=1, min_width=320, elem_classes="controls"):
            selection = gr.Dropdown(choices=CHOICES, value="__ALL__", label="Scope")
            min_crit = gr.Slider(1, 5, value=3, step=1, label="Criticality floor")
            engine = gr.Radio(
                choices=["Agent — use a model if configured", "Deterministic baseline only"],
                value="Deterministic baseline only",
                label="Engine",
            )
            with gr.Accordion("Advanced", open=False):
                model_pick = gr.Dropdown(
                    choices=MODEL_CHOICES, value="", label="Model override",
                    info="Auto reads the router catalogue and takes the first tool-capable "
                         "model your token can reach.",
                )
                probe = gr.Checkbox(
                    value=False, label="Prompt-injection probe",
                    info="Injects a hostile instruction into one hazard record.",
                )
            run_button = gr.Button("Run assessment", variant="primary", elem_id="run-btn")
            status = gr.Markdown("Ready.")

        with gr.Column(scale=3, min_width=340):
            with gr.Tabs():
                with gr.Tab("Verified"):
                    verified = gr.Markdown()
                with gr.Tab("Model"):
                    narrative = gr.Markdown()
                with gr.Tab("Table"):
                    table = gr.Dataframe(wrap=True, interactive=False, label=None,
                                         elem_classes="risk-table")
                with gr.Tab("Trace"):
                    trace = gr.JSON(label=None)
                with gr.Tab("Audit"):
                    gr.Markdown("Append-only, hash-chained. Editing an earlier entry breaks "
                                "verification.")
                    audit_code = gr.Code(language="json", label="audit_log.jsonl", lines=12)
                    audit_file = gr.File(label="Download")

    # Phone-only sticky action bar. Hidden by CSS above 640px.
    with gr.Row(elem_classes="mobile-bar"):
        run_mobile = gr.Button("Run assessment", variant="primary")

    with gr.Accordion("About this demo", open=False):
        gr.Markdown(ABOUT)

    inputs = [selection, min_crit, engine, probe, model_pick]
    outputs = [narrative, verified, table, trace, audit_code, audit_file, status]

    run_button.click(on_run, inputs=inputs, outputs=outputs)
    run_mobile.click(on_run, inputs=inputs, outputs=outputs)
    demo.load(on_run, inputs=inputs, outputs=outputs)

demo.queue(default_concurrency_limit=2)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", 7860)),
    )
