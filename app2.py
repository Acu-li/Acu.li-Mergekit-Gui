import os
import pathlib
import random
import string
import tempfile
from typing import Iterable, List

import gradio as gr
import torch
import yaml
import webbrowser
from gradio_logsview.logsview import Log, LogsView, LogsViewRunner
from mergekit.config import MergeConfiguration

url = 'http://127.0.0.1:7860'
webbrowser.open(url)

has_gpu = torch.cuda.is_available()

cli = "mergekit-yaml config.yaml merge --copy-tokenizer" + (
    " --cuda --low-cpu-memory" if has_gpu else " --allow-crimes --out-shard-size 1B --lazy-unpickle --transformers-cache ./cache1 --lora-merge-cache ./cache"
)

MARKDOWN_DESCRIPTION = """
# Aculi-mergekit-gui

The local way, to do a model merge.

Well, just paste in the ``.Yaml`` here.
"""

MARKDOWN_ARTICLE = """
___

## Merge Methods

Supported Merge Methods:

| Method                                                                                       | `merge_method` value | Multi-Model | Uses base model |
| -------------------------------------------------------------------------------------------- | -------------------- | ----------- | --------------- |
| Linear ([Model Soups](https://arxiv.org/abs/2203.05482))                                     | `linear`             | ✅          | ❌              |
| SLERP                                                                                        | `slerp`              | ❌          | ✅              |
| [Task Arithmetic](https://arxiv.org/abs/2212.04089)                                          | `task_arithmetic`    | ✅          | ✅              |
| [TIES](https://arxiv.org/abs/2306.01708)                                                     | `ties`               | ✅          | ✅              |
| [DARE](https://arxiv.org/abs/2311.03099) [TIES](https://arxiv.org/abs/2306.01708)            | `dare_ties`          | ✅          | ✅              |
| [DARE](https://arxiv.org/abs/2311.03099) [Task Arithmetic](https://arxiv.org/abs/2212.04089) | `dare_linear`        | ✅          | ✅              |
| Passthrough                                                                                  | `passthrough`        | ❌          | ❌              |
| [Model Stock](https://arxiv.org/abs/2403.19522)                                              | `model_stock`        | ✅          | ✅              |


## 😱

This Program is based on Mergekit-GUI on huggingface spaces.
"""


def merge(yaml_config: str, local_path: str) -> Iterable[List[Log]]:
    runner = LogsViewRunner()

    if not yaml_config:
        yield runner.log("Empty yaml", level="ERROR")
        return
    try:
        merge_config = MergeConfiguration.model_validate(yaml.safe_load(yaml_config))
    except Exception as e:
        yield runner.log(f"Invalid yaml {e}", level="ERROR")
        return

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdirname:
        tmpdir = pathlib.Path(tmpdirname)
        merged_path = tmpdir / "merged"
        merged_path.mkdir(parents=True, exist_ok=True)
        config_path = merged_path / "config.yaml"
        config_path.write_text(yaml_config)
        yield runner.log(f"Merge configuration saved in {config_path}")

        if not local_path:
            yield runner.log("No local path provided. Generating a random folder name.")
            local_path = f"Aculi-mergekit-{merge_config.merge_method}"
            # Make local_path "unique" (no need to be extra careful on uniqueness)
            local_path += "-" + "".join(random.choices(string.ascii_lowercase, k=7))
            local_path = local_path.replace("/", "-").strip("-")
            local_path = os.path.join(os.getcwd(), local_path)
        
        local_path = pathlib.Path(local_path)
        local_path.mkdir(parents=True, exist_ok=True)

        tmp_env = os.environ.copy()
        tmp_env["HF_HOME"] = f"{tmpdirname}/.cache"
        yield from runner.run_command(cli.split(), cwd=merged_path, env=tmp_env)

        if runner.exit_code != 0:
            yield runner.log("Merge failed.", level="ERROR")
            return

        yield runner.log("Model merged successfully. Saving to local path.")
        
        # Copy all files from merged_path to local_path
        for item in merged_path.iterdir():
            if item.is_file():
                dest = local_path / item.name
                with open(dest, 'wb') as f_dest:
                    f_dest.write(item.read_bytes())
                yield runner.log(f"Copied {item.name} to {dest}")
            elif item.is_dir():
                dest = local_path / item.name
                dest.mkdir(parents=True, exist_ok=True)
                for subitem in item.rglob('*'):
                    if subitem.is_file():
                        subdest = dest / subitem.relative_to(item)
                        subdest.parent.mkdir(parents=True, exist_ok=True)
                        with open(subdest, 'wb') as f_subdest:
                            f_subdest.write(subitem.read_bytes())
                        yield runner.log(f"Copied {subitem.name} to {subdest}")

        yield runner.log(f"Model successfully saved to: {local_path}")

with gr.Blocks() as demo:
    gr.Markdown(MARKDOWN_DESCRIPTION)

    with gr.Row():
        filename = gr.Textbox(visible=False, label="filename")
        config = gr.Code(language="yaml", lines=10, label="config.yaml")
        with gr.Column():
            local_path = gr.Textbox(
                lines=1,
                label="Local Path",
                info="Please select a folder path:",
                placeholder="If you leave this field empty, a new folder will be created in the same directory as the program.",
            )
    button = gr.Button("Merge", variant="primary")
    logs = LogsView(label="Terminal output")
    gr.Markdown(MARKDOWN_ARTICLE)

    button.click(fn=merge, inputs=[config, local_path], outputs=[logs])

demo.queue(default_concurrency_limit=1).launch()
