import marimo

__generated_with = "0.11.17"
app = marimo.App(width="medium")


@app.cell
def __(__file__):
    import sys
    from os import path
    sys.path.append(path.abspath(path.join(path.dirname(__file__), '../../..')))
    import marimo as mo
    import pandas as pd
    from dnallm import load_config, load_model_and_tokenizer, DNADataset, DNATrainer

    return sys, pd, mo, load_config, load_model_and_tokenizer, DNADataset, DNATrainer


@app.cell
def __(mo):
    title = mo.md(
        "<center><h2>Finetune a DNA model with a custom dataset</h2></center>"
    )
    config_text = mo.ui.text(value="finetune_config.yaml", placeholder="config.yaml",
                             label="Config file (*.yaml)", full_width=True)
    model_text = mo.ui.text(value="zhangtaolab/plant-dnagpt-BPE", placeholder="zhangtaolab/plant-dnagpt-BPE",
                            label="Model name or path", full_width=True)
    source1_text = mo.ui.dropdown(['local', 'huggingface', 'modelscope'], value="modelscope", label="Model source", full_width=True)
    datasets_text = mo.ui.text(value="/mnt/extend2/plant-genomic-benchmark/pro_seq/m_esculenta_train.fa", placeholder="zhangtaolab/plant-multi-species-core-promoters",
                               label="Datasets name or path", full_width=True)
    source2_text = mo.ui.dropdown(['local', 'huggingface', 'modelscope'], value="modelscope", label="Dataset source", full_width=True)
    seq_col_text = mo.ui.text(value="sequence", placeholder="sequence", label="Sequence column name", full_width=True)
    label_col_text = mo.ui.text(value="labels", placeholder="labels", label="Label column name", full_width=True)
    maxlen_text = mo.ui.text(value="512", placeholder="512", label="Max token length", full_width=True)
    model_stack = mo.hstack([model_text.style(width="75ch"), source1_text], align='center', justify='center')
    datasets_stack = mo.hstack([datasets_text.style(width="75ch"), source2_text], align='center', justify='center')
    options_stack = mo.hstack([seq_col_text, label_col_text, maxlen_text], align='center', justify='center')
    mo.vstack([title, config_text.style(width="30ch"), model_stack, datasets_stack, options_stack],
              align='center', justify='center')
    return (config_text, model_text, source1_text, datasets_text, source2_text, seq_col_text, label_col_text, maxlen_text, )


@app.cell
def __(mo, config_text, load_config,
       model_text, source1_text, load_model_and_tokenizer, 
       datasets_text, source2_text, seq_col_text, label_col_text, maxlen_text, DNADataset):
    def prepare(config_text, load_config,
        model_text, source1_text, load_model_and_tokenizer, 
        datasets_text, source2_text, seq_col_text, label_col_text, maxlen_text, DNADataset):
        # Load configs
        configs = load_config(config_text.value)
        # Load model and tokenizer
        model_name = model_text.value
        source1 = source1_text.value
        if model_name:
            model, tokenizer = load_model_and_tokenizer(model_name, task_config=configs['task'], source=source1)
        else:
            model = None
            tokenizer = None
        # Load datasets
        datasets_name = datasets_text.value
        seq_col = seq_col_text.value
        label_col = label_col_text.value
        max_length = int(maxlen_text.value)
        source2 = source2_text.value
        if datasets_name:
            if source2 == "huggingface":
                datasets = DNADataset.from_huggingface(datasets_name, seq_col=seq_col, label_col=label_col,
                                                    tokenizer=tokenizer, max_length=max_length)
            elif source2 == "modelscope":
                datasets = DNADataset.from_modelscope(datasets_name, seq_col=seq_col, label_col=label_col,
                                                    tokenizer=tokenizer, max_length=max_length)
            else:
                datasets = DNADataset.load_local_data(datasets_name, seq_col=seq_col, label_col=label_col,
                                                    tokenizer=tokenizer, max_length=max_length)
        else:
            datasets = None
        # Process datasets
        if datasets:
            datasets.encode_sequences(task=configs['task'].task_type, remove_unused_columns=True)
            if isinstance(datasets.dataset, dict):
                pass
            else:
                datasets.split_data()
        else:
            pass
        
        return (configs, model, tokenizer, datasets,)
    train_button = mo.ui.button(label="Start Training", 
                                on_click=lambda _: prepare(
                                    config_text, load_config,
                                    model_text, source1_text, load_model_and_tokenizer,
                                    datasets_text, source2_text, seq_col_text, label_col_text, maxlen_text, DNADataset
                                ))
    mo.vstack([train_button], align='center', justify='center')
    return (train_button,)


@app.cell
def __(mo):
    text_output = mo.output
    return text_output


@app.cell
def __(mo, train_button, DNATrainer, text_output):
    from transformers import TrainerCallback
    from math import ceil
    def get_total_steps(trainer):
        if trainer.args.max_steps and trainer.args.max_steps > 0:
            total_steps = trainer.args.max_steps
        else:
            # 2️⃣ else compute from dataloader
            train_dl = trainer.get_train_dataloader()
            # number of optimizer updates per epoch
            steps_per_epoch = ceil(
                len(train_dl) / trainer.args.gradient_accumulation_steps
            )
            total_steps = steps_per_epoch * trainer.args.num_train_epochs
        return total_steps

    class MarimoCallback(TrainerCallback):
        def __init__(self, text_out):
            self.text_out = text_out
            self.steps = []
            self.epochs = []
            self.all_logs = ""

        def on_log(self, args, state, control, logs=None, **kwargs):
            # logs might contain 'loss', 'eval_loss', 'eval_accuracy', etc.
            step = state.global_step
            self.steps.append(step)
            increment = self.steps[-1] - self.steps[-2] if len(self.steps) > 1 else 0
            # update progress bar
            self.bar.update(increment=increment)

            # collect
            if "loss" in logs:
                txt = f"**Step {step}**<br>" + ", ".join(
                    f"{k}: {v:.4f}" for k, v in logs.items()
                )
            if "eval_loss" in logs:
                txt = ", ".join(
                    f"{k}: {v:.4f}" for k, v in logs.items()
                )
            if txt:
                self.all_logs += txt + "<br>"
            self.text_out.clear()
            self.text_out.replace(mo.md(self.all_logs))

    if train_button.value:
        configs, model, _, datasets = train_button.value
        trainer = DNATrainer(
            model=model,
            config=configs,
            datasets=datasets
        )
        trainer.trainer.add_callback(MarimoCallback(text_output))
        trainer.train()
    else:
        trainer = None
    return trainer


# @app.cell
# def __():
#     return


if __name__ == "__main__":
    app.run()