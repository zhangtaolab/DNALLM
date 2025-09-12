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
    config_title = mo.md("<h3>Finetune configuration</h3>")
    config_text = mo.ui.text(value="finetune_config.yaml", placeholder="config.yaml",
                             label="Config file (*.yaml)", full_width=True)
    model_text = mo.ui.text(value="zhangtaolab/plant-dnagpt-BPE", placeholder="zhangtaolab/plant-dnagpt-BPE",
                            label="Model name or path", full_width=True)
    source1_text = mo.ui.dropdown(['local', 'huggingface', 'modelscope'], value="modelscope", label="Model source", full_width=True)
    datasets_text = mo.ui.text(value="zhangtaolab/plant-multi-species-core-promoters",
                               placeholder="zhangtaolab/plant-multi-species-core-promoters",
                               label="Datasets name or path", full_width=True)
    source2_text = mo.ui.dropdown(['local', 'huggingface', 'modelscope'], value="modelscope", label="Dataset source", full_width=True)
    seq_col_text = mo.ui.text(value="sequence", placeholder="sequence","
        "label="Sequence column name", full_width=True)
    label_col_text = mo.ui.text(value="labels", placeholder="labels","
        "label="Label column name", full_width=True)
    maxlen_text = mo.ui.text(value="512", placeholder="512","
        "label="Max token length", full_width=True)
    mo.vstack([title, config_title, config_text.style(width="30ch")],
              align='center', justify='center')
    return (
        config_text,
        model_text,
        source1_text,
        datasets_text,
        source2_text,
        seq_col_text,
        label_col_text,
        maxlen_text,
        
    )


@app.cell
def __(mo, config_text, load_config):
    if config_text.value:
        raw_configs = load_config(config_text.value)
        raw_task_configs = raw_configs['task']
        raw_train_configs = raw_configs['finetune']
        states = {}
        states['label_separator'] = mo.state(',')
        for att in dir(raw_task_configs):
            if not att.startswith("_"):
                if att == "label_names":
                    all_labels = getattr(raw_task_configs, att)
                    print(all_labels)
                    states[att] = mo.state(",".join(all_labels))
                else:
                    states[att] = mo.state(getattr(raw_task_configs, att))
        for att in dir(raw_train_configs):
            if not att.startswith("_"):
                states[att] = mo.state(getattr(raw_train_configs, att))
        if raw_train_configs.bf16:
            states['precision'] = mo.state('bf16')
        elif raw_train_configs.fp16:
            states['precision'] = mo.state('fp16')
        else:
            states['precision'] = mo.state('float32')
        configs = raw_configs
    else:
        raw_task_configs = None
        raw_train_configs = None
        configs = None
    return (raw_task_configs, raw_train_configs, states, configs, )


@app.cell
def __(mo, model_text, source1_text, datasets_text, source2_text, seq_col_text, label_col_text,
       maxlen_text, states):
    config_dict = mo.ui.dictionary({
        "task_type": mo.ui.dropdown(options=['mask', 'generation',
                                             'binary', 'multiclass', 'multilabel', 'regression', 'token'],
                                    value=states["task_type"][0](),
                                    on_change=states["task_type"][1],
                                    label="task_type", full_width=True),
        "num_labels": mo.ui.number(value=states["num_labels"][0](),
                                   on_change=states["num_labels"][1],
                                   start=0, step=1,
                                   label="num_labels", full_width=True),
        "label_separator": mo.ui.dropdown(options=[',', ';', '|', '/', '&'],
                                          value=states["label_separator"][0](),
                                          on_change=states["label_separator"][1],
                                          label="label_separator", full_width=True),
        "label_names": mo.ui.text(value=states["label_names"][0](),
                                  on_change=states["label_names"][1],
                                  label="label_names", full_width=True),
        "num_train_epochs": mo.ui.number(value=states["num_train_epochs"][0](),
                                         on_change=states["num_train_epochs"][1],
                                         start=1, step=1,
                                         label="num_train_epochs", full_width=True),
        "per_device_train_batch_size": mo.ui.number(value=states["per_device_train_batch_size"][0](),
                                                    on_change=states["per_device_train_batch_size"][1],
                                                    start=1, step=1,
                                                    label="per_device_train_batch_size", full_width=True),
        "per_device_eval_batch_size": mo.ui.number(value=states["per_device_train_batch_size"][0](),
                                                   on_change=states["per_device_train_batch_size"][1],
                                                   start=1, step=1,
                                                   label="per_device_eval_batch_size", full_width=True),
        "gradient_accumulation_steps": mo.ui.number(value=states["gradient_accumulation_steps"][0](),
                                                    on_change=states["gradient_accumulation_steps"][1],
                                                    start=1, step=1,
                                                    label="gradient_accumulation_steps", full_width=True),
        "logging_strategy": mo.ui.dropdown(options=['steps', 'epoch'],
                                           value=states["logging_strategy"][0](),
                                           on_change=states["logging_strategy"][1],
                                           label="logging_strategy", full_width=True),
        "logging_steps": mo.ui.number(value=states["logging_steps"][0](),
                                      on_change=states["logging_steps"][1],
                                      start=0, step=5,
                                      label="logging_steps", full_width=True),
        "eval_strategy": mo.ui.dropdown(options=['steps', 'epoch'],
                                        value=states["logging_strategy"][0](),
                                        on_change=states["logging_strategy"][1],
                                        label="eval_strategy", full_width=True),
        "eval_steps": mo.ui.number(value=states["eval_steps"][0](),
                                   on_change=states["eval_steps"][1],
                                   start=0, step=5,
                                   label="eval_steps", full_width=True),
        "save_strategy": mo.ui.dropdown(options=['steps', 'epoch'],
                                        value=states["logging_strategy"][0](),
                                        on_change=states["logging_strategy"][1],
                                        label="save_strategy", full_width=True),
        "save_steps": mo.ui.number(value=states["save_steps"][0](),
                                   on_change=states["save_steps"][1],
                                   start=0, step=5,
                                   label="save_steps", full_width=True),
        "save_total_limit": mo.ui.number(value=states["save_total_limit"][0](),
                                         on_change=states["save_total_limit"][1],
                                         start=1, step=1,
                                         label="save_total_limit", full_width=True),
        "learning_rate": mo.ui.number(value=states["learning_rate"][0](),
                                      on_change=states["learning_rate"][1],
                                      start=1e-10, stop=1, step=1e-6,
                                      label="learning_rate", full_width=True),
        "weight_decay": mo.ui.number(value=states["weight_decay"][0](),
                                     on_change=states["weight_decay"][1],
                                     start=0.0, stop=1, step=0.005,
                                     label="weight_decay", full_width=True),
        "adam_beta1": mo.ui.number(value=states["adam_beta1"][0](),
                                   on_change=states["adam_beta1"][1],
                                   start=0.0, stop=1.0, step=0.001,
                                   label="adam_beta1", full_width=True),
        "adam_beta2": mo.ui.number(value=states["adam_beta2"][0](),
                                   on_change=states["adam_beta2"][1],
                                   start=0.0, stop=1.0, step=0.001,
                                   label="adam_beta2", full_width=True),
        "adam_epsilon": mo.ui.number(value=states["adam_epsilon"][0](),
                                     on_change=states["adam_epsilon"][1],
                                     start=1e-10, stop=1.0, step=1e-8,
                                     label="adam_epsilon", full_width=True),
        "max_grad_norm": mo.ui.number(value=states["max_grad_norm"][0](),
                                      on_change=states["max_grad_norm"][1],
                                      start=0.0, step=0.1,
                                      label="max_grad_norm", full_width=True),
        "warmup_ratio": mo.ui.number(value=states["warmup_ratio"][0](),
                                     on_change=states["warmup_ratio"][1],
                                     start=0.0, step=0.01,
                                     label="warmup_ratio", full_width=True),
        "lr_scheduler_type": mo.ui.dropdown(options=['linear', 'cosine', 'cosine_with_restarts',
                                                     'polynomial', 'constant',
                                                     'constant_with_warmup', 'inverse_sqrt'],
                                            value=states["lr_scheduler_type"][0](),
                                            on_change=states["lr_scheduler_type"][1],
                                            label="lr_scheduler_type", full_width=True),
        "precision": mo.ui.dropdown(options=['float32', 'fp16', 'bf16'],
                                    value=states["precision"][0](),
                                    on_change=states["precision"][1],
                                    label="precision", full_width=True),
        "output_dir": mo.ui.text(value=states["output_dir"][0](),
                                 on_change=states["output_dir"][1],
                                 label="output_dir", full_width=True),
    })
    elems = list(config_dict.values())
    rows = [
        mo.hstack(
            elems[i : i+4], widths="equal", align="stretch", gap=0.5
        ) for i in range(0, len(elems), 4)
    ]
    config_stack = mo.vstack(rows, align="stretch", gap=0.5)
    model_title = mo.md("<h3>Model and dataset</h3>")
    model_stack = mo.hstack([model_text.style(width="75ch"), source1_text], align='center', justify='center')
    datasets_stack = mo.hstack([datasets_text.style(width="75ch"), source2_text], align='center', justify='center')
    options_stack = mo.hstack(
        [seq_col_text,
        label_col_text,
        maxlen_text],
        align='center',
        justify='center'
    )
    mo.vstack([config_stack, model_title, model_stack, datasets_stack, options_stack],
              align='center', justify='center')
    return (config_dict, )


@app.cell
def __(configs, config_dict):
    if configs:
        for arg in config_dict:
            if arg in ['task_type', 'num_labels']:
                setattr(configs['task'], arg, config_dict[arg].value)
            if arg == "label_names":
                sep = config_dict['label_separator'].value
                setattr(
                    configs['task'],
                    arg,
                    config_dict[arg].value.split(sep)
                )
            if arg in dir(configs['finetune']):
                setattr(configs['finetune'], arg, config_dict[arg].value)
            if arg == "precision":
                if config_dict[arg].value == "bf16":
                    configs['finetune'].bf16 = True
                elif config_dict[arg].value == "fp16":
                    configs['finetune'].fp16 = True
                else:
                    pass
    print(configs)
    return (configs, )


@app.cell
def __(mo, configs, model_text, source1_text, load_model_and_tokenizer, 
       datasets_text, source2_text, seq_col_text, label_col_text, maxlen_text, DNADataset):
    def prepare(configs, model_text, source1_text, load_model_and_tokenizer, 
        datasets_text, source2_text, seq_col_text, label_col_text, maxlen_text, DNADataset):
        # Load model and tokenizer
        model_name = model_text.value
        source1 = source1_text.value
        if model_name:
            model, tokenizer = load_model_and_tokenizer(
                model_name,
                task_config=configs['task'],
                source=source1
            )
        else:
            model = None
            tokenizer = None
        # Load datasets
        datasets_name = datasets_text.value
        seq_col = seq_col_text.value
        label_col = label_col_text.value
        max_length = int(maxlen_text.value)
        source2 = source2_text.value
        print(datasets_name, source2)
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
        if datasets is not None:
            datasets.encode_sequences(task=configs['task'].task_type, remove_unused_columns=True)
            if isinstance(datasets.dataset, dict):
                pass
            else:
                datasets.split_data()
        else:
            pass
        
        return (model, tokenizer, datasets,)
    train_button = mo.ui.button(label="Start Training", 
                                on_click=lambda _: prepare(
                                    configs, model_text, source1_text, load_model_and_tokenizer,
                                    datasets_text, source2_text, seq_col_text, label_col_text, maxlen_text, DNADataset
                                ))
    mo.vstack([train_button], align='center', justify='center')
    return (train_button,)


@app.cell
def __(mo):
    text_output = mo.output
    return text_output


@app.cell
def __(mo, configs, train_button, DNATrainer, text_output):
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
            # self.bar.update(increment=increment)

            # collect
            txt = ''
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
        model, _, datasets = train_button.value
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