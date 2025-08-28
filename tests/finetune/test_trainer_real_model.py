from dnallm import load_config, load_model_and_tokenizer, DNADataset, DNATrainer

# Load the config file
configs = load_config("./test_finetune_config.yaml")

# Load the model and tokenizer
model_name = "zhangtaolab/plant-dnabert-BPE"
# from Hugging Face
# model, tokenizer = load_model_and_tokenizer(model_name, task_config=configs['task'], source="huggingface")
# from ModelScope
model, tokenizer = load_model_and_tokenizer(model_name, task_config=configs['task'], source="modelscope")

# Load the datasets
data_name = "zhangtaolab/plant-multi-species-core-promoters"
# from Hugging Face
# datasets = DNADataset.from_huggingface(data_name, seq_col="sequence", label_col="label", tokenizer=tokenizer, max_length=512)
# from ModelScope
datasets = DNADataset.from_modelscope(data_name, seq_col="sequence", label_col="label", tokenizer=tokenizer, max_length=512)

# Encode the datasets
datasets.encode_sequences()

# sample datasets
sampled_datasets = datasets.sampling(0.05, overwrite=True)

# Initialize the trainer
trainer = DNATrainer(
    model=model,
    config=configs,
    datasets=sampled_datasets
)

# Start training
metrics = trainer.train()
print(metrics)

# Do prediction on the test set
trainer.predict()