# DNALLM
DNA Large Language Model Package

Functionality:
- dnallm
  - models
    * load_model (读取模型)
    * load_tokenizer (读取tokenizer)
    * load_model_config (读取模型参数)
    * custom_tokenizer (自定义tokenizer)
  - datasets
    * load_datasets (读取数据集)
    * generate_datasets (生成数据集)
    * data_clean (数据清洗/过滤)
    * data_collator (数据格式整理)
  - tasks
    * load_task_config (读取任务参数)
      * binary
      * multi-class
      * multi-label
      * regression
      * ner (optional)
    * compute_metrics (根据任务选择对应评估函数)
  - finetune
    * load_trainer_config (读取训练参数)
    * trainer (训练器)
    * accelerate (分布式训练)
    * peft (高效微调)
  - inference
    * predictor (推理)
    * evaluator (带标签数据评估)
    * mutagenesis (变异效应评估)
  - cli
    * train (命令行开始训练)
    * predict (命令行开始推理)
  - mcp (模型上下文协议)
    * server
    * client
- test (测试所有功能)
- example
  - marimo (交互式模型训练和推理)
    - interactive_demo
    - finetune_exmaple
    - inference_exmaple
  - notebook (可调用式模型训练和推理)
    - function_introduction
    - finetune
    - inference
  - run_finetune_cli (命令行微调示例)
  - run_inference_cli (命令行推理示例)

https://docs.pydantic.dev/latest/
