# DNALLM Configuration Generator - Gradio UI

A web-based interface for generating configuration files for DNALLM fine-tuning, inference, and benchmarking tasks.

## 🌟 Features

### **Smart Auto-Configuration**
- **Automatic Model Detection**: Automatically reads model information from `model_info.yaml`
- **zhangtaolab Model Support**: Special support for zhangtaolab models with pre-configured task parameters
- **Auto-Fill Functionality**: Click a button to automatically populate task configuration based on model path

### **Three Configuration Types**
1. **🔧 Fine-tuning Configuration**: Generate configs for model training
2. **🔮 Inference Configuration**: Generate configs for model prediction
3. **📊 Benchmark Configuration**: Generate configs for model evaluation

### **User-Friendly Interface**
- **Dropdown Menus**: All options presented as easy-to-use dropdown selections
- **Real-time Updates**: Dynamic UI updates based on user selections
- **Model Information Display**: Shows detailed model descriptions and task information
- **Local Access Only**: Secure localhost-only access for privacy

## 🚀 Quick Start

### **Installation**
```bash
cd dnallm/gradio
pip install -r requirements.txt
```

### **Launch the Application**
```bash
python run_app.py
```

### **Access the UI**
Open your browser and navigate to: `http://127.0.0.1:7860`

## 🔍 Auto-Fill Feature

### **How It Works**
1. **Enter Model Path**: Input a model identifier (e.g., `zhangtaolab/plant-dnabert-BPE-promoter`)
2. **Click Auto-Fill**: Use the "🔍 Auto-Fill from Model" button
3. **Automatic Configuration**: The system automatically populates:
   - Task type (binary_classification, multiclass_classification, regression)
   - Number of labels
   - Label names
   - Classification threshold
   - Model description

### **Supported Models**
The system automatically recognizes models from `model_info.yaml`, including:
- **zhangtaolab Models**: Plant DNA models for various tasks
- **Pretrained Models**: Foundation models for DNA sequence understanding
- **Finetuned Models**: Task-specific models with pre-configured parameters

### **Example Auto-Fill**
```
Model Path: zhangtaolab/plant-dnabert-BPE-promoter
↓ Click Auto-Fill ↓
Task Type: binary_classification
Number of Labels: 2
Label Names: Not promoter, Core promoter
Threshold: 0.5
Description: Predict whether a DNA sequence is a core promoter in plants.
```

## 📋 Configuration Parameters

### **Fine-tuning Configuration**
- **Task Settings**: Task type, number of labels, threshold, label names
- **Training Parameters**: Epochs, batch sizes, learning rate, weight decay
- **Advanced Options**: Gradient accumulation, mixed precision, logging strategies

### **Inference Configuration**
- **Task Settings**: Task type, number of labels
- **Inference Parameters**: Batch size, sequence length, device selection
- **Performance Options**: Number of workers, mixed precision

### **Benchmark Configuration**
- **Model Settings**: Model selection, task type, data type
- **Dataset Configuration**: File format, column mappings, task type
- **Evaluation Options**: Batch size, device, metrics, output format

## 🧪 Testing

### **Test Auto-Fill Functionality**
```bash
python test_auto_fill.py
```

This will verify:
- YAML file loading
- Auto-fill method creation
- Model information retrieval

### **Test Application Launch**
```bash
python quick_test.py
```

## 📁 File Structure

```
dnallm/gradio/
├── config_generator_app.py    # Main Gradio application
├── run_app.py                 # Simplified launch script
├── requirements.txt           # Python dependencies
├── README.md                 # This documentation
└── test_auto_fill.py         # Auto-fill functionality tests
```

## 🔧 Customization

### **Adding New Models**
1. Update `model_info.yaml` with new model information
2. Include task details: `task_type`, `num_labels`, `label_names`, `threshold`, `describe`
3. Restart the application to load new models

### **Modifying Task Types**
1. Update the `TASK_TYPES` mapping in `config_generator_app.py`
2. Add corresponding UI components as needed
3. Update configuration generation methods

## 🐛 Troubleshooting

### **Common Issues**

1. **Import Errors**
   - Ensure you're running from the correct directory
   - Check that all dependencies are installed
   - Verify Python path configuration

2. **YAML Loading Issues**
   - Check if `model_info.yaml` exists in `dnallm/models/`
   - Verify YAML file format and syntax
   - Check file permissions

3. **Auto-Fill Not Working**
   - Verify model path format (e.g., `zhangtaolab/model-name`)
   - Check if model exists in `model_info.yaml`
   - Ensure model has task information

### **Debug Mode**
```bash
python run_app.py --debug
```

## 📚 Related Links

- [DNALLM Documentation](https://dnallm.readthedocs.io/)
- [Gradio Documentation](https://www.gradio.app/docs)
- [Model Information](https://github.com/zhangtaolab/DNALLM)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the same license as DNALLM.
