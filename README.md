# ModelTry

A comprehensive deep learning experimentation repository focused on Large Language Model (LLM) implementations, transformer architectures, and neural network components for hands-on learning and research.

## 🚀 Project Overview

ModelTry is an evolving repository dedicated to implementing and experimenting with various deep learning architectures, with a primary focus on Large Language Models (LLMs) and transformer-based systems. This project serves as a comprehensive learning resource for understanding modern AI architectures through practical implementations, from foundational components to complete model architectures.

## 📁 Project Structure

```
ModelTry/
├── alexnet_test.py          # AlexNet implementation and training script
├── alexnet_try.py           # AlexNet model definition
├── decoder_try.py           # LLaMA decoder layer implementation
├── mha_try.py              # Multi-head attention mechanism
├── rope_try.py             # Rotary Position Embedding (RoPE)
├── rms_norm_try.py         # RMS normalization implementation
├── sentencepiece_try.py    # SentencePiece tokenizer experiments
├── main.py                 # Main entry point
├── pyproject.toml          # Project dependencies
├── dataset/                # Dataset storage directory
├── DeepSeek-V2/            # DeepSeek V2 model implementation
├── DeepSeek-V3/            # DeepSeek V3 model implementation
├── [Future LLM Models]/    # Additional LLM implementations (coming soon)
├── [Training Scripts]/     # Model training and fine-tuning scripts
├── [Evaluation Tools]/     # Model evaluation and benchmarking
└── README.md               # This file
```

> **Note**: This repository is actively expanding with new LLM implementations, training scripts, and evaluation tools. The structure will grow to include more comprehensive language model architectures and utilities.

## 🧠 Implemented Models & Components

### Large Language Models (LLMs)
- **DeepSeek V2 & V3**: Modern MoE (Mixture of Experts) language models
  - Complete model architecture
  - Configuration management
  - Tokenizer integration
- **LLaMA Components**: Decoder layer and attention mechanisms
- **[Future LLMs]**: Additional language model implementations planned

### Transformer Architecture Components
- **Multi-Head Attention**: Scaled dot-product attention with visualization
- **Rotary Position Embedding (RoPE)**: Advanced positional encoding
- **RMS Normalization**: Alternative to Layer Normalization
- **LLaMA Decoder Layer**: Complete transformer decoder implementation
- **Attention Mechanisms**: Various attention implementations and optimizations

### Computer Vision (Legacy/Reference)
- **AlexNet**: Classic CNN architecture for image classification
  - CIFAR-10 dataset training
  - TensorBoard logging
  - Configurable hyperparameters

### Tokenization & Preprocessing
- **SentencePiece**: Tokenizer implementation and experiments
- **Text Processing**: Various text preprocessing utilities

## 🛠️ Dependencies

The project uses the following key dependencies:

- **PyTorch** (>=2.8.0): Deep learning framework
- **TorchVision** (>=0.23.0): Computer vision utilities
- **NumPy** (>=2.3.2): Numerical computing
- **TensorBoard** (>=2.20.0): Experiment tracking

## 🚀 Getting Started

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ModelTry
```

2. Install dependencies using uv:
```bash
uv sync
```

Or using pip:
```bash
pip install -r requirements.txt
```

### Running Experiments

#### AlexNet Training
```bash
python alexnet_test.py --max_epoch 40 --batch_size 32 --lr 0.001
```

Available parameters:
- `--max_epoch`: Number of training epochs (default: 40)
- `--batch_size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 0.001)
- `--image_size`: Input image size (default: 224)
- `--momentum`: SGD momentum (default: 0.90)
- `--device_num`: Device selection ('cpu' or 'cuda')
- `--SummerWriter_log`: TensorBoard log directory

#### Multi-Head Attention Visualization
```bash
python mha_try.py
```

This will generate attention weight visualizations for different attention heads.

#### Testing Other Components
```bash
python rope_try.py          # Test RoPE implementation
python rms_norm_try.py      # Test RMS normalization
python sentencepiece_try.py # Test tokenizer
```

## 📊 Features

### AlexNet Implementation
- Complete CNN architecture with 5 convolutional layers
- CIFAR-10 dataset support with automatic download
- Real-time training monitoring with TensorBoard
- Configurable hyperparameters via command line
- GPU/CPU automatic detection

### Transformer Components
- **Multi-Head Attention**: Full implementation with attention weight visualization
- **RoPE**: Efficient positional encoding for long sequences
- **RMS Norm**: Stable normalization technique
- **LLaMA Decoder**: Production-ready transformer decoder layer

### DeepSeek Models
- Complete model implementations for V2 and V3
- Configuration management system
- Tokenizer integration
- Ready for inference and fine-tuning

## 📈 Monitoring & Visualization

The project includes comprehensive monitoring capabilities:

- **TensorBoard Integration**: Real-time loss and accuracy tracking
- **Attention Visualization**: Heatmaps showing attention patterns
- **Model Parameter Counting**: Automatic parameter estimation
- **Training Progress**: Detailed logging of training metrics

## 🎯 Use Cases

This project is ideal for:

- **LLM Research & Development**: Implementing and experimenting with language models
- **Transformer Architecture Learning**: Understanding attention mechanisms and positional encodings
- **Model Fine-tuning**: Training and adapting pre-trained language models
- **Educational Purposes**: Hands-on learning of modern AI architectures
- **Research Experiments**: Testing new architectures and techniques
- **Custom Model Development**: Building specialized language models
- **Benchmarking & Evaluation**: Comparing different model implementations

## 🔧 Customization & Extension

### Adding New LLM Models
1. Create a new directory for your model (e.g., `llama-v2/`, `gpt-variant/`)
2. Implement the model architecture with proper configuration management
3. Add training, fine-tuning, and inference scripts
4. Include evaluation and benchmarking tools
5. Update this README with your additions

### Adding New Components
1. Create a new Python file for your component
2. Implement the class inheriting from `nn.Module`
3. Add comprehensive tests and examples
4. Document usage and parameters

### Modifying Existing Models
- All models are modular and easily extensible
- Configuration parameters are clearly documented
- Training scripts support various hyperparameters
- Easy to adapt for different use cases and datasets

## 📚 Learning Resources

This project implements concepts from:
- **Attention Mechanisms**: "Attention Is All You Need" (Vaswani et al.)
- **RoPE**: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
- **LLaMA Architecture**: "LLaMA: Open and Efficient Foundation Language Models"
- **DeepSeek Models**: DeepSeek-V2 and V3 technical reports
- **MoE Architectures**: Mixture of Experts implementations
- **Classic CNNs**: AlexNet and computer vision foundations

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- **Add new LLM implementations** (GPT variants, LLaMA variants, etc.)
- **Implement new transformer components** (attention mechanisms, normalization layers)
- **Add training and fine-tuning scripts** for different models
- **Create evaluation and benchmarking tools**
- **Improve existing implementations** with optimizations
- **Add comprehensive tests and documentation**
- **Share interesting experiments and results**

## 📄 License

This project is for educational and research purposes. Please respect the licenses of the individual model implementations, especially the DeepSeek models.

## 🆘 Support

If you encounter any issues or have questions:
1. Check the existing code comments for implementation details
2. Review the PyTorch documentation for framework-specific questions
3. Open an issue for bugs or feature requests

---

## 🚀 Roadmap

This repository is continuously evolving. Planned additions include:

- **More LLM Architectures**: GPT variants, PaLM, T5, and other major language models
- **Training Infrastructure**: Distributed training scripts and utilities
- **Evaluation Suite**: Comprehensive benchmarking tools and metrics
- **Fine-tuning Tools**: LoRA, QLoRA, and other parameter-efficient methods
- **Optimization Techniques**: Quantization, pruning, and inference optimizations
- **Dataset Utilities**: Data loading, preprocessing, and augmentation tools

---

**Happy Learning! 🎓**

This project is designed to be a comprehensive learning resource for LLM practitioners, researchers, and students at all levels.
