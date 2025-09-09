# ModelTry

A comprehensive machine learning experimentation project featuring implementations of various neural network architectures and deep learning components.

## 🚀 Project Overview

ModelTry is a learning and experimentation repository that contains implementations of different neural network architectures, from classic computer vision models to modern transformer components. This project serves as a hands-on exploration of deep learning concepts and their practical implementations.

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
└── README.md               # This file
```

## 🧠 Implemented Models & Components

### Computer Vision
- **AlexNet**: Classic CNN architecture for image classification
  - CIFAR-10 dataset training
  - TensorBoard logging
  - Configurable hyperparameters

### Transformer Components
- **Multi-Head Attention**: Scaled dot-product attention with visualization
- **Rotary Position Embedding (RoPE)**: Advanced positional encoding
- **RMS Normalization**: Alternative to Layer Normalization
- **LLaMA Decoder Layer**: Complete transformer decoder implementation

### Language Models
- **DeepSeek V2 & V3**: Modern MoE (Mixture of Experts) language models
  - Complete model architecture
  - Configuration management
  - Tokenizer integration

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

- **Learning Deep Learning**: Hands-on implementation of key concepts
- **Research Experiments**: Testing new architectures and techniques
- **Educational Purposes**: Understanding how different components work
- **Model Development**: Building custom neural network architectures

## 🔧 Customization

### Adding New Models
1. Create a new Python file for your model
2. Implement the model class inheriting from `nn.Module`
3. Add training/testing scripts as needed
4. Update this README with your additions

### Modifying Existing Models
- All models are modular and easily extensible
- Configuration parameters are clearly documented
- Training scripts support various hyperparameters

## 📚 Learning Resources

This project implements concepts from:
- AlexNet paper: "ImageNet Classification with Deep Convolutional Neural Networks"
- Attention mechanisms from "Attention Is All You Need"
- RoPE from "RoFormer: Enhanced Transformer with Rotary Position Embedding"
- LLaMA architecture and DeepSeek models

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- Add new model implementations
- Improve existing code
- Add more comprehensive tests
- Enhance documentation

## 📄 License

This project is for educational and research purposes. Please respect the licenses of the individual model implementations, especially the DeepSeek models.

## 🆘 Support

If you encounter any issues or have questions:
1. Check the existing code comments for implementation details
2. Review the PyTorch documentation for framework-specific questions
3. Open an issue for bugs or feature requests

---

**Happy Learning! 🎓**

This project is designed to be a comprehensive learning resource for deep learning practitioners at all levels.
