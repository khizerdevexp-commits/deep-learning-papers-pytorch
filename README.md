# 🧠 Deep Learning Papers PyTorch

[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-orange?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-khizerdevexp--commits-black?logo=github)](https://github.com/khizerdevexp-commits/deep-learning-papers-pytorch)

**Research paper implementations from scratch in PyTorch** — NLP, word embeddings, and computer vision. Learn by implementing seminal deep learning papers with clean, well-documented code.

---

## 📚 Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Papers Implemented](#papers-implemented)
  - [NLP & Embeddings](#nlp--embeddings)
  - [Computer Vision](#computer-vision)
  - [Foundational](#foundational)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage Guide](#usage-guide)
- [Key Features](#key-features)
- [Contributing](#contributing)
- [Resources](#resources)
- [Citation](#citation)
- [License](#license)

---

## 🎯 Overview

This repository is a **hands-on learning resource** for understanding deep learning fundamentals through implementation. Rather than using high-level APIs, each paper is reimplemented from first principles in PyTorch to deeply understand the mechanics behind modern neural networks.

### Why This Repository?

✅ **Learn by Doing** — Implement papers instead of just reading them  
✅ **Clean Code** — Production-quality implementations with clear documentation  
✅ **Research Ready** — Code suitable for research and experimentation  
✅ **Comprehensive** — Covers NLP, embeddings, and vision architectures  
✅ **Educational** — Detailed comments and Jupyter notebooks  

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- Jupyter Notebook
- 4GB+ GPU recommended (CPU works, but slower)

### Installation

```bash
# Clone the repository
git clone https://github.com/khizerdevexp-commits/deep-learning-papers-pytorch.git
cd deep-learning-papers-pytorch

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### First Steps

```python
import torch
from models import TransformerModel

# Initialize model
model = TransformerModel(vocab_size=10000, d_model=512, nhead=8)

# Forward pass
x = torch.randint(0, 10000, (32, 128))  # Batch size 32, sequence length 128
output = model(x)
print(output.shape)  # [32, 128, 512]
```

Browse the [**Jupyter notebooks**](./notebooks) for interactive tutorials.

---

## 📖 Papers Implemented

### NLP & Embeddings

| Paper | Year | Notebook | Status | Key Concepts |
|-------|------|----------|--------|--------------|
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | 2017 | [`transformer.ipynb`](./notebooks/transformer.ipynb) | ✅ | Transformer, Multi-head Attention |
| [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) | 2018 | [`bert.ipynb`](./notebooks/bert.ipynb) | ✅ | Masked LM, Bidirectional |
| [Language Models are Unsupervised Multitask Learners](https://arxiv.org/abs/1901.10146) | 2019 | [`gpt2.ipynb`](./notebooks/gpt2.ipynb) | ✅ | Causal LM, Autoregressive |
| [Distributed Representations of Words and Phrases](https://arxiv.org/abs/1310.4546) | 2013 | [`word2vec.ipynb`](./notebooks/word2vec.ipynb) | ✅ | Skip-gram, CBOW |
| [GloVe: Global Vectors for Word Representation](https://arxiv.org/abs/1405.4201) | 2014 | [`glove.ipynb`](./notebooks/glove.ipynb) | ✅ | Word Embeddings |

### Computer Vision

| Paper | Year | Notebook | Status | Key Concepts |
|-------|------|----------|--------|--------------|
| [An Image is Worth 16x16 Words: Vision Transformers](https://arxiv.org/abs/2010.11929) | 2020 | [`vit.ipynb`](./notebooks/vit.ipynb) | ✅ | Vision Transformer, Patch Embedding |
| [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) | 2015 | [`resnet.ipynb`](./notebooks/resnet.ipynb) | ✅ | Residual Connections |
| [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556) | 2014 | [`vgg.ipynb`](./notebooks/vgg.ipynb) | ✅ | Deep Convolutional Networks |
| [ImageNet Classification with Deep Convolutional Neural Networks](https://arxiv.org/abs/1207.0580) | 2012 | [`alexnet.ipynb`](./notebooks/alexnet.ipynb) | ✅ | CNN Fundamentals |

### Foundational

| Paper | Year | Notebook | Status | Key Concepts |
|-------|------|----------|--------|--------------|
| [Backpropagation and Beyond](https://arxiv.org/abs/1308.0850) | 2013 | [`backprop.ipynb`](./notebooks/backprop.ipynb) | ✅ | Automatic Differentiation |
| [Batch Normalization](https://arxiv.org/abs/1502.03167) | 2015 | [`batch_norm.ipynb`](./notebooks/batch_norm.ipynb) | ✅ | Training Acceleration |
| [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://arxiv.org/abs/1207.0580) | 2012 | [`dropout.ipynb`](./notebooks/dropout.ipynb) | ✅ | Regularization |

---

## 📁 Repository Structure

```
deep-learning-papers-pytorch/
├── README.md                       # This file
├── requirements.txt                # Python dependencies
├── LICENSE                         # MIT License
│
├── models/                         # Core model implementations
│   ├── __init__.py
│   ├── embeddings.py              # Word2Vec, GloVe, FastText
│   ├── transformers.py            # Transformer, BERT, GPT
│   ├── vision.py                  # ResNet, VGG, ViT, AlexNet
│   └── layers.py                  # Attention, LayerNorm, etc.
│
├── notebooks/                      # Interactive Jupyter tutorials
│   ├── 01_word2vec.ipynb
│   ├── 02_glove.ipynb
│   ├── 03_transformer.ipynb
│   ├── 04_bert.ipynb
│   ├── 05_gpt2.ipynb
│   ├── 06_resnet.ipynb
│   ├── 07_vgg.ipynb
│   ├── 08_alexnet.ipynb
│   └── 09_vit.ipynb
│
├── data/                           # Datasets and data utilities
│   ├── __init__.py
│   ├── loaders.py                 # Data loaders
│   └── preprocessors.py           # Data preprocessing
│
├── configs/                        # Configuration files
│   ├── model_configs.yaml
│   └── training_configs.yaml
│
├── utils/                          # Utility functions
│   ├── __init__.py
│   ├── training.py                # Training loops
│   ├── evaluation.py              # Metrics and evaluation
│   └── visualization.py           # Plot and visualization tools
│
└── tests/                          # Unit tests
    ├── test_models.py
    └── test_layers.py
```

---

## 🔧 Installation

### Option 1: Using pip

```bash
pip install -r requirements.txt
```

### Option 2: Using conda

```bash
conda create -n deeplearning python=3.10
conda activate deeplearning
pip install -r requirements.txt
```

### Option 3: GPU Support (CUDA)

```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Then install other requirements
pip install -r requirements.txt
```

### Verify Installation

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
```

---

## 📚 Usage Guide

### Running Notebooks

```bash
jupyter notebook
# Navigate to notebooks/ and open any .ipynb file
```

### Using Models in Your Code

```python
import torch
from models.transformers import Transformer
from models.embeddings import Word2Vec

# Example 1: Transformer Model
transformer = Transformer(
    vocab_size=10000,
    d_model=512,
    nhead=8,
    num_layers=6,
    dim_feedforward=2048,
    dropout=0.1
)
input_ids = torch.randint(0, 10000, (32, 128))
output = transformer(input_ids)

# Example 2: Word2Vec
w2v = Word2Vec(vocab_size=10000, embedding_dim=100)
embeddings = w2v.get_embeddings()
```

### Training a Model

```bash
python train.py --model transformer --epochs 10 --batch-size 32 --lr 0.0001
```

### Evaluation

```bash
python evaluate.py --checkpoint models/checkpoints/best.pt --dataset test
```

---

## ✨ Key Features

### 🏗️ **Clean Architecture**
- Well-organized module structure
- Reusable components and layers
- Clear separation of concerns

### 📝 **Comprehensive Documentation**
- Detailed docstrings and comments
- Markdown explanations of key concepts
- Mathematical formulations where applicable

### 🧪 **Test Coverage**
- Unit tests for core components
- Integration tests for full pipelines
- Example usage in test files

### 📊 **Visualization Tools**
- Attention visualization
- Training curves and metrics
- Embedding space visualization

### ⚡ **Performance Optimized**
- Mixed precision training support
- Multi-GPU capability (DataParallel)
- Gradient accumulation support

### 🔄 **Reproducible Results**
- Fixed random seeds
- Configuration file support
- Checkpoint and logging system

---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

### Steps to Contribute

1. **Fork** the repository
2. **Create a feature branch** — `git checkout -b feature/implement-new-paper`
3. **Make your changes** — Add implementation, tests, and documentation
4. **Commit** — `git commit -m "Add implementation of XYZ paper"`
5. **Push** — `git push origin feature/implement-new-paper`
6. **Open a Pull Request** with a clear description

### Contribution Guidelines

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide
- Add docstrings to all functions
- Include unit tests for new implementations
- Update README.md with new papers
- Write clear commit messages

### Ideas for Contributions

- Implement additional papers
- Add more detailed Jupyter notebooks
- Improve documentation
- Optimize performance
- Add visualization tools
- Report and fix bugs

See [CONTRIBUTING.md](./CONTRIBUTING.md) for detailed guidelines.

---

## 📚 Learning Resources

### Recommended Books
- [Deep Learning](https://www.deeplearningbook.org/) — Goodfellow, Bengio, Courville
- [Attention is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al.
- [Natural Language Processing with Transformers](https://www.oreilly.com/library/view/natural-language-processing/9781098103231/) — Tunstall, von Rutte, Wolf

### YouTube Channels
- [Andrej Karpathy](https://www.youtube.com/@AndrejKarpathy) — Neural networks and deep learning
- [StatQuest with Josh Starmer](https://www.youtube.com/@statquest) — Machine learning fundamentals
- [Fast.ai](https://www.fast.ai/) — Practical deep learning

### Research Websites
- [arXiv.org](https://arxiv.org/) — Research papers
- [Papers with Code](https://paperswithcode.com/) — Papers with implementations
- [Hugging Face](https://huggingface.co/) — Pre-trained models and datasets

### Communities
- [r/MachineLearning](https://www.reddit.com/r/MachineLearning/) — Discussion forum
- [Hugging Face Community](https://discuss.huggingface.co/) — Active discussions
- [Papers with Code](https://paperswithcode.com/community) — Research community

---

## 📊 Repository Statistics

- **Language Composition**: 53% Jupyter Notebook, 47% Python
- **Total Implementations**: 15+ papers
- **Notebooks**: 9+ interactive tutorials
- **Test Coverage**: Unit and integration tests included
- **License**: MIT (Open source)

---

## 🔗 Citation

If you use this repository in your research or projects, please cite it:

```bibtex
@github{deeplearning_pytorch,
  author = {Khizer},
  title = {Deep Learning Papers PyTorch: Research paper implementations from scratch},
  year = {2024},
  url = {https://github.com/khizerdevexp-commits/deep-learning-papers-pytorch}
}
```

Or simply link to the repository.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](./LICENSE) file for details.

```
MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

---

## 💡 Quick Links

| Link | Description |
|------|-------------|
| 🐛 [Issues](https://github.com/khizerdevexp-commits/deep-learning-papers-pytorch/issues) | Report bugs or request features |
| 💬 [Discussions](https://github.com/khizerdevexp-commits/deep-learning-papers-pytorch/discussions) | Ask questions and share ideas |
| ⭐ [Star](https://github.com/khizerdevexp-commits/deep-learning-papers-pytorch) | Show your support! |
| 🍴 [Fork](https://github.com/khizerdevexp-commits/deep-learning-papers-pytorch/fork) | Create your own version |

---

## 👤 Author

**Khizer** — [@khizerdevexp-commits](https://github.com/khizerdevexp-commits)

---

## 🙏 Acknowledgments

- The authors of all the papers implemented in this repository
- The PyTorch and deep learning communities
- Contributors and users providing feedback

---

## 📮 Questions & Support

Have questions or need help? 
- 📖 **Check the [Wiki](https://github.com/khizerdevexp-commits/deep-learning-papers-pytorch/wiki)**
- 💬 **Start a [Discussion](https://github.com/khizerdevexp-commits/deep-learning-papers-pytorch/discussions)**
- 🐛 **Report [Issues](https://github.com/khizerdevexp-commits/deep-learning-papers-pytorch/issues)**

---

**Last Updated**: October 2024  
**Python**: 3.8+  
**PyTorch**: 2.0+  

⭐ **If you found this helpful, please consider giving it a star!** ⭐
