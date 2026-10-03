# Deep Learning Papers in PyTorch

<div align="center">

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Open Issues](https://img.shields.io/github/issues/khizerdevexp-commits/deep-learning-papers-pytorch)](https://github.com/khizerdevexp-commits/deep-learning-papers-pytorch/issues)
[![Last Commit](https://img.shields.io/github/last-commit/khizerdevexp-commits/deep-learning-papers-pytorch)](https://github.com/khizerdevexp-commits/deep-learning-papers-pytorch/commits/main)

A structured repository for implementing research papers from scratch in PyTorch with a focus on **NLP**, **Word Embeddings**, and **Computer Vision**.

[Quick Start](#-quick-start) • [Workflow](#-implementation-workflow) • [Papers](#-recommended-papers) • [Documentation](#-documentation) • [Contributing](#-contributing)

</div>

---

## Overview

This repository bridges the gap between research papers and practical implementations. Each paper is translated into clear, modular, and reusable PyTorch code while preserving the original methodological intent.

**Key Features:**
- 📝 Well-documented implementations with equation references
- 🔄 Reusable components (attention, embeddings, losses)
- 📊 Jupyter notebooks for experimentation and visualization
- ⚙️ Configuration-driven experiments
- 📚 Comprehensive summaries and learning notes
- ✅ Checklist-based workflow for consistent development

---

## 📋 Table of Contents

- [Repository Structure](#-repository-structure)
- [Quick Start](#-quick-start)
- [Implementation Workflow](#-implementation-workflow)
- [Code Examples](#-code-examples)
- [Recommended Papers](#-recommended-papers)
- [Documentation](#-documentation)
- [Contributing](#-contributing)
- [License](#-license)

---

## 📁 Repository Structure

```
deep-learning-papers-pytorch/
├── papers/                          # Original PDF research papers
│   ├── glove.pdf
│   ├── attention_is_all_you_need.pdf
│   └── ...
│
├── implementations/                 # Reusable, production-ready code
│   ├── models/
│   │   ├── __init__.py
│   │   ├── glove.py                # GloVe model implementation
│   │   ├── word2vec.py             # Word2Vec (CBOW & Skip-gram)
│   │   ├── transformer.py          # Transformer architecture
│   │   └── ...
│   │
│   ├── layers/
│   │   ├── __init__.py
│   │   ├── attention.py            # Attention mechanisms
│   │   ├── embeddings.py           # Embedding layers
│   │   └── ...
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── data.py                 # Data loading & preprocessing
│   │   ├── training.py             # Training loops, evaluation
│   │   ├── visualization.py        # Plotting & analysis tools
│   │   └── constants.py            # Shared constants
│   │
│   └── losses/
│       ├── __init__.py
│       └── custom_losses.py        # Paper-specific loss functions
│
├── notebooks/                       # Jupyter notebooks - Learning & Experimentation
│   ├── glove_learning.ipynb
│   ├── transformer_learning.ipynb
│   └── ...
│
├── configs/                         # Configuration files (JSON/YAML)
│   ├── glove_config.json
│   ├── transformer_config.json
│   └── ...
│
├── data/                           # Small datasets for experimentation
│   ├── sample_text.txt
│   └── vocab.pkl
│
├── results/                        # Model checkpoints & outputs
│   ├── glove/
│   │   ├── model.pt
│   │   └── embeddings.npy
│   └── ...
│
├── docs/                           # Learning notes & paper summaries
│   ├── glove_summary.md
│   ├── transformer_summary.md
│   └── ...
│
├── requirements.txt
├── setup.py
├── LICENSE
└── .gitignore
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- Git
- Virtual environment tool (venv, conda, etc.)

### Installation

**1. Clone the repository**

```bash
git clone https://github.com/khizerdevexp-commits/deep-learning-papers-pytorch.git
cd deep-learning-papers-pytorch
```

**2. Create and activate a virtual environment**

```bash
# Using venv
python -m venv .venv
source .venv/bin/activate           # On Linux/macOS
.venv\Scripts\activate              # On Windows

# Or using conda
conda create -n paper-impl python=3.10
conda activate paper-impl
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. (Optional) Install in development mode**

```bash
pip install -e .
```

### Verify Installation

```python
import torch
from implementations.models import glove

model = glove.GloVe(vocab_size=10000, embedding_dim=300)
print(f"Model created successfully: {model}")
```

---

## 🔄 Implementation Workflow

Each paper follows a consistent, structured approach:

### Step 1: Add the Paper

Save the original PDF in the `papers/` directory.

```bash
papers/your_paper.pdf
```

### Step 2: Create Implementation

Create a new module in `implementations/models/` with docstring-guided structure.

```bash
implementations/models/your_paper.py
```

### Step 3: Create Learning Notebook

Add experiments and visualizations in a Jupyter notebook.

```bash
notebooks/your_paper_learning.ipynb
```

### Step 4: Document the Paper

Write a technical summary in `docs/`.

```bash
docs/your_paper_summary.md
```

### Suggested Workflow Timeline

| Phase | Time | Activity |
|-------|------|----------|
| **Review** | 20 min | Read abstract, key equations, diagrams from PDF |
| **Implement** | 50 min | Code model in `implementations/models/` with Copilot |
| **Experiment** | 40 min | Test in notebook, add visualizations |
| **Document** | 10 min | Write summary, commit with meaningful message |

---

## 💻 Code Examples

### Basic Model Implementation

```python
"""
GloVe: Global Vectors for Word Representation
Reference: https://nlp.stanford.edu/pubs/glove.pdf
Citation: Pennington et al., 2014

Key Equations:
    Eq. 1: w_i · w_j + b_i + b_j = log(X_ij)
    Eq. 2: L = Σ f(X_ij) * (w_i · w_j + b_i + b_j - log(X_ij))^2
"""

import torch
import torch.nn as nn
from typing import Optional

class GloVe(nn.Module):
    """
    GloVe embedding model.
    
    Args:
        vocab_size (int): Size of the vocabulary
        embedding_dim (int): Dimension of word embeddings
        context_dim (int): Dimension of context embeddings
    """
    
    def __init__(
        self, 
        vocab_size: int, 
        embedding_dim: int, 
        context_dim: int = 300
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Section 3.1: Word and context embeddings
        self.word_embed = nn.Embedding(vocab_size, embedding_dim)
        self.context_embed = nn.Embedding(vocab_size, context_dim)
        
        # Bias terms
        self.word_bias = nn.Parameter(torch.zeros(vocab_size))
        self.context_bias = nn.Parameter(torch.zeros(vocab_size))
    
    def forward(
        self, 
        word_ids: torch.Tensor,          # [batch_size]
        context_ids: torch.Tensor        # [batch_size]
    ) -> torch.Tensor:                   # [batch_size]
        """
        Forward pass implementing Eq. 1 from paper.
        
        Args:
            word_ids: Word indices
            context_ids: Context word indices
            
        Returns:
            Dot product scores
        """
        word_vecs = self.word_embed(word_ids)
        context_vecs = self.context_embed(context_ids)
        
        dot_product = (word_vecs * context_vecs).sum(dim=-1)
        return dot_product + self.word_bias[word_ids] + self.context_bias[context_ids]
```

### Using in a Notebook

```python
# Cell 1: Imports
import torch
import numpy as np
from implementations.models import glove
from implementations.utils import training, visualization

# Cell 2: Model Setup
model = glove.GloVe(vocab_size=10000, embedding_dim=300)
optimizer = torch.optim.Adagrad(model.parameters(), lr=0.05)

# Cell 3: Training Loop
for epoch in range(10):
    loss = training.train_epoch(model, optimizer, train_loader)
    print(f"Epoch {epoch}: Loss = {loss:.4f}")

# Cell 4: Visualization
embeddings = model.word_embed.weight.detach()
visualization.plot_tsne(embeddings, vocab, top_n=500)
```

---

## 📚 Recommended Papers to Start

| # | Paper | Topic | Difficulty | Status |
|---|-------|-------|------------|--------|
| 1 | [GloVe](https://nlp.stanford.edu/pubs/glove.pdf) | Word Embeddings | 🟢 Beginner | ✓ |
| 2 | [Word2Vec](https://arxiv.org/pdf/1310.4546.pdf) | Word Embeddings | 🟢 Beginner | ✓ |
| 3 | [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf) | Transformers | 🟡 Intermediate | ✓ |
| 4 | [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf) | Vision | 🟡 Intermediate | ✓ |
| 5 | [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/pdf/1810.04805.pdf) | NLP | 🔴 Advanced | ⏳ |

---

## 📖 Documentation

### Paper Summaries

Detailed summaries and notes for each implemented paper are available in [`docs/`](docs/):

- [GloVe Summary](docs/glove_summary.md)
- [Transformer Summary](docs/transformer_summary.md)
- [Implementation Guide](docs/IMPLEMENTATION_GUIDE.md)

### Learning Resources

- **Notebooks**: Interactive experiments in [`notebooks/`](notebooks/)
- **Code Templates**: Reusable patterns in [`implementations/`](implementations/)
- **Configs**: Experiment configurations in [`configs/`](configs/)

### Best Practices

#### Pattern 1: Equation-to-Code
Use docstrings to bridge equations to code implementation.

```python
def compute_loss(predictions, targets):
    """
    Equation (5) from paper:
    L = -sum(y_i * log(p_i)) + lambda * ||w||^2
    """
    # Implementation follows naturally from equation
```

#### Pattern 2: Section-Based Comments
Reference paper sections for context.

```python
# Section 3.2: Co-occurrence Matrix Construction
# Based on equation (1): w_ij represents co-occurrence count

def build_cooccurrence_matrix(corpus, window_size):
    # Implementation
```

#### Pattern 3: Type Hints + Docstrings
Clear signatures for reproducibility.

```python
def forward(
    self, 
    input_ids: torch.Tensor,              # [batch_size, seq_len]
    attention_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:                        # [batch_size, seq_len, hidden_dim]
    """Forward pass implementing Section 3.1 equations."""
```

---

## ✅ Implementation Checklist

Use this checklist for each new paper:

- [ ] PDF saved in `papers/`
- [ ] `implementations/models/paper_name.py` created with stubs
- [ ] Core model implemented with docstring guidance
- [ ] `notebooks/paper_name_learning.ipynb` created
- [ ] Basic training loop in notebook
- [ ] Evaluation metrics implemented
- [ ] `docs/paper_name_summary.md` written
- [ ] All code committed with meaningful messages
- [ ] Results saved in `results/`
- [ ] Tests added for critical components

---

## 🤝 Contributing

Contributions are welcome! To add a new paper implementation:

### How to Contribute

1. **Fork the repository**

```bash
git clone https://github.com/YOUR-USERNAME/deep-learning-papers-pytorch.git
cd deep-learning-papers-pytorch
git checkout -b feature/new-paper-impl
```

2. **Follow the implementation workflow** (see above)

3. **Ensure code quality**

```bash
# Add type hints
# Write comprehensive docstrings
# Include unit tests for critical components
```

4. **Commit and push**

```bash
git add .
git commit -m "feat: Add GloVe implementation with notebook and docs"
git push origin feature/new-paper-impl
```

5. **Create a Pull Request**

Open a PR with:
- Clear description of the paper
- Link to the original paper
- Summary of implementation decisions
- Any experimental results

### Code Guidelines

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/)
- Use type hints throughout
- Include docstrings with equation references
- Add unit tests for models and utilities
- Keep notebooks clean and well-commented

---

## 📞 Support & Community

### Get Help

- **Issues**: [Open an issue](https://github.com/khizerdevexp-commits/deep-learning-papers-pytorch/issues) for bugs or questions
- **Discussions**: Use [GitHub Discussions](https://github.com/khizerdevexp-commits/deep-learning-papers-pytorch/discussions) for general questions
- **Documentation**: Check [`docs/`](docs/) for guides and summaries

### Stay Updated

- 👀 **Watch** this repo to be notified of updates
- ⭐ **Star** if you find it useful!

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 📊 Project Status

| Component | Status | Last Updated |
|-----------|--------|--------------|
| GloVe | ✅ Complete | Oct 2024 |
| Word2Vec | ✅ Complete | Oct 2024 |
| Transformer | 🟡 In Progress | Oct 2024 |
| ResNet | ⏳ Planned | - |
| BERT | ⏳ Planned | - |

---

## 🎯 Roadmap

- [x] Core repository structure
- [x] GloVe implementation
- [ ] Word2Vec (CBOW + Skip-gram)
- [ ] Transformer from scratch
- [ ] Vision models (ResNet)
- [ ] Benchmark suite
- [ ] Pre-trained model zoo
- [ ] Community contributions guide

See [Projects](https://github.com/khizerdevexp-commits/deep-learning-papers-pytorch/projects) for detailed progress.

---

## 🔗 Useful Links

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Papers with Code](https://paperswithcode.com/)
- [arXiv](https://arxiv.org/)
- [Stanford NLP](https://nlp.stanford.edu/)

---

<div align="center">

**Made with ❤️ for the deep learning community**

[⬆ Back to Top](#deep-learning-papers-in-pytorch)

</div>