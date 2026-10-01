# Advanced VS Code + Python Productivity: Deep Learning Research Papers Edition

A focused guide for mid-level Python developers implementing papers. Skip the basics—this covers advanced patterns, configurations, and workflows optimized for your specific pain point: eliminating repetitive copy-paste between ChatGPT, notebooks, and code.

---

## Table of Contents

1. [Advanced Workspace Configuration](#advanced-workspace-configuration)
2. [Copilot Mastery Patterns](#copilot-mastery-patterns)
3. [Jupyter Notebook Workflow Optimization](#jupyter-notebook-workflow-optimization)
4. [Debugging Deep Learning Code](#debugging-deep-learning-code)
5. [Multi-File Refactoring & Navigation](#multi-file-refactoring--navigation)
6. [Git + Paper Implementation Lifecycle](#git--paper-implementation-lifecycle)
7. [VS Code Tasks for Automation](#vs-code-tasks-for-automation)
8. [Extension Configuration for Data Science](#extension-configuration-for-data-science)
9. [Performance Optimization for Large Projects](#performance-optimization-for-large-projects)

---

## Advanced Workspace Configuration

### 1. Professional `.vscode/settings.json`

This configuration is tuned for PyTorch deep learning projects:

```json
{
  // ============ PYTHON INTERPRETER ============
  "python.defaultInterpreterPath": "${workspaceFolder}/venv/bin/python",
  "python.formatting.provider": "black",
  "[python]": {
    "editor.defaultFormatter": "ms-python.python",
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
      "source.organizeImports": true
    },
    "editor.rulers": [88, 120]
  },

  // ============ LINTING ============
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": true,
  "python.linting.pylintArgs": [
    "--max-line-length=120",
    "--disable=C0111",
    "--disable=R0913",
    "--disable=W0212"
  ],
  "python.analysis.typeCheckingMode": "off",

  // ============ EDITOR BEHAVIOR ============
  "editor.wordWrap": "on",
  "editor.wordWrapColumn": 120,
  "editor.formatOnSave": true,
  "editor.formatOnPaste": true,
  "editor.bracketPairColorization.enabled": true,
  "editor.guides.bracketPairs": "active",
  "editor.trimAutoWhitespace": true,
  "files.trimTrailingWhitespace": true,
  "files.insertFinalNewline": true,
  "files.autoSave": "afterDelay",
  "files.autoSaveDelay": 1000,

  // ============ SMART AUTOCOMPLETE ============
  "editor.quickSuggestions": {
    "other": "on",
    "comments": false,
    "strings": false
  },
  "editor.suggestSelection": "recentlyUsedByPrefix",
  "editor.acceptSuggestionOnCommitCharacter": false,
  "editor.acceptSuggestionOnEnter": "smart",

  // ============ FILE ORGANIZATION ============
  "files.exclude": {
    "**/__pycache__": true,
    "**/*.pyc": true,
    "**/.pytest_cache": true,
    "**/.ipynb_checkpoints": true,
    "**/*.egg-info": true,
    "**/node_modules": true,
    "**/.mypy_cache": true,
    "**/.venv": true,
    "**/venv": true
  },

  "search.exclude": {
    "**/venv": true,
    "**/__pycache__": true,
    "**/results": true,
    "**/data": true,
    "**/.ipynb_checkpoints": true,
    "**/papers": true
  },

  // ============ JUPYTER/NOTEBOOKS ============
  "jupyter.experiments.enabled": false,
  "notebook.lineNumbers": "on",
  "notebook.cellToolbarLocation": "right",
  "notebook.formatOnSave": true,
  "notebook.formatOnCellExecution": true,
  "[markdown]": {
    "editor.defaultFormatter": "ms-python.python"
  },

  // ============ COPILOT ============
  "github.copilot.enable": {
    "*": true,
    "plaintext": false,
    "markdown": true,
    "scminput": false
  },
  "github.copilot.advanced": {
    "listTopN": 1,
    "inlineSuggestCount": 3
  },

  // ============ GIT & SOURCE CONTROL ============
  "git.ignoreLimitWarning": true,
  "git.autofetch": true,
  "git.autoStash": true,
  "gitlens.advanced.telemetry.enabled": false,

  // ============ TERMINAL ============
  "terminal.integrated.defaultProfile.linux": "bash",
  "terminal.integrated.defaultProfile.osx": "bash",
  "terminal.integrated.defaultProfile.windows": "PowerShell",
  "terminal.integrated.fontSize": 12,
  "terminal.integrated.env.linux": {
    "PYTHONPATH": "${workspaceFolder}"
  },

  // ============ DEBUGGING ============
  "debug.console.fontSize": 12,
  "debug.showBreakpointsInOverviewRuler": true,
  "debug.onTaskErrors": "abort",

  // ============ EXTENSIONS ============
  "error-lens.severity": "warning",
  "[jsonc]": {
    "editor.defaultFormatter": "ms-python.python"
  }
}
```

**Key insights:**
- Line rulers at 88 (Black default) and 120 (hard limit)
- Auto-format + organize imports on save
- PYTHONPATH set in terminal (important for imports across projects)
- Copilot tuned to single suggestion (less noise)
- Notebook cells format on execution (no manual formatting)

---

### 2. Launch Configuration for Debugging

Create `.vscode/launch.json`:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python: Current File",
      "type": "python",
      "request": "launch",
      "program": "${file}",
      "console": "integratedTerminal",
      "justMyCode": false,
      "env": {"PYTHONPATH": "${workspaceFolder}"}
    },
    {
      "name": "Python: Module (Training)",
      "type": "python",
      "request": "launch",
      "module": "implementations.models.glove",
      "args": ["--config", "configs/glove_config.json"],
      "console": "integratedTerminal",
      "justMyCode": false
    },
    {
      "name": "Python: Debug Tests",
      "type": "python",
      "request": "launch",
      "module": "pytest",
      "args": ["-v", "--tb=short"],
      "console": "integratedTerminal",
      "justMyCode": false
    }
  ]
}
```

**Use cases:**
- Debug current file: F5 (for one-off scripts)
- Debug module: Run a model directly with config
- Debug tests: Run pytest with breakpoints

---

## Copilot Mastery Patterns

You already know Copilot basics. Here are advanced patterns for paper implementation:

### Pattern 1: Equation-Driven Forward Pass

**The old way (bad):**
```python
# Ask ChatGPT: "Write PyTorch code for GloVe forward pass"
# Copy-paste messy code, fix it manually, waste 20 min
```

**The new way (good):**

```python
def forward(self, i_idx: torch.Tensor, j_idx: torch.Tensor) -> torch.Tensor:
    """
    Compute GloVe prediction from Eq. (2) of Pennington et al. 2014.
    
    Eq. (2): log(X_ij) ≈ w_i^T * w_j + b_i + b_j
    
    Where:
        w_i: word vector for word i from W [embedding_dim]
        w_j: context vector for word j from W' [embedding_dim]
        b_i, b_j: scalar bias terms
        X_ij: co-occurrence count (handled outside this function)
    
    Args:
        i_idx: target word indices [batch_size]
        j_idx: context word indices [batch_size]
    
    Returns:
        predictions: predicted log co-occurrence values [batch_size]
    """
    # Press Ctrl+Alt+\ for Copilot suggestion
```

**Copilot generates:**
```python
    w_i = self.W[i_idx]  # [batch_size, embedding_dim]
    w_j = self.W_context[j_idx]  # [batch_size, embedding_dim]
    
    # Dot product: sum over embedding dimension
    dot_product = torch.sum(w_i * w_j, dim=1)  # [batch_size]
    
    # Add biases from Eq. (2)
    predictions = dot_product + self.b[i_idx] + self.b_context[j_idx]
    
    return predictions
```

**Why this works:**
- Equation in docstring → Copilot understands context
- Type hints → Copilot knows tensor shapes
- Variable names match paper → Copilot generates correct operations
- Result: Perfect code on first try, no manual fixes needed

---

### Pattern 2: Loss Function from Mathematical Notation

```python
class GloVeLoss(nn.Module):
    """
    Weighted MSE loss implementing Eq. (3) from Pennington et al. 2014.
    
    Eq. (3): J = Σ_ij f(X_ij) * (w_i^T * w_j + b_i + b_j - log(X_ij))^2
    
    Where:
        f(X_ij): weighting function from Eq. (4)
        Σ_ij: sum over all word pairs in co-occurrence matrix
    
    The weighting function (Eq. 4) prevents high-frequency pairs from 
    dominating the loss. Implementation:
        f(x) = (x / x_max)^α  if x < x_max
               1              if x ≥ x_max
    """
    
    def __init__(self, x_max: float = 100.0, alpha: float = 0.75):
        """
        Args:
            x_max: clipping parameter from Eq. (4), typically 100
            alpha: exponent in weighting function, typically 0.75
        """
        super().__init__()
        self.x_max = x_max
        self.alpha = alpha
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute loss from Eq. (3).
        
        Args:
            predictions: w_i^T * w_j + b_i + b_j from model forward() [N]
            targets: log(X_ij + 1) [N]  # +1 to avoid log(0)
            weights: f(X_ij) from Eq. (4) [N]
        
        Returns:
            scalar loss
        """
        # Copilot fills in implementation
```

**Copilot generates:**
```python
        # Eq. (3): Weighted MSE loss
        diff = predictions - targets  # prediction error
        squared_diff = diff ** 2
        
        # Weight each pair by f(X_ij)
        weighted_loss = weights * squared_diff
        
        return weighted_loss.mean()
```

**Key advantage:** You keep the math in comments, Copilot does the coding.

---

### Pattern 3: Data Processing with Comments

```python
def build_cooccurrence_matrix(
    tokens: List[int],
    window_size: int = 5,
) -> Dict[Tuple[int, int], int]:
    """
    Build co-occurrence matrix from Eq. (1).
    
    Eq. (1): X_ij = count(word_i, word_j in context_window)
    
    Pseudocode:
        for each position t in corpus:
            for each word i in window around t:
                for each word j in window around t where j != i:
                    increment X_ij
    """
    # Copilot generates implementation
```

**Copilot generates:**
```python
    cooccurrence = defaultdict(lambda: defaultdict(int))
    
    for i, word_i in enumerate(tokens):
        # Define context window boundaries
        context_start = max(0, i - window_size)
        context_end = min(len(tokens), i + window_size + 1)
        
        # Count co-occurrences within the window
        for j in range(context_start, context_end):
            if i != j:  # Don't count word with itself
                word_j = tokens[j]
                cooccurrence[word_i][word_j] += 1
    
    return cooccurrence
```

---

### Pattern 4: Ask Copilot for Edge Cases

**In Copilot Chat (Ctrl+I):**

```
Select the loss function code, then ask:
"What happens if X_ij is 0? How do we prevent NaN?"
```

**Copilot responds:**
```
When X_ij = 0:
- log(0) → -∞ (problematic)
- Solution: use log(X_ij + 1) instead of log(X_ij)
  This gives log(1) = 0 when X_ij = 0

The weighting function f(X_ij) = 0 when X_ij = 0,
so these pairs don't contribute to loss anyway.
```

**This saves you debugging time.**

---

### Pattern 5: Copilot as Code Explainer

```python
# Hover over complex line, Copilot Chat:
"Explain what torch.clamp does here"

# Copilot:
# torch.clamp(x, min=0, max=1) restricts values to range [0, 1]
# In this case: weights = torch.clamp((cooccurrence / x_max) ** alpha, max=1.0)
# This caps the weight at 1.0, ensuring high-frequency pairs don't exceed weight=1
```

---

## Jupyter Notebook Workflow Optimization

### Problem You're Solving

Currently: Read paper → Ask ChatGPT → Copy-paste into notebook → Fix alignment → Repeat

**New approach:** Code lives in `.py` files, notebooks import and experiment.

---

### Notebook Structure

**File: `notebooks/glove_learning.ipynb`**

```python
# Cell 1: Imports and Setup
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project to path
import sys
sys.path.insert(0, '..')

from implementations.models.glove import GloVeModel, GloVeLoss
from implementations.utils.glove_data import CooccurrenceDataset
from implementations.utils.training import train_epoch, evaluate
from implementations.utils.visualization import plot_embeddings, plot_training_curve

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Cell 2: Paper Summary (Markdown)
# # GloVe: Global Vectors for Word Representation
# 
# **Paper:** Pennington, Socher, Manning (2014)
# 
# **Problem:** Word2Vec captures local context but ignores global corpus statistics.
# GloVe combines both through weighted factorization of co-occurrence matrix.
# 
# **Key Equations:**
# - Eq. (1): X_ij = count(word_i, word_j in context)
# - Eq. (2): log(X_ij) ≈ w_i^T * w_j + b_i + b_j
# - Eq. (3): J = Σ f(X_ij) * (prediction - log(X_ij))^2

# Cell 3: Load or Create Data
from collections import Counter

# For testing, use toy corpus
toy_corpus = [
    "the cat sat on the mat",
    "the dog sat on the rug",
    "a cat is an animal",
    "a dog is an animal",
]

# Tokenize
tokens_list = [sent.split() for sent in toy_corpus]
print(f"Corpus: {len(tokens_list)} sentences")

# Build vocabulary
vocab = sorted(set(word for tokens in tokens_list for word in tokens))
word2idx = {w: i for i, w in enumerate(vocab)}
print(f"Vocabulary size: {len(vocab)}")

# Convert to indices
corpus_indices = [[word2idx[w] for w in tokens] for tokens in tokens_list]

# Cell 4: Build Co-occurrence Matrix
from implementations.utils.glove_data import build_cooccurrence_matrix

cooccurrence = build_cooccurrence_matrix(
    tokens=[w for tokens in corpus_indices for w in tokens],
    window_size=2
)
print(f"Co-occurrence pairs: {sum(len(v) for v in cooccurrence.values())}")

# Cell 5: Initialize Model and Optimizer
vocab_size = len(vocab)
embedding_dim = 10  # Small for testing
x_max = 5
alpha = 0.75

model = GloVeModel(vocab_size, embedding_dim, x_max, alpha).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = GloVeLoss(x_max, alpha)

print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

# Cell 6: Training Loop
train_losses = []

for epoch in range(5):
    total_loss = 0
    
    # Iterate through co-occurrence pairs
    for (i, j), count in cooccurrence.items():
        i_idx = torch.tensor([i], device=device)
        j_idx = torch.tensor([j], device=device)
        count_tensor = torch.tensor([count], dtype=torch.float32, device=device)
        
        # Weight: f(X_ij) from Eq. (4)
        weight = torch.clamp((count_tensor / x_max) ** alpha, max=1.0)
        target = torch.log(count_tensor + 1)
        
        # Forward
        pred = model(i_idx, j_idx)
        loss = loss_fn(pred, target.unsqueeze(0), weight.unsqueeze(0))
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(cooccurrence)
    train_losses.append(avg_loss)
    print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")

# Cell 7: Visualize Training Curve
plot_training_curve(train_losses)

# Cell 8: Inspect Learned Embeddings
embeddings = model.W.detach().cpu().numpy()  # [vocab_size, embedding_dim]
print(f"Embeddings shape: {embeddings.shape}")

# For visualization, reduce to 2D
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings)

# Plot
fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6)
for i, word in enumerate(vocab):
    ax.annotate(word, (embeddings_2d[i, 0], embeddings_2d[i, 1]), fontsize=8)
ax.set_title("GloVe Word Embeddings (PCA)")
plt.show()

# Cell 9: Evaluate on Word Similarity (Optional)
# If you have a similarity benchmark dataset, test here

# Cell 10: Key Insights (Markdown)
# ## Observations
# 
# 1. **Loss decreased as expected** - model learned to fit co-occurrence data
# 2. **Embeddings clustered by meaning** - "cat" and "dog" close, "the" isolated
# 3. **Weighting function crucial** - prevents high-freq words from dominating
# 
# ## Implementation Details
# - Used co-occurrence matrix from Eq. (1)
# - Implemented weighted MSE loss from Eq. (3)
# - Weight function from Eq. (4) with x_max=5, alpha=0.75
# 
# ## Differences from Paper
# - Toy corpus (4 sentences) vs large corpus (6B tokens)
# - Smaller embedding dimension (10 vs 300)
# - AdaGrad recommended in paper, used Adam for testing
```

**Key advantages:**
- ✅ Code lives in `.py` (reusable, testable)
- ✅ Notebook is 90% imports + experiments
- ✅ Easy to change hyperparameters
- ✅ Markdown cells stay clean (not mixed with code)
- ✅ No copy-paste from ChatGPT

---

### Use Notebook Powerline

In VS Code, while editing notebook:

**Split pane:**
- Left: `glove.py` (implementation)
- Right: `glove_learning.ipynb` (experiments)

Jump between them:
- `Ctrl+P` to switch files
- `F12` to go to definition in `.py` file
- `Ctrl+Shift+O` to see functions in `.py`

**Result:** You see code and notebook side-by-side, no context switching.

---

## Debugging Deep Learning Code

### Setup for PyTorch Debugging

In `.vscode/launch.json`, add:

```json
{
  "name": "Python: Debug PyTorch",
  "type": "python",
  "request": "launch",
  "program": "${file}",
  "console": "integratedTerminal",
  "justMyCode": false,
  "env": {"PYTHONPATH": "${workspaceFolder}", "CUDA_LAUNCH_BLOCKING": "1"},
  "logToFile": true
}
```

**Why `CUDA_LAUNCH_BLOCKING=1`?**
- Makes GPU errors synchronous (easier to debug)
- Without it, GPU errors may be delayed/hidden

---

### Advanced Breakpoint Patterns

#### Pattern 1: Shape Validation Breakpoint

```python
def forward(self, i_idx, j_idx):
    w_i = self.W[i_idx]
    w_j = self.W_context[j_idx]
    
    # Conditional breakpoint: RIGHT-CLICK on line, "Add Conditional Breakpoint"
    # Condition: w_i.shape[0] > 100
    dot_product = torch.sum(w_i * w_j, dim=1)  # ← Break only if batch > 100
    
    return dot_product
```

---

#### Pattern 2: Watch NaN/Inf

```python
# In Watch panel (bottom of debugger), add:
# torch.isnan(loss).any()
# torch.isinf(loss).any()
# loss.item()

# These evaluate in real-time as you step through
```

---

#### Pattern 3: Inspect Tensor Stats

```python
# In Watch panel:
# embeddings.mean().item()
# embeddings.std().item()
# embeddings.min().item()
# embeddings.max().item()
# embeddings.grad.abs().max().item()  # Check gradient magnitude
```

**Useful for spotting:**
- NaN/Inf values early
- Exploding gradients
- Dead neurons (all zeros)

---

### Debug Actual Training Run

```python
# File: debug_glove_training.py

import torch
from implementations.models.glove import GloVeModel, GloVeLoss

# Small toy setup for fast debugging
model = GloVeModel(vocab_size=100, embedding_dim=10)
loss_fn = GloVeLoss()

# Create dummy batch
i_idx = torch.randint(0, 100, (4,))
j_idx = torch.randint(0, 100, (4,))
cooccurrence = torch.randint(1, 50, (4,)).float()

# Set breakpoint here (F9 on this line)
pred = model(i_idx, j_idx)

# Breakpoint will pause execution
# Inspect in VS Code debugger:
# - pred.shape
# - pred.requires_grad
# - model.W.shape
```

**Run with:**
```
F5 (or Ctrl+Shift+D)
```

VS Code will pause at breakpoint, let you inspect everything.

---

## Multi-File Refactoring & Navigation

### Scenario: You realized a function should be shared

**Current state:**
- `implementations/models/glove.py` has `build_cooccurrence_matrix()`
- `implementations/models/word2vec.py` needs the same function

**Old way (bad):** Copy-paste, maintain two versions

**New way (good):** Extract to utils, import both places

**Steps:**

1. **Select function in glove.py:**
   ```python
   def build_cooccurrence_matrix(...):
       # ... entire function ...
   ```

2. **Right-click → Extract Method → Move to New File**
   VS Code creates `implementations/utils/cooccurrence.py`

3. **Add imports:**
   ```python
   # glove.py
   from implementations.utils.cooccurrence import build_cooccurrence_matrix
   
   # word2vec.py
   from implementations.utils.cooccurrence import build_cooccurrence_matrix
   ```

4. **Verify all references updated:**
   - `Ctrl+Shift+F` to find all uses
   - Check they all import correctly

---

### Find All References Across Project

```
Shift+F12 (on any symbol)
```

Shows every place a function/class is used.

**Example:** Rename `forward()` safely

```python
# In glove.py, on "def forward":
Shift+F12
# Shows all 7 places where forward() is called
# Now you can safely rename with F2
```

---

### Quick File Jumping

**Go to file by partial name:**
```
Ctrl+P "co" 
# Shows: cooccurrence.py, config.json, constants.py
```

**Go to specific symbol across all files:**
```
Ctrl+T
# Type "GloVeLoss" 
# Jump to definition anywhere in project
```

---

## Git + Paper Implementation Lifecycle

### Commit Strategy

Each paper should follow this commit pattern:

```bash
# Step 1: Implement model
git add implementations/models/glove.py
git commit -m "Implement GloVeModel with forward() computing Eq. (2)"

# Step 2: Add loss function
git add implementations/models/glove.py
git commit -m "Add GloVeLoss implementing Eq. (3) with weighting function"

# Step 3: Add data utilities
git add implementations/utils/glove_data.py
git commit -m "Add co-occurrence matrix builder (Eq. 1)"

# Step 4: Create learning notebook
git add notebooks/glove_learning.ipynb
git commit -m "Add GloVe training notebook with toy corpus"

# Step 5: Document
git add docs/glove_summary.md
git commit -m "Add GloVe paper summary and implementation notes"
```

**Benefits:**
- Each commit is atomic (one logical piece)
- Equation numbers in commit message = searchable
- Easy to revert one piece if needed
- History shows implementation progression

---

### Use Git Inside VS Code

**Source Control view:**
- `Ctrl+Shift+G`

**From here:**
- Stage files (drag to "Staged Changes")
- Write commit message
- Press `Ctrl+Enter` to commit
- Push with one click

**Never leave VS Code for git.**

---

### Branch for Experiments

**Testing hyperparameters?**

```bash
# Create branch
git checkout -b experiment/glove-dim-200

# Modify hyperparameters, run experiments
# If good results:
git checkout main
git merge experiment/glove-dim-200

# If bad:
git branch -D experiment/glove-dim-200
```

---

## VS Code Tasks for Automation

### Common Paper Implementation Tasks

Create `.vscode/tasks.json`:

```json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "Format All Python",
      "type": "shell",
      "command": "black",
      "args": ["implementations", "notebooks", "docs"],
      "group": {"kind": "build", "isDefault": true}
    },
    {
      "label": "Run Linting",
      "type": "shell",
      "command": "pylint",
      "args": ["implementations/models"],
      "presentation": {"echo": true, "reveal": "always"}
    },
    {
      "label": "Run Unit Tests",
      "type": "shell",
      "command": "python",
      "args": ["-m", "pytest", "-v", "tests/"],
      "group": {"kind": "test"}
    },
    {
      "label": "Build Docs",
      "type": "shell",
      "command": "jupyter",
      "args": ["nbconvert", "--to", "html", "notebooks/glove_learning.ipynb"],
      "presentation": {"echo": false}
    },
    {
      "label": "Create New Paper Template",
      "type": "shell",
      "command": "bash",
      "args": ["-c", "cp implementations/models/template.py implementations/models/${input:paperName}.py && echo 'Created ${input:paperName}.py'"],
      "inputs": [
        {
          "id": "paperName",
          "type": "promptString",
          "description": "Paper name (lowercase, no spaces)"
        }
      ]
    }
  ]
}
```

**Run with:**
```
Ctrl+Shift+B (build)
Ctrl+Shift+P → "Tasks: Run Task"
```

---

## Extension Configuration for Data Science

### Minimal Essential Extensions

```json
// In settings.json, under "[python]":
{
  "extensions.enabled": true,
  "extensions.autoUpdate": true
}
```

**Install these specific versions:**
1. **Python** (Microsoft) - Always latest
2. **Pylance** - Better IntelliSense than default
3. **Jupyter** (Microsoft) - For notebook support
4. **GitHub Copilot** - AI assistance
5. **Black Formatter** - Code formatting

**Optional:**
- **GitLens** - Git blame/history
- **Error Lens** - Show errors inline
- **Markdown Preview Enhanced** - Better markdown preview
- **Trailing Spaces** - See whitespace issues

---

### Disable Unnecessary Extensions

Many VS Code installs have bloat. Disable:
- Remote containers (unless using)
- C++ tools
- Live Share (unless pair programming)
- Azure extensions (unless using Azure)

**Check: Extensions → Find "Disabled" to see what you turned off**

---

## Performance Optimization for Large Projects

### Issue: VS Code gets slow with many files

**Solution 1: Configure file watching**

```json
{
  "files.watcherExclude": {
    "**/venv": true,
    "**/__pycache__": true,
    "**/results/**": true,
    "**/data/**": true,
    "**/.ipynb_checkpoints": true
  }
}
```

**Solution 2: Disable heavy extensions**

```
Ctrl+Shift+X
Search "Python"
Disable anything except: Python, Pylance, Jupyter
```

**Solution 3: Use TypeScript efficiently**

```json
{
  "typescript.tsdk": "node_modules/typescript/lib",
  "typescript.enablePromptUseWorkspaceTsdk": true
}
```

---

### Monitor Performance

```
Ctrl+Shift+P → "Developer: Open Timeline"
```

Shows what's taking time.

---

## Advanced Keyboard Shortcuts You Don't Know

For mid-level developers:

| Command | Shortcut | Use Case |
|---------|----------|----------|
| Multi-cursor select occurrences | `Ctrl+Shift+L` | Rename var across line |
| Add cursor above/below | `Ctrl+Alt+Up/Down` | Edit multiple lines at once |
| Expand/shrink selection | `Ctrl+Shift+→/←` | Select by word |
| Delete line | `Ctrl+Shift+K` | Remove debug line |
| Duplicate line | `Ctrl+Shift+D` or `Alt+Shift+Down` | Copy line without clipboard |
| Join lines | `Ctrl+J` | Merge lines |
| Sort lines | `Ctrl+K Ctrl+S` | Organize imports (with plugin) |
| Column selection | `Shift+Alt+Click` | Edit multiple columns |
| Breadcrumb navigation | `Ctrl+Shift+;` | Jump in outline |

---

## Your Optimized Workflow (Step-by-Step)

### Paper Implementation in 90 Minutes

**Setup (once):**
1. Create `.vscode/settings.json` from above
2. Install extensions: Python, Pylance, Jupyter, Copilot
3. Select interpreter: `Ctrl+Shift+P` → Python: Select Interpreter

**For each paper:**

**Phase 1 (20 min): Read & Plan**
1. Open PDF in external viewer (or in VS Code via extension)
2. Take notes in `docs/paper_notes.md`
3. Identify equations, hyperparameters, data requirements

**Phase 2 (40 min): Implement**
1. Copy template: `cp implementations/models/template.py implementations/models/glove.py`
2. Add header with equations
3. Write docstrings with Eq. references
4. Use Copilot (Ctrl+Alt+\) to generate code
5. Format (Shift+Alt+F) and save
6. Quick test: Open terminal (`Ctrl+``), run model import

**Phase 3 (15 min): Experiment Notebook**
1. Create `notebooks/glove_learning.ipynb`
2. Cell 1: Import model
3. Cell 2: Paper summary (markdown)
4. Cell 3-5: Load data, create model, training loop
5. Cell 6: Visualizations
6. Run and verify works

**Phase 4 (10 min): Document & Commit**
1. Write `docs/glove_summary.md`
2. `Ctrl+Shift+G` (Git) → Stage all → Commit
3. Commit message includes equation numbers

**Phase 5 (5 min): Optional Debugging**
- If training is weird: Set breakpoint (F9) → F5 to debug
- Inspect variables in watch panel

---

## Quick Answers to Common Mid-Level Issues

### "Copilot gives bad suggestions"
→ Write better docstrings with equations. Poor input = poor output.

### "IntelliSense is wrong"
→ Check interpreter is correct (bottom-left shows `Python 3.x (venv)`)

### "Debugging is slow on GPU"
→ Add to launch.json: `"env": {"CUDA_LAUNCH_BLOCKING": "1"}`

### "Notebook output is huge"
→ In notebook, right-click cell → Clear Outputs

### "Can't find where a function is called"
→ `Shift+F12` on the function name

### "Accidentally deleted file"
→ `Ctrl+Z` undo, or `git checkout filename`

### "Terminal can't find Python packages"
→ Check terminal has venv activated: `which python` should show `/venv/bin/python`

---

## Final Checklist: Ready to Implement

- [ ] `.vscode/settings.json` copied to project
- [ ] `.vscode/launch.json` created
- [ ] Python extension installed
- [ ] Pylance installed (better intellisense)
- [ ] Jupyter extension installed
- [ ] GitHub Copilot installed
- [ ] Interpreter selected (Ctrl+Shift+P → Python: Select Interpreter)
- [ ] Can run `python -c "import torch; print(torch.__version__)"` in VS Code terminal
- [ ] Opened a `.py` file, Copilot suggestions appear (Ctrl+Alt+\)
- [ ] Created test notebook, can import from implementations/

**When all ✓: You're ready to implement papers efficiently.**

---

## Your Action Plan

**This week:**
1. Copy the `settings.json` and `launch.json` from above into your project `.vscode/`
2. Restart VS Code
3. Implement one paper (GloVe recommended, 2-3 hours)
4. Notice how much time you save by NOT copy-pasting from ChatGPT
5. Commit to git with equation references in messages

**By week 2:**
- You'll naturally use Copilot for most code generation
- Debugging will feel faster
- Notebooks will be clean (imports only)
- Git workflow automated

**By week 4:**
- You'll have 4-5 papers implemented
- Building a reusable library
- Can implement a new paper in ~60 minutes
- Your repo becomes a portfolio

---

## Reference: All Settings in One Place

Keep this as a template for new projects:

**`.vscode/settings.json`** → Copy from "Advanced Workspace Configuration" section above

**`.vscode/launch.json`** → Copy from there

**`.vscode/tasks.json`** → Copy from "VS Code Tasks" section

**`.vscode/extensions.json`** (optional, for sharing with team):
```json
{
  "recommendations": [
    "ms-python.python",
    "ms-python.vscode-pylance",
    "ms-toolsai.jupyter",
    "GitHub.copilot"
  ]
}
```

---

**You now have a professional, production-grade VS Code setup optimized for implementing deep learning papers without wasting time on copy-paste work.**

Get started this week—pick GloVe and implement it end-to-end using this guide. 🚀
