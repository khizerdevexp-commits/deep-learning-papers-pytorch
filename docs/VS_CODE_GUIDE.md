# VS Code Mastery Guide for Deep Learning Paper Implementation

A practical guide to maximizing VS Code productivity for your workflow. Based on official VS Code docs + optimizations specific to your use case.

---

## Table of Contents

1. [Essential Setup](#essential-setup)
2. [Must-Know Features](#must-know-features)
3. [Python-Specific Productivity](#python-specific-productivity)
4. [Copilot Integration](#copilot-integration)
5. [Keyboard Shortcuts You Need](#keyboard-shortcuts-you-need)
6. [Workspace Organization](#workspace-organization)
7. [Debugging for Paper Implementation](#debugging-for-paper-implementation)
8. [Extensions Recommended](#extensions-recommended-for-your-workflow)

---

## Essential Setup

### Step 1: Install VS Code

Download from: https://code.visualstudio.com/

**Recommended:** Install the "User Installer" (not System-wide)

---

### Step 2: Install Python Extension

1. Open VS Code
2. Press `Ctrl+Shift+X` (or click Extensions icon on left sidebar)
3. Search: `Python`
4. Install **"Python"** by Microsoft (official, 70+ million downloads)

**Why:** This extension unlocks:
- IntelliSense (autocomplete)
- Linting (error detection)
- Debugging
- Testing
- Jupyter notebook support
- Virtual environment detection

---

### Step 3: Select Python Interpreter

1. Press `Ctrl+Shift+P` (Command Palette)
2. Type: `Python: Select Interpreter`
3. Choose your virtual environment (the one you created with `python -m venv venv`)

**Verify it works:**
- Bottom-left corner should show: `Python 3.x.x ('./venv': venv)`

---

### Step 4: Create Workspace Settings

In your project root, create `.vscode/settings.json`:

```json
{
  // Python paths
  "python.defaultInterpreterPath": "${workspaceFolder}/venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": true,
  "python.formatting.provider": "black",
  
  // Editor settings
  "editor.formatOnSave": true,
  "editor.defaultFormatter": "ms-python.python",
  "editor.rulers": [80, 120],
  "editor.wordWrap": "on",
  
  // File exclusions
  "files.exclude": {
    "**/__pycache__": true,
    "**/*.pyc": true,
    "**/.pytest_cache": true,
    "**/node_modules": true
  },
  
  // Search exclusions
  "search.exclude": {
    "**/.venv": true,
    "**/venv": true,
    "**/__pycache__": true
  },
  
  // Jupyter
  "jupyter.experiments.enabled": false,
  "notebook.lineNumbers": "on"
}
```

**What this does:**
- Formats code on save using Black
- Shows vertical rulers at 80 and 120 characters (PEP 8 guidelines)
- Hides `__pycache__` from file explorer
- Enables linting with Pylint

---

## Must-Know Features

### 1. IntelliSense (Autocomplete)

**What it is:** Real-time code suggestions based on your project and Python libraries.

**How to use:**

```python
# Start typing - suggestions appear automatically
import torch

model = torch.nn.  # Press Ctrl+Space to trigger
# Suggestions: Linear, Conv2d, LSTM, etc.

# IntelliSense also shows:
# - Parameter hints
# - Documentation
# - Type information
```

**Keyboard shortcuts:**
- `Ctrl+Space`: Trigger suggestions manually
- `Tab` or `Enter`: Accept suggestion
- `Esc`: Dismiss suggestions
- `Ctrl+J`: Toggle suggestion details

**Pro tip:** Hover over any variable to see its type and docstring.

---

### 2. Go to Definition

**Navigate to where a function or class is defined.**

**How:**
```python
# Position cursor on a function name
from implementations.models.glove import GloVeModel

model = GloVeModel()  # Hover and press Ctrl+Click to jump to definition
                      # OR right-click → "Go to Definition"
                      # OR press F12
```

**This jumps to:** `implementations/models/glove.py` line where `GloVeModel` is defined

**Keyboard:**
- `F12`: Go to definition
- `Ctrl+Alt+Click`: Open definition in side panel (don't leave current file)
- `Shift+F12`: Find all references to this symbol

---

### 3. Find and Replace

**The most powerful search in VS Code.**

```
Ctrl+H: Open Find and Replace
```

**Features:**
- Find: `Ctrl+F`
- Replace: `Ctrl+H`
- Find in files: `Ctrl+Shift+F`
- Replace in files: `Ctrl+Shift+H`

**Example:**

```
Find:    "def forward("
Replace: "def forward_pass("
```

**Advanced: Use Regex**

```
Find:    "Eq\. \((\d+)\)"
Replace: "Equation ($1):"
```

Replaces all instances of "Eq. (1)" → "Equation (1):", etc.

Toggle regex: Click `.*` icon in find box (or Alt+R)

---

### 4. Outline View

**See all functions, classes, and variables in the current file.**

```
Ctrl+Shift+O: Go to Symbol
```

**In your glove.py file:**

```python
class GloVeModel(nn.Module):
    def __init__(self, ...):
    def forward(self, ...):

class GloVeLoss(nn.Module):
    def __init__(self, ...):
    def forward(self, ...):

def build_cooccurrence_matrix(...):
```

When you press `Ctrl+Shift+O`, you get:

```
GloVeModel
  ├── __init__
  └── forward
GloVeLoss
  ├── __init__
  └── forward
build_cooccurrence_matrix
```

Click any to jump there instantly.

---

### 5. Breadcrumb Navigation

Shows your current location in the file hierarchy.

**Enable:**
- View menu → Show Breadcrumb
- OR set in settings.json: `"breadcrumbs.enabled": true`

**Example while editing `forward()` method:**

```
deep-learning-papers-pytorch > implementations/models > glove.py > GloVeModel > forward
```

Click any segment to jump to that location or see siblings.

---

## Python-Specific Productivity

### 1. Linting (Error Detection)

**What:** Catches bugs and style issues in real-time.

**Setup:**
```bash
# Install pylint
pip install pylint

# Install in VS Code: Settings → Search "linting" → Enable Pylint
```

**Example:**

```python
import torch

def compute_loss(predictions, targets):
    unused_variable = 42  # ← Pylint warns: "unused variable"
    
    diff = predictions - target  # ← Pylint warns: "undefined name 'target'"
    return (diff ** 2).mean()
```

**Red squigglies** appear under errors. Hover to see explanation.

**Auto-fix:**
- Click the lightbulb icon (or `Ctrl+.`)
- Select suggested fix

---

### 2. Code Formatting

**What:** Automatically formats code to PEP 8 standard.

**Setup:**
```bash
pip install black
```

**Auto-format:**
- `Shift+Alt+F`: Format document
- `Ctrl+K Ctrl+F`: Format selection
- Enabled on save (if `"editor.formatOnSave": true` in settings.json)

**Before:**
```python
def forward(self,i_idx,j_idx):
    w_i=self.W[i_idx]
    w_j=self.W_context[j_idx]
    dot_product=torch.sum(w_i*w_j,dim=1)
    predictions=dot_product+self.b[i_idx]+self.b_context[j_idx]
    return predictions
```

**After (one keystroke):**
```python
def forward(self, i_idx, j_idx):
    w_i = self.W[i_idx]
    w_j = self.W_context[j_idx]
    dot_product = torch.sum(w_i * w_j, dim=1)
    predictions = dot_product + self.b[i_idx] + self.b_context[j_idx]
    return predictions
```

---

### 3. Refactoring

**Rename a symbol everywhere it's used.**

```python
# In glove.py
class GloVeModel(nn.Module):
    pass

# In notebooks/glove_learning.ipynb
from implementations.models.glove import GloVeModel
model = GloVeModel()
```

**Rename refactoring:**
1. Right-click on `GloVeModel` in glove.py
2. Select "Rename Symbol"
3. Type new name
4. Press Enter
5. ✅ ALL instances updated automatically (in all files!)

**Keyboard:** `F2` on the symbol

---

### 4. Extract Method

**Convert selected code into a separate function.**

**Before:**
```python
def forward(self, x):
    # Lots of code
    w_i = self.W[x]
    w_j = self.W_context[x]
    dot_product = torch.sum(w_i * w_j, dim=1)
    # More code
```

**After:**
1. Select the 3 lines
2. Right-click → "Extract Method"
3. Name it: `compute_embeddings`
4. ✅ New function created, call inserted

```python
def compute_embeddings(self, x):
    w_i = self.W[x]
    w_j = self.W_context[x]
    dot_product = torch.sum(w_i * w_j, dim=1)
    return dot_product

def forward(self, x):
    dot_product = self.compute_embeddings(x)
    # Rest of code
```

---

## Copilot Integration

### 1. Install GitHub Copilot

**In VS Code:**
1. Press `Ctrl+Shift+X` (Extensions)
2. Search: `GitHub Copilot`
3. Click "Install" (requires GitHub sign-in)
4. Sign in with your GitHub account

---

### 2. Using Copilot Effectively

#### **Pattern 1: Line Completion**

```python
# Type this:
def build_cooccurrence_matrix(tokens, window_size):
    """
    Equation (1): X_ij = count(word_i, word_j in window)
    """
    # Press Tab after incomplete line - Copilot suggests
```

Copilot suggests:
```python
    cooccurrence = defaultdict(lambda: defaultdict(int))
    
    for i, word_i in enumerate(tokens):
        for j in range(max(0, i - window_size), min(len(tokens), i + window_size + 1)):
            if i != j:
                cooccurrence[word_i][tokens[j]] += 1
    
    return cooccurrence
```

**How to trigger:**
- `Ctrl+Alt+\`: Open Copilot suggestion panel
- `Tab`: Accept suggestion
- `Esc`: Reject
- `Alt+]`: Next suggestion
- `Alt+[`: Previous suggestion

#### **Pattern 2: Function Generation from Docstring**

```python
def compute_loss(predictions, targets, weights):
    """
    Compute weighted MSE loss from Equation (3).
    
    Args:
        predictions: Model outputs [batch]
        targets: Target log values [batch]
        weights: Sample weights f(X_ij) [batch]
    
    Returns:
        loss: Scalar MSE value
    """
    # Press Ctrl+Alt+\ for Copilot suggestion
```

Copilot generates:
```python
    diff = predictions - targets
    squared_error = (diff ** 2)
    weighted_mse = weights * squared_error
    return weighted_mse.mean()
```

#### **Pattern 3: Comment to Code**

```python
# Section 2.3: Weighting Function
# Implement Equation (4): f(x) = (x / x_max)^alpha if x < x_max else 1

# Start typing and Copilot suggests
def weighting_function(cooccurrence, x_max, alpha):
    # Copilot generates full implementation
```

---

### 3. Copilot Chat (Inline Discussions)

**New feature:** Ask Copilot questions without leaving the editor.

```
Ctrl+I: Open Copilot Chat inline
```

**Example in your code:**

```python
def forward(self, i_idx, j_idx):
    w_i = self.W[i_idx]  # ← Select this line
    
    # Ctrl+I → Type question:
    # "Explain what this line does in the context of GloVe"
```

**Copilot responds in a side panel without interrupting your code.**

---

## Keyboard Shortcuts You Need

### Navigation

| Action | Shortcut |
|--------|----------|
| Go to file | `Ctrl+P` |
| Go to line | `Ctrl+G` |
| Go to definition | `F12` |
| Go to symbol in file | `Ctrl+Shift+O` |
| Find in project | `Ctrl+Shift+F` |
| Toggle sidebar | `Ctrl+B` |
| Toggle terminal | `Ctrl+` ` |

### Editing

| Action | Shortcut |
|--------|----------|
| Format code | `Shift+Alt+F` |
| Cut line | `Ctrl+X` |
| Copy line | `Ctrl+C` |
| Delete line | `Ctrl+Shift+K` |
| Duplicate line | `Ctrl+Shift+D` or `Alt+Shift+Down` |
| Move line up | `Alt+Up` |
| Move line down | `Alt+Down` |
| Comment line | `Ctrl+/` |
| Multi-line comment | `Shift+Alt+A` |

### Search & Replace

| Action | Shortcut |
|--------|----------|
| Find | `Ctrl+F` |
| Replace | `Ctrl+H` |
| Find in files | `Ctrl+Shift+F` |
| Replace in files | `Ctrl+Shift+H` |

### Debugging

| Action | Shortcut |
|--------|----------|
| Start/Continue | `F5` |
| Pause | `F6` |
| Step over | `F10` |
| Step into | `F11` |
| Step out | `Shift+F11` |
| Toggle breakpoint | `F9` |

### Selection & Multi-Cursor

| Action | Shortcut |
|--------|----------|
| Select word | `Ctrl+D` |
| Select all occurrences | `Ctrl+Shift+L` |
| Add cursor above | `Ctrl+Alt+Up` |
| Add cursor below | `Ctrl+Alt+Down` |
| Undo last cursor | `Ctrl+U` |

---

## Workspace Organization

### 1. Project Structure in VS Code

**Open folder (entire project):**

```
File → Open Folder
Select: deep-learning-papers-pytorch
```

**Explorer view (left sidebar):**

```
📁 deep-learning-papers-pytorch
  📁 implementations/
    📁 models/
      glove.py
      word2vec.py
      template.py
    📁 utils/
      training.py
      visualization.py
  📁 notebooks/
    glove_learning.ipynb
    word2vec_learning.ipynb
  📁 docs/
    glove_summary.md
    WORKFLOW_GUIDE.md
  📁 papers/
  requirements.txt
  README.md
```

---

### 2. Create Multiple Folders in Explorer

**Right-click in Explorer → New Folder**

Best practice:
```
deep-learning-papers-pytorch/
├── .vscode/          ← VS Code settings
│   └── settings.json
├── implementations/
│   ├── models/
│   ├── layers/
│   ├── utils/
│   └── losses/
├── notebooks/        ← Learning notebooks
├── papers/           ← PDF research papers
├── docs/             ← Markdown summaries
├── data/             ← Training data
├── results/          ← Model checkpoints
├── .gitignore
├── requirements.txt
└── README.md
```

---

### 3. Minimize Explorer Clutter

In `.vscode/settings.json`:

```json
{
  "files.exclude": {
    "**/__pycache__": true,
    "**/*.pyc": true,
    "**/.pytest_cache": true,
    "**/results/**/*.pt": true,
    "**/.ipynb_checkpoints": true
  },
  "search.exclude": {
    "**/venv": true,
    "**/__pycache__": true,
    "**/results": true
  }
}
```

Now the Explorer only shows source code, not clutter.

---

### 4. Quick File Switching

**Open file by name (no navigation):**

```
Ctrl+P → Type "glove" → See all matches
```

Results:
```
glove.py (implementations/models/)
glove.py (notebooks/)
glove_summary.md (docs/)
```

Click or press Enter to jump there.

---

## Debugging for Paper Implementation

### 1. Set a Breakpoint

```python
def forward(self, i_idx, j_idx):
    w_i = self.W[i_idx]  # ← Click left margin here to add breakpoint
    w_j = self.W_context[j_idx]
    
    dot_product = torch.sum(w_i * w_j, dim=1)  # Red dot = breakpoint
    return predictions
```

**Or use `F9` with cursor on the line**

---

### 2. Start Debugging

**Press `F5` to start debugger**

```
Choose environment: Python
```

VS Code runs your code and pauses at the breakpoint.

---

### 3. Debug Controls

```
F5   = Continue execution
F10  = Step over (run next line)
F11  = Step into (enter function)
Shift+F11 = Step out (exit function)
```

---

### 4. Inspect Variables

When paused at breakpoint:

**Left panel "Variables":**
- Shows all local variables
- Hover over any variable to see value
- Expand to see nested properties

**Example:**
```python
w_i = tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])  # Hover to see

# Shows: tensor (shape=[2, 3], dtype=float32)
```

---

### 5. Conditional Breakpoint

**Right-click breakpoint → Edit Breakpoint**

```
Expression: loss > 1.0
```

Pauses only when loss exceeds 1.0 (useful for catching anomalies in training)

---

### 6. Watch Expressions

**Bottom of debugger → "Watch" section**

Click `+` and add:
```
loss.item()
predictions.shape
torch.cuda.memory_allocated()
```

Values update as you step through code.

---

## Extensions Recommended for Your Workflow

### Essential

1. **Python** (Microsoft) - Already installed
2. **Pylint** - Linting (catches errors)
3. **Black Formatter** - Code formatting

```bash
pip install pylint black
```

### Highly Recommended

4. **Jupyter** (Microsoft)
   - Run and edit `.ipynb` notebooks inside VS Code
   - No need to switch to browser

5. **Pylance**
   - Better IntelliSense than default Python extension
   - Type checking
   - Install: Search "Pylance" in Extensions

6. **Thunder Client** or **REST Client**
   - Test API endpoints (useful if your model has a server)

### Optional but Useful

7. **GitLens**
   - See git blame, history per line
   - Understand who changed what

8. **Even Better TOML**
   - Better syntax highlighting for config files

9. **Markdown Preview Enhanced**
   - Live preview of your `.md` files
   - See how docs/glove_summary.md will render

10. **Error Lens**
    - Show errors inline instead of only on hover
    - Saves time spotting mistakes

---

## Practical Workflow: Putting It All Together

### Scenario: Implementing GloVe

**Step 1: Open project**
```
Ctrl+K Ctrl+O → Select deep-learning-papers-pytorch folder
```

**Step 2: Open glove.py**
```
Ctrl+P → Type "glove" → Select glove.py
```

**Step 3: Implement with Copilot**
```python
def forward(self, i_idx, j_idx):
    """
    Equation (2): log(X_ij) ≈ w_i^T * w_j + b_i + b_j
    """
    # Ctrl+Alt+\ for Copilot suggestion
    # Copilot generates the full implementation
```

**Step 4: Format on save**
```
Ctrl+S → Black auto-formats the code
```

**Step 5: Check for errors**
```
Pylint underlines any bugs in red
Hover to see explanation
Ctrl+. to apply auto-fix
```

**Step 6: Navigate to data loader**
```
Ctrl+Shift+O → See outline of file
Click "build_cooccurrence_matrix" → Jump there
Or: F12 on function call → Jump to definition
```

**Step 7: Quick test**
```
Ctrl+` → Open terminal
python -c "from implementations.models.glove import GloVeModel; print('✓ Imports work')"
```

**Step 8: Debug if needed**
```
Set breakpoint: Click left margin
F5 → Start debugger
F10 → Step through code
Watch panel → Inspect variables
```

**Step 9: Find all usages**
```
Right-click GloVeModel → "Find All References"
Shows all places where this class is used
```

**Step 10: Rename safely**
```
Right-click class name → "Rename Symbol"
All references updated automatically
```

---

## Settings for Your Specific Use Case

**Create `.vscode/settings.json` in your project:**

```json
{
  // Python interpreter
  "python.defaultInterpreterPath": "${workspaceFolder}/venv/bin/python",
  
  // Linting
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": true,
  "python.linting.pylintArgs": ["--max-line-length=120"],
  
  // Formatting
  "python.formatting.provider": "black",
  "[python]": {
    "editor.defaultFormatter": "ms-python.python",
    "editor.formatOnSave": true,
    "editor.rulers": [80, 120]
  },
  
  // Code style
  "editor.wordWrap": "on",
  "editor.trimAutoWhitespace": true,
  "files.trimTrailingWhitespace": true,
  "files.insertFinalNewline": true,
  
  // File exclusions
  "files.exclude": {
    "**/__pycache__": true,
    "**/*.pyc": true,
    "**/.pytest_cache": true,
    "**/node_modules": true,
    "**/.ipynb_checkpoints": true
  },
  
  "search.exclude": {
    "**/venv": true,
    "**/__pycache__": true,
    "**/results": true,
    "**/data": true
  },
  
  // Jupyter
  "jupyter.experiments.enabled": false,
  "notebook.lineNumbers": "on",
  
  // Git
  "git.ignoreLimitWarning": true,
  
  // Terminal
  "terminal.integrated.defaultProfile.linux": "bash",
  "terminal.integrated.defaultProfile.osx": "bash",
  "terminal.integrated.defaultProfile.windows": "PowerShell"
}
```

---

## Troubleshooting Common Issues

### Issue 1: Copilot Not Suggesting

**Solution:**
1. Check GitHub Copilot is installed: `Ctrl+Shift+X`
2. Restart VS Code
3. Make sure you're in a `.py` file (not `.txt`)
4. Try `Ctrl+Alt+\` to explicitly open Copilot

---

### Issue 2: Linting Not Working

**Solution:**
```bash
# Install pylint
pip install pylint

# In VS Code, restart Python extension:
# Ctrl+Shift+P → "Python: Restart Language Server"
```

---

### Issue 3: IntelliSense Not Working

**Solution:**
1. Check you selected the right Python interpreter
   - Bottom-left shows Python version
   - Click to see if venv is selected
2. Make sure venv is activated:
   ```bash
   source venv/bin/activate  # macOS/Linux
   venv\Scripts\activate     # Windows
   ```
3. Restart VS Code

---

### Issue 4: Debugger Won't Start

**Solution:**
1. Make sure Python extension is installed
2. Create `.vscode/launch.json`:
   ```bash
   Ctrl+Shift+D → "Create a launch.json file" → Python
   ```
3. Try running a simple script first (not Jupyter)

---

## Next Steps

Now that you understand VS Code:

1. **Practice the shortcuts:** Spend 10 minutes just using them
   - `Ctrl+P`: 5 times
   - `Ctrl+H`: Replace something 3 times
   - `F12`: Jump to definition 5 times
   - `Ctrl+/`: Comment/uncomment 10 lines

2. **Set up your workspace:**
   - Create `.vscode/settings.json` from above
   - Install Pylance extension
   - Test debugging with a simple script

3. **Implement your first paper:**
   - Open glove.py (create it if you haven't)
   - Use `Ctrl+Alt+\` for Copilot suggestions
   - Format with `Shift+Alt+F`
   - Test with debugger

4. **Master Git:**
   - `Ctrl+Shift+G`: Open Git panel
   - Stage changes, commit, push—all in VS Code

---

## Official Resources

**When stuck, check official docs:**

- [VS Code Python Docs](https://code.visualstudio.com/docs/languages/python)
- [VS Code Debugging Guide](https://code.visualstudio.com/docs/editor/debugging)
- [VS Code Keyboard Shortcuts Cheat Sheet](https://code.visualstudio.com/docs/getstarted/keybindings)
- [GitHub Copilot Docs](https://docs.github.com/en/copilot)

---

**You're now ready to use VS Code like a pro for your deep learning paper implementations!** 🚀
