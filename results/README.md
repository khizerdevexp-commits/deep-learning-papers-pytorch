# Results

This directory stores generated model checkpoints and other experiment outputs. Generated binaries are ignored by Git; keep important artifacts in local storage or a durable cloud location.

## GloVe WikiText-2 Sample

`glove_wikitext2_sample.pt` is a PyTorch checkpoint produced by the Kaggle notebook at `kaggle/glove-wikitext2/glove_wikitext2.ipynb`. It contains:

- `state_dict`: target/context embedding tables and biases
- `vocabulary`: token-to-ID mapping
- `embeddings`: combined target and context vectors (`w_i + w_tilde_i`)

The sample run uses the first 2,000 lines of WikiText-2 raw training data, a vocabulary capped at 10,000 tokens, and 100-dimensional vectors. It is for implementation and workflow experiments, not a paper reproduction. No benchmark accuracy is included; evaluate on a held-out similarity or analogy benchmark before making quality claims.

### Download the latest Kaggle output

Run from `C:\Users\Dell\Desktop\practice` in PowerShell:

```powershell
& '.\.venv\Scripts\kaggle.exe' kernels output khizercheema/glove-wikitext-2-gpu-training -p '.\deep-learning-papers-pytorch' --file-pattern '.*glove_wikitext2_sample\.pt$'
```

The Kaggle output retains the remote `results/` subfolder, so the checkpoint lands directly in this directory. Do not set `-p` to this `results` directory, or the downloaded path will be nested as `results\results`.

### Load the checkpoint

Run from the repository root with the project environment:

```python
from pathlib import Path
import torch

from implementations.models.glove import GloveModel

checkpoint = torch.load(
	Path("results/glove_wikitext2_sample.pt"),
	map_location="cpu",
	weights_only=False,
)
vocabulary = checkpoint["vocabulary"]
model = GloveModel(
	vocab_size=len(vocabulary),
	embedding_dim=checkpoint["embeddings"].shape[1],
)
model.load_state_dict(checkpoint["state_dict"])
embeddings = checkpoint["embeddings"]
```

Use `map_location="cpu"` for inspection or move the loaded model to an available device for inference. `weights_only=False` is needed because this artifact also stores the vocabulary dictionary; only load checkpoints you trust.
