## 4. LoRA

For a targeted pretrained matrix $W_0 \in \mathbb{R}^{d \times k}$, LoRA represents the adaptation update as

$$
\Delta W \;=\; B A, \qquad B \in \mathbb{R}^{d \times r}, \;\; A \in \mathbb{R}^{r \times k}, \;\; r \ll \min(d, k).
$$

The effective weight used in the forward pass is $W = W_0 + \Delta W = W_0 + B A$. For an input $x$,

$$
h \;=\; W_0\, x \;+\; \frac{\alpha}{r}\, B A\, x .
$$

Four design choices from the paper are worth pausing on:

- **Initialization.** $A$ is initialized with a Gaussian, and $B$ is initialized to zero. At step 0, $\Delta W = B A = 0$, so the model starts exactly at the pretrained function — a safe starting point.
- **Scaling $\alpha / r$.** The update is multiplied by $\alpha/r$. This makes training roughly rank-agnostic: when we change $r$ we do not need to re-tune the learning rate. The paper sets $\alpha$ to the first $r$ they try and leaves it there.
- **Why small $r$?** The paper's hypothesis (supported empirically later in the paper) is that the task-specific update lives in a very low-dimensional subspace. The extreme result from the paper's Table 6 is that $r = 1$ or $r = 2$ is often enough for GPT-3 adaptation.
- **No extra parameters in the base path.** $W_0$ is frozen. No optimizer state is allocated for it, which is where the memory savings come from.

Below is a matrix-shape view of the factorization.

![Lora factorization](assets/factorization.png)


```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset

DEVICE = "cpu"
print("PyTorch:", torch.__version__, "| Device:", DEVICE)
```


```python
class LoraLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int,
                 r: int = 4, alpha: float = 1.0, dropout: float = 0.0):
        super().__init__()
        self.r     = r
        self.scale = alpha / r 

        self.weight = nn.Parameter(torch.empty(out_features,in_features), requires_grad=False)

        self.lora_A = nn.Parameter(torch.empty(self.r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, self.r))

        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight)
        lora = F.linear(F.linear(self.dropout(x), self.lora_a),self.lora_B) * self.scale
        return base + lora

    def mergeweights(self) -> None:
        with torch.no_grad():
            self.weight.data += (self.lora_B @ self.lora_A) * self.scale


    def extra_repr(self):
        return (f"in={self.weight.shape[1]}, out={self.weight.shape[0]}, "
            f"r={self.r}, scale={self.scale:.3f}")
```

```python
import urllib.request
from datasets import Dataset
from transformers import GPT2Tokenizer

url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
urllib.request.urlretrieve(url, "shakespeare.txt")

with open("shakespeare.txt", "r", encoding="utf-8") as f:
    lines = [l.strip() for l in f if l.strip()]

entries = lines[:200]

d = Dataset.from_dict({"text": entries})

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  

tok = tokenizer(entries, padding=True, truncation=True, return_tensors="pt")
print(tok["input_ids"].shape)  
```

```python
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.requires_grad_(False)  

for head in model.transformer.h[-3:]:
    orig_c_attn = head.attn.c_attn
    orig_c_proj = head.attn.c_proj

    head.attn.c_attn = LoraLayer(orig_c_attn.weight.shape[0], orig_c_attn.weight.shape[1], r=4, alpha=1.0)
    head.attn.c_proj = LoraLayer(orig_c_proj.weight.shape[0], orig_c_proj.weight.shape[1], r=4, alpha=1.0)

    head.attn.c_attn.weight.data.copy_(orig_c_attn.weight.t())
    head.attn.c_proj.weight.data.copy_(orig_c_proj.weight.t())
    head.attn.c_attn.bias.data.copy_(orig_c_attn.bias)
    head.attn.c_proj.bias.data.copy_(orig_c_proj.bias)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
```