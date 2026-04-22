## 3. Why Existing Solutions Are Not Enough

Before LoRA, two broad families of parameter-efficient adaptation were already well known. The paper argues that each has a drawback LoRA is designed to avoid.

**Adapter layers**. Insert small bottleneck MLP modules between existing Transformer sublayers and train only those. They have few parameters, but they sit serially in the forward pass. In latency-sensitive, small-batch inference, even a few extra sequential matmuls add measurable latency — the paper reports up to ~20–30% slowdown on GPT-2 medium at batch size 1.

**Prompt / prefix tuning**. Prepend a learnable sequence of "soft" tokens to the input and train only those embeddings (or the activations they induce at each layer). This reserves part of the context window for adaptation, shortening the effective input available to the downstream task, and the paper reports that prefix tuning is difficult to optimize and that performance is non-monotone in the number of prefix tokens.