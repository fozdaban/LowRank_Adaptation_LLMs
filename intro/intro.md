# LoRA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS

![LoRA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS - Abstract](assets/paper.png)

## 1. Background

Large language models such as GPT, LLaMA and other LLMs are built with billions of parameters. These parameters are the values the model learns during training, and they allow the model to understand and generate language. A common way to adapt a pretrained model to a new task is full fine-tuning, which updates every parameter in the model using task-specific training data.

This can work well, but it creates several practical problems. Large models need a huge amount of GPU memory during training, and they also require extra memory for optimizer states, which makes training even more expensive. Each fine-tuned version of the model must store a full new set of weights, so checkpoint files become large. As models are getting larger and larger with more parameters and tokens, this problem increases to a critical point hard to ignore.

LoRA stands for **Low-Rank Adaptation**. It was introduced as a more efficient way to adapt large pretrained models. Instead of changing all of the original model weights, LoRA keeps the pretrained weights frozen and learns only a small update. This update is represented using two low-rank matrices, which together approximate the changes needed for the new task. Because the rank is small, the number of trainable parameters is much lower than in full fine-tuning.

This design gives some advantages. Training becomes cheaper because only a small number of parameters need to be updated, memory usage is reduced, and the final task-specific checkpoint is much smaller, which makes it easier to save, share, and deploy. Finally, the original pretrained model stays unchanged, so the same base model can support many different LoRA adapters for different tasks.