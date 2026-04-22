# LoRA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS

![LoRA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS - Abstract](assets/paper.png)

## 1. Background

Large language models such as GPT, LLaMA and other LLMs are built with billions of parameters. 
These parameters are the values the model learns during training, and they allow the model to understand and generate language.

A common way to adapt a pretrained model to a new task is full fine-tuning. 
In full fine-tuning, every parameter in the model is updated using task-specific training data. 
This can work well, but it creates several practical problems. Large models need a huge amount of GPU memory during training. 
They also require extra memory for optimizer states, which makes training even more expensive. 
In addition, each fine-tuned version of the model must store a full new set of weights, so checkpoint files become large. 
If we want separate models for many tasks, deployment and storage costs grow.

LoRA stands for **Low-Rank Adaptation**. It was introduced as a more efficient way to adapt large pretrained models. 
Instead of changing all of the original model weights, LoRA keeps the pretrained weights frozen and learns only a small update. 
This update is represented using two low-rank matrices, which together approximate the changes needed for the new task. 
Because the rank is small, the number of trainable parameters is much lower than in full fine-tuning.

This design gives some advantages. First, training becomes cheaper because only a small number of parameters need to be updated. 
Second, memory usage is reduced because fewer gradients and optimizer states must be stored. 
Third, the final task-specific checkpoint is much smaller, which makes it easier to save, share, and deploy. 
Finally, the original pretrained model stays unchanged, so the same base model can support many different LoRA adapters for different tasks.