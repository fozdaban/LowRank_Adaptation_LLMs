## 2. Problem Statement

LoRA's central claim is that the *update* $\Delta W$ learned during adaptation has low intrinsic rank, so it is wasteful to represent $\Delta W$ as a full matrix. Instead, LoRA factorizes it as $\Delta W = B A$ with $B \in \mathbb{R}^{d\times r}$, $A \in \mathbb{R}^{r\times k}$ and $r \ll \min(d, k)$. The pretrained $W_0$ is frozen; only $B$ and $A$ are trained.

Suppose we have a pretrained model $P_{\Phi_0}(y \mid x)$ with parameters $\Phi_0$, and we want to adapt it to a downstream dataset $\mathcal{Z} = \{(x_i, y_i)\}_{i=1}^N$. 

**Full fine-tuning** searches for a new parameter vector $\Phi_0 + \Delta\Phi$ by maximizing

$$
\max_{\Phi}\; \sum_{(x, y) \in \mathcal{Z}} \sum_{t=1}^{|y|} \log P_{\Phi}(y_t \mid x, y_{<t})
$$

and the update $\Delta\Phi$ has the same dimension as $\Phi_0$.

**Parameter-efficient adaptation** factorizes the update through a much smaller parameter set $\Theta$ with $|\Theta| \ll |\Phi_0|$:

$$
\Delta\Phi \;=\; \Delta\Phi(\Theta), \qquad
\max_{\Theta}\; \sum_{(x, y) \in \mathcal{Z}} \sum_{t=1}^{|y|} \log P_{\Phi_0 + \Delta\Phi(\Theta)}(y_t \mid x, y_{<t}).
$$

LoRA is one particular choice of the encoding $\Delta\Phi(\Theta)$: for each targeted pretrained matrix $W_0 \in \mathbb{R}^{d \times k}$ the update is

$$
\Delta W \;=\; B A, \qquad B \in \mathbb{R}^{d \times r}, \;\; A \in \mathbb{R}^{r \times k}, \;\; r \ll \min(d, k).
$$

At inference, the model sees a single effective matrix $W = W_0 + B A$, so there is no architectural overhead — more on that shortly.

The schematic below shows the two computational paths (frozen base and trainable low-rank update) that are combined at the output.

![Lora Image](assets/lora.png)

[2] 

