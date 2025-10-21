import type { Post } from "@/types/post";

const post: Post = {
  title: "How LoRA Shrinks the Training Matrix",
  summary: "Inside the low-rank adaptation that reshaped AI fine-tuning.",
  content: `

Low-Rank Adaptation (LoRA) is a training (fine-tuning) technique that is one of the most popular Parameter-Efficient Fine-Tuning (PEFT) methods.
It was introduced in the paper [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) by 
Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, and Weizhu Chen in 2021.\n
LoRA reduces the number of trainable parameters of a large model by freezing the orginal (pretrained) weights
and injecting trainable rank-decompostion matrices into some layers of the Transformer architecture. Wah wah wah,

too many jargons at once... Let's break it down, starting with the concept of "rank" in linear algebra.

## Table of Contents
- [What is Rank?](#what-is-rank)
- [Matrix Decomposition](#matrix-decomposition)
- [Parameter Reduction](#parameter-reduction)
- [LoRA Mechanism](#lora-mechanism)
- [How All Weights Work Togather](#how-all-weights-work-togather)
- [LoRA Matrices Initialization](#lora-matrices-initialization)
- [What to LoRA?](#what-to-lora)
- [LoRA Inside Transformers](#lora-inside-transformers)

<a id="what-is-rank"></a>
## What is Rank?
The rank of a matrix is the smallest number of *linearly independent* rows (or columns) in the matrix.
linear independence means that no row (or column) can be represented as a linear combination of the others.
For example, consider the following matrix:
<p align="center">
  <img src="images/mat-rank-2.png" alt="LoRA Matrix" width="400"/>
</p>
\n
This matrix has a rank of 2 because there are two linearly independent rows,
meaning that we can represent the entire matrix using just 2 rows (or columns), these two are 
the first and second rows. With these two rows, we can construct the others, allowing to reconstruct the
the entire orginal matrix as shown below:

$$
X =
\\begin{bmatrix}
1 & 2 & 3 & 4 & 5 \\\\
0 & 1 & 1 & 2 & 3 \\\\
1 & 3 & 4 & 6 & 8 \\\\
2 & 3 & 5 & 6 & 7 \\\\
-1 & 2 & 1 & 4 & 7
\\end{bmatrix}
$$
We can express rows 3, 4, and 5 as linear combinations (weighted sum) of rows 1 and 2:
$$
\\begin{aligned}
\\text{Row}_3 &= \\text{Row}_1 + \\text{Row}_2, \\\\
\\text{Row}_4 &= 2\\,\\text{Row}_1 - \\text{Row}_2, \\\\
\\text{Row}_5 &= -\\text{Row}_1 + 4\\,\\text{Row}_2.
\\end{aligned}
$$

Hence, the rank($$X$$) = **2**; (only two inearly independent rows). \n
Another way to think about rank is that it measures the information content of a matrix, 
how much unique information it contains, i.e., how many rows (or columns) are needed to represent the entire matrix. 
> 📝 **Note:** Higher-rank matrices indicate more information content (less compressible), while lower-rank matrices indicate redundancy (more compressible), since with a few rows, we can represent the other rows, thus reconstructing the orginal matrix.\n
But how does this help in reducing parameters in LoRA?


<a id="matrix-decomposition"></a>
## Matrix Decomposition
Matrix decomposition is the process of breaking down a matrix into a product of two or more matrices.
For example, the above matrix $$X$$ can be decomposed into two matrices smaller matrices $$A$$ and $$B$$ such that:
$$
X = A B
$$
where $$A$$ is a 5x2 ($$R \\times r$$) matrix and $$B$$ is a 2x5 ($$r \\times C$$) matrix representing
the rank-2 factorization of $$X$$, where R is the number of rows (5), C is the number of columns (5),
and r is the rank (here r set to 2). The matrices $$A$$ and $$B$$ are constructed 
such that when we multiply $$A$$ and $$B$$, we get back the original matrix $$X$$:
Originally we have:

$$
X =
\\begin{bmatrix}
1 & 2 & 3 & 4 & 5 \\\\
0 & 1 & 1 & 2 & 3 \\\\
1 & 3 & 4 & 6 & 8 \\\\
2 & 3 & 5 & 6 & 7 \\\\
-1 & 2 & 1 & 4 & 7
\\end{bmatrix}
$$

After decomposition, we have:
$$
X = A B =
\\begin{bmatrix}
1 & 2 & 3 & 4 & 5 \\\\
0 & 1 & 1 & 2 & 3 \\\\
1 & 3 & 4 & 6 & 8 \\\\
2 & 3 & 5 & 6 & 7 \\\\
-1 & 2 & 1 & 4 & 7
\\end{bmatrix}
$$
$$
=
\\begin{bmatrix}
1 & 0 \\\\
0 & 1 \\\\
1 & 1 \\\\
2 & -1 \\\\
-1 & 4
\\end{bmatrix}
\\begin{bmatrix}
1 & 2 & 3 & 4 & 5 \\\\
0 & 1 & 1 & 2 & 3
\\end{bmatrix}
$$

In this example, we used rank $$r$$ = rank($$X$$) = 2, meaning, the rank of the decomposition is the same as 
the rank of the matrix, this makes the original matrix $$X$$ perfectly reconstructable from $$A$$ and $$B$$ without 
any loss of information. However, we can use lower rank like $$r$$ = 1 even when the rank of the original
matrix is 2, though you should expect some loss of information, fortunately, the loss is negligibile in some cases 
when $$r$$ is not too small and original matrix is inherently low-rank.
We can clearly see that $$r$$ is a hyperparameter we select that determines the size of the decomposed matrices, and thus,
the number of parameters needed to represent the original matrix. \n
Let's see how this helps in reducing parameters.

<a id="parameter-reduction"></a>
### Parameter Reduction
By choosing a smaller rank, we can reduce the number of parameters significantly.
For example, the original matrix $$X$$ has 5x5 = 25 parameters, while the rank-2 factorization uses 
5x2 + 2x5 = 20 parameters only! That's already **20%** reduction in parameters with relatively high rank
rank($$X$$) = $$r$$!\n
If we use rank-1 factorization, we get even more reduction with a total of 5x1 + 1x5 = 10 parameters
representing the entire $$X$$ matrix. We can expect much more reduction as the matrix gets larger, like
the weight matrices in large language models, which can have millions (or even billions) of parameters,
this idea of parameter reduction is the core concept behind LoRA.

<a id="lora-mechanism"></a>
## LoRA Mechanism

In LoRA, we add small *Low-Rank Adapters* (hence the name) to selected layers of a pretrained Transformer -which
is an archeticture we'll dig into in another post-, then
fine-tune only these adapters while keeping the original weights frozen, these adapters are represented as
two low-rank matrices $$A$$ and $$B$$ that are multiplied together to form a low-rank update 
to the original weight matrix as shown earlier.
The rank $$r$$ is a hyperparameter that you can choose based on your compute and performance needs. Keep in mind
that lower $$r$$ means more parameter reduction but potentially more information loss, $$r$$ is usually set 
to a small value like 4 or 8, but can go higher up to 64 and beyond depending on the model size, task and 
resources avaiable.\n
The image below is taken from the original LoRA paper showing how LoRA adapters are injected into a pretrained model.

<p align="center">
  <img src="images/lora-diagram.png" alt="LoRA Matrix" width="600"/>
</p>
**Assumption for this post:** all weight matrices are square $$(d \\times d$$) (i.e., $$(d_{in}=d_{out}=d$$)).
We can see that during fine-tuning, only these LoRA adapter matrices are updated,
while the pretrained weights remain unchanged. This results in significant savings in terms of trainable parameters.

<a id="how-all-weights-work-togather"></a>
### How All Weights Work Togather
During training, the original weight matrix $$W$$ is frozen while the LoRA matrices $$A$$ and $$B$$ are
updated through backpropagation producing the update of the low-rank matrices defined as the  $$\\Delta W = A B$$.
During inference (prediction), the original weights and the LoRA weights are combined together
to produce the final output:

$$
W_{\t{final}} = W + \\Delta W = W_{\t{pretrained}} + A B,
$$
$$
{where}\\ W \\in{R}^{d \\times d}, \\\\ 
A \\in {R}^{d \\times r},
$$
$$
B \\in {R}^{r \\times d},
$$
$$
r \\ll d
$$

Where $$W_{\t{final}}$$ is the final weight matrix that is used in 
the forward pass of model as usual, $$d$$ (aka $$d_{model}$$) is the model dimension or hidden size 
(in BERT-Large d=1024), and $$AB$$ is the low-rank update from LoRA.\n
This way, the model benefits from both the pretrained knowledge inside $$W$$ and the task-specific 
adaptations learned through the LoRA matrices $$A$$ and $$B$$ during fine-tuning.


<a id="lora-matrices-initialization"></a>
### LoRA Matrices Initialization

The LoRA matrices $$A$$ and $$B$$ are usually initialized such that the product $$AB$$
is close to zero at the start of training.
This is often done by initializing $$A$$ with small random values (e.g. samples from a normal distribution with
zero-mean and some variance $\\sigma^2$) and $$B$$ with zeros, ensuring that the **initial output**
of the LoRA adapters does 
not significantly change the behavior of the pretrained model, preventing Catastrophic Forgetting,
 as shown in the LoRA diagram found in
in the original paper, and shown above.\n
You might ask, why not just initialize both $$A$$ and $$B$$ with zeros? The answer is we 
can't have both matrices initialized with zeros because then the gradients would also zero out,
preventing any learning from happening. What about initializing both with small random values? 
The problem with that is the initial output $$AB$$ might not be close to zero, which can lead to large
initial changes to the model's behavior, potentially destorting the pretrained weights. It's
worth mentioning that we can swap the initialization (A=0, B=random) but
this is less common due to slower convergence observed in practice.

<a id="what-to-lora"></a>
### What to LoRA?

LoRA can be applied to different weight matrices across various layers of the Transformer architecture,
including:
- Attention projection matrices (query, key, value, output)
- Feed-forward network weights
- Cross-attention layers (Common in Stable Diffusion fine-tuning)
- Other linear layers within the model

The choice of which to LoRA depends on the model architecture, available resources, and the specific task at hand.
The original paper mainly applied LoRA to the query $$W_Q$$ and value $$W_V$$ projection matrices in the attention layers
which was shown to be very effective, in fact, as effective as full fine-tuning in some cases.\n

<a id="lora-inside-transformers"></a>
### LoRA Inside Transformers
We'll take BERT-Large as an example to see LoRA placement. BERT-Large has 24 layers (Transformer blocks) and 
345 million parameters, this means when we want to fine-tune BERT-Large, we need to update all 345M parameters
and store gradients for all these parameters during training (that's ~16-32 GB, we're GPU poor here folks), 
which can be very resource-intensive, and even this is considered small in today's standards. \n
The image below shows the vanilla BERT architecture (One encoder block). Some Transformer
parts are omitted for simplicity.
<p align="center">
  <img src="images/bert.png" alt="one-block-bert" width="600"/>
</p>

Now let's see where LoRA adapters are placed inside BERT.

<p align="center">
  <img src="images/bert-lora.png" alt="one-block-bert-lora" width="600"/>
</p>

These adapters shown in the image above are the low-rank matrices $$A$$ and $$B$$, which are basically 
small linear layers added in parallel to the original weight matrices (Across all heads) in the attention
and feed-forward layers. Now instead of updating all 345M parameters during fine-tuning, we only update the LoRA
adapters, which can be as low as 1-2% of the total parameters, depending on the selected rank $$r$$. This results in
huge savings in terms of memory and computation during training. \n
The plot below shows the relationship between the LoRA rank $$r$$ and the percentage of trainable parameters
when LoRA'ing (😎) the query $$W_Q$$ and value $$W_V$$ projection matrices across all 
24 layers of BERT-Large.

<p align="center">
  <img src="images/lora_params_vs_r.png" alt="params-vs-rank" width="600"/>
</p>

From this plot, we can see that trainable parameters increase *linearly* with the rank $$r$$,
which is expected since the number of parameters in LoRA for square matrices ($$d //times d)  is given by:

$$
\\{#Params}_{LoRA} = 2 \\times d_{model} \\times r \\times L_{LoRA}
$$
Where $$d_{model}$$ is the model dimension (1024 for BERT-Large), 
$$L_{LoRA}$$ is the number of target layers(or matrices) LoRA'ed (e.g., $$2x24=48$$ for $$W_Q$$ and $$W_V$$
across all BERT-Large 24 layers), and of course $$r$$ is the desired LoRA rank.\n
Also, we can see that even with a relatively high rank like $$r$$ = 64,
we only need to fine-tune less than 2% (~6M) of the total parameters, which is a huge reduction compared to
full fine-tuning, while performance remains comparable to full fine-tuning in many tasks. \n

> 📝 **Note:** During inference, LoRA can be merged into the base weights, 
(eliminating adapters and keeping size unchanged), with means no delay is introduced during inference.
 However, many setups don't merge, especially with *quantized* models, but even without merging since the matrices
 are low-rank, the additional computation is minimal, and practically there is no additional latency 
 introduced during prediction, which is a big advantage of LoRA over other PEFT techniques.

 # Key Takeaways


 # Test Your Understanding
 
  `,
  tags: ["Compression", "Fine-Tuning", "LoRA"],
  image: "images/lora.png",
  date: "2025-10-24",
  readTime: "30 min read"

};
export default post;
