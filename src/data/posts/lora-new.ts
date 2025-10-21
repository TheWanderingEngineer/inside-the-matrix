import type { Post } from "@/types/post";

const post: Post = {
  title: "How LoRA Shrinks the Training Matrix",
  summary: "Inside the low-rank adaptation that reshaped AI fine-tuning.",
  content: `
# Inside the Matrix: LoRA Explained
Low-Rank Adaptation (LoRA) is a training (fine-tuning) technique that is one of the most popular Parameter-Efficient Fine-Tuning (PEFT) methods.
It was introduced in the paper [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) by 
Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, and Weizhu Chen in 2021.\n
LoRA reduces the number of trainable parameters of a large model by freezing the orginal (pretrained) weights
and injecting trainable rank-decompostion matrices into some layers of the Transformer architecture. Wah wah wah,
too many jargons at once... Let's break it down, starting with the concept of "rank" in linear algebra.

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

Hence, the rank of $$X$$ is **2** (only two linearly independent rows). \n
One important note is that the rank of a matrix gives us an idea of "information content" of a matrix,
i.e., how many rows (or columns) are needed to represent the entire matrix. **Higher-rank matrices 
indicate more information content (less compressible), while lower-rank matrices indicate redundancy 
(more compressible), since with a few rows, we can represent the other rows, thus reconstructing the orginal
matrix.**
But how does this help in reducing parameters in LoRA?


## Matrix Decomposition
Matrix decomposition is the process of breaking down a matrix into a product of two or more matrices.
For example, the above matrix $$X$$ can be decomposed into two matrices smaller matrices $$A$$ and $$B$$ such that:
$$
X = A B
$$
where $$A$$ is a 5x2 ($$R \\times r$$) matrix and $$B$$ is a 2x5 ($$r \\times C$$) matrix representing
the rank-2 factorization of $$X$$, where R is the number of rows (5), C is the number of columns (5),
and r is the rank (2). The matrices $$A$$ and **B** are constructed 
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

In this example, we used full rank ($$r$$ = 2) here, meaning, the rank of the decomposition is the same as 
the rank of the matrix, however, we can use lower rank like $$r$$ = 1 even when the rank of the original
matrix is 2, though you should expect some loss of information (sometime can be neglibile).
We can clearly see that $$r$$ is a hyperparameter that determines the size of the decomposed matrices, and thus,
the number of parameters needed to represent the original matrix. \n
Let's see how this helps in reducing parameters in LoRA.

### Parameter Reduction
By choosing a smaller rank (or even the full rank), we can reduce the number of parameters significantly.
For example, the original matrix $$X$$ has 5x5 = 25 parameters, while the rank-2 factorization uses 
5x2 + 2x5 = 20 parameters only! That's already **20%** reduction in parameters even with full rank! \n
If we use rank-1 factorization, we get even more reduction with a total of 5x1 + 1x5 = 10 parameters
representing the entire $$X$$ matrix. We can expect much more reduction when dealing with larger matrices, like
the weight matrices in large language models, which can have millions (or even billions) of parameters.

## LoRA in Mechanism

In LoRA, we add small *low-rank adapters* (hence the name) to selected layers of a pretrained Transformer, then
fine-tune only these adapters while keeping the original weights frozen, these adapters are represented as
two low-rank matrices $$A$$ and $$B$$ that are multiplied together to form a low-rank update 
to the original weight matrix as shown earlier.
The rank $$r$$ is a hyperparameter that you can choose based on your compute and performance needs. Keep in mind
that lower $$r$$ means more parameter reduction but potentially more information loss, $$r$$ is usually set 
to a small value like 4 or 8, but can go higher up to 64 and beyond depending on the model size and task.\n
The image below is taken from the original LoRA paper showing how LoRA adaptors are injected into a pretrained model.

<p align="center">
  <img src="images/lora-diagram.png" alt="LoRA Matrix" width="600"/>
</p>

We can see that during fine-tuning, only these LoRA adaptor matrices are updates,
while the pretrained weights remain unchanged. This results in significant savings in terms of trainable parameters.

### How Original Weights and LoRA Adaptors Work Together
During training, the original weight matrix $$W$$ is frozen while the LoRA matrices $$A$$ and $$B$$ are
updated through backpropagation producing the the update, the low-rank $$\\Delta W = A B$$. During inference
(prediction), the original weights and the LoRA weights are combined together
to produce the final output:


$$
W_{\t{final}} = W + \\Delta W = W_{\t{pretrained}} + A B,
$$
$$
{where}\\ W \\in{R}^{d \\times d} \; A \\in {R}^{d \\times r},
$$
$$
B \\in {R}^{r \\times d},
$$
$$
r \\ll d
$$

Where $$W_{\t{final}}$$ is the final weight matrix that is used in 
the forward pass of model as usal, d (aka $$d_{model}$$) is the model dimension or hidden size (in BERT-Large d=1024)
, and $$AB$$ is the low-rank update from LoRA.\n
This way, the model benefits from both the pretrained knowledge inside $$W$$ and the task-specific 
adaptations learned through the LoRA matrices $$A$$ and $$B$$ during fine-tuning.


### LoRA Matrices Initialization

The LoRA matrices $$A$$ and $$B$$ are usually initialized such that the product $$AB$$
is close to zero at the start of training.
This is often done by initializing $$A$$ with small random values (e.g. samples from a normal distribution with
zero-mean and some variance) and $$B$$ with zeros, ensuring that the initial output of the LoRA adaptors does 
not significantly alter the behavior of the pretrained model, as shown in the LoRA diagram found in
in the original paper, and shown above.\n
The matrix $$A$$ is chosen to have small values to prevent large initial and sudden updates, 
while $$B$$ is set to zero to ensure that the **initial contribution** of the LoRA adaptors is negligible. We
can't have both matrices initialized with zeros because then the gradients would also zero out,
preventing any learning from happening. We can swap the initialization (A=0, B=random) but
this is less common.

### What to LoRA

LoRA can be applied to different weight matrices across various layers of the Transformer architecture,
including:
- Attention projection matrices (query, key, value, output)
- Feed-forward network weights
- Cross-attention layers (Common in Stable Diffusion fine-tuning)
- Other linear layers within the model

The choice of which to LoRA depends on the model architecture, available resources, and the specific task at hand.
The original paper applied LoRA to the query $$W_Q$$ and value $$W_v$$ projection matrices in the attention layers
which was shown to be very effective.\n

### LoRA Inside Transformers
We'll take BERT-Large as an example to see LoRA placement. BERT-Large has 24 layers (Transformer blocks) and 
345 million parameters, this means when we want to fine-tune BERT-Large, we need to update all 345M parameters,
and store gradients for all these parameters during training (that's multiple terabytes), 
which can be very resource-intensive, and this is considered small in today's standards. \n
The image below shows vanilla BERT architecture (One encoder block). Some Transformers 
parts are omitted for simplicity.
<p align="center">
  <img src="images/bert.png" alt="one-block-bert" width="600"/>
</p>
Now let's see where LoRA adaptors are placed inside BERT.

<p align="center">
  <img src="images/bert-lora.png" alt="one-block-bert-lora" width="600"/>
</p>

These adaptors shown in the image above are the low-rank matrices $$A$$ and $$B$$, which are basically 
small linear layers added in parallel to the original weight matrices (Across all heads) in the attention
and feed-forward layers. Now instead of updating all 345M parameters during fine-tuning, we only update the LoRA
adaptors, which can be as low as 1-2% of the total parameters, depending on the selected rank $$r$$. This results in
huge savings in terms of memory and computation during training. **Note that during inference, the LoRA weights are 
merged with the original weights, so there is no additional latency introduced during prediction, which is 
a big advantage of LoRA over other PEFT techniques.**
  `,
  tags: ["Compression", "Fine-Tuning", "LoRA"],
  image: "images/lora.png",
  date: "2025-10-24",
  readTime: "30 min read"

};
export default post;        // <-- no "id" here
