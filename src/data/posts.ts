// Blog posts data structure
// To add a new post: Add a new object to the posts array below

export interface BlogPost {
  id: string;
  title: string;
  summary: string;
  content: string;
  tags: string[];
  image: string;
  date: string;
  readTime: string;
}

export const posts: BlogPost[] = [
  {
    id: "1",
    title: "Understanding Neural Networks: A Deep Dive",
    summary: "Explore the fundamental concepts of neural networks and how they power modern AI applications.",
    content: `
# Understanding Neural Networks: A Deep Dive

Neural networks are the backbone of modern artificial intelligence. In this article, we'll explore how these powerful computational models work and why they're so effective.

## What is a Neural Network?
 Einstein’s formula: $E = mc^2$
A neural network is a computational model inspired by the human brain's structure. It consists of interconnected nodes (neurons) organized in layers:

- **Input Layer**: Receives the raw data
- **Hidden Layers**: Process and transform the data
- **Output Layer**: Produces the final prediction

## Key Components

### Neurons and Weights

Each neuron performs a weighted sum of its inputs and applies an activation function. The weights are learned during training through backpropagation.

### Activation Functions

Common activation functions include:

\`\`\`python
# ReLU (Rectified Linear Unit)
def relu(x):
    return max(0, x)

# Sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Tanh
def tanh(x):
    return np.tanh(x)
\`\`\`

## Training Process

The training process involves:

1. Forward propagation
2. Loss calculation
3. Backward propagation
4. Weight updates

## Applications

Neural networks power:
- Image recognition
- Natural language processing
- Speech recognition
- Autonomous vehicles
- Medical diagnosis

## Conclusion

Neural networks represent a paradigm shift in how we approach problem-solving with computers. Their ability to learn from data makes them invaluable for complex tasks.
    `,
    tags: ["Deep Learning", "Neural Networks", "AI"],
    image: "images/neural-network.jpg",
    date: "2024-01-15",
    readTime: "8 min read"
  },
  {
    id: "2",
    title: "The Rise of Transformers in NLP",
    summary: "How attention mechanisms revolutionized natural language processing and gave birth to models like GPT and BERT.",
    content: `
# The Rise of Transformers in NLP

The introduction of the Transformer architecture in 2017 revolutionized natural language processing. Let's explore why this architecture became the foundation for modern language models.

## The Attention Mechanism

The core innovation of Transformers is the **self-attention mechanism**, which allows the model to weigh the importance of different words in a sentence.

### How Attention Works

\`\`\`python
def attention(Q, K, V):
    """
    Q: Query matrix
    K: Key matrix
    V: Value matrix
    """
    scores = Q @ K.T / sqrt(d_k)
    weights = softmax(scores)
    return weights @ V
\`\`\`

## Key Advantages

1. **Parallelization**: Unlike RNNs, Transformers can process all tokens simultaneously
2. **Long-range Dependencies**: Attention mechanisms capture relationships between distant words
3. **Scalability**: Easily scale to billions of parameters

## Famous Transformer Models

### GPT Series
- Generative Pre-trained Transformers
- Autoregressive language models
- Powers ChatGPT and similar applications

### BERT
- Bidirectional Encoder Representations
- Excellent for understanding tasks
- Used in search engines and classification

## Impact on Industry

Transformers have enabled:
- Advanced chatbots and virtual assistants
- High-quality machine translation
- Code generation tools
- Content creation assistance

## The Future

The Transformer architecture continues to evolve with:
- More efficient attention mechanisms
- Multimodal capabilities
- Improved reasoning abilities

The future of AI is being written in the language of attention.
    `,
    tags: ["NLP", "Transformers", "GPT", "BERT"],
    image: "images/transformers.jpg",
    date: "2024-01-20",
    readTime: "10 min read"
  },
  {
    id: "3",
    title: "Machine Learning Model Deployment Best Practices",
    summary: "Learn the essential strategies for deploying ML models to production environments efficiently and reliably.",
    content: `
# Machine Learning Model Deployment Best Practices

Deploying machine learning models to production is a critical step that requires careful planning and execution. Here's your comprehensive guide.

## Pre-Deployment Checklist

### Model Validation
- Test on diverse datasets
- Measure performance metrics
- Validate edge cases
- Check for bias

### Infrastructure Requirements
- Compute resources
- Storage capacity
- Network bandwidth
- Monitoring systems

## Deployment Strategies

### 1. Containerization

Using Docker for consistent environments:

\`\`\`dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY model.pkl .
COPY app.py .

CMD ["python", "app.py"]
\`\`\`

### 2. API Development

Create robust APIs for model serving:

\`\`\`python
from fastapi import FastAPI
import joblib

app = FastAPI()
model = joblib.load('model.pkl')

@app.post("/predict")
async def predict(data: InputData):
    prediction = model.predict(data.features)
    return {"prediction": prediction}
\`\`\`

### 3. Monitoring

Essential metrics to track:
- Prediction latency
- Throughput
- Error rates
- Model drift
- Resource utilization

## CI/CD for ML

Implement automated pipelines:
1. Data validation
2. Model training
3. Model testing
4. Deployment
5. Monitoring

## Scaling Considerations

### Horizontal Scaling
- Load balancing
- Multiple replicas
- Auto-scaling policies

### Optimization Techniques
- Model quantization
- Batch prediction
- Caching strategies

## Security Best Practices

- Input validation
- Authentication and authorization
- Encrypted communications
- Regular security audits

## Conclusion

Successful ML deployment requires attention to:
- Infrastructure design
- Monitoring and logging
- Scalability planning
- Security measures

Remember: A model is only valuable when it's successfully deployed and serving predictions.
    `,
    tags: ["MLOps", "Deployment", "Production", "DevOps"],
    image: "images/deployment.jpg",
    date: "2024-01-25",
    readTime: "12 min read"
  },
  {
  id: "4",
  title: "Exploring the Matrix",
  summary: "Dive into the hidden layers of reality and AI.",
  content: `
# Exploring the Matrix

Welcome to the Matrix...

## The Real World

Here's an image inside the article:

![Neo entering the Matrix](images/matrix-neo.jpg)

Or using HTML for size control:

<img src="images/matrix-neo.jpg" alt="Neo entering" width="500"/>

More text here...

## Code Glitch Example

\`\`\`python
print("Wake up, Neo.")
\`\`\`
  `,
  tags: ["AI", "Matrix", "Philosophy"],
  image: "images/matrix-cover.jpg",
  date: "2025-10-17",
  readTime: "6 min read"
},
{
  id: "lora-article",
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
linearly independent means that no row (or column) can be represented as a linear combination of the others.
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

Hence, the rank of **X** is **2** (only two linearly independent rows). \n
One important note is that the rank of a matrix gives us an idea of "information content" of a matrix,
i.e., how many rows (or columns) are needed to represent the entire matrix. **Higher-rank matrices 
indicate more information content (less compressible), while lower-rank matrices indicate redundancy 
(more compressible), since with a few rows, we can represent the other rows, thus reconstructing the orginal
matrix.**\n
But how does this help in reducing parameters in LoRA?


## Matrix Decomposition
Matrix decomposition is the process of breaking down a matrix into a product of two or more matrices.
For example, the above matrix **X** can be decomposed into two matrices smaller matrices **A** and **B** such that:
$$
X = A B
$$
where **A** is a (Rxr) 5x2 matrix and **B** is a 2x5 (rxC) matrix representing
the rank-2 factorization of **X**, where R is the number of rows (5), C is the number of columns (5),
and r is the rank (2). The matrices **A** and **B** are constructed 
such that when we multiply **A** and **B**, we get back the original matrix **X**:
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

In this example, we used full rank (**r** = 2) here, meaning, the rank of the decomposition is the same as 
the rank of the matrix, however, we can use lower rank like **r** = 1 even when the rank of the original
matrix is 2, though you should expect some loss of information (sometime can be neglibile).
We can clearly see that **r** is a hyperparameter that determines the size of the decomposed matrices, and thus,
the number of parameters needed to represent the original matrix. \n
Let's see how this helps in reducing parameters in LoRA.

### Parameter Reduction
By choosing a smaller rank (or even the full rank), we can reduce the number of parameters significantly.
For example, the original matrix **X** has 5x5 = 25 parameters, while the rank-2 factorization uses 
5x2 + 2x5 = 20 parameters only! That's already **20%** reduction in parameters even with full rank! \n
If we use rank-1 factorization, we get even more reduction with a total of 5x1 + 1x5 = 10 parameters
representing the entire **X** matrix. We can expect much more reduction when dealing with larger matrices, like
the weight matrices in large language models, which can have millions (or even billions) of parameters.

## LoRA in Mechanism
In LoRA, we apply this concept of matrix decomposition to the weight matrices of certain layers in a 
pretrained Transformer model. Instead of updating the full weight matrix during fine-tuning, we freeze the
original, pretrained weights and inject low-rank matrices (A & B) that are trainable, where the rank **r**
is a hyperparameter that you can choose based on your compute and performance needs. Keep in mind
that lower **r** means more parameter reduction but potentially more information loss, **r** is usually set 
to a small value like 4 or 8, but can go higher up to 64 and beyond depending on the model size and task.\n
The image below is taken from the original LoRA paper showing how LoRA adaptors are injected into a pretrained model.

<p align="center">
  <img src="images/lora-diagram.png" alt="LoRA Matrix" width="600"/>
</p>

We can see that during fine-tuning, only these LoRA adaptor matrices are updates,
while the pretrained weights remain unchanged. This results in significant savings in terms of trainable parameters.

### How Original Weights and LoRA Adaptors Work Together
During training, the original weight matrix $$W$$ is frozen while the LoRA matrices $$A$$ and $$B$$ are
updated through backpropagation producing the low-rank $$\\Delta W = A B$$. During inference
(prediction), the original weights and the LoRA weights are combined together
to produce the final output:


$$
W_{\t{final}} = W + \\Delta W = W_{\t{pretrained}} + A B,
$$
$$
{where}\\ W \\in{R}^{d_{\t{out}}\\times\t d_{\t{in}}} \; A \\in \m{R}^{d_{\t{out}}\\times\t r},
$$
$$
B \\in {R}^{r \\times\t d_{\t{in}}} \; 
r \\ll \\min(d_{\t{out}}, d_{\t{in}})
$$

where $$W_{\t{final}}$$ is the original weight matrix which are the weights used in 
the forward pass of model as usal, $$\\Delta W$$ is the update,
and **AB** is the low-rank update from LoRA.\n
This way, the model benefits from both the pretrained knowledge in **W** and the task-specific adaptations learned
through the LoRA matrices $$A$$ and $$B$$.


### LoRA Matrices Initialization
The LoRA matrices **A** and **B** are usually initialized such that the product **AB** 
is close to zero at the start of training.
This is often done by initializing **A** with small random values (e.g. samples from a normal distribution with
zero-mean and some variance) and **B** with zeros, ensuring that the initial output of the LoRA adaptors does 
not significantly alter the behavior of the pretrained model. \n
The matrix **A** is choosen to have small values to prevent large initial and sudden updates, 
while **B** is set to zero to ensure that the **initial contribution** of the LoRA adaptors is negligible. We
can't have both matrices initialized with zeros because then the gradients would also zero out,
preventing any learning from happening.\n

### What to LoRA
LoRA can be applied to different weight matrices across various layers of the Transformer architecture,
including:
- Attention projection matrices (query, key, value, output)
- Feed-forward network weights
- Cross-attention layers (Common in Stable Diffusion fine-tuning)
- Other linear layers within the model

The choice of which to LoRA depends on the model architecture, resources, and the specific task at hand.
The original paper applied LoRA to the query and value projection matrices in the attention layers
which was shown to be very effective.\n

  `,
  tags: ["Compression", "Fine-Tuning", "LoRA"],
  image: "images/lora.png",
  date: "2025-10-17",
  readTime: "6 min read"
}
];


// Markdown syntax you can use inside `content`:
//
// New line: \n
// # Heading 1
// ## Heading 2
// **bold**
// *italic*
// `inline code`
// ```python ... ```  ← for code blocks
// - bullet list
// 1. numbered list
// > quote
// [text](url) ← link
// ![alt](image-path) ← image v(if not working use full path: ![alt](/inside-the-matrix/images/lora-mat.png))

// centered image:
// <p align="center">
// <img src="images/lora-mat.png" alt="LoRA Matrix" width="500"/>
// </p>


// Math syntax (requires remark-math + rehype-katex):
// Inline: $E = mc^2$
// Block:
// $$
// \nabla_\theta L(\theta) = \frac{\partial L}{\partial \theta}
// $$

