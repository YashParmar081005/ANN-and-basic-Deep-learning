📌 1. What is Deep Learning?

A subset of Machine Learning that uses neural networks with many layers (deep).

Learns patterns from large datasets using representation learning.

Good for: Images, audio, text, time series, classification, regression, generation.

🧩 2. Artificial Neural Network (ANN) Basics
Structure

Input Layer – receives data

Hidden Layers – extract features

Output Layer – prediction

Neuron

Performs:
Weighted Sum → Add Bias → Activation Function

Equation
𝑦
=
𝑓
(
𝑤
1
𝑥
1
+
𝑤
2
𝑥
2
+
.
.
.
+
𝑏
)
y=f(w
1
	​

x
1
	​

+w
2
	​

x
2
	​

+...+b)
⚡ 3. Activation Functions (Must Know)
Function	Formula	Use-case
ReLU	max(0, x)	Hidden layers, fast, reduces vanishing gradient
Sigmoid	1/(1+e^-x)	Binary classification
Tanh	(e^x − e^-x)/(e^x + e^-x)	Zero-centered → better than sigmoid
Softmax	e^x / sum(e^x)	Multiclass outputs
📈 4. Loss Functions
Classification

Binary Cross-Entropy

Categorical Cross-Entropy

Regression

MSE (Mean Squared Error)

MAE (Mean Absolute Error)

🏋️‍♂️ 5. Training a Neural Network
Forward Propagation

Data flows input → hidden → output

Prediction is generated

Loss Calculation

Compare prediction vs actual

Backward Propagation

Calculate gradients

Update weights using gradient descent

Optimization Algorithms

SGD – Simple but slow

Momentum – Faster

Adam – Most used (adaptive learning rates)

🔧 6. Important Hyperparameters

Learning Rate → controls how fast weights update

Epochs → full passes over dataset

Batch Size → samples processed at once

Hidden Layers & Neurons → model capacity

Dropout Rate → prevents overfitting

📉 7. Overfitting & Underfitting
Overfitting

Model memorizes data.
Fixes: Regularization, dropout, more data.

Underfitting

Model too simple.
Fixes: More layers, more epochs.

🏗 8. Types of Neural Networks
1. CNN (Convolutional Neural Network)

Used for images, videos

Performs convolution → pooling → classification

2. RNN (Recurrent Neural Network)

Used for sequential data: text, time series

Types: LSTM, GRU

3. Autoencoders

Compression + reconstruction

Used for dimensionality reduction & anomaly detection

4. GAN (Generative Adversarial Network)

Generator + Discriminator

Used for image generation, deepfakes

🧪 9. Train/Validation/Test Split

Train (70%)

Validation (15%)

Test (15%)

🧰 10. Deep Learning Workflow

Load data

Normalize/Preprocess

Design ANN architecture

Choose optimizer + loss

Train

Validate

Tune hyperparameters

Test & deploy

⚙️ 11. Popular Frameworks

TensorFlow (Keras)

PyTorch

JAX

FastAI
