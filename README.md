# Linear Regression with Gradient Descent

## Problem Statement
Implement gradient descent optimization from scratch to fit a linear model to synthetic data, demonstrating understanding of calculus-based optimization and responsible AI integration.

## Mathematical Approach

**Dataset Generation:**
- Synthetic data: y = 2.5x + 5.0 + ε, where ε ~ N(0, 4)
- 150 samples with Gaussian noise to simulate real-world data imperfection
- Creates file named data_plot.png to visualize data plot points compared to the true line

**Algorithm:**
- Hypothesis: h(x) = θ₀ + θ₁·x
- Cost Function (MSE): J(θ) = (1/2m) Σ(h(xᵢ) - yᵢ)²
- Gradient Formulas (derived via calculus):
  - ∂J/∂θ₀ = (1/m) Σ(h(xᵢ) - yᵢ)
  - ∂J/∂θ₁ = (1/m) Σ(h(xᵢ) - yᵢ) · xᵢ
- Update Rule: θⱼ := θⱼ - α · ∂J/∂θⱼ

**Hyperparameters:**
- Learning rate (α): 0.02 (chosen through experimentation)
- Iterations: 2000 (ensures convergence)
- Convergence criterion: Cost stabilization

## AI Integration
Used Claude API (Anthropic) to generate a detailed explanation of the gradient descent algorithm. Rather than having AI write the implementation, AI was used for:
- **Mathematical articulation:** Explaining how cost function, gradients, and update rules work together
- **Documentation support:** Providing clear, expert-level explanation of the optimization process
- **Validation of reasoning:** Confirming algorithmic understanding through explanation request

This demonstrates responsible AI usage: AI supports understanding and documentation rather than replacing implementation reasoning.

## Results
- True parameters: θ₀ = 5.0, θ₁ = 2.5
- Learned parameters: θ₀ = 4.7018, θ₁ = 2.5267
- Final cost: 1.7612
- Successfully recovered parameters with high accuracy despite noise

## Installation & Running

**Requirements:**
```bash
pip3 install numpy anthropic python-dotenv
```

**Setup:**
1. Create your graph plotting the data points
```bash
Python3 data_generator.py
```
2. Create `.env` file with your Anthropic API key:
```
   ANTHROPIC_API_KEY=your-key-here
```

**Run:**
```bash
Python3 main.py
```

**Project Structure:**
- `data_generator.py` - Synthetic linear dataset generation
- `algorithm.py` - Gradient descent implementation
- `ai_assistant.py` - Claude API integration
- `main.py` - Main execution script
