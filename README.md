# Neural Network Tests

Educational neural network implementations demonstrating fundamental machine learning concepts using NumPy and PyTorch. Includes logical gate implementations and linear regression from scratch.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows
pip install matplotlib numpy torch
```

## Usage

```bash
python main.py  # Interactive menu
```

## Project Structure

```
neural-network-tests/
├── main.py                     # Interactive menu
├── linear-regression/          # Linear regression implementation
├── logical-gate-and/           # AND gate implementations
├── logical-gate-or/            # OR gate implementations  
├── logical-gate-xor/           # XOR gate implementations
├── logical-gate-xnor/          # XNOR gate implementations
├── logical-gate-not/           # NOT gate implementations
├── logical-gate-nand/          # NAND gate implementations
└── README.md
```

**Direct Links:**
- [AND Gate](./logical-gate-and/) - Simple neural networks (start here)
- [OR Gate](./logical-gate-or/) - Basic classification
- [NOT Gate](./logical-gate-not/) - Single input networks
- [NAND Gate](./logical-gate-nand/) - Inverted logic
- [XOR Gate](./logical-gate-xor/) - Non-linear problems
- [XNOR Gate](./logical-gate-xnor/) - Complex architectures
- [Linear Regression](./linear-regression/) - Mathematical foundations

## Architecture Guide

| Gate | Layers | Architecture | Reason |
|------|--------|--------------|--------|
| AND, OR, NOT, NAND | 1 | 2→1 or 1→1 | Linearly separable |
| XOR, XNOR | 2+ | 2→4→1 | Not linearly separable |

**Rule**: Start simple, add complexity only if needed.

## Learning Modules

### 1. Simple Logical Gates ([`logical-gate-and/`](./logical-gate-and/), [`logical-gate-or/`](./logical-gate-or/), [`logical-gate-not/`](./logical-gate-not/), [`logical-gate-nand/`](./logical-gate-nand/))
Perfect introduction to neural networks using familiar boolean logic:
- **Binary Classification**: Clear inputs (0/1) and outputs (True/False)
- **Immediate Intuition**: Everyone understands AND, OR, NOT operations
- **Single Layer Networks**: Linearly separable problems need minimal architecture
- **Multiple Implementations**: Both NumPy and PyTorch versions
- **Quick Wins**: Fast training, clear success metrics

**Key Learning**: How neural networks learn simple patterns and basic classification.

### 2. Complex Logical Gates ([`logical-gate-xor/`](./logical-gate-xor/), [`logical-gate-xnor/`](./logical-gate-xnor/))
Introduction to non-linear problems and deeper architectures:
- **Non-Linear Separability**: Why some problems need hidden layers
- **Architecture Design**: When to add complexity to your network
- **Historical Significance**: The XOR problem that drove neural network research
- **Multi-Layer Networks**: Introduction to hidden layers and non-linear activation

**Key Learning**: Why network depth matters and when simple solutions aren't enough.

### 3. Linear Regression ([`linear-regression/`](./linear-regression/))
Advanced foundation module with deep mathematical understanding:
- **Gradient Descent Mastery**: Complete mathematical derivation from first principles
- **Continuous Outputs**: Move beyond binary classification to regression
- **Cost Function Theory**: Deep dive into MSE and optimization landscapes
- **Mathematical Rigor**: Full calculus derivations showing the "why" behind algorithms
- **Vectorized Implementation**: Efficient NumPy operations at scale

**Key Learning**: Mathematical foundations of optimization - the theory behind all neural networks.

## Getting Started

### Quick Demo (Interactive Menu)
```bash
python main.py
```
*Note: The main menu currently covers [logical gates](./logical-gate-and/). [Linear regression](./linear-regression/) can be run directly from its folder.*

## Educational Philosophy

This project follows a **"foundations-first"** approach to understanding neural networks:

### Core Concepts Covered
- **Gradient Descent**: The optimization engine behind all neural networks
- **Cost Functions**: How we measure and improve model performance  
- **Matrix Operations**: Efficient vectorized computations
- **Architecture Design**: Matching network complexity to problem complexity
- **Mathematical Foundations**: Complete derivations showing the "why" behind algorithms

### Why This Approach?
1. **Build Intuition**: Start with interpretable problems (logical gates, then linear regression)
2. **Show the Math**: Complete mathematical derivations demystify the algorithms
3. **Multiple Implementations**: Compare pure NumPy vs PyTorch to understand abstractions
4. **Hands-On Learning**: Run code, see results, modify parameters
5. **Progression**: From simple binary classification to complex architectures to mathematical mastery

### Learning Outcomes
After completing these modules, you'll understand:
- How neural networks actually learn (gradient descent)
- When to use simple vs complex architectures
- The mathematical foundations of machine learning
- How high-level frameworks (PyTorch) relate to the underlying mathematics
- Why certain problems need deeper networks than others

**Perfect for**: Students, educators, or anyone wanting to understand neural networks from first principles.
