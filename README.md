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
├── main.py                  # Interactive menu
├── linear-regression/       # Linear regression implementation
├── logical_gate_and/        # AND gate implementations
├── logical_gate_or/         # OR gate implementations  
├── logical_gate_xor/        # XOR gate implementations
├── logical_gate_xnor/       # XNOR gate implementations
├── logical_gate_not/        # NOT gate implementations
├── logical_gate_nand/       # NAND gate implementations
└── README.md
```

## Architecture Guide

| Gate | Layers | Architecture | Reason |
|------|--------|--------------|--------|
| AND, OR, NOT, NAND | 1 | 2→1 or 1→1 | Linearly separable |
| XOR, XNOR | 2+ | 2→4→1 | Not linearly separable |

**Rule**: Start simple, add complexity only if needed.

## Learning Modules

### Linear Regression (`linear-regression/`)
Foundation module demonstrating core machine learning concepts:
- **Gradient Descent Algorithm**: Core optimization technique used in neural networks
- **Cost Function Minimization**: Shows how algorithms learn from data
- **Mathematical Derivation**: Complete derivation from MSE to gradient formula
- **Pure NumPy Implementation**: Understanding without high-level frameworks
- **Visualization**: See the learning process in action

**Key Learning**: Understanding gradient-based optimization - the foundation of all neural networks.

### Logical Gates (`logical_gate_*/`)
Neural network implementations for basic logical operations:
- **Educational Progression**: From simple (AND) to complex (XOR) problems
- **Architecture Comparison**: When to use simple vs complex networks  
- **Multiple Implementations**: Both NumPy and PyTorch versions
- **Optimization Techniques**: Weight initialization, learning rates, early stopping

**Key Learning**: How network architecture relates to problem complexity.

## Getting Started

### Recommended Learning Path

1. **Start with Linear Regression** (`linear-regression/`):
   ```bash
   cd linear-regression
   python LinearRegressionTest1.py
   ```
   - Understand gradient descent fundamentals
   - See mathematical derivations in action
   - Visualize the learning process

2. **Simple Logical Gates** (`logical_gate_and/`, `logical_gate_or/`):
   ```bash
   cd logical_gate_and
   python AndLogicalGateTest2.py  # Optimal PyTorch version
   ```
   - Learn basic neural network architecture
   - See how networks learn simple patterns

3. **Complex Logical Gates** (`logical_gate_xor/`):
   ```bash
   cd logical_gate_xor  
   python XorLogicalGateTest1.py
   ```
   - Understand why some problems need deeper networks
   - See non-linearly separable problems in action

### Quick Demo (Interactive Menu)
```bash
python main.py
```
*Note: The main menu currently covers logical gates. Linear regression can be run directly from its folder.*

## Educational Philosophy

This project follows a **"foundations-first"** approach to understanding neural networks:

### Core Concepts Covered
- **Gradient Descent**: The optimization engine behind all neural networks
- **Cost Functions**: How we measure and improve model performance  
- **Matrix Operations**: Efficient vectorized computations
- **Architecture Design**: Matching network complexity to problem complexity
- **Mathematical Foundations**: Complete derivations showing the "why" behind algorithms

### Why This Approach?
1. **Build Intuition**: Start with interpretable problems (linear regression, logical gates)
2. **Show the Math**: Complete mathematical derivations demystify the algorithms
3. **Multiple Implementations**: Compare pure NumPy vs PyTorch to understand abstractions
4. **Hands-On Learning**: Run code, see results, modify parameters
5. **Progression**: From simple (linear) to complex (non-linear) problems

### Learning Outcomes
After completing these modules, you'll understand:
- How neural networks actually learn (gradient descent)
- When to use simple vs complex architectures
- The mathematical foundations of machine learning
- How high-level frameworks (PyTorch) relate to the underlying mathematics
- Why certain problems need deeper networks than others

**Perfect for**: Students, educators, or anyone wanting to understand neural networks from first principles.
