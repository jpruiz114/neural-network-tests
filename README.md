# Neural Network Tests

Neural network implementations for logical gates using NumPy and PyTorch.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows
pip install numpy torch
```

## Usage

```bash
python main.py  # Interactive menu
```

## Project Structure

```
neural-network-tests/
├── main.py                  # Interactive menu
├── logical_gate_and/        # AND gate implementations
├── logical_gate_or/         # OR gate implementations  
├── logical_gate_xor/        # XOR gate implementations
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
