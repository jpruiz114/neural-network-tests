# AND Gate Tests

This folder contains multiple implementations of the AND gate using different approaches and optimizations. Each version demonstrates different neural network concepts and training strategies.

## Files Overview

### PyTorch Implementations
- `AndLogicalGateTest1.py` - PyTorch implementation with 2→4→1 architecture
- `AndLogicalGateTest2.py` - PyTorch implementation with optimal 2→1 architecture

### NumPy Implementations (numpy folder)
- `numpy/AndLogicalGateTest1.py` - Original NumPy implementation
- `numpy/AndLogicalGateTest1_Fast.py` - Optimized with smart initialization
- `numpy/AndLogicalGateTest1_Simple.py` - Sample-by-sample training approach
- `numpy/AndLogicalGateTest1_Optimized.py` - Best practices implementation

## Key Differences Summary

| Feature | Original | Fast | Simple | Optimized |
|---------|----------|------|--------|-----------|
| **Weight Init** | Random (-1 to 1) | Smart (1, 1) | Smart (2, 2) | Smart (2, 2) |
| **Bias Init** | Random (0 to 1) | Smart (-1.5) | Smart (-3.0) | Smart (-3.0) |
| **Learning Rate** | 0.1 (slow) | 2.0 (fast) | 1.0 (good) | 1.0 (good) |
| **Max Iterations** | 10,000 | 1,000 | 500 | 1,000 |
| **Progress Monitoring** | ❌ No | ✅ Every 50 | ✅ Every 25 | ✅ Every 100 |
| **Early Stopping** | ❌ No | ✅ Yes | ✅ Yes | ✅ Yes |
| **Training Method** | Batch | Batch | Sample-by-sample | Batch |
| **Convergence** | ~Never | ~Never | ~400 epochs | ~400 epochs |

## Implementation Details

### PyTorch Implementations

#### AndLogicalGateTest1.py (Overcomplicated)
- **Purpose**: PyTorch implementation with unnecessary complexity
- **Architecture**: 2→4→1 with ReLU (overkill for AND gate)
- **Learning**: Shows that complex networks can solve simple problems
- **Note**: Works but inefficient

#### AndLogicalGateTest2.py (Optimal)
- **Purpose**: PyTorch implementation with optimal architecture
- **Architecture**: 2→1 (perfect for linearly separable AND gate)
- **Features**: Uses PyTorch's automatic differentiation
- **Learning**: Shows the right architecture for the problem

### NumPy Implementations (numpy folder)

#### AndLogicalGateTest1.py (Original)
- **Purpose**: Basic NumPy implementation
- **Issues**: Poor initialization, slow learning rate, no monitoring
- **Learning**: Shows what NOT to do

#### AndLogicalGateTest1_Fast.py (Fast)
- **Purpose**: Smart initialization approach
- **Features**: Better starting weights, higher learning rate
- **Learning**: Importance of good initialization

#### AndLogicalGateTest1_Simple.py (Simple)
- **Purpose**: Sample-by-sample training
- **Features**: Different update strategy, fastest convergence
- **Learning**: Batch vs sample-by-sample training

#### AndLogicalGateTest1_Optimized.py (Optimized)
- **Purpose**: Best practices implementation
- **Features**: All optimizations combined
- **Learning**: Production-ready neural network training

## Architecture Comparison

| Implementation | Architecture | Parameters | Efficiency | Learning Value |
|----------------|--------------|------------|------------|----------------|
| **AndLogicalGateTest1.py** | 2→4→1 | 17 | Low | Shows overcomplicated approach |
| **AndLogicalGateTest2.py** | 2→1 | 3 | High | Shows optimal approach |
| **NumPy versions** | 2→1 | 3 | High | Shows manual implementation |

**Key Insight**: AND gate is linearly separable, so it only needs a single perceptron (2→1). The 2→4→1 architecture works but is unnecessarily complex.

## Key Learning Points

1. **Architecture Matters**: Choose the simplest architecture that can solve the problem
2. **Weight Initialization Matters**: Starting close to the solution dramatically improves convergence
3. **Learning Rate is Critical**: Too low = slow, too high = unstable
4. **Early Stopping Prevents Overfitting**: Stop when converged, not after fixed iterations
5. **Progress Monitoring is Essential**: See what's happening during training
6. **Different Training Methods**: Batch vs sample-by-sample have different characteristics

## Running the Tests

```bash
# Run PyTorch implementations
python AndLogicalGateTest1.py
python AndLogicalGateTest2.py

# Run NumPy implementations
python numpy/AndLogicalGateTest1.py
python numpy/AndLogicalGateTest1_Fast.py
python numpy/AndLogicalGateTest1_Simple.py
python numpy/AndLogicalGateTest1_Optimized.py

# Or run from main directory
python main.py
```

## Expected Results

All implementations should converge to approximately:
- Input [0,0] → Output ~0.000
- Input [0,1] → Output ~0.000  
- Input [1,0] → Output ~0.000
- Input [1,1] → Output ~1.000

The key difference is **how fast** they reach this solution and **how much monitoring** they provide during training.
