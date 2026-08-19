
markdown[![arXiv](https://img.shields.io/badge/arXiv-coming%20soon-b31b1b.svg)](https://arxiv.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000.svg)](https://github.com/psf/black)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)
markdown[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

# su3-gauge-gnn
PyTorch implementation of SU(3) Gauge-Covariant Resonance Graph Neural Networks.
markdown## Quickstart
```python
from su3_gauge import SU3GaugeBlock

block = SU3GaugeBlock(in_channels=8, out_channels=16, num_edges=100)
Z_out, U, diag = block(Z, edge_index)
# su3-gauge-gnn

A PyTorch implementation of SU(3) Gauge-Covariant Resonance Graph Neural Networks.

This library provides a message-passing layer where node features transform in the fundamental representation of SU(3): Z_i ∈ C^(F × 3). Edge-wise gauge fields U_ij ∈ SU(3) are learned via the Cayley transform and used for parallel transport.

The core idea is a gauge-invariant "resonance" score that weights messages between nodes, analogous to attention but respecting local gauge symmetry.

### Key Features
- Gauge Covariance: Guaranteed by construction using U_ij Z_j transport
- Resonance Attention: R_ij = |<U_ij Z_j, Z_i>| / (|| ||) weights messages invariantly
- Gauge-Preserving Nonlinearity: Activations applied to magnitude only, preserving SU(3) direction
- Complex-Valued Layers: Full support for torch.complex64 features
- Self-Tests: Built-in tests for unitarity, det(U)=1, and gauge equivariance

### Installation
`bash
pip install torch torch_geometric
git clone https://github.com/yourname/su3-gauge-gnn
