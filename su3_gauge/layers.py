"""
SU(3) Gauge-Covariant Resonance Graph Layer
===========================================

Research implementation of an SU(3)-covariant complex-valued GNN layer.

Mathematical structure
----------------------

Node state:

    Z_i in C^(F x 3)

where:
    F = latent feature dimension
    3 = fundamental SU(3) color representation

Gauge field on directed edge j -> i:

    U_ij = Cayley(A_ij)

with

    A_ij = i * sum_a phi_ij^a T_a

and

    T_a = lambda_a / 2

being the eight Gell-Mann generators.

Gauge transformation:

    Z_i -> G_i Z_i

    U_ij -> G_i U_ij G_j^dagger

Parallel transport:

    Z_j -> U_ij Z_j

Gauge-invariant resonance:

    R_ij =
        | <U_ij Z_j, Z_i> |
        --------------------------------
        U_ij Z_j Z_i + eps

Since U_ij is unitary:

    U_ij Z_j = Z_j

The layer computes:

    Z_i' =
        residual(Z_i)
        +
        self_projection(Z_i)
        +
        sum_j
            norm_ij *
            R_ij *
            U_ij *
            message_projection(Z_j)

The layer does NOT claim physical QCD dynamics.
It implements a mathematically motivated SU(3)-covariant
message-passing architecture.

Author: Universe Zero Research
"""

from future import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 1. GELL-MANN GENERATORS
# =============================================================================

def get_gell_mann_matrices(
    *,
    dtype: torch.dtype = torch.complex64,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Return SU(3) generators T_a = lambda_a / 2.

    Normalization:

        Tr(T_a T_b) = 1/2 delta_ab

    Each generator is Hermitian and traceless.
    """

    T = torch.zeros(
        8,
        3,
        3,
        dtype=dtype,
        device=device,
    )

    # T1
    T[0, 0, 1] = 0.5
    T[0, 1, 0] = 0.5

    # T2
    T[1, 0, 1] = -0.5j
    T[1, 1, 0] = 0.5j

    # T3
    T[2, 0, 0] = 0.5
    T[2, 1, 1] = -0.5

    # T4
    T[3, 0, 2] = 0.5
    T[3, 2, 0] = 0.5

    # T5
    T[4, 0, 2] = -0.5j
    T[4, 2, 0] = 0.5j

    # T6
    T[5, 1, 2] = 0.5
    T[5, 2, 1] = 0.5

    # T7
    T[6, 1, 2] = -0.5j
    T[6, 2, 1] = 0.5j

    # T8
    T[7, 0, 0] = 1.0 / (2.0 * math.sqrt(3.0))
    T[7, 1, 1] = 1.0 / (2.0 * math.sqrt(3.0))
    T[7, 2, 2] = -1.0 / math.sqrt(3.0)

    return T


# =============================================================================
# 2. SU(3) GAUGE FIELD
# =============================================================================

class SU3GaugeField(nn.Module):
    """
    Trainable SU(3) gauge field living on directed graph edges.

    Each edge carries eight real coefficients:

        phi_ij in R^8

    which parameterize the su(3) Lie algebra.

    Transport:

        U_ij = Cayley(i * phi_ij^a T_a)

    The Cayley transform maps anti-Hermitian generators
    to unitary matrices without explicitly using matrix_exp.
    """

    def init(
        self,
        num_edges: int,
        init_scale: float = 1e-2,
        dtype: torch.dtype = torch.complex64,
    ):
        super().init()

        self.num_edges = num_edges

        self.phi = nn.Parameter(
            torch.randn(num_edges, 8) * init_scale
        )

        self.register_buffer(
            "T",
            get_gell_mann_matrices(dtype=dtype),
            persistent=False,
        )

    def lie_algebra_element(
        self,
        phi: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Construct

            A = i * phi^a T_a

        where A is anti-Hermitian.
        """

        if phi is None:
            phi = self.phi

        generator = torch.einsum(
            "ea,abc->ebc",
            phi.to(self.T.dtype),
            self.T,
        )

        return 1j * generator

    @staticmethod
    def cayley(A: torch.Tensor) -> torch.Tensor:
        """
        Cayley transform:
U = (I - A/2)^(-1) (I + A/2)

        For anti-Hermitian A, U is unitary.
        """

        E = A.shape[0]

        I = torch.eye(
            A.shape[-1],
            dtype=A.dtype,
            device=A.device,
        ).unsqueeze(0).expand(E, -1, -1)

        left = I - 0.5 * A
        right = I + 0.5 * A

        return torch.linalg.solve(left, right)

    def transport_matrix(
        self,
        phi: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Return U_ij in SU(3).
        """

        A = self.lie_algebra_element(phi)

        U = self.cayley(A)

        return U

    @staticmethod
    def unitarity_error(U: torch.Tensor) -> torch.Tensor:
        """
        Mean Frobenius deviation from U U^dagger = I.
        """

        I = torch.eye(
            3,
            dtype=U.dtype,
            device=U.device,
        )

        error = (
            U @ U.conj().transpose(-2, -1) - I
        )

        return torch.linalg.matrix_norm(error).mean()

    @staticmethod
    def determinant_error(U: torch.Tensor) -> torch.Tensor:
        """
        Mean |det(U) - 1|.
        """

        return torch.abs(
            torch.linalg.det(U) - 1.0
        ).mean()


# =============================================================================
# 3. SU(3) RESONANCE LAYER
# =============================================================================

class SU3ResonanceLayer(nn.Module):
    """
    Gauge-covariant SU(3) message-passing layer.

    Input:

        Z : [N, F_in, 3]

    Output:

        Z_out : [N, F_out, 3]

    The last dimension transforms under the fundamental representation
    of SU(3).

    The feature dimension is an ordinary learnable latent space.
    """

    def init(
        self,
        in_channels: int,
        out_channels: int,
        *,
        dropout: float = 0.0,
        residual: bool = True,
        resonance_power: float = 1.0,
        eps: float = 1e-8,
    ):
        super().init()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.residual = residual
        self.resonance_power = resonance_power
        self.eps = eps

        # ---------------------------------------------------------------
        # Feature-space projections.
        #
        # These act on F but DO NOT mix the SU(3) color index.
        # ---------------------------------------------------------------

        scale = 1.0 / math.sqrt(max(in_channels, 1))

        self.W_msg = nn.Parameter(
            torch.randn(
                in_channels,
                out_channels,
                dtype=torch.complex64,
            ) * scale * 0.05
        )

        self.W_self = nn.Parameter(
            torch.randn(
                in_channels,
                out_channels,
                dtype=torch.complex64,
            ) * scale * 0.05
        )

        # ---------------------------------------------------------------
        # Residual projection if feature dimensions differ.
        # ---------------------------------------------------------------

        if residual and in_channels != out_channels:
            self.W_residual = nn.Parameter(
                torch.randn(
                    in_channels,
                    out_channels,
                    dtype=torch.complex64,
                ) * scale * 0.05
            )
        else:
            self.W_residual = None

        self.dropout = (
            nn.Dropout(dropout)
            if dropout > 0.0
            else nn.Identity()
        )

        self.norm = nn.LayerNorm(
            out_channels,
            elementwise_affine=True,
        )

    # -----------------------------------------------------------------
    # Feature projection
    # -----------------------------------------------------------------

    def project_features(
        self,
        Z: torch.Tensor,
        W: torch.Tensor,
    ) -> torch.Tensor:
        """
        Z: [N, F_in, 3]
        W: [F_in, F_out]

        Returns:
            [N, F_out, 3]
        """
return torch.einsum(
            "nfc,fg->ngc",
            Z,
            W,
        )

    # -----------------------------------------------------------------
    # SU(3) parallel transport
    # -----------------------------------------------------------------

    @staticmethod
    def parallel_transport(
        Z_src: torch.Tensor,
        U: torch.Tensor,
    ) -> torch.Tensor:
        """
        Z_src: [E, F, 3]
        U:     [E, 3, 3]

        Returns:
            [E, F, 3]
        """

        return torch.einsum(
            "eab,efb->efa",
            U,
            Z_src,
        )

    # -----------------------------------------------------------------
    # Gauge-invariant resonance
    # -----------------------------------------------------------------

    def compute_resonance(
        self,
        Z_transported: torch.Tensor,
        Z_dst: torch.Tensor,
    ) -> torch.Tensor:
        """
        Gauge-invariant overlap:

            R =
                |<Z_transported, Z_dst>|
                -----------------------
                Z_transported
                Z_dst

        Result:

            [E, F]
        """

        inner = torch.sum(
            torch.conj(Z_dst) * Z_transported,
            dim=-1,
        )

        numerator = torch.abs(inner)

        denominator = (
            torch.linalg.vector_norm(
                Z_transported,
                dim=-1,
            )
            *
            torch.linalg.vector_norm(
                Z_dst,
                dim=-1,
            )
            + self.eps
        )

        R = numerator / denominator

        # Numerical protection.
        return torch.clamp(R, 0.0, 1.0)

    # -----------------------------------------------------------------
    # Symmetric graph normalization
    # -----------------------------------------------------------------

    @staticmethod
    def edge_normalization(
        edge_index: torch.Tensor,
        num_nodes: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:

        src, dst = edge_index

        deg = torch.zeros(
            num_nodes,
            dtype=dtype,
            device=device,
        )

        ones = torch.ones(
            dst.shape[0],
            dtype=dtype,
            device=device,
        )

        deg.index_add_(0, dst, ones)

        deg_inv_sqrt = torch.rsqrt(
            deg.clamp_min(1.0)
        )

        return (
            deg_inv_sqrt[src]
            *
            deg_inv_sqrt[dst]
        )

    # -----------------------------------------------------------------
    # Gauge-invariant activation
    # -----------------------------------------------------------------

    def gauge_activation(
        self,
        Z: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply a nonlinear transformation only to gauge-invariant
        amplitudes while preserving the SU(3) direction.

        This avoids applying an ordinary element-wise activation
        independently to color components.
        """

        magnitude = torch.linalg.vector_norm(
            Z,
            dim=-1,
            keepdim=True,
        )

        direction = Z / (
            magnitude + self.eps
        )

        new_magnitude = (
            F.softplus(magnitude)
            + 1e-3
        )

        return direction * new_magnitude

    # -----------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------

    def forward(
        self,
        Z: torch.Tensor,
        edge_index: torch.Tensor,
        U: torch.Tensor,
        *,
        return_diagnostics: bool = False,
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        """
        Parameters
        ----------

        Z:
            Complex node states [N, F_in, 3].

        edge_index:
            Directed edges [2, E].
            Each edge means src -> dst.

        U:
            SU(3) transport matrices [E, 3, 3].

        Returns
        -------
Z_out:
            [N, F_out, 3]

        diagnostics:
            Optional dictionary containing resonance statistics.
        """

        if not Z.is_complex():
            raise TypeError(
                "SU3ResonanceLayer requires a complex-valued Z tensor."
            )

        if Z.ndim != 3 or Z.shape[-1] != 3:
            raise ValueError(
                "Expected Z with shape [N, F, 3]."
            )

        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(
                "edge_index must have shape [2, E]."
            )

        src, dst = edge_index
        N = Z.shape[0]

        # ---------------------------------------------------------------
        # 1. Feature projections
        # ---------------------------------------------------------------

        Z_msg = self.project_features(
            Z,
            self.W_msg,
        )

        Z_self = self.project_features(
            Z,
            self.W_self,
        )

        # ---------------------------------------------------------------
        # 2. Symmetric degree normalization
        # ---------------------------------------------------------------

        norm = self.edge_normalization(
            edge_index,
            N,
            dtype=Z.real.dtype,
            device=Z.device,
        )

        # ---------------------------------------------------------------
        # 3. Parallel transport
        # ---------------------------------------------------------------

        Z_src = Z_msg[src]

        Z_src_transported = self.parallel_transport(
            Z_src,
            U,
        )

        Z_dst = Z_msg[dst]

        # ---------------------------------------------------------------
        # 4. Gauge-invariant resonance
        # ---------------------------------------------------------------

        resonance = self.compute_resonance(
            Z_src_transported,
            Z_dst,
        )

        if self.resonance_power != 1.0:
            resonance = resonance.pow(
                self.resonance_power
            )

        # ---------------------------------------------------------------
        # 5. Message weighting
        # ---------------------------------------------------------------

        msg = (
            Z_src_transported
            *
            resonance.unsqueeze(-1)
            *
            norm.view(-1, 1, 1)
        )

        msg = self.dropout(msg)

        # ---------------------------------------------------------------
        # 6. Aggregation
        # ---------------------------------------------------------------

        aggregated = torch.zeros(
            N,
            self.out_channels,
            3,
            dtype=Z.dtype,
            device=Z.device,
        )

        aggregated.index_add_(
            0,
            dst,
            msg,
        )

        # ---------------------------------------------------------------
        # 7. Self + interaction
        # ---------------------------------------------------------------

        out = Z_self + aggregated

        # ---------------------------------------------------------------
        # 8. Residual connection
        # ---------------------------------------------------------------

        if self.residual:

            if self.W_residual is None:
                out = out + Z
            else:
                residual = self.project_features(
                    Z,
                    self.W_residual,
                )
                out = out + residual

        # ---------------------------------------------------------------
        # 9. Gauge-preserving activation
        # ---------------------------------------------------------------

        out = self.gauge_activation(out)
# ---------------------------------------------------------------
        # 10. Feature normalization
        #
        # We normalize magnitude over feature channels, not color.
        # This avoids breaking the SU(3) representation.
        # ---------------------------------------------------------------

        magnitude = torch.linalg.vector_norm(
            out,
            dim=-1,
        )

        magnitude = self.norm(
            magnitude.real
        )

        scale = (
            magnitude
            /
            (
                torch.linalg.vector_norm(
                    out,
                    dim=-1,
                )
                + self.eps
            )
        )

        out = out * scale.unsqueeze(-1)

        # ---------------------------------------------------------------
        # Diagnostics
        # ---------------------------------------------------------------

        diagnostics = None

        if return_diagnostics:

            diagnostics = {
                "mean_resonance":
                    resonance.mean().detach(),

                "std_resonance":
                    resonance.std().detach(),

                "mean_state_norm":
                    torch.linalg.vector_norm(
                        out,
                        dim=-1,
                    ).mean().detach(),

                "mean_message_norm":
                    torch.linalg.vector_norm(
                        msg,
                        dim=-1,
                    ).mean().detach(),
            }

        return out, diagnostics


# =============================================================================
# 4. CURVATURE / WILSON LOOP
# =============================================================================

def wilson_loop(
    U_ij: torch.Tensor,
    U_jk: torch.Tensor,
    U_ki: torch.Tensor,
) -> torch.Tensor:
    """
    Ordered Wilson loop:

        W = U_ij U_jk U_ki

    Returns:

        [3, 3]

    For a correctly oriented closed loop, the trace is gauge invariant.
    """

    return (
        U_ij
        @ U_jk
        @ U_ki
    )


def wilson_loop_trace(
    U_ij: torch.Tensor,
    U_jk: torch.Tensor,
    U_ki: torch.Tensor,
) -> torch.Tensor:
    """
    Gauge-invariant Wilson observable:

        W = Re Tr(U_ij U_jk U_ki)
    """

    W = wilson_loop(
        U_ij,
        U_jk,
        U_ki,
    )

    return torch.real(
        torch.trace(W)
    )


def wilson_loop_curvature_loss(
    U_ij: torch.Tensor,
    U_jk: torch.Tensor,
    U_ki: torch.Tensor,
) -> torch.Tensor:
    """
    Soft curvature penalty.

    This is NOT required for gauge covariance.

    It encourages low holonomy:

        U_ij U_jk U_ki ~ I

    and therefore acts as a smoothness / low-curvature prior.
    """

    W = wilson_loop(
        U_ij,
        U_jk,
        U_ki,
    )

    I = torch.eye(
        3,
        dtype=W.dtype,
        device=W.device,
    )

    return torch.linalg.matrix_norm(
        W - I
    ).pow(2)


# =============================================================================
# 5. COMPLETE SU(3) GAUGE BLOCK
# =============================================================================

class SU3GaugeBlock(nn.Module):
    """
    Complete trainable SU(3) gauge-covariant graph block.

    Components:

        1. SU(3) gauge field
        2. Parallel transport
        3. Gauge-invariant resonance
        4. Feature mixing
        5. Residual dynamics
        6. Gauge-preserving activation

    This class intentionally keeps curvature regularization outside
    the forward pass. The training loop can decide whether curvature
    should be penalized.
    """

    def init(
        self,
        in_channels: int,
        out_channels: int,
        num_edges: int,
        *,
        residual: bool = True,
        resonance_power: float = 1.0,
        dropout: float = 0.0,
    ):
        super().init()

        self.gauge = SU3GaugeField(
            num_edges=num_edges
        )
self.layer = SU3ResonanceLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            residual=residual,
            resonance_power=resonance_power,
            dropout=dropout,
        )

    def forward(
        self,
        Z: torch.Tensor,
        edge_index: torch.Tensor,
        *,
        return_diagnostics: bool = False,
    ):
        U = self.gauge.transport_matrix()

        Z_out, diagnostics = self.layer(
            Z,
            edge_index,
            U,
            return_diagnostics=return_diagnostics,
        )

        return Z_out, U, diagnostics

    def gauge_diagnostics(
        self,
        U: torch.Tensor,
    ) -> dict:
        """
        Numerical diagnostics for SU(3) transport.
        """

        return {
            "unitarity_error":
                SU3GaugeField.unitarity_error(U),

            "determinant_error":
                SU3GaugeField.determinant_error(U),
        }


# =============================================================================
# 6. GAUGE COVARIANCE TEST
# =============================================================================

@torch.no_grad()
def gauge_covariance_test(
    layer: SU3ResonanceLayer,
    Z: torch.Tensor,
    edge_index: torch.Tensor,
    U: torch.Tensor,
    G: torch.Tensor,
) -> float:
    """
    Numerical covariance test.

    Gauge transformation:

        Z_i' = G_i Z_i

        U_ij' =
            G_dst U_ij G_src^dagger

    Expected:

        Layer(Z', U') =
            G_dst Layer(Z, U)

    Returns relative error.
    """

    src, dst = edge_index

    # Transform node states.
    Z_g = torch.einsum(
        "nab,nfb->nfa",
        G,
        Z,
    )

    # Transform edge links.
    U_g = torch.einsum(
        "dab,eab,sec->edc",
        G[dst],
        U,
        G[src].conj().transpose(-2, -1),
    )

    # Original output.
    out, _ = layer(
        Z,
        edge_index,
        U,
    )

    # Gauge transformed output.
    out_g, _ = layer(
        Z_g,
        edge_index,
        U_g,
    )

    # Expected transformed output.
    expected = torch.einsum(
        "nab,nfb->nfa",
        G,
        out,
    )

    numerator = torch.linalg.vector_norm(
        out_g - expected
    )

    denominator = (
        torch.linalg.vector_norm(
            expected
        )
        + 1e-8
    )

    return (
        numerator / denominator
    ).item()


# =============================================================================
# 7. RANDOM SU(3) TRANSFORMATION
# =============================================================================

@torch.no_grad()
def random_su3(
    num_nodes: int,
    *,
    scale: float = 0.1,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Generate random local SU(3) transformations using
    the same Lie-algebra/Cayley construction.
    """

    T = get_gell_mann_matrices(
        device=device
    )

    phi = torch.randn(
        num_nodes,
        8,
        device=device,
    ) * scale

    A = 1j * torch.einsum(
        "na,abc->nbc",
        phi.to(T.dtype),
        T,
    )

    return SU3GaugeField.cayley(A)


# =============================================================================
# 8. SELF TEST
# =============================================================================

def run_self_test() -> None:

    torch.manual_seed(42)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    N = 32
    E = 96
    F_in = 8
    F_out = 16

    edge_index = torch.randint(
        0,
        N,
        (2, E),
        device=device,
    )

    Z = torch.randn(
        N,
        F_in,
        3,
        device=device,
        dtype=torch.complex64,
    )

    gauge = SU3GaugeField(
        num_edges=E
    ).to(device)

    layer = SU3ResonanceLayer(
        F_in,
        F_out,
        residual=True,
    ).to(device)
# ---------------------------------------------------------------
    # Transport
    # ---------------------------------------------------------------

    U = gauge.transport_matrix()

    unit_error = gauge.unitarity_error(U)
    det_error = gauge.determinant_error(U)

    # ---------------------------------------------------------------
    # Forward
    # ---------------------------------------------------------------

    Z_out, diagnostics = layer(
        Z,
        edge_index,
        U,
        return_diagnostics=True,
    )

    # ---------------------------------------------------------------
    # Gauge covariance
    # ---------------------------------------------------------------

    G = random_su3(
        N,
        device=device,
    )

    covariance_error = gauge_covariance_test(
        layer,
        Z,
        edge_index,
        U,
        G,
    )

    # ---------------------------------------------------------------
    # Report
    # ---------------------------------------------------------------

    print("=" * 70)
    print("SU(3) GAUGE-COVARIANT RESONANCE LAYER — SELF TEST")
    print("=" * 70)

    print(f"device:                 {device}")
    print(f"input state:            {tuple(Z.shape)}")
    print(f"output state:           {tuple(Z_out.shape)}")
    print(f"edges:                  {E}")

    print()
    print(f"unitarity error:        {unit_error.item():.3e}")
    print(f"det(U)-1 error:         {det_error.item():.3e}")
    print(f"equivariance error:     {covariance_error:.3e}")

    if diagnostics is not None:
        print()
        print(
            f"mean resonance:         "
            f"{diagnostics['mean_resonance'].item():.4f}"
        )

        print(
            f"resonance std:          "
            f"{diagnostics['std_resonance'].item():.4f}"
        )

        print(
            f"mean state norm:        "
            f"{diagnostics['mean_state_norm'].item():.4f}"
        )

    print()
    print("PASS CONDITIONS")
    print(
        "  unitary:               ",
        unit_error.item() < 1e-4,
    )
    print(
        "  determinant:           ",
        det_error.item() < 1e-4,
    )
    print(
        "  gauge covariance:     ",
        covariance_error < 1e-4,
    )

    print("=" * 70)


if name == "main":
    run_self_test()
