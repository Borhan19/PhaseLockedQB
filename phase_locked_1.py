#!/usr/bin/env python3
"""
Canonical numerical and plotting code for:

    Phase-Locked Exponential Growth:
    High-SNR Quantum Energy Storage in Dissipative Broken Regimes

This file replaces the legacy scripts and directly generates the combined manuscript figures.  It uses one
consistent convention throughout:

    kappa * D[a]  ->  amplitude decay kappa / 2.

The direct dissipative model is parametrized nonredundantly by

    Gamma_col, theta, phi,

with normalized collective jump

    z = cos(theta) a + exp(i phi) sin(theta) b.

The code generates combined manuscript PDFs and supporting individual PDFs, never PNG files.  It uses the user's
preferred compact Nature-style plotting setup, calls plt.show(), never calls
plt.close(), prints important numerical values, and saves the plotted data.

Main commands
-------------
python phase_locked_paper_figures.py validate
python phase_locked_paper_figures.py stability
python phase_locked_paper_figures.py normalized-scans
python phase_locked_paper_figures.py dynamics
python phase_locked_paper_figures.py phase-locking
python phase_locked_paper_figures.py finite-capacity
python phase_locked_paper_figures.py gross-input
python phase_locked_paper_figures.py theta-tradeoff
python phase_locked_paper_figures.py detuning
python phase_locked_paper_figures.py modal-diagnostics
python phase_locked_paper_figures.py mediator
python phase_locked_paper_figures.py saturation
python phase_locked_paper_figures.py all-linear

The saturation calculation requires QuTiP.  All other calculations require only
NumPy, SciPy, Matplotlib, and pandas.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, Normalize, LogNorm
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
from scipy.linalg import eig
from scipy.optimize import brentq, minimize_scalar
from scipy.signal import find_peaks
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply


# =============================================================================
# Plot style
# =============================================================================

NPG_COLORS = (
    "#E64B35",
    "#4DBBD5",
    "#00A087",
    "#3C5488",
    "#F39B7F",
    "#8491B4",
    "#91D1C2",
    "#DC0000",
    "#7E6148",
)

NPG_DIVERGING = LinearSegmentedColormap.from_list(
    "npg_diverging",
    (NPG_COLORS[3], NPG_COLORS[1], "#FFFFFF", NPG_COLORS[4], NPG_COLORS[0]),
)
NPG_SEQUENTIAL = LinearSegmentedColormap.from_list(
    "npg_sequential",
    (NPG_COLORS[3], NPG_COLORS[1], NPG_COLORS[6], NPG_COLORS[4], NPG_COLORS[0]),
)


# =============================================================================
# Manuscript units and physical figure dimensions
# =============================================================================

# One common rate is used for every manuscript-facing axis and parameter table.
# The numerical propagation is performed in the original rates, while plots and
# exported dimensionless columns use:
#
#     kappa_0 = 0.01,
#     tau = kappa_0 t,
#     tilde{x} = x / kappa_0
#
# for every rate or frequency x.
REFERENCE_RATE = 1.0e-2
REFERENCE_RATE_SYMBOL = r"\kappa_0"

# REVTeX PRL reprint text is 10 pt.  Axis labels, tick labels, legends, panel
# labels, annotations, and colorbar text all use this same size.
MAIN_TEXT_FONT_SIZE = 10.0
FIG1_FONT_SIZE = 15.0
SINGLE_COLUMN_WIDTH = 3.37
DOUBLE_COLUMN_WIDTH = 6.00

# Final moderate-composition benchmark requested after the collaborator review.
# We keep the off-diagonal dissipative drift strength |g| fixed at the value used
# in the matched coherent benchmark, but move theta away from pi/2.  This raises
# the battery participation of the quasi-dark mode by orders of magnitude while
# retaining high ergotropy and SNR.
PAPER_MATCHED_G = 0.032
PAPER_THETA = 1.15
PAPER_GAMMA_COL = (
    2.0 * PAPER_MATCHED_G
    / (math.sin(PAPER_THETA) * math.cos(PAPER_THETA))
)

# Secondary stability benchmark requested for Fig. 1.  The off-diagonal
# dissipative strength |g| is kept identical to the main benchmark, so changing
# theta changes only the collective-mode composition and the Gamma_col required
# to realize the same |g|.
PAPER_THETA_SECONDARY = 1.30
PAPER_GAMMA_COL_SECONDARY = (
    2.0 * PAPER_MATCHED_G
    / (
        math.sin(PAPER_THETA_SECONDARY)
        * math.cos(PAPER_THETA_SECONDARY)
    )
)

# Exact endpoint used only as a stability reference in Fig. 1 panel (d).
# At theta = pi/2 the collective jump is z=b, so the off-diagonal
# dissipative coupling g=(Gamma_col/2) sin(theta) cos(theta) vanishes exactly.
# Therefore the fixed-|g| construction is singular at this endpoint.  We keep
# the finite reservoir strength from the theta=1.30 panel and rotate only the
# collective-mode composition to theta=pi/2.  Panel (d) is thus an endpoint
# stability reference, not a matched charging operating point.
PAPER_THETA_ENDPOINT = 0.5 * math.pi
PAPER_GAMMA_COL_ENDPOINT = PAPER_GAMMA_COL_SECONDARY

# Fig. 3 intentionally uses a different pair of colors from Fig. 2 because the
# curves represent protocol-level finite-capacity metrics rather than the energy
# decomposition shown in Fig. 2.
FIG3_COHERENT_COLOR = "#54BC88"
FIG3_DISSIPATIVE_COLOR = "#B088E4"

# End-Matter gross-input convention.  The incident coherent field is connected
# to the Hamiltonian drive amplitude through the usual input-output magnitude
# relation
#
#     |E|^2 = kappa_ext |a_in|^2.
#
# The default value kappa_ext = kappa_a_down is the fully external-coupling
# limit of the local charger damping.  The End-Matter figure also scans the
# physically allowed fraction kappa_ext/kappa_a_down, so the implementation
# dependence is explicit rather than hidden in a single arbitrary choice.
GROSS_INPUT_EXTERNAL_COUPLING_FRACTION = 1.0
GROSS_INPUT_SCAN_MIN_FRACTION = 0.10
GROSS_INPUT_SCAN_MAX_FRACTION = 1.00
GROSS_INPUT_SCAN_POINTS = 91
GROSS_INPUT_OMEGA_DRIVE_OVER_OMEGA_A = 1.0


def normalized_rate(value: Any) -> Any:
    return np.asarray(value) / REFERENCE_RATE


def normalized_time(value: Any) -> Any:
    return REFERENCE_RATE * np.asarray(value)


def normalized_power(value: Any, omega_b: float = 1.0) -> Any:
    return np.asarray(value) / (omega_b * REFERENCE_RATE)


def unit_metadata() -> dict[str, Any]:
    return {
        "reference_rate_symbol": "kappa_0",
        "reference_rate": REFERENCE_RATE,
        "dimensionless_time": "kappa_0 * t",
        "dimensionless_rate": "rate / kappa_0",
        "dimensionless_ergotropic_power": "P_W_star / (omega_b * kappa_0)",
        "main_text_font_size_pt": MAIN_TEXT_FONT_SIZE,
        "single_column_width_in": SINGLE_COLUMN_WIDTH,
        "double_column_width_in": DOUBLE_COLUMN_WIDTH,
    }


def normalized_two_mode_parameters(params: "TwoModeParams") -> dict[str, Any]:
    return {
        "kappa_a_down_over_kappa0": float(params.kappa_a_down / REFERENCE_RATE),
        "kappa_a_up_over_kappa0": float(params.kappa_a_up / REFERENCE_RATE),
        "kappa_b_over_kappa0": float(params.kappa_b / REFERENCE_RATE),
        "delta_over_kappa0": float(params.delta / REFERENCE_RATE),
        "drive_over_kappa0": complex(params.drive / REFERENCE_RATE),
        "J_over_kappa0": complex(params.coherent_J / REFERENCE_RATE),
        "Gamma_col_over_kappa0": float(params.Gamma_col / REFERENCE_RATE),
        "g_over_kappa0": complex(params.dissipative_g / REFERENCE_RATE),
        "theta": float(params.theta),
        "phi": float(params.phi),
    }


def _panel_label(ax: mpl.axes.Axes, label: str) -> None:
    ax.text(
        0.04,
        0.95,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=MAIN_TEXT_FONT_SIZE,
        color="black",
        zorder=20,
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.78,
            "pad": 0.6,
        },
    )


def _style_colorbar(colorbar: mpl.colorbar.Colorbar) -> None:
    colorbar.ax.tick_params(labelsize=MAIN_TEXT_FONT_SIZE)
    colorbar.ax.yaxis.label.set_size(MAIN_TEXT_FONT_SIZE)


def _choose_serif_font() -> str:
    """Use Times New Roman when available, with quiet professional fallbacks."""
    candidates = (
        "Times New Roman",
        "Times",
        "Nimbus Roman",
        "Nimbus Roman No9 L",
        "TeX Gyre Termes",
        "STIXGeneral",
        "DejaVu Serif",
    )
    for family in candidates:
        try:
            font_manager.findfont(family, fallback_to_default=False)
            return family
        except Exception:
            continue
    return "DejaVu Serif"


def configure_plot_style() -> str:
    chosen = _choose_serif_font()
    mpl.rcParams.update(
        {
            "text.usetex": False,
            "font.family": "serif",
            "font.serif": [chosen],
            "mathtext.fontset": "stix",
            "figure.max_open_warning": 0,
            "font.size": MAIN_TEXT_FONT_SIZE,
            "axes.labelsize": MAIN_TEXT_FONT_SIZE,
            "axes.titlesize": MAIN_TEXT_FONT_SIZE,
            "legend.fontsize": MAIN_TEXT_FONT_SIZE,
            "xtick.labelsize": MAIN_TEXT_FONT_SIZE,
            "ytick.labelsize": MAIN_TEXT_FONT_SIZE,
            "axes.linewidth": 1.0,
            "lines.linewidth": 1.5,
            "lines.markersize": 4.0,
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
            "xtick.minor.width": 0.7,
            "ytick.minor.width": 0.7,
            "xtick.major.size": 4.0,
            "ytick.major.size": 4.0,
            "xtick.minor.size": 2.5,
            "ytick.minor.size": 2.5,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "axes.prop_cycle": cycler(color=NPG_COLORS),
            "legend.frameon": False,
            "savefig.format": "pdf",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.01,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 160,
        }
    )
    print(f"Plot font: {chosen}")
    print(
        f"Manuscript font size: {MAIN_TEXT_FONT_SIZE:g} pt; "
        f"reference rate kappa_0={REFERENCE_RATE:g}"
    )
    return chosen


# =============================================================================
# Paths and serialization
# =============================================================================

@dataclass(frozen=True)
class OutputPaths:
    root: Path
    figures: Path
    data: Path
    summaries: Path
    diagnostics: Path

    @classmethod
    def create(cls, root: str | Path) -> "OutputPaths":
        root_path = Path(root).expanduser().resolve()
        figures = root_path / "Figs_New"
        data = root_path / "data"
        summaries = root_path / "summaries"
        diagnostics = root_path / "diagnostics"
        for directory in (
            root_path,
            figures,
            data,
            summaries,
            diagnostics,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        return cls(root_path, figures, data, summaries, diagnostics)


def _json_ready(value: Any) -> Any:
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


def save_json(payload: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(_json_ready(payload), indent=2), encoding="utf-8")
    print(f"Wrote: {path}")


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)
    print(f"Wrote: {path}")


def save_figure(
    fig: mpl.figure.Figure,
    path: Path,
    apply_tight_layout: bool = True,
) -> None:
    if path.suffix.lower() != ".pdf":
        raise ValueError("All manuscript figures must be saved as PDF.")
    if apply_tight_layout and not fig.get_constrained_layout():
        fig.tight_layout(pad=0.25)
    fig.savefig(
        path,
        bbox_inches="tight",
        pad_inches=0.01,
        transparent=False,
    )
    print(f"Wrote: {path}")
    plt.show()


# =============================================================================
# Gaussian toolbox
# =============================================================================

def symplectic_form(n_modes: int) -> np.ndarray:
    one_mode = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=float)
    return np.kron(np.eye(n_modes), one_mode)


def annihilation_quadrature_vector(coefficients: Sequence[complex]) -> np.ndarray:
    """Return l such that L = l^T r = sum_j coefficients[j] a_j."""
    coefficients = np.asarray(coefficients, dtype=complex)
    lvec = np.zeros(2 * len(coefficients), dtype=complex)
    for j, coefficient in enumerate(coefficients):
        lvec[2 * j] = coefficient / np.sqrt(2.0)
        lvec[2 * j + 1] = 1j * coefficient / np.sqrt(2.0)
    return lvec


def creation_quadrature_vector(coefficients: Sequence[complex]) -> np.ndarray:
    """Return l such that L = l^T r = sum_j coefficients[j] a_j^dagger."""
    coefficients = np.asarray(coefficients, dtype=complex)
    lvec = np.zeros(2 * len(coefficients), dtype=complex)
    for j, coefficient in enumerate(coefficients):
        lvec[2 * j] = coefficient / np.sqrt(2.0)
        lvec[2 * j + 1] = -1j * coefficient / np.sqrt(2.0)
    return lvec


def number_conserving_hamiltonian_matrix(h: np.ndarray) -> np.ndarray:
    """
    Convert H = c^dagger h c to H = 1/2 r^T H_R r, up to a constant.

    h must be Hermitian.  The off-diagonal block for h_ij is
        [[Re h_ij, -Im h_ij],
         [Im h_ij,  Re h_ij]].
    """
    h = np.asarray(h, dtype=complex)
    if h.ndim != 2 or h.shape[0] != h.shape[1]:
        raise ValueError("h must be a square matrix.")
    if not np.allclose(h, h.conj().T, atol=1e-12):
        raise ValueError("h must be Hermitian.")
    n_modes = h.shape[0]
    h_r = np.zeros((2 * n_modes, 2 * n_modes), dtype=float)
    for i in range(n_modes):
        for j in range(n_modes):
            hij = h[i, j]
            block = np.array(
                [[np.real(hij), -np.imag(hij)],
                 [np.imag(hij),  np.real(hij)]],
                dtype=float,
            )
            h_r[2 * i:2 * i + 2, 2 * j:2 * j + 2] = block
    return 0.5 * (h_r + h_r.T)


def drive_vector(n_modes: int, mode: int, amplitude: complex) -> np.ndarray:
    """
    H_drive = E a_mode^dagger + E* a_mode.
    The quadrature force vector is d=(sqrt(2) Re E, sqrt(2) Im E).
    """
    dvec = np.zeros(2 * n_modes, dtype=float)
    dvec[2 * mode] = np.sqrt(2.0) * np.real(amplitude)
    dvec[2 * mode + 1] = np.sqrt(2.0) * np.imag(amplitude)
    return dvec


def gaussian_matrices(
    h_r: np.ndarray,
    dvec: np.ndarray,
    lindblad_vectors: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For H = 1/2 r^T H_R r + d^T r and L_k = l_k^T r:

        dot m = A m + s
        dot V = A V + V A^T + D

        A = Omega H_R - Omega Im(Y)
        D = Omega Re(Y) Omega^T
        s = Omega d
        Y = sum_k l_k l_k^dagger
    """
    h_r = np.asarray(h_r, dtype=float)
    dvec = np.asarray(dvec, dtype=float)
    n_modes = h_r.shape[0] // 2
    omega = symplectic_form(n_modes)
    ymat = np.zeros_like(h_r, dtype=complex)
    for lvec in lindblad_vectors:
        lvec = np.asarray(lvec, dtype=complex)
        ymat += np.outer(lvec, lvec.conj())
    amat = omega @ h_r - omega @ np.imag(ymat)
    dmat = omega @ np.real(ymat) @ omega.T
    source = omega @ dvec
    amat = np.real_if_close(amat).astype(float)
    dmat = np.real_if_close(dmat).astype(float)
    dmat = 0.5 * (dmat + dmat.T)
    return amat, dmat, source


def factorize_kossakowski(kmat: np.ndarray, atol: float = 1e-12) -> np.ndarray:
    """
    Return R such that K = R^dagger R.  Rows of R are jump coefficients.
    Small negative eigenvalues caused by roundoff are clipped.
    """
    kmat = np.asarray(kmat, dtype=complex)
    kmat = 0.5 * (kmat + kmat.conj().T)
    values, vectors = np.linalg.eigh(kmat)
    if np.min(values) < -atol:
        raise ValueError(
            f"Kossakowski matrix is not positive semidefinite: min eigenvalue={np.min(values):.3e}"
        )
    values = np.clip(values, 0.0, None)
    keep = values > atol
    if not np.any(keep):
        return np.zeros((0, kmat.shape[0]), dtype=complex)
    return np.diag(np.sqrt(values[keep])) @ vectors[:, keep].conj().T


def simulate_constant_gaussian(
    amat: np.ndarray,
    dmat: np.ndarray,
    source: np.ndarray,
    times: np.ndarray,
    mean0: np.ndarray | None = None,
    covariance0: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Exact propagation of a constant linear Gaussian system by expm_multiply.

    The covariance is vectorized in Fortran order:
        vec(dot V) = (I kron A + A kron I) vec(V) + vec(D).
    """
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or times.size < 2:
        raise ValueError("times must be a one-dimensional array with at least two points.")
    if not np.all(np.diff(times) > 0.0):
        raise ValueError("times must be strictly increasing.")
    if not np.allclose(np.diff(times), np.diff(times)[0], rtol=1e-10, atol=1e-13):
        raise ValueError("Exact propagation currently requires an equally spaced time grid.")
    dim = amat.shape[0]
    if mean0 is None:
        mean0 = np.zeros(dim, dtype=float)
    if covariance0 is None:
        covariance0 = 0.5 * np.eye(dim, dtype=float)
    mean0 = np.asarray(mean0, dtype=float)
    covariance0 = np.asarray(covariance0, dtype=float)

    covariance_generator = np.kron(np.eye(dim), amat) + np.kron(amat, np.eye(dim))
    total_dim = dim + dim * dim + 1
    generator = np.zeros((total_dim, total_dim), dtype=float)
    generator[:dim, :dim] = amat
    generator[:dim, -1] = source
    start = dim
    stop = dim + dim * dim
    generator[start:stop, start:stop] = covariance_generator
    generator[start:stop, -1] = dmat.reshape(-1, order="F")

    initial = np.zeros(total_dim, dtype=float)
    initial[:dim] = mean0
    initial[start:stop] = covariance0.reshape(-1, order="F")
    initial[-1] = 1.0

    shifted_times = times - times[0]
    propagated = expm_multiply(
        csr_matrix(generator),
        initial,
        start=float(shifted_times[0]),
        stop=float(shifted_times[-1]),
        num=times.size,
        endpoint=True,
    )
    means = propagated[:, :dim]
    covariances = propagated[:, start:stop].reshape(
        times.size, dim, dim, order="F"
    )
    covariances = 0.5 * (covariances + np.swapaxes(covariances, 1, 2))
    return means, covariances


# =============================================================================
# Direct two-mode model
# =============================================================================

@dataclass(frozen=True)
class TwoModeParams:
    kappa_a_down: float = 0.01
    kappa_a_up: float = 0.0
    kappa_b: float = 0.01
    delta: float = 0.0
    drive: complex = 0.1 + 0.0j
    coherent_J: complex = 0.0 + 0.0j
    Gamma_col: float = 0.0
    theta: float = np.pi / 4.0
    phi: float = 0.0
    omega_a: float = 1.0
    omega_b: float = 1.0

    def validate(self) -> None:
        rates = (
            self.kappa_a_down,
            self.kappa_a_up,
            self.kappa_b,
            self.Gamma_col,
            self.omega_a,
            self.omega_b,
        )
        if any(rate < 0.0 for rate in rates):
            raise ValueError("All rates and oscillator frequencies must be nonnegative.")
        if not (0.0 <= self.theta <= np.pi / 2.0):
            raise ValueError("theta must lie in [0, pi/2].")

    @property
    def dissipative_g(self) -> complex:
        return (
            0.5
            * self.Gamma_col
            * np.exp(1j * self.phi)
            * np.sin(self.theta)
            * np.cos(self.theta)
        )

    @property
    def gamma_a(self) -> float:
        return 0.5 * (
            self.kappa_a_down
            - self.kappa_a_up
            + self.Gamma_col * np.cos(self.theta) ** 2
        )

    @property
    def gamma_b(self) -> float:
        return 0.5 * (
            self.kappa_b
            + self.Gamma_col * np.sin(self.theta) ** 2
        )


def two_mode_complex_drift(params: TwoModeParams) -> np.ndarray:
    params.validate()
    dissipative_g = params.dissipative_g
    return np.array(
        [
            [
                -params.gamma_a - 1j * params.delta,
                -1j * params.coherent_J - dissipative_g,
            ],
            [
                -1j * np.conjugate(params.coherent_J) - np.conjugate(dissipative_g),
                -params.gamma_b + 1j * params.delta,
            ],
        ],
        dtype=complex,
    )


def two_mode_gaussian_matrices(
    params: TwoModeParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    params.validate()
    h = np.array(
        [
            [params.delta, params.coherent_J],
            [np.conjugate(params.coherent_J), -params.delta],
        ],
        dtype=complex,
    )
    h_r = number_conserving_hamiltonian_matrix(h)
    dvec = drive_vector(2, 0, params.drive)

    jumps: list[np.ndarray] = []
    if params.kappa_a_down > 0.0:
        jumps.append(
            np.sqrt(params.kappa_a_down)
            * annihilation_quadrature_vector([1.0, 0.0])
        )
    if params.kappa_a_up > 0.0:
        jumps.append(
            np.sqrt(params.kappa_a_up)
            * creation_quadrature_vector([1.0, 0.0])
        )
    if params.kappa_b > 0.0:
        jumps.append(
            np.sqrt(params.kappa_b)
            * annihilation_quadrature_vector([0.0, 1.0])
        )
    if params.Gamma_col > 0.0:
        collective = [
            np.cos(params.theta),
            np.exp(1j * params.phi) * np.sin(params.theta),
        ]
        jumps.append(
            np.sqrt(params.Gamma_col)
            * annihilation_quadrature_vector(collective)
        )
    return gaussian_matrices(h_r, dvec, jumps)


def two_mode_stability_margin(params: TwoModeParams) -> float:
    return float(np.max(np.real(np.linalg.eigvals(two_mode_complex_drift(params)))))


def two_mode_ep_discriminant(params: TwoModeParams) -> complex:
    matrix = two_mode_complex_drift(params)
    return ((matrix[0, 0] - matrix[1, 1]) / 2.0) ** 2 + matrix[0, 1] * matrix[1, 0]


def dissipative_critical_pump_resonance(params: TwoModeParams) -> float:
    """Exact normalized threshold at delta=0 and coherent_J=0."""
    if abs(params.delta) > 1e-14 or abs(params.coherent_J) > 1e-14:
        raise ValueError("This closed form requires delta=0 and coherent_J=0.")
    numerator = (
        params.Gamma_col
        * params.kappa_b
        * np.cos(params.theta) ** 2
    )
    denominator = params.kappa_b + params.Gamma_col * np.sin(params.theta) ** 2
    return params.kappa_a_down + numerator / denominator


def find_first_root(
    function,
    lower: float,
    upper: float,
    samples: int = 300,
    zero_tolerance: float = 1e-12,
) -> float:
    grid = np.linspace(lower, upper, samples)
    values = np.array([function(x) for x in grid], dtype=float)
    close = np.where(np.abs(values) <= zero_tolerance)[0]
    if close.size:
        return float(grid[close[0]])
    signs = np.sign(values)
    crossings = np.where(signs[:-1] * signs[1:] < 0.0)[0]
    if not crossings.size:
        raise RuntimeError(
            f"No root bracketed in [{lower}, {upper}]. "
            f"Endpoint values are {values[0]:+.3e}, {values[-1]:+.3e}."
        )
    index = int(crossings[0])
    return float(brentq(function, grid[index], grid[index + 1], xtol=1e-13, rtol=1e-12))


def critical_pump_numerical(
    params: TwoModeParams,
    lower: float = 0.0,
    upper: float = 1.0,
) -> float:
    return find_first_root(
        lambda pump: two_mode_stability_margin(replace(params, kappa_a_up=pump)),
        lower,
        upper,
    )


def pump_for_target_margin(
    params: TwoModeParams,
    target_margin: float,
    lower: float = 0.0,
    upper: float = 2.0,
) -> float:
    return find_first_root(
        lambda pump: two_mode_stability_margin(replace(params, kappa_a_up=pump))
        - target_margin,
        lower,
        upper,
    )


# =============================================================================
# Observables
# =============================================================================

def safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    result = np.full(np.broadcast_shapes(numerator.shape, denominator.shape), np.nan)
    np.divide(
        numerator,
        denominator,
        out=result,
        where=np.abs(denominator) > 1e-15,
    )
    return result


def gaussian_mode_observables(
    times: np.ndarray,
    means: np.ndarray,
    covariances: np.ndarray,
    mode: int,
    omega: float = 1.0,
) -> pd.DataFrame:
    i = 2 * mode
    xmean = means[:, i]
    pmean = means[:, i + 1]
    var_x = covariances[:, i, i]
    var_p = covariances[:, i + 1, i + 1]
    cov_xp = covariances[:, i, i + 1]

    amplitude = (xmean + 1j * pmean) / np.sqrt(2.0)
    coherent_occupation = np.abs(amplitude) ** 2
    total_occupation = 0.5 * (
        var_x + var_p + xmean ** 2 + pmean ** 2 - 1.0
    )
    incoherent_occupation = total_occupation - coherent_occupation
    incoherent_occupation[np.abs(incoherent_occupation) < 5e-13] = 0.0

    determinant = var_x * var_p - cov_xp ** 2
    determinant = np.maximum(determinant, 0.0)
    gaussian_ergotropy = 0.5 * omega * (
        var_x + var_p + xmean ** 2 + pmean ** 2
        - 2.0 * np.sqrt(determinant)
    )
    coherent_energy = omega * coherent_occupation
    total_energy = omega * total_occupation
    passive_energy = total_energy - gaussian_ergotropy
    ergotropy_fraction = safe_ratio(gaussian_ergotropy, total_energy)
    snr = coherent_occupation / np.maximum(0.5 * (var_x + var_p), 1e-300)
    anomalous_centered = 0.5 * (var_x - var_p + 2j * cov_xp)

    return pd.DataFrame(
        {
            "t": times,
            "x_mean": xmean,
            "p_mean": pmean,
            "amplitude_real": amplitude.real,
            "amplitude_imag": amplitude.imag,
            "n_total": total_occupation,
            "n_coherent": coherent_occupation,
            "n_incoherent": incoherent_occupation,
            "energy_total": total_energy,
            "ergotropy": gaussian_ergotropy,
            "coherent_energy": coherent_energy,
            "passive_energy": passive_energy,
            "ergotropy_fraction": ergotropy_fraction,
            "snr": snr,
            "var_x": var_x,
            "var_p": var_p,
            "cov_xp": cov_xp,
            "anomalous_real": anomalous_centered.real,
            "anomalous_imag": anomalous_centered.imag,
        }
    )



def relative_coherent_phase(
    charger: pd.DataFrame,
    battery: pd.DataFrame,
    amplitude_fraction_tolerance: float = 5.0e-2,
) -> np.ndarray:
    """
    The principal relative phase Delta phi(t) = arg(v_b) - arg(v_a), modulo 2pi,
    with v_a(t) = <a(t)> and v_b(t) = <b(t)>.

    This is a dynamical phase diagnostic, not the time-independent difference
    between components of a single drift eigenvector.  The two phases are
    undefined whenever either coherent amplitude vanishes, in particular at
    t=0 for vacuum initial conditions.  Those samples are represented by NaN.
    """
    a = (
        charger["amplitude_real"].to_numpy(dtype=float)
        + 1j * charger["amplitude_imag"].to_numpy(dtype=float)
    )
    b = (
        battery["amplitude_real"].to_numpy(dtype=float)
        + 1j * battery["amplitude_imag"].to_numpy(dtype=float)
    )
    if a.shape != b.shape:
        raise ValueError("Coherent charger and battery amplitudes have unequal shapes.")
    # Compute the phase of b * conj(a), equivalent to the difference of args.
    phase = np.angle(b * np.conjugate(a)).astype(float)
    threshold_a = max(1.0e-14, amplitude_fraction_tolerance * float(np.max(np.abs(a))))
    threshold_b = max(1.0e-14, amplitude_fraction_tolerance * float(np.max(np.abs(b))))
    undefined = (np.abs(a) <= threshold_a) | (np.abs(b) <= threshold_b)
    phase[undefined] = np.nan
    # +pi and -pi denote exactly the same anti-phase relation.  Map both to
    # +pi near the branch cut to keep the phase-locked curve continuous.
    near_negative_pi = np.isfinite(phase) & np.isclose(
        phase, -np.pi, atol=1.0e-10, rtol=0.0
    )
    phase[near_negative_pi] = np.pi
    return phase


def phase_curve_without_false_jumps(phase: np.ndarray) -> np.ndarray:
    """Break the plotted line at abrupt phase changes or undefined phases."""
    curve = np.asarray(phase, dtype=float).copy()
    for i in range(1, curve.size):
        if np.isfinite(phase[i - 1]) and np.isfinite(phase[i]):
            if abs(phase[i] - phase[i - 1]) >= 0.75 * np.pi:
                curve[i] = np.nan
    return curve


def complex_amplitude_ratio(
    charger: pd.DataFrame,
    battery: pd.DataFrame,
    amplitude_fraction_tolerance: float = 5.0e-2,
) -> np.ndarray:
    """Return the complex ratio <b>/<a> wherever the charger amplitude is defined."""
    a = (
        charger["amplitude_real"].to_numpy(dtype=float)
        + 1j * charger["amplitude_imag"].to_numpy(dtype=float)
    )
    b = (
        battery["amplitude_real"].to_numpy(dtype=float)
        + 1j * battery["amplitude_imag"].to_numpy(dtype=float)
    )
    if a.shape != b.shape:
        raise ValueError("Coherent charger and battery amplitudes have unequal shapes.")
    threshold_a = max(1.0e-14, amplitude_fraction_tolerance * float(np.max(np.abs(a))))
    ratio = np.full(a.shape, np.nan + 1j * np.nan, dtype=complex)
    valid = np.abs(a) > threshold_a
    ratio[valid] = b[valid] / a[valid]
    return ratio


def _set_complex_plane_limits(ax: mpl.axes.Axes, values: np.ndarray) -> None:
    finite = np.isfinite(values.real) & np.isfinite(values.imag)
    if not np.any(finite):
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-1.0, 1.0)
        return
    xr = values.real[finite]
    yi = values.imag[finite]
    xmin, xmax = float(np.min(xr)), float(np.max(xr))
    ymin, ymax = float(np.min(yi)), float(np.max(yi))
    xspan = xmax - xmin
    yspan = ymax - ymin
    dominant_span = max(xspan, yspan, 1.0e-3)
    min_span = max(0.08 * dominant_span, 0.05)
    xspan = max(xspan, min_span)
    yspan = max(yspan, min_span)
    xmid = 0.5 * (xmin + xmax)
    ymid = 0.5 * (ymin + ymax)
    pad_x = 0.10 * xspan
    pad_y = 0.10 * yspan
    ax.set_xlim(xmid - 0.5 * xspan - pad_x, xmid + 0.5 * xspan + pad_x)
    ax.set_ylim(ymid - 0.5 * yspan - pad_y, ymid + 0.5 * yspan + pad_y)


def _add_fig2_phase_inset(
    parent: mpl.axes.Axes,
    result: dict[str, Any],
    color: str,
    linewidth: float,
) -> np.ndarray:
    """Show the early-time relative phase of the coherent first moments.

    The complete charging dynamics may amplify both mean fields by many orders
    of magnitude.  Using a *fraction of the maximum over the entire run* to
    determine whether an early-time phase is defined incorrectly removes the
    whole visible inset (particularly for the dissipative battery).  Here the
    phase is considered undefined only if a first moment is numerically zero.

    For the coherent protocol, the principal phase switches between -pi/2
    and +pi/2 across nodes of a first moment.  A staircase displays those
    discontinuous branch switches clearly; its vertical edges are visual
    indicators of jumps, not resolved continuous phase sweeps.  For the
    dissipative protocol, the relative phase is locked at pi.
    """
    time = normalized_time(result["a"]["t"].to_numpy(dtype=float))
    # The default relative_coherent_phase tolerance of 5% is appropriate only
    # for evaluating a late-time/global-amplitude signal. It is much too large
    # for the 0 <= kappa_0*t <= 8 inset because amplitudes grow dramatically
    # during the rest of the simulation.  The absolute 1e-14 floor in that
    # function still masks exactly zero (initial vacuum) first moments.
    phase = relative_coherent_phase(
        result["a"], result["b"], amplitude_fraction_tolerance=0.0
    )
    is_dissipative = abs(result["params"].dissipative_g) > 0.0
    inset = parent.inset_axes([0.295, 0.340, 0.510, 0.375])
    inset.set_facecolor("white")
    inset.set_zorder(12)
    inset.plot(
        time,
        phase,
        color=color,
        linestyle="-",
        drawstyle="default" if is_dissipative else "steps-mid",
        linewidth=linewidth,
        solid_capstyle="butt",
        zorder=3,
    )

    inset_end = min(8.0, float(time[-1]))
    inset.set_xlim(0.0, inset_end)
    if is_dissipative:
        # Provide headroom above pi so the phase-locked line is not hidden
        # against the inset's top border.
        inset.set_ylim(-0.20 * np.pi, 1.20 * np.pi)
        inset.set_yticks([0.0, np.pi])
        inset.set_yticklabels(["$0$", r"$\pi$"])
    else:
        inset.set_ylim(-0.80 * np.pi, 0.80 * np.pi)
        inset.set_yticks([-0.5 * np.pi, 0.5 * np.pi])
        inset.set_yticklabels([r"$-\pi/2$", r"$\pi/2$"])
    inset.set_xticks([0.0, 0.5 * inset_end, inset_end])
    inset.set_ylabel(r"$\Delta\phi$", fontsize=8.5, labelpad=-1.0)
    inset.set_xlabel(r"$\kappa_0 t$", fontsize=8.5, labelpad=0.2)
    inset.tick_params(
        direction="in", top=True, right=True,
        labelsize=7.5, width=0.8, length=2.2, pad=0.5,
    )
    for spine in inset.spines.values():
        spine.set_linewidth(0.9)
    # Preserve the complete, now correctly defined phase in the Fig. 2 CSV.
    return phase


def simulate_two_mode(
    params: TwoModeParams,
    times: np.ndarray,
    mean0: np.ndarray | None = None,
    covariance0: np.ndarray | None = None,
) -> dict[str, Any]:
    amat, dmat, source = two_mode_gaussian_matrices(params)
    means, covariances = simulate_constant_gaussian(
        amat, dmat, source, times, mean0, covariance0
    )
    obs_a = gaussian_mode_observables(
        times, means, covariances, mode=0, omega=params.omega_a
    )
    obs_b = gaussian_mode_observables(
        times, means, covariances, mode=1, omega=params.omega_b
    )
    return {
        "params": params,
        "times": times,
        "A": amat,
        "D": dmat,
        "source": source,
        "means": means,
        "covariances": covariances,
        "a": obs_a,
        "b": obs_b,
        "margin": two_mode_stability_margin(params),
        "complex_drift": two_mode_complex_drift(params),
    }


def first_crossing(
    times: np.ndarray,
    values: np.ndarray,
    target: float,
) -> tuple[float, int, float]:
    values = np.asarray(values, dtype=float)
    indices = np.where(values >= target)[0]
    if not indices.size:
        raise RuntimeError(f"Target {target:g} was not reached.")
    index = int(indices[0])
    if index == 0:
        return float(times[0]), 0, 0.0
    t0, t1 = times[index - 1], times[index]
    y0, y1 = values[index - 1], values[index]
    if abs(y1 - y0) < 1e-15:
        weight = 1.0
    else:
        weight = float((target - y0) / (y1 - y0))
    crossing_time = float(t0 + weight * (t1 - t0))
    return crossing_time, index, weight


def interpolate_dataframe_row(
    dataframe: pd.DataFrame,
    index: int,
    weight: float,
) -> pd.Series:
    if index == 0:
        return dataframe.iloc[0].copy()
    row0 = dataframe.iloc[index - 1]
    row1 = dataframe.iloc[index]
    return row0 + weight * (row1 - row0)


def simulate_until_target(
    params: TwoModeParams,
    maximum_target: float,
    initial_t_end: float | None = None,
    points: int = 5001,
    maximum_doublings: int = 14,
) -> dict[str, Any]:
    margin = two_mode_stability_margin(params)
    if initial_t_end is None:
        if margin > 0.0:
            initial_t_end = max(100.0, 1.4 * np.log1p(maximum_target) / (2.0 * margin))
        else:
            initial_t_end = 1000.0
    t_end = float(initial_t_end)
    for _ in range(maximum_doublings):
        times = np.linspace(0.0, t_end, points)
        result = simulate_two_mode(params, times)
        if np.nanmax(result["b"]["n_total"].to_numpy()) >= maximum_target:
            return result
        t_end *= 2.0
    raise RuntimeError(
        f"Target {maximum_target:g} was not reached after extending to t={t_end:g}."
    )


def cumulative_trapezoid_numpy(values: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Cumulative trapezoidal integral with a zero initial value."""
    values = np.asarray(values, dtype=float)
    times = np.asarray(times, dtype=float)
    if values.shape != times.shape:
        raise ValueError("values and times must have the same shape.")
    if values.ndim != 1:
        raise ValueError("cumulative_trapezoid_numpy expects one-dimensional arrays.")
    if values.size == 0:
        return values.copy()
    increments = 0.5 * (values[1:] + values[:-1]) * np.diff(times)
    return np.concatenate(([0.0], np.cumsum(increments)))


def active_energy_accounting(result: dict[str, Any]) -> pd.DataFrame:
    """
    Net energetic cost of the active resources used by the charging protocol.

    We count the coherent charger drive and the incoherent gain channel as the
    active resources.  Local loss and the collective engineered reservoir are
    passive channels and are therefore not included in the supplied-energy cost.

    For H_drive = E a^dagger + E* a,

        d<n_a>/dt |_drive = -2 Im[E* <a>],

    while the Lindblad pump kappa_up D[a^dagger] contributes

        d<n_a>/dt |_gain = kappa_up (<n_a> + 1).

    Multiplication by omega_a gives the corresponding active powers.  The main
    charging efficiency uses the signed (net-work) integral

        E_act^net = integral [P_drive + P_gain] dt.

    Energy returned from the system to the coherent source therefore enters
    with a negative sign.  A separate End-Matter diagnostic below implements
    the experimentally oriented gross incident-field convention instead of the
    previously considered positive-part construction.
    """
    params = result["params"]
    charger = result["a"]
    times = charger["t"].to_numpy(dtype=float)
    amplitude = (
        charger["amplitude_real"].to_numpy(dtype=float)
        + 1j * charger["amplitude_imag"].to_numpy(dtype=float)
    )
    n_a = charger["n_total"].to_numpy(dtype=float)

    drive_power = (
        -2.0
        * float(params.omega_a)
        * np.imag(np.conjugate(params.drive) * amplitude)
    )
    gain_power = (
        float(params.omega_a)
        * float(params.kappa_a_up)
        * (n_a + 1.0)
    )
    total_active_power = drive_power + gain_power

    active_energy_net = cumulative_trapezoid_numpy(
        total_active_power,
        times,
    )

    return pd.DataFrame(
        {
            "t": times,
            "drive_power": drive_power,
            "gain_power": gain_power,
            "active_power": total_active_power,
            "active_energy_cost": active_energy_net,
        }
    )


def finite_capacity_table(
    result: dict[str, Any],
    targets: Sequence[float],
    label: str,
) -> pd.DataFrame:
    battery = result["b"]
    active = active_energy_accounting(result)
    drive_work = cumulative_trapezoid_numpy(
        active["drive_power"].to_numpy(dtype=float),
        active["t"].to_numpy(dtype=float),
    )
    gain_energy = cumulative_trapezoid_numpy(
        active["gain_power"].to_numpy(dtype=float),
        active["t"].to_numpy(dtype=float),
    )
    active = active.copy()
    active["drive_work"] = drive_work
    active["gain_energy"] = gain_energy
    rows: list[dict[str, float | str]] = []
    omega_b = float(result["params"].omega_b)
    for target in targets:
        crossing_time, index, weight = first_crossing(
            battery["t"].to_numpy(),
            battery["n_total"].to_numpy(),
            float(target),
        )
        row = interpolate_dataframe_row(battery, index, weight)
        active_row = interpolate_dataframe_row(active, index, weight)
        ergotropic_power = (
            float(row["ergotropy"] / crossing_time)
            if crossing_time > 0.0
            else np.nan
        )
        active_cost = float(active_row["active_energy_cost"])
        omega_a = float(result["params"].omega_a)
        kappa_up = float(result["params"].kappa_a_up)
        integrated_n_a_plus_one = (
            float(active_row["gain_energy"]) / (omega_a * kappa_up)
            if omega_a > 0.0 and kappa_up > 0.0
            else np.nan
        )
        charging_efficiency = (
            float(row["ergotropy"] / active_cost)
            if active_cost > 1.0e-15
            else np.nan
        )
        rows.append(
            {
                "case": label,
                "omega_a": float(result["params"].omega_a),
                "n_target": float(target),
                "t_ch": crossing_time,
                "kappa0_t_ch": float(crossing_time * REFERENCE_RATE),
                "ergotropy_at_target": float(row["ergotropy"]),
                "ergotropy_fraction_at_target": float(row["ergotropy_fraction"]),
                "snr_at_target": float(row["snr"]),
                "drive_work_at_target": float(active_row["drive_work"]),
                "gain_energy_at_target": float(active_row["gain_energy"]),
                "integrated_n_a_plus_one_at_target": integrated_n_a_plus_one,
                "kappa0_integrated_n_a_plus_one_at_target": float(
                    REFERENCE_RATE * integrated_n_a_plus_one
                ),
                "active_energy_cost_at_target": active_cost,
                "charging_efficiency_at_target": charging_efficiency,
                "ergotropic_power": ergotropic_power,
                "ergotropic_power_over_omega_b_kappa0": float(
                    ergotropic_power / (omega_b * REFERENCE_RATE)
                ),
            }
        )
    return pd.DataFrame(rows)



def gross_input_table(
    metrics: pd.DataFrame,
    coherent_params: TwoModeParams,
    dissipative_params: TwoModeParams,
    external_coupling_fraction: float = GROSS_INPUT_EXTERNAL_COUPLING_FRACTION,
    omega_drive_over_omega_a: float = GROSS_INPUT_OMEGA_DRIVE_OVER_OMEGA_A,
) -> pd.DataFrame:
    """
    Add an implementation-oriented gross coherent-input resource count.

    In the input-output realization we use, up to the phase convention,

        |E|^2 = kappa_ext |a_in|^2,

    so for a constant coherent drive

        E_in^coh(t) = omega_d |E|^2 t / kappa_ext          (hbar = 1).

    The default kappa_ext = kappa_a_down is the fully external-coupling limit
    of the local charger damping.  The result depends on kappa_ext and is
    therefore kept separate from the thermodynamic net-work efficiency used in
    the main text.

    The gross active resource measure used here is

        E_act^gross = E_gain + E_in^coh,

    and

        eta_ch^gross = W_b / E_act^gross.

    This is a port-level resource measure, not a wall-plug efficiency: the
    Lindblad gain term is still counted by the net energy it delivers to the
    charger, not by a microscopic source-side cost of producing that reservoir.
    """
    fraction = float(external_coupling_fraction)
    if not (0.0 < fraction <= 1.0):
        raise ValueError(
            "external_coupling_fraction must lie in (0, 1]; "
            "kappa_ext cannot exceed the local charger damping used here."
        )
    if omega_drive_over_omega_a <= 0.0:
        raise ValueError("omega_drive_over_omega_a must be positive.")

    params_by_case = {
        "coherent": coherent_params,
        "dissipative": dissipative_params,
    }
    rows: list[dict[str, Any]] = []

    for _, row in metrics.iterrows():
        case = str(row["case"])
        if case not in params_by_case:
            raise KeyError(f"Unknown finite-capacity case: {case!r}")
        params = params_by_case[case]
        kappa_ext = fraction * float(params.kappa_a_down)
        if kappa_ext <= 0.0:
            raise ValueError("kappa_ext must be positive.")

        omega_a = float(params.omega_a)
        omega_drive = float(omega_drive_over_omega_a) * omega_a
        t_ch = float(row["t_ch"])
        incident_coherent_energy = (
            omega_drive
            * abs(complex(params.drive)) ** 2
            * t_ch
            / kappa_ext
        )
        gross_active_energy = (
            float(row["gain_energy_at_target"])
            + incident_coherent_energy
        )
        gross_efficiency = (
            float(row["ergotropy_at_target"] / gross_active_energy)
            if gross_active_energy > 1.0e-15
            else np.nan
        )

        payload = dict(row)
        payload.update(
            {
                "kappa_ext": kappa_ext,
                "kappa_ext_over_kappa0": kappa_ext / REFERENCE_RATE,
                "kappa_ext_over_kappa_a_down": fraction,
                "omega_drive": omega_drive,
                "omega_drive_over_omega_a": float(omega_drive_over_omega_a),
                "incident_coherent_energy_at_target": incident_coherent_energy,
                "incident_coherent_energy_over_omega_a_at_target": (
                    incident_coherent_energy / omega_a
                ),
                "gross_active_energy_cost_at_target": gross_active_energy,
                "gross_active_energy_cost_over_omega_a_at_target": (
                    gross_active_energy / omega_a
                ),
                "gross_charging_efficiency_at_target": gross_efficiency,
            }
        )
        rows.append(payload)

    return pd.DataFrame(rows)


def gross_input_efficiency_scan_at_target(
    metrics: pd.DataFrame,
    coherent_params: TwoModeParams,
    dissipative_params: TwoModeParams,
    target: float,
    fractions: np.ndarray | None = None,
) -> pd.DataFrame:
    """Scan eta_ch^gross over kappa_ext/kappa_a_down at one target."""
    if fractions is None:
        fractions = np.linspace(
            GROSS_INPUT_SCAN_MIN_FRACTION,
            GROSS_INPUT_SCAN_MAX_FRACTION,
            GROSS_INPUT_SCAN_POINTS,
        )
    fractions = np.asarray(fractions, dtype=float)
    selected = metrics[np.isclose(metrics["n_target"], float(target))].copy()
    if selected.empty:
        raise ValueError(f"No finite-capacity data found at n_target={target:g}.")

    frames = [
        gross_input_table(
            selected,
            coherent_params,
            dissipative_params,
            external_coupling_fraction=float(fraction),
        )
        for fraction in fractions
    ]
    return pd.concat(frames, ignore_index=True)


# =============================================================================
# Gauge-invariant modal diagnostics
# =============================================================================

def modal_source_noise_diagnostics(
    amat: np.ndarray,
    dmat: np.ndarray,
    source: np.ndarray,
    real_part_tolerance: float = 1e-9,
) -> pd.DataFrame:
    """
    Report gauge-invariant left-eigenvector projections.

    If several modes share the largest real part, all are reported.  This is
    important for the coherent example, where a conjugate pair can have equal
    growth rates.
    """
    values, left_vectors, right_vectors = eig(amat, left=True, right=True)
    max_real = float(np.max(np.real(values)))
    selected = np.where(np.real(values) >= max_real - real_part_tolerance)[0]
    rows = []
    for index in selected:
        left = left_vectors[:, index]
        right = right_vectors[:, index]
        overlap = np.vdot(left, right)
        if abs(overlap) < 1e-13:
            continue
        left = left / np.conjugate(overlap)
        source_projection = abs(np.vdot(left, source)) ** 2
        noise_projection = float(np.real(np.vdot(left, dmat @ left)))
        rows.append(
            {
                "mode_index": int(index),
                "eigenvalue_real": float(np.real(values[index])),
                "eigenvalue_imag": float(np.imag(values[index])),
                "source_projection": float(source_projection),
                "noise_projection": noise_projection,
                "source_to_noise": float(source_projection / noise_projection)
                if noise_projection > 0.0
                else np.inf,
            }
        )
    return pd.DataFrame(rows)


# =============================================================================
# Three-mode microscopic model and effective elimination
# =============================================================================

@dataclass(frozen=True)
class ThreeModeParams:
    kappa_a_down: float = 0.01
    kappa_a_up: float = 0.011
    kappa_b: float = 0.01
    kappa_c: float = 1.0
    delta: float = 0.0
    Delta_c: float = 0.0
    drive: complex = 0.1 + 0.0j
    Gamma_ac: float = 1.0
    Gamma_bc: float = 1.0
    theta_a: float = np.pi / 4.0
    theta_b: float = np.pi / 4.0
    phi_a: float = 0.0
    phi_b: float = 0.0
    omega_b: float = 1.0

    def validate(self) -> None:
        rates = (
            self.kappa_a_down,
            self.kappa_a_up,
            self.kappa_b,
            self.kappa_c,
            self.Gamma_ac,
            self.Gamma_bc,
        )
        if any(rate < 0.0 for rate in rates):
            raise ValueError("Three-mode rates must be nonnegative.")
        if not (0.0 <= self.theta_a <= np.pi / 2.0):
            raise ValueError("theta_a must lie in [0, pi/2].")
        if not (0.0 <= self.theta_b <= np.pi / 2.0):
            raise ValueError("theta_b must lie in [0, pi/2].")

    @property
    def k_a(self) -> float:
        return self.kappa_a_down + self.Gamma_ac * np.cos(self.theta_a) ** 2

    @property
    def k_b_passive(self) -> float:
        return self.kappa_b + self.Gamma_bc * np.cos(self.theta_b) ** 2

    @property
    def k_c_total(self) -> float:
        return (
            self.kappa_c
            + self.Gamma_ac * np.sin(self.theta_a) ** 2
            + self.Gamma_bc * np.sin(self.theta_b) ** 2
        )

    @property
    def xi_a(self) -> complex:
        return (
            self.Gamma_ac
            * np.exp(1j * self.phi_a)
            * np.sin(self.theta_a)
            * np.cos(self.theta_a)
        )

    @property
    def xi_b(self) -> complex:
        return (
            self.Gamma_bc
            * np.exp(1j * self.phi_b)
            * np.sin(self.theta_b)
            * np.cos(self.theta_b)
        )


def three_mode_full_gaussian_matrices(
    params: ThreeModeParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    params.validate()
    h = np.diag([params.delta, -params.delta, params.Delta_c]).astype(complex)
    h_r = number_conserving_hamiltonian_matrix(h)
    dvec = drive_vector(3, 0, params.drive)
    jumps: list[np.ndarray] = []

    if params.kappa_a_down > 0.0:
        jumps.append(
            np.sqrt(params.kappa_a_down)
            * annihilation_quadrature_vector([1.0, 0.0, 0.0])
        )
    if params.kappa_a_up > 0.0:
        jumps.append(
            np.sqrt(params.kappa_a_up)
            * creation_quadrature_vector([1.0, 0.0, 0.0])
        )
    if params.kappa_b > 0.0:
        jumps.append(
            np.sqrt(params.kappa_b)
            * annihilation_quadrature_vector([0.0, 1.0, 0.0])
        )
    if params.kappa_c > 0.0:
        jumps.append(
            np.sqrt(params.kappa_c)
            * annihilation_quadrature_vector([0.0, 0.0, 1.0])
        )
    if params.Gamma_ac > 0.0:
        coefficients_ac = [
            np.cos(params.theta_a),
            0.0,
            np.exp(1j * params.phi_a) * np.sin(params.theta_a),
        ]
        jumps.append(
            np.sqrt(params.Gamma_ac)
            * annihilation_quadrature_vector(coefficients_ac)
        )
    if params.Gamma_bc > 0.0:
        coefficients_bc = [
            0.0,
            np.cos(params.theta_b),
            np.exp(1j * params.phi_b) * np.sin(params.theta_b),
        ]
        jumps.append(
            np.sqrt(params.Gamma_bc)
            * annihilation_quadrature_vector(coefficients_bc)
        )
    return gaussian_matrices(h_r, dvec, jumps)


def effective_two_mode_from_mediator(
    params: ThreeModeParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    params.validate()
    xi = np.array([params.xi_a, params.xi_b], dtype=complex)
    kss = np.diag([params.k_a, params.k_b_passive]).astype(complex)
    eta_d = params.k_c_total / (
        params.k_c_total ** 2 + 4.0 * params.Delta_c ** 2
    )
    eta_ls = params.Delta_c / (
        params.k_c_total ** 2 + 4.0 * params.Delta_c ** 2
    )
    k_eff = kss - eta_d * np.outer(xi, xi.conj())
    h_s = np.diag([params.delta, -params.delta]).astype(complex)
    h_eff = h_s + eta_ls * np.outer(xi, xi.conj())

    rows = factorize_kossakowski(k_eff)
    jumps = [
        annihilation_quadrature_vector(row)
        for row in rows
    ]
    if params.kappa_a_up > 0.0:
        jumps.append(
            np.sqrt(params.kappa_a_up)
            * creation_quadrature_vector([1.0, 0.0])
        )
    h_r = number_conserving_hamiltonian_matrix(h_eff)
    dvec = drive_vector(2, 0, params.drive)
    amat, dmat, source = gaussian_matrices(h_r, dvec, jumps)
    metadata = {
        "K_eff": k_eff,
        "H_eff_single_particle": h_eff,
        "eta_d": eta_d,
        "eta_ls": eta_ls,
        "xi": xi,
        "K_eff_eigenvalues": np.linalg.eigvalsh(k_eff),
    }
    return amat, dmat, source, metadata


def simulate_three_mode_full_and_effective(
    params: ThreeModeParams,
    times: np.ndarray,
) -> dict[str, Any]:
    a3, d3, s3 = three_mode_full_gaussian_matrices(params)
    means3, covariances3 = simulate_constant_gaussian(a3, d3, s3, times)
    full_b = gaussian_mode_observables(
        times, means3, covariances3, mode=1, omega=params.omega_b
    )

    a2, d2, s2, metadata = effective_two_mode_from_mediator(params)
    means2, covariances2 = simulate_constant_gaussian(a2, d2, s2, times)
    effective_b = gaussian_mode_observables(
        times, means2, covariances2, mode=1, omega=params.omega_b
    )

    return {
        "params": params,
        "full": {
            "A": a3,
            "D": d3,
            "source": s3,
            "means": means3,
            "covariances": covariances3,
            "b": full_b,
        },
        "effective": {
            "A": a2,
            "D": d2,
            "source": s2,
            "means": means2,
            "covariances": covariances2,
            "b": effective_b,
            "metadata": metadata,
        },
    }


# =============================================================================
# Plot helpers
# =============================================================================

def _axis_common(ax: mpl.axes.Axes) -> None:
    ax.minorticks_on()
    ax.tick_params(which="both", direction="in", top=True, right=True)


def _compute_stability_landscape(
    params: TwoModeParams,
    pump_values: np.ndarray,
    delta_values: np.ndarray,
    show_ep: bool,
) -> dict[str, Any]:
    zmat = np.empty((delta_values.size, pump_values.size), dtype=float)
    for i, delta in enumerate(delta_values):
        for j, pump in enumerate(pump_values):
            zmat[i, j] = two_mode_stability_margin(
                replace(params, delta=float(delta), kappa_a_up=float(pump))
            )

    boundary = np.full(delta_values.size, np.nan)
    for i, row in enumerate(zmat):
        signs = np.sign(row)
        crossings = np.where(signs[:-1] * signs[1:] < 0.0)[0]
        if crossings.size:
            j = int(crossings[0])
            boundary[i] = brentq(
                lambda pump: two_mode_stability_margin(
                    replace(
                        params,
                        delta=float(delta_values[i]),
                        kappa_a_up=pump,
                    )
                ),
                pump_values[j],
                pump_values[j + 1],
            )

    ep_payload: dict[str, float] = {}
    if show_ep and abs(params.coherent_J) < 1e-14 and params.Gamma_col > 0.0:
        pump_ep = (
            params.kappa_a_down
            - params.kappa_b
            + params.Gamma_col * np.cos(2.0 * params.theta)
        )
        delta_ep = abs(params.dissipative_g)
        if (
            pump_values[0] <= pump_ep <= pump_values[-1]
            and delta_values[0] <= delta_ep <= delta_values[-1]
        ):
            ep_payload = {
                "kappa_a_up_EP": float(pump_ep),
                "delta_EP": float(delta_ep),
            }

    return {
        "params": params,
        "pump_values": np.asarray(pump_values, dtype=float),
        "delta_values": np.asarray(delta_values, dtype=float),
        "stability_margin": zmat,
        "boundary": boundary,
        "ep": ep_payload,
    }


def _draw_stability_panel(
    ax: mpl.axes.Axes,
    landscape: dict[str, Any],
    panel_label: str | None = None,
    show_legend: bool = False,
) -> mpl.image.AxesImage:
    params = landscape["params"]
    pump = normalized_rate(landscape["pump_values"])
    delta = normalized_rate(landscape["delta_values"])
    margin = normalized_rate(landscape["stability_margin"])
    boundary = normalized_rate(landscape["boundary"])

    maximum = float(np.nanmax(np.abs(margin)))
    norm = TwoSlopeNorm(vmin=-maximum, vcenter=0.0, vmax=maximum)
    image = ax.imshow(
        margin,
        origin="lower",
        aspect="auto",
        extent=[
            float(pump[0]),
            float(pump[-1]),
            float(delta[0]),
            float(delta[-1]),
        ],
        cmap=NPG_DIVERGING,
        norm=norm,
        interpolation="nearest",
        rasterized=True,
    )
    ax.plot(
        boundary,
        delta,
        linestyle="--",
        color="black",
        linewidth=1.3,
        label=r"$\max_j\,\mathrm{Re}\lambda_j=0$",
    )
    ax.axvline(
        params.kappa_a_down / REFERENCE_RATE,
        linestyle=":",
        color="darkgreen",
        linewidth=1.2,
        label=r"$\kappa_{a,\uparrow}=\kappa_{a,\downarrow}$",
    )

    ep = landscape["ep"]
    if ep:
        ax.plot(
            ep["kappa_a_up_EP"] / REFERENCE_RATE,
            ep["delta_EP"] / REFERENCE_RATE,
            marker="o",
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=0.9,
            markersize=5.0,
            linestyle="none",
        )

    ax.set_xlabel(r"$\kappa_{a,\uparrow}/\kappa_0$")
    ax.set_ylabel(r"$\delta/\kappa_0$")
    _axis_common(ax)
    if panel_label is not None:
        _panel_label(ax, panel_label)
    if show_legend:
        ax.legend(loc="upper left", frameon=False)
    return image


def _save_stability_data(
    landscape: dict[str, Any],
    output: OutputPaths,
    stem: str,
) -> pd.DataFrame:
    pump_values = landscape["pump_values"]
    delta_values = landscape["delta_values"]
    zmat = landscape["stability_margin"]
    boundary = landscape["boundary"]
    records = []
    for i, delta in enumerate(delta_values):
        for j, pump in enumerate(pump_values):
            records.append(
                {
                    "delta": float(delta),
                    "delta_over_kappa0": float(delta / REFERENCE_RATE),
                    "kappa_a_up": float(pump),
                    "kappa_a_up_over_kappa0": float(pump / REFERENCE_RATE),
                    "stability_margin": float(zmat[i, j]),
                    "stability_margin_over_kappa0": float(
                        zmat[i, j] / REFERENCE_RATE
                    ),
                    "boundary_kappa_a_up": float(boundary[i])
                    if np.isfinite(boundary[i])
                    else np.nan,
                    "boundary_kappa_a_up_over_kappa0": float(
                        boundary[i] / REFERENCE_RATE
                    )
                    if np.isfinite(boundary[i])
                    else np.nan,
                }
            )
    dataframe = pd.DataFrame(records)
    save_dataframe(dataframe, output.data / f"{stem}.csv")
    save_json(
        {
            "units": unit_metadata(),
            "params": asdict(landscape["params"]),
            "params_dimensionless": normalized_two_mode_parameters(
                landscape["params"]
            ),
            "ep": landscape["ep"],
        },
        output.summaries / f"{stem}.json",
    )
    return dataframe


def plot_stability_landscape(
    params: TwoModeParams,
    pump_values: np.ndarray,
    delta_values: np.ndarray,
    output: OutputPaths,
    stem: str,
    show_ep: bool = False,
) -> dict[str, Any]:
    landscape = _compute_stability_landscape(
        params,
        pump_values,
        delta_values,
        show_ep,
    )
    fig, ax = plt.subplots(
        figsize=(SINGLE_COLUMN_WIDTH, 2.50),
        constrained_layout=True,
    )
    image = _draw_stability_panel(ax, landscape, show_legend=True)
    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label(
        r"$\max_j\,\mathrm{Re}\lambda_j/\kappa_0$"
    )
    _style_colorbar(colorbar)
    save_figure(
        fig,
        output.figures / f"{stem}.pdf",
        apply_tight_layout=False,
    )
    _save_stability_data(landscape, output, stem)
    return landscape


def plot_main_figure_1(
    coherent_landscape: dict[str, Any],
    dissipative_landscape: dict[str, Any],
    dissipative_landscape_secondary: dict[str, Any],
    dissipative_landscape_endpoint: dict[str, Any],
    output: OutputPaths,
) -> None:
    """
    Generate the four-panel stability figure.

    Panel (a) is the coherent reference. Panels (b), (c), and (d) are the
    dissipative scheme at theta=PAPER_THETA, theta=PAPER_THETA_SECONDARY, and
    theta=pi/2, respectively.

    In each dissipative panel, hatching marks the parameter region where the
    dissipative scheme is already unstable while the coherent reference at the
    same (kappa_a_up, delta) remains strictly stable. This directly visualizes
    the lower dissipative instability threshold.

    The theta=pi/2 panel is an exact endpoint stability reference. At that
    endpoint cos(theta)=0 and the off-diagonal dissipative coupling vanishes,
    so it is not a matched charging operating point.
    """

    def add_lower_threshold_hatching(
        ax: mpl.axes.Axes,
        coherent_reference: dict[str, Any],
        dissipative_reference: dict[str, Any],
    ) -> None:
        pump_dis_raw = np.asarray(
            dissipative_reference["pump_values"], dtype=float
        )
        delta_dis_raw = np.asarray(
            dissipative_reference["delta_values"], dtype=float
        )
        margin_dis = np.asarray(
            dissipative_reference["stability_margin"], dtype=float
        )

        coherent_params = coherent_reference["params"]
        margin_coh_on_dis_grid = np.empty_like(margin_dis)
        for i, delta_value in enumerate(delta_dis_raw):
            for j, pump_value in enumerate(pump_dis_raw):
                margin_coh_on_dis_grid[i, j] = two_mode_stability_margin(
                    replace(
                        coherent_params,
                        delta=float(delta_value),
                        kappa_a_up=float(pump_value),
                    )
                )

        lower_threshold_region = (
            (margin_dis > 1.0e-12)
            & (margin_coh_on_dis_grid < -1.0e-12)
        )

        if np.any(lower_threshold_region):
            hatch_field = np.where(
                lower_threshold_region,
                1.0,
                np.nan,
            )
            with mpl.rc_context(
                {"hatch.color": "black", "hatch.linewidth": 0.8}
            ):
                ax.contourf(
                    normalized_rate(pump_dis_raw),
                    normalized_rate(delta_dis_raw),
                    hatch_field,
                    levels=[0.5, 1.5],
                    colors="none",
                    hatches=["//////"],
                    zorder=1.5,
                )

    fig = plt.figure(
        figsize=(DOUBLE_COLUMN_WIDTH, 4.85),
        constrained_layout=True,
    )
    grid = fig.add_gridspec(
        2,
        3,
        width_ratios=(1.00, 1.00, 0.045),
        wspace=0.050,
        hspace=0.05,
    )

    ax_coh = fig.add_subplot(grid[0, 0])
    ax_dis_115 = fig.add_subplot(grid[0, 1])
    cax_top = fig.add_subplot(grid[0, 2])
    ax_dis_130 = fig.add_subplot(grid[1, 0])
    ax_dis_endpoint = fig.add_subplot(grid[1, 1])
    cax_bottom = fig.add_subplot(grid[1, 2])

    image_coh = _draw_stability_panel(
        ax_coh,
        coherent_landscape,
        panel_label="(a)",
        show_legend=False,
    )
    image_dis_115 = _draw_stability_panel(
        ax_dis_115,
        dissipative_landscape,
        panel_label="(b)",
        show_legend=False,
    )
    image_dis_130 = _draw_stability_panel(
        ax_dis_130,
        dissipative_landscape_secondary,
        panel_label="(c)",
        show_legend=False,
    )
    image_dis_endpoint = _draw_stability_panel(
        ax_dis_endpoint,
        dissipative_landscape_endpoint,
        panel_label="(d)",
        show_legend=False,
    )

    # Top row: remove only the x-axis labels to save vertical space, while
    # keeping the x-axis tick numbers visible.
    for ax_top in (ax_coh, ax_dis_115):
        ax_top.set_xlabel("")

    # Panel (a): retain the even-numbered major ticks used for the coherent
    # pump axis in the previous figure.
    x_left, x_right = ax_coh.get_xlim()
    first_even = 2.0 * math.ceil(x_left / 2.0)
    last_even = 2.0 * math.floor(x_right / 2.0)
    if last_even >= first_even:
        ax_coh.set_xticks(
            np.arange(first_even, last_even + 0.5, 2.0)
        )

    add_lower_threshold_hatching(
        ax_dis_115,
        coherent_landscape,
        dissipative_landscape,
    )
    add_lower_threshold_hatching(
        ax_dis_130,
        coherent_landscape,
        dissipative_landscape_secondary,
    )
    add_lower_threshold_hatching(
        ax_dis_endpoint,
        coherent_landscape,
        dissipative_landscape_endpoint,
    )

    ax_dis_115.text(
        0.97,
        0.06,
        rf"$\theta={PAPER_THETA:.2f}$",
        transform=ax_dis_115.transAxes,
        ha="right",
        va="bottom",
    )
    ax_dis_130.text(
        0.97,
        0.06,
        rf"$\theta={PAPER_THETA_SECONDARY:.2f}$",
        transform=ax_dis_130.transAxes,
        ha="right",
        va="bottom",
    )
    ax_dis_endpoint.text(
        0.97,
        0.06,
        r"$\theta=\pi/2$",
        transform=ax_dis_endpoint.transAxes,
        ha="right",
        va="bottom",
    )

    # Right-column panels share the y scale with the left panel in each row.
    for ax_right in (ax_dis_115, ax_dis_endpoint):
        ax_right.set_ylabel("")
        ax_right.tick_params(axis="y", labelleft=False)

    # Keep only one legend for the dashed/dotted line conventions.
    ax_coh.legend(loc="upper right", frameon=False)

    cb_top = fig.colorbar(image_dis_115, cax=cax_top)
    cb_top.set_label(r"$\max_j\,\mathrm{Re}\lambda_j/\kappa_0$")
    _style_colorbar(cb_top)
    cb_bottom = fig.colorbar(image_dis_endpoint, cax=cax_bottom)
    cb_bottom.set_label(r"$\max_j\,\mathrm{Re}\lambda_j/\kappa_0$")
    _style_colorbar(cb_bottom)

    # Fig. 1 keeps its existing dedicated font size.
    for axis in fig.axes:
        axis.title.set_fontsize(FIG1_FONT_SIZE)
        axis.xaxis.label.set_size(FIG1_FONT_SIZE)
        axis.yaxis.label.set_size(FIG1_FONT_SIZE)
        axis.tick_params(axis="both", which="both", labelsize=FIG1_FONT_SIZE)
        for text_artist in axis.texts:
            text_artist.set_fontsize(FIG1_FONT_SIZE)
        legend = axis.get_legend()
        if legend is not None:
            for legend_text in legend.get_texts():
                legend_text.set_fontsize(FIG1_FONT_SIZE)

    save_figure(
        fig,
        output.figures / "Fig1_model_and_stability.pdf",
        apply_tight_layout=False,
    )


def plot_energy_decomposition(
    result: dict[str, Any],
    output: OutputPaths,
    stem: str,
    time_label: str = r"$\kappa_0 t$",
    x_values: np.ndarray | None = None,
) -> None:
    battery = result["b"].copy()
    omega_b = float(result["params"].omega_b)
    x = (
        normalized_time(battery["t"].to_numpy())
        if x_values is None
        else np.asarray(x_values)
    )
    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_WIDTH, 2.50))
    ax.plot(x, battery["energy_total"] / omega_b, label=r"$E_b$")
    ax.plot(
        x,
        battery["ergotropy"] / omega_b,
        linestyle="--",
        label=r"$\mathcal{W}_b$",
    )
    ax.plot(
        x,
        battery["passive_energy"] / omega_b,
        linestyle=":",
        label=r"$E_b^{\rm pas}$",
    )
    ax.set_xlabel(time_label)
    ax.set_ylabel(r"Energy$/\omega_b$")
    _axis_common(ax)
    ax.legend(frameon=False)
    save_figure(fig, output.figures / f"{stem}.pdf")
    battery["kappa0_t"] = normalized_time(battery["t"].to_numpy())
    battery["energy_total_over_omega_b"] = battery["energy_total"] / omega_b
    battery["ergotropy_over_omega_b"] = battery["ergotropy"] / omega_b
    battery["passive_energy_over_omega_b"] = battery["passive_energy"] / omega_b
    save_dataframe(battery, output.data / f"{stem}.csv")


def plot_snr(
    result: dict[str, Any],
    output: OutputPaths,
    stem: str,
    time_label: str = r"$\kappa_0 t$",
    x_values: np.ndarray | None = None,
) -> None:
    battery = result["b"].copy()
    x = (
        normalized_time(battery["t"].to_numpy())
        if x_values is None
        else np.asarray(x_values)
    )
    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_WIDTH, 2.50))
    ax.semilogy(x, np.maximum(battery["snr"], 1e-16))
    ax.set_xlabel(time_label)
    ax.set_ylabel(r"$\mathrm{SNR}_b$")
    _axis_common(ax)
    save_figure(fig, output.figures / f"{stem}.pdf")
    battery["kappa0_t"] = normalized_time(battery["t"].to_numpy())
    save_dataframe(
        battery[["t", "kappa0_t", "snr", "ergotropy_fraction"]],
        output.data / f"{stem}.csv",
    )


def plot_ergotropy_fraction(
    result: dict[str, Any],
    output: OutputPaths,
    stem: str,
    time_label: str = r"$\kappa_0 t$",
    x_values: np.ndarray | None = None,
) -> None:
    battery = result["b"].copy()
    x = (
        normalized_time(battery["t"].to_numpy())
        if x_values is None
        else np.asarray(x_values)
    )
    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_WIDTH, 2.50))
    ax.plot(x, battery["ergotropy_fraction"])
    ax.set_xlabel(time_label)
    ax.set_ylabel(r"$\eta_{\mathcal{W}}$")
    ax.set_ylim(-0.02, 1.02)
    _axis_common(ax)
    save_figure(fig, output.figures / f"{stem}.pdf")
    battery["kappa0_t"] = normalized_time(battery["t"].to_numpy())
    save_dataframe(
        battery[["t", "kappa0_t", "ergotropy_fraction"]],
        output.data / f"{stem}.csv",
    )



def dissipative_critical_pump_general_fast(
    params: TwoModeParams,
    lower: float,
    upper: float,
    samples: int = 80,
) -> float:
    """
    Fast threshold solver for the purely dissipative two-mode model.

    It evaluates the exact real-Omega marginal condition directly instead of
    repeatedly diagonalizing the drift matrix.  The accepted root is verified
    against the full spectral stability margin.
    """
    if abs(params.coherent_J) > 1.0e-14:
        raise ValueError("Fast threshold solver requires coherent_J=0.")
    gamma_b = params.gamma_b
    coupling_squared = abs(params.dissipative_g) ** 2
    passive_a = (
        params.kappa_a_down
        + params.Gamma_col * np.cos(params.theta) ** 2
    )

    def condition(pump: float) -> float:
        gamma_a = 0.5 * (passive_a - pump)
        gamma_sum = gamma_a + gamma_b
        if gamma_sum <= 0.0:
            return np.nan
        return (
            coupling_squared
            - gamma_a
            * gamma_b
            * (
                1.0
                + 4.0 * params.delta ** 2 / gamma_sum ** 2
            )
        )

    trace_boundary = passive_a + 2.0 * gamma_b
    safe_upper = min(
        float(upper),
        np.nextafter(trace_boundary, -np.inf),
    )
    if safe_upper <= lower:
        raise RuntimeError("No positive-trace branch in the requested interval.")

    grid = np.linspace(float(lower), safe_upper, samples)
    values = np.asarray([condition(value) for value in grid])
    finite = np.isfinite(values)
    crossing = np.where(
        finite[:-1]
        & finite[1:]
        & (values[:-1] * values[1:] <= 0.0)
    )[0]
    if crossing.size == 0:
        raise RuntimeError("No dissipative threshold in the requested interval.")
    index = int(crossing[0])
    root = brentq(condition, grid[index], grid[index + 1])

    spectral_error = abs(
        two_mode_stability_margin(
            replace(params, kappa_a_up=float(root))
        )
    )
    if spectral_error > 1.0e-9:
        raise RuntimeError(
            "Fast threshold root failed spectral verification: "
            f"{spectral_error:.3e}"
        )
    return float(root)


def plot_critical_pump_theta_delta(
    base: TwoModeParams,
    theta_values: np.ndarray,
    delta_values: np.ndarray,
    output: OutputPaths,
    stem: str,
    pump_upper: float = 1.0,
) -> pd.DataFrame:
    critical = np.full((delta_values.size, theta_values.size), np.nan)
    for i, delta in enumerate(delta_values):
        for j, theta in enumerate(theta_values):
            params = replace(
                base,
                coherent_J=0.0 + 0.0j,
                delta=float(delta),
                theta=float(theta),
            )
            try:
                critical[i, j] = dissipative_critical_pump_general_fast(
                    params,
                    lower=0.0,
                    upper=pump_upper,
                )
            except RuntimeError:
                pass

    fig, ax = plt.subplots(
        figsize=(SINGLE_COLUMN_WIDTH, 2.50),
        constrained_layout=True,
    )
    image = ax.imshow(
        normalized_rate(critical),
        origin="lower",
        aspect="auto",
        extent=[
            float(theta_values[0]),
            float(theta_values[-1]),
            float(delta_values[0] / REFERENCE_RATE),
            float(delta_values[-1] / REFERENCE_RATE),
        ],
        cmap=NPG_SEQUENTIAL,
        interpolation="nearest",
        rasterized=True,
    )
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$\delta/\kappa_0$")
    _axis_common(ax)
    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label(
        r"$\kappa_{a,\uparrow}^{\rm crit}/\kappa_0$"
    )
    _style_colorbar(colorbar)
    save_figure(
        fig,
        output.figures / f"{stem}.pdf",
        apply_tight_layout=False,
    )

    records = [
        {
            "theta": float(theta_values[j]),
            "delta": float(delta_values[i]),
            "delta_over_kappa0": float(delta_values[i] / REFERENCE_RATE),
            "kappa_a_up_crit": float(critical[i, j]),
            "kappa_a_up_crit_over_kappa0": float(
                critical[i, j] / REFERENCE_RATE
            ),
        }
        for i in range(delta_values.size)
        for j in range(theta_values.size)
    ]
    dataframe = pd.DataFrame(records)
    save_dataframe(dataframe, output.data / f"{stem}.csv")
    return dataframe


def plot_critical_pump_theta_gamma(
    base: TwoModeParams,
    theta_values: np.ndarray,
    gamma_values: np.ndarray,
    output: OutputPaths,
    stem: str,
    pump_upper: float = 1.0,
) -> pd.DataFrame:
    critical = np.full((gamma_values.size, theta_values.size), np.nan)
    for i, gamma_col in enumerate(gamma_values):
        for j, theta in enumerate(theta_values):
            params = replace(
                base,
                coherent_J=0.0 + 0.0j,
                Gamma_col=float(gamma_col),
                theta=float(theta),
            )
            try:
                threshold = dissipative_critical_pump_resonance(params)
                if 0.0 <= threshold <= pump_upper:
                    critical[i, j] = threshold
            except RuntimeError:
                pass

    fig, ax = plt.subplots(
        figsize=(SINGLE_COLUMN_WIDTH, 2.50),
        constrained_layout=True,
    )
    image = ax.imshow(
        normalized_rate(critical),
        origin="lower",
        aspect="auto",
        extent=[
            float(theta_values[0]),
            float(theta_values[-1]),
            float(gamma_values[0] / REFERENCE_RATE),
            float(gamma_values[-1] / REFERENCE_RATE),
        ],
        cmap=NPG_SEQUENTIAL,
        interpolation="nearest",
        rasterized=True,
    )
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$\Gamma_{\rm col}/\kappa_0$")
    _axis_common(ax)
    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label(
        r"$\kappa_{a,\uparrow}^{\rm crit}/\kappa_0$"
    )
    _style_colorbar(colorbar)
    save_figure(
        fig,
        output.figures / f"{stem}.pdf",
        apply_tight_layout=False,
    )

    records = [
        {
            "theta": float(theta_values[j]),
            "Gamma_col": float(gamma_values[i]),
            "Gamma_col_over_kappa0": float(
                gamma_values[i] / REFERENCE_RATE
            ),
            "kappa_a_up_crit": float(critical[i, j]),
            "kappa_a_up_crit_over_kappa0": float(
                critical[i, j] / REFERENCE_RATE
            ),
        }
        for i in range(gamma_values.size)
        for j in range(theta_values.size)
    ]
    dataframe = pd.DataFrame(records)
    save_dataframe(dataframe, output.data / f"{stem}.csv")
    return dataframe


def coherent_finite_capacity_knees(
    coherent_result: dict[str, Any],
    n_min: float = 0.2,
    n_max: float = 100.0,
) -> np.ndarray:
    """Return record-high local occupation maxima of the coherent trajectory.

    First passage jumps when the requested capacity exceeds a preceding
    local maximum and must wait for the next beating cycle. Only record-high
    maxima generate these first-passage jumps; smaller peaks are excluded.
    """
    n_b = coherent_result["b"]["n_total"].to_numpy(dtype=float)
    peak_indices, _ = find_peaks(n_b)
    record = n_min
    knees: list[float] = []
    for index in peak_indices:
        value = float(n_b[index])
        if value > record + 1e-6:
            record = value
            if n_min < value < n_max:
                knees.append(value)
        if record >= n_max:
            break
    if not knees:
        raise RuntimeError("No record-high coherent charging peak in the capacity range.")
    return np.asarray(knees, dtype=float)


def dense_finite_capacity_targets(
    coherent_result: dict[str, Any],
    n_min: float = 0.2,
    n_max: float = 100.0,
    n_uniform: int = 701,
) -> tuple[np.ndarray, np.ndarray]:
    """Dense, nearly uniform capacity scan resolving the physical jumps.

    The former 24-point geometric grid was too sparse at high occupations.
    In addition to uniform sampling, bracket each coherent first-passage
    jump by nearby capacity targets; no artificial step is inserted into t_ch.
    """
    knees = coherent_finite_capacity_knees(coherent_result, n_min, n_max)
    bracketed: list[float] = []
    for knee in knees:
        step = max(0.003, 1.0e-4 * knee)
        bracketed.extend((knee - step, knee, knee + step))
    targets = np.unique(np.concatenate((
        np.linspace(n_min, n_max, n_uniform),
        np.asarray(bracketed, dtype=float),
    )))
    targets = targets[(targets >= n_min) & (targets <= n_max)]
    return targets, knees


def add_capacity_knee_marker(
    axes: Iterable[mpl.axes.Axes],
    n_transition: float,
) -> None:
    """Mark the first coherent first-passage jump on capacity axes only."""
    for ax in axes:
        ax.axvline(
            n_transition,
            color="#303030", linestyle=":",
            linewidth=1.12, alpha=0.9,
            zorder=5,
        )


def plot_finite_capacity_comparison(
    coherent_result: dict[str, Any],
    dissipative_result: dict[str, Any],
    targets: np.ndarray,
    output: OutputPaths,
    stem_prefix: str,
    first_coherent_knee: float,
) -> pd.DataFrame:
    coherent_table = finite_capacity_table(
        coherent_result,
        targets,
        "coherent",
    )
    dissipative_table = finite_capacity_table(
        dissipative_result,
        targets,
        "dissipative",
    )
    combined = pd.concat(
        [coherent_table, dissipative_table],
        ignore_index=True,
    )
    save_dataframe(combined, output.data / f"{stem_prefix}_metrics.csv")

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(SINGLE_COLUMN_WIDTH, 3.45),
        sharex=True,
    )
    axes = np.asarray(axes)
    styles = {
        "coherent": dict(
            color=FIG3_COHERENT_COLOR, marker=None, linestyle="-", linewidth=1.15
        ),
        "dissipative": dict(
            color=FIG3_DISSIPATIVE_COLOR, marker=None, linestyle="-", linewidth=1.15
        ),
    }

    for label, group in combined.groupby("case"):
        style = styles[label]
        axes[0, 0].plot(
            group["n_target"],
            group["kappa0_t_ch"],
            label=label.capitalize(),
            **style,
        )
        axes[0, 1].plot(
            group["n_target"],
            group["charging_efficiency_at_target"],
            **style,
        )
        axes[1, 0].semilogy(
            group["n_target"],
            np.maximum(group["snr_at_target"], 1e-16),
            **style,
        )
        axes[1, 1].plot(
            group["n_target"],
            group["ergotropic_power_over_omega_b_kappa0"],
            **style,
        )

    # Exact first record-high coherent peak: the n_* threshold for the first
    # discontinuous delay in the first-passage time.  All panels share n_*.
    add_capacity_knee_marker(axes.flat, first_coherent_knee)

    axes[0, 0].set_ylabel(r"$\kappa_0t_{\rm ch}$")
    axes[0, 1].set_ylabel(r"$\eta_{\rm ch,\star}$")
    axes[0, 1].set_ylim(-0.02, 1.02)
    axes[1, 0].set_ylabel(r"$\mathrm{SNR}_{b,\star}$")
    axes[1, 1].set_ylabel(
        r"$P_{\mathcal{W},\star}/(\omega_b\kappa_0)$"
    )
    axes[1, 0].set_xlabel(r"$n_\star$")
    axes[1, 1].set_xlabel(r"$n_\star$")

    for label, ax in zip(("(a)", "(b)", "(c)", "(d)"), axes.flat):
        _axis_common(ax)
        _panel_label(ax, label)

    handles = [
        Line2D(
            [0],
            [0],
            color=styles["coherent"]["color"],
            marker=styles["coherent"]["marker"],
            linestyle="-",
            label="Coherent",
        ),
        Line2D(
            [0],
            [0],
            color=styles["dissipative"]["color"],
            marker=styles["dissipative"]["marker"],
            linestyle="-",
            label="Dissipative",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
        fontsize=MAIN_TEXT_FONT_SIZE,
    )
    fig.subplots_adjust(
        left=0.18,
        right=0.98,
        bottom=0.13,
        top=0.88,
        wspace=0.52,
        hspace=0.22,
    )
    save_figure(
        fig,
        output.figures / "Fig3_finite_capacity_single_column.pdf",
        apply_tight_layout=False,
    )
    return combined



def plot_active_energy_accounting(
    metrics: pd.DataFrame,
    output: OutputPaths,
) -> None:
    """
    Decompose the active energetic cost behind the charging efficiency.

    All quantities are evaluated at each first-passage target n_star.  Panel
    (a) shows the time integral that enters the stimulated gain cost directly,

        kappa_0 int_0^{t_ch} [n_a(t)+1] dt.

    Panels (b)-(d) then show the cumulative gain energy, coherent-drive work,
    and their sum E_act.  This figure is intended as a compact supplementary
    diagnostic accompanying Fig. 3(b).
    """
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(SINGLE_COLUMN_WIDTH, 3.45),
        sharex=True,
    )
    axes = np.asarray(axes)
    styles = {
        "coherent": dict(
            color=FIG3_COHERENT_COLOR, marker="o", linestyle="-"
        ),
        "dissipative": dict(
            color=FIG3_DISSIPATIVE_COLOR, marker="s", linestyle="-"
        ),
    }

    for label, group in metrics.groupby("case"):
        style = styles[label]
        omega_a = float(group["omega_a"].iloc[0])
        axes[0, 0].plot(
            group["n_target"],
            group["kappa0_integrated_n_a_plus_one_at_target"],
            label=label.capitalize(),
            **style,
        )
        axes[0, 1].plot(
            group["n_target"],
            group["gain_energy_at_target"] / omega_a,
            **style,
        )
        axes[1, 0].plot(
            group["n_target"],
            group["drive_work_at_target"] / omega_a,
            **style,
        )
        axes[1, 1].plot(
            group["n_target"],
            group["active_energy_cost_at_target"] / omega_a,
            **style,
        )

    axes[0, 0].set_ylabel(
        r"$\kappa_0\!\int_0^{t_{\rm ch}}\!(n_a+1)\,dt$"
    )
    axes[0, 1].set_ylabel(r"$E_{\rm gain,\star}/\omega_a$")
    axes[1, 0].set_ylabel(r"$E_{\rm drive,\star}/\omega_a$")
    axes[1, 1].set_ylabel(r"$E_{\rm act,\star}/\omega_a$")
    axes[1, 0].set_xlabel(r"$n_\star$")
    axes[1, 1].set_xlabel(r"$n_\star$")

    for panel_label, ax in zip(("(a)", "(b)", "(c)", "(d)"), axes.flat):
        _axis_common(ax)
        _panel_label(ax, panel_label)

    handles = [
        Line2D(
            [0],
            [0],
            color=styles["coherent"]["color"],
            marker=styles["coherent"]["marker"],
            linestyle="-",
            label="Coherent",
        ),
        Line2D(
            [0],
            [0],
            color=styles["dissipative"]["color"],
            marker=styles["dissipative"]["marker"],
            linestyle="-",
            label="Dissipative",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
        fontsize=MAIN_TEXT_FONT_SIZE,
    )
    fig.subplots_adjust(
        left=0.20,
        right=0.98,
        bottom=0.13,
        top=0.88,
        wspace=0.60,
        hspace=0.22,
    )
    save_figure(
        fig,
        output.figures / "FigS_active_energy_accounting.pdf",
        apply_tight_layout=False,
    )

    target = float(metrics["n_target"].max())
    summary = metrics[np.isclose(metrics["n_target"], target)].copy()
    if not summary.empty:
        save_dataframe(
            summary,
            output.data / "active_energy_accounting_at_largest_target.csv",
        )
        print("\nActive-energy accounting at the largest target")
        print(
            summary[
                [
                    "case",
                    "n_target",
                    "kappa0_t_ch",
                    "kappa0_integrated_n_a_plus_one_at_target",
                    "gain_energy_at_target",
                    "drive_work_at_target",
                    "active_energy_cost_at_target",
                    "charging_efficiency_at_target",
                ]
            ].to_string(index=False)
        )



def plot_gross_input_endmatter(
    metrics: pd.DataFrame,
    coherent_params: TwoModeParams,
    dissipative_params: TwoModeParams,
    output: OutputPaths,
    external_coupling_fraction: float = GROSS_INPUT_EXTERNAL_COUPLING_FRACTION,
    first_coherent_knee: float | None = None,
) -> pd.DataFrame:
    """
    Generate the compact End-Matter gross-input comparison.

    Panels (a)-(c) use the fully external-coupling reference
    kappa_ext = external_coupling_fraction * kappa_a_down.  Panel (d) makes the
    implementation dependence explicit by scanning kappa_ext/kappa_a_down at
    the largest finite-capacity target.
    """
    gross = gross_input_table(
        metrics,
        coherent_params,
        dissipative_params,
        external_coupling_fraction=external_coupling_fraction,
    )
    target = float(metrics["n_target"].max())
    scan = gross_input_efficiency_scan_at_target(
        metrics,
        coherent_params,
        dissipative_params,
        target=target,
    )

    save_dataframe(
        gross,
        output.data / "gross_input_finite_capacity_metrics.csv",
    )
    save_dataframe(
        scan,
        output.data / "gross_input_kappa_ext_scan_at_largest_target.csv",
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(SINGLE_COLUMN_WIDTH, 3.45),
    )
    axes = np.asarray(axes)
    styles = {
        "coherent": dict(
            color=FIG3_COHERENT_COLOR, marker=None, linestyle="-", linewidth=1.15
        ),
        "dissipative": dict(
            color=FIG3_DISSIPATIVE_COLOR, marker=None, linestyle="-", linewidth=1.15
        ),
    }

    for label, group in gross.groupby("case"):
        style = styles[label]
        axes[0, 0].plot(
            group["n_target"],
            group["incident_coherent_energy_over_omega_a_at_target"],
            label=label.capitalize(),
            **style,
        )
        axes[0, 1].plot(
            group["n_target"],
            group["gross_active_energy_cost_over_omega_a_at_target"],
            **style,
        )
        axes[1, 0].plot(
            group["n_target"],
            group["gross_charging_efficiency_at_target"],
            **style,
        )

    for label, group in scan.groupby("case"):
        style = styles[label]
        axes[1, 1].plot(
            group["kappa_ext_over_kappa_a_down"],
            group["gross_charging_efficiency_at_target"],
            **style,
        )

    # The same coherent first-passage knee as Fig. 3, but only for axes
    # against n_*; panel (d) scans kappa_ext, not target capacity.
    if first_coherent_knee is not None:
        add_capacity_knee_marker(
            (axes[0, 0], axes[0, 1], axes[1, 0]),
            first_coherent_knee,
        )

    axes[0, 0].set_ylabel(r"$E_{{\rm in},\star}^{\rm coh}/\omega_a$")
    axes[0, 1].set_ylabel(r"$E_{{\rm act},\star}^{\rm gross}/\omega_a$")
    axes[1, 0].set_ylabel(r"$\eta_{{\rm ch},\star}^{\rm gross}$")
    axes[1, 1].set_ylabel(r"$\eta_{{\rm ch},\star}^{\rm gross}$")
    axes[0, 0].set_xlabel(r"$n_\star$")
    axes[0, 1].set_xlabel(r"$n_\star$")
    axes[1, 0].set_xlabel(r"$n_\star$")
    axes[1, 1].set_xlabel(
        r"$\kappa_{\rm ext}/\kappa_{a,\downarrow}$"
    )
    axes[1, 0].set_ylim(bottom=0.0)
    axes[1, 1].set_ylim(bottom=0.0)

    for panel_label, ax in zip(("(a)", "(b)", "(c)", "(d)"), axes.flat):
        _axis_common(ax)
        _panel_label(ax, panel_label)

    handles = [
        Line2D(
            [0],
            [0],
            color=styles["coherent"]["color"],
            marker=styles["coherent"]["marker"],
            linestyle="-",
            label="Coherent",
        ),
        Line2D(
            [0],
            [0],
            color=styles["dissipative"]["color"],
            marker=styles["dissipative"]["marker"],
            linestyle="-",
            label="Dissipative",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
        fontsize=MAIN_TEXT_FONT_SIZE,
    )
    fig.subplots_adjust(
        left=0.20,
        right=0.98,
        bottom=0.13,
        top=0.88,
        wspace=0.62,
        hspace=0.32,
    )
    save_figure(
        fig,
        output.figures / "Fig5_gross_input_end_matter.pdf",
        apply_tight_layout=False,
    )

    # Also generate the requested compact standalone version containing only
    # the total gross active-input cost and gross-input charging efficiency.
    # The four-panel version is also kept and receives the same capacity
    # sampling and first-passage marker in its capacity-dependent panels.
    plot_gross_input_endmatter_compact(
        gross, output, first_coherent_knee=first_coherent_knee,
        coherent_params=coherent_params,
        external_coupling_fraction=external_coupling_fraction,
    )

    largest = gross[np.isclose(gross["n_target"], target)].copy()
    if not largest.empty:
        save_dataframe(
            largest,
            output.data / "gross_input_at_largest_target.csv",
        )
        print("\nGross-input accounting at the largest target")
        print(
            largest[
                [
                    "case",
                    "n_target",
                    "kappa0_t_ch",
                    "kappa_ext_over_kappa_a_down",
                    "incident_coherent_energy_over_omega_a_at_target",
                    "gain_energy_at_target",
                    "gross_active_energy_cost_over_omega_a_at_target",
                    "gross_charging_efficiency_at_target",
                ]
            ].to_string(index=False)
        )

    save_json(
        {
            "definition": (
                "E_in^coh = omega_drive * |E|^2 * t_ch / kappa_ext; "
                "E_act^gross = E_gain + E_in^coh; "
                "eta_ch^gross = W_b / E_act^gross"
            ),
            "input_output_magnitude_relation": (
                "|E|^2 = kappa_ext |a_in|^2"
            ),
            "hbar": 1.0,
            "omega_drive_over_omega_a": GROSS_INPUT_OMEGA_DRIVE_OVER_OMEGA_A,
            "default_kappa_ext_over_kappa_a_down": (
                external_coupling_fraction
            ),
            "scan_range_kappa_ext_over_kappa_a_down": [
                GROSS_INPUT_SCAN_MIN_FRACTION,
                GROSS_INPUT_SCAN_MAX_FRACTION,
            ],
            "interpretation": (
                "Port-level gross coherent-input resource measure. "
                "It is not a wall-plug efficiency because the Lindblad gain "
                "reservoir is still counted by the net energy delivered to "
                "the charger rather than by a microscopic source-side cost."
            ),
        },
        output.summaries / "gross_input_convention.json",
    )
    return gross


# Only the compact gross-input panel uses these helpers to add the
# frequency-optimized coherent curve.  All existing simulations remain intact.
_OPT_GAUSS_NODES, _OPT_GAUSS_WEIGHTS = np.polynomial.legendre.leggauss(128)


def _opt_int_exp(z: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Stable (exp(z*t)-1)/z for scalar/array complex z and real t."""
    z, t = np.broadcast_arrays(np.asarray(z, complex), np.asarray(t, float))
    value = np.empty(z.shape, dtype=complex)
    small = np.abs(z * t) < 1.0e-5
    y = (z * t)[small]
    value[small] = t[small] * (
        1.0 + y/2.0 + y*y/6.0 + y*y*y/24.0 + y**4/120.0
    )
    value[~small] = np.expm1((z * t)[~small]) / z[~small]
    return value


class _OptimizedCoherentGaussian:
    """Exact Gaussian coherent response in dimensionless kappa_0 units."""

    def __init__(self, params: TwoModeParams):
        if abs(params.Gamma_col) > 1.0e-14:
            raise ValueError("Frequency optimizer is for coherent coupling only.")
        self.params = params
        self.M = two_mode_complex_drift(params) / REFERENCE_RATE
        self.lam, self.V = np.linalg.eig(self.M)
        if np.linalg.cond(self.V) > 1.0e8:
            raise RuntimeError("Nearly defective drift: use matrix exponential.")
        U = np.linalg.inv(self.V)
        force = np.array([-1j * params.drive / REFERENCE_RATE, 0.0], complex)
        noise = np.diag([params.kappa_a_up / REFERENCE_RATE, 0.0]).astype(complex)
        src_modes = U @ force
        noise_modes = U @ noise @ U.conj().T
        self.source_proj = np.array([self.V[i, :] * src_modes for i in range(2)])
        self.noise_proj = np.array([
            np.outer(self.V[i, :], self.V[i, :].conj()) * noise_modes
            for i in range(2)
        ])
        self.noise_exponents = self.lam[:, None] + self.lam.conj()[None, :]

    def amplitude(self, tau: np.ndarray | float, nu: float, mode: int = 1):
        tau = np.asarray(tau, float)
        shape = (2,) + (1,) * tau.ndim
        phases = self.lam + 1j * float(nu)
        terms = _opt_int_exp(phases.reshape(shape), tau)
        response = np.sum(self.source_proj[mode].reshape(shape) * terms, axis=0)
        return complex(response) if response.ndim == 0 else response

    def amplitude_prime(self, tau: np.ndarray | float, nu: float, mode: int = 1):
        tau = np.asarray(tau, float)
        shape = (2,) + (1,) * tau.ndim
        z = self.lam + 1j * float(nu)
        terms = np.exp(z.reshape(shape) * tau)
        response = np.sum(self.source_proj[mode].reshape(shape) * terms, axis=0)
        return complex(response) if response.ndim == 0 else response

    def n_noise(self, tau: np.ndarray | float, mode: int = 1):
        tau = np.asarray(tau, float)
        shape = (2, 2) + (1,) * tau.ndim
        z = self.noise_exponents.reshape(shape)
        weight = self.noise_proj[mode].reshape(shape)
        out = np.sum(weight * _opt_int_exp(z, tau), axis=(0, 1)).real
        if np.any(out < -1.0e-8):
            raise RuntimeError("Negative Gaussian occupation")
        out = np.maximum(0.0, out)
        return float(out) if out.ndim == 0 else out

    def n_noise_prime(self, tau: np.ndarray | float, mode: int = 1):
        tau = np.asarray(tau, float)
        shape = (2, 2) + (1,) * tau.ndim
        z = self.noise_exponents.reshape(shape)
        weight = self.noise_proj[mode].reshape(shape)
        out = np.sum(weight * np.exp(z * tau), axis=(0, 1)).real
        return float(out) if out.ndim == 0 else out

    def battery_occ(self, tau, nu):
        return np.abs(self.amplitude(tau, nu, 1))**2 + self.n_noise(tau, 1)

    def battery_occ_prime(self, tau, nu):
        b = self.amplitude(tau, nu, 1)
        return (2.0 * np.real(np.conj(b) * self.amplitude_prime(tau, nu, 1))
                + self.n_noise_prime(tau, 1))

    def gain_energy(self, tau: float, nu: float) -> float:
        """Original manuscript's integrated gain-channel energy (hbar=1)."""
        t = float(tau)
        if t <= 0:
            return 0.0
        nodes_t = 0.5 * t * (1.0 + _OPT_GAUSS_NODES)
        n_a = np.abs(self.amplitude(nodes_t, nu, 0))**2 + self.n_noise(nodes_t, 0)
        k_up = self.params.kappa_a_up / REFERENCE_RATE
        return float(self.params.omega_a * k_up * 0.5 * t
                     * np.dot(_OPT_GAUSS_WEIGHTS, n_a + 1.0))


def _opt_stationary_partition(s: _OptimizedCoherentGaussian,
                              nu: float, tmax: float, step: float):
    """Find all turning points to resolve TRUE earliest target crossings."""
    grid = np.linspace(0.0, tmax, int(np.ceil(tmax / step)) + 1)
    derivative = s.battery_occ_prime(grid, nu)
    indices = np.flatnonzero(derivative[:-1] * derivative[1:] < 0.0)
    roots = [0.0]
    for i in indices:
        r = brentq(lambda t: s.battery_occ_prime(t, nu),
                   grid[i], grid[i+1], xtol=1.0e-11)
        if r > roots[-1] + 1.0e-8:
            roots.append(float(r))
    roots.append(float(tmax))
    pts = np.asarray(roots, dtype=float)
    return pts, np.asarray(s.battery_occ(pts, nu), dtype=float)


def _opt_first_passages(s: _OptimizedCoherentGaussian, targets: np.ndarray,
                        nu: float, part) -> np.ndarray:
    pts, values = part
    records = np.maximum.accumulate(values)
    idx = np.searchsorted(records, targets, side="left")
    ans = np.full(targets.shape, np.nan, dtype=float)
    ans[targets <= values[0]] = pts[0]
    good = (idx > 0) & (idx < len(pts))
    if np.any(good):
        low = pts[idx[good] - 1].copy()
        high = pts[idx[good]].copy()
        goal = targets[good]
        for _ in range(36):
            mid = 0.5 * (low + high)
            reached = s.battery_occ(mid, nu) >= goal
            high = np.where(reached, mid, high)
            low = np.where(reached, low, mid)
        ans[good] = 0.5 * (low + high)
    return ans


def _optimized_coherent_gross_base(coherent_params: TwoModeParams,
                                    targets: np.ndarray,
                                    nu_max: float = 6.5,
                                    nu_points: int = 161,
                                    tmax: float = 3.0,
                                    step: float = 0.025) -> pd.DataFrame:
    """Fastest-frequency coherent first passages and exact gain-energy integrals.

    The drive frequency is optimized for EACH target occupation, ONLY by
    minimizing first-passage time. The same nu is used for ergotropy and gain
    cost at that target. No parameters of the physical coherent device change.
    """
    targets = np.asarray(targets, dtype=float)
    s = _OptimizedCoherentGaussian(coherent_params)
    nu_grid = np.linspace(-nu_max, nu_max, nu_points)
    best_t = np.full(len(targets), np.inf, dtype=float)
    best_nu = np.full(len(targets), np.nan, dtype=float)
    best_idx = np.zeros(len(targets), dtype=int)
    for j, nu in enumerate(nu_grid):
        part = _opt_stationary_partition(s, float(nu), tmax, step)
        passage = _opt_first_passages(s, targets, float(nu), part)
        improved = np.isfinite(passage) & (passage < best_t)
        best_t[improved] = passage[improved]
        best_nu[improved] = nu
        best_idx[improved] = j
    if np.any(~np.isfinite(best_t)):
        raise RuntimeError("Coherent frequency scan missed some target occupations")
    # Match the frequency refinement used for the final target of Fig. 3.
    k = len(targets)-1
    j = int(best_idx[k]); target = float(targets[k])
    lo = float(nu_grid[max(0, j-1)])
    hi = float(nu_grid[min(nu_points-1, j+1)])
    if hi > lo:
        def earliest(nu):
            part = _opt_stationary_partition(s, float(nu), tmax, step)
            return float(_opt_first_passages(
                s, np.array([target]), float(nu), part)[0])
        polished = minimize_scalar(earliest, bounds=(lo, hi), method="bounded",
                                   options={"xatol":1.0e-6, "maxiter":60})
        for nu in (lo, float(polished.x), hi):
            t = earliest(nu)
            if np.isfinite(t) and t < best_t[k]:
                best_t[k], best_nu[k] = t, nu
    rows = []
    for tau, nu, target in zip(best_t, best_nu, targets):
        passive = float(s.n_noise(tau, mode=1))
        w = float(np.abs(s.amplitude(tau, nu, mode=1))**2)
        n = w + passive
        if abs(n-target) > 2.0e-7:
            raise RuntimeError(f"Optimized first-passage residual: {n-target:.3e}")
        rows.append({
            "case": "coherent_opt",
            "omega_a": float(coherent_params.omega_a),
            "n_target": float(target),
            "t_ch": float(tau / REFERENCE_RATE),
            "kappa0_t_ch": float(tau),
            "ergotropy_at_target": float(coherent_params.omega_b * w),
            "snr_at_target": float(w / (passive + 0.5)),
            "ergotropy_fraction_at_target": float(w/n),
            "gain_energy_at_target": s.gain_energy(float(tau), float(nu)),
            "drive_frequency_offset_over_kappa0": float(nu),
            "n_incoherent_at_target": passive,
        })
    return pd.DataFrame(rows)



def plot_gross_input_endmatter_compact(
    gross: pd.DataFrame,
    output: OutputPaths,
    first_coherent_knee: float | None = None,
    coherent_params: TwoModeParams | None = None,
    external_coupling_fraction: float = GROSS_INPUT_EXTERNAL_COUPLING_FRACTION,
) -> None:
    """
    Generate a compact standalone version of the gross-input comparison using
    only the quantities shown in panels (b) and (c) of
    Fig5_gross_input_end_matter.pdf.

    Because this is a standalone two-panel figure, those quantities are
    relabeled (a) and (b). The four-panel version is generated as well.
    """
    styles = {
        "coherent": dict(
            color=FIG3_COHERENT_COLOR, marker=None, linestyle="-", linewidth=1.15
        ),
        "dissipative": dict(
            color=FIG3_DISSIPATIVE_COLOR, marker=None, linestyle="-", linewidth=1.15
        ),
    }

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(SINGLE_COLUMN_WIDTH, 1.85),
    )

    for label, group in gross.groupby("case"):
        style = styles[label]
        axes[0].plot(
            group["n_target"],
            group["gross_active_energy_cost_over_omega_a_at_target"],
            label=label.capitalize(),
            **style,
        )
        axes[1].plot(
            group["n_target"],
            group["gross_charging_efficiency_at_target"],
            **style,
        )

    # Add only the frequency-optimized coherent curve.  The existing midpoint
    # coherent and dissipative data and their plotting commands stay unchanged.
    if coherent_params is not None:
        opt = _optimized_coherent_gross_base(
            coherent_params,
            gross.loc[gross["case"] == "coherent", "n_target"].to_numpy(dtype=float),
        )
        kappa_ext = external_coupling_fraction * coherent_params.kappa_a_down
        drive_frequency = (
            GROSS_INPUT_OMEGA_DRIVE_OVER_OMEGA_A * coherent_params.omega_a
            + REFERENCE_RATE * opt["drive_frequency_offset_over_kappa0"].to_numpy()
        )
        if np.any(drive_frequency <= 0.0) or kappa_ext <= 0.0:
            raise ValueError("Nonphysical coherent drive frequency or coupling rate")
        incident = (
            drive_frequency * abs(coherent_params.drive)**2
            * opt["t_ch"].to_numpy() / kappa_ext
        )
        gross_opt = opt["gain_energy_at_target"].to_numpy() + incident
        eta_opt = opt["ergotropy_at_target"].to_numpy() / gross_opt
        axes[0].plot(
            opt["n_target"], gross_opt / coherent_params.omega_a,
            color="#167B55", linestyle="--", linewidth=1.15,
        )
        axes[1].plot(
            opt["n_target"], eta_opt,
            color="#167B55", linestyle="--", linewidth=1.15,
        )

    if first_coherent_knee is not None:
        add_capacity_knee_marker(axes, first_coherent_knee)

    axes[0].set_xlabel(r"$n_\star$")
    axes[0].set_ylabel(r"$E_{{\rm act},\star}^{\rm gross}/\omega_a$")
    axes[1].set_xlabel(r"$n_\star$")
    axes[1].set_ylabel(r"$\eta_{{\rm ch},\star}^{\rm gross}$")
    axes[1].set_ylim(bottom=0.0)

    for panel_label, ax in zip(("(a)", "(b)"), axes):
        _axis_common(ax)
        _panel_label(ax, panel_label)

    handles = [
        Line2D(
            [0],
            [0],
            color=styles["coherent"]["color"],
            marker=styles["coherent"]["marker"],
            linestyle="-",
            label="Coherent",
        ),
        Line2D(
            [0],
            [0],
            color=styles["dissipative"]["color"],
            marker=styles["dissipative"]["marker"],
            linestyle="-",
            label="Dissipative",
        ),
    ]
    if coherent_params is not None:
        handles.insert(1, Line2D(
            [0], [0], color="#167B55", linestyle="--", linewidth=1.15,
            label="Coherent (opt.)",
        ))
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=3 if coherent_params is not None else 2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.995) if coherent_params is not None else (0.6, 0.995),
        fontsize=7.3 if coherent_params is not None else MAIN_TEXT_FONT_SIZE,
        handlelength=1.5 if coherent_params is not None else 2.0,
        columnspacing=0.55 if coherent_params is not None else 2.0,
        handletextpad=0.35 if coherent_params is not None else 0.8,
    )
    fig.subplots_adjust(
        left=0.20,
        right=0.98,
        bottom=0.22,
        top=0.78,
        wspace=0.6,
    )
    save_figure(
        fig,
        output.figures / "Fig5_gross_input_end_matter_compact.pdf",
        apply_tight_layout=False,
    )



def plot_full_effective_comparison(
    comparison: dict[str, Any],
    output: OutputPaths,
    stem_prefix: str,
) -> None:
    full = comparison["full"]["b"]
    effective = comparison["effective"]["b"]
    params = comparison["params"]
    omega_b = float(params.omega_b)
    tau = normalized_time(full["t"].to_numpy())

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(DOUBLE_COLUMN_WIDTH, 2.05),
    )
    specifications = (
        ("energy_total", r"$E_b/\omega_b$", False),
        ("ergotropy", r"$\mathcal{W}_b/\omega_b$", False),
        ("snr", r"$\mathrm{SNR}_b$", True),
    )
    for panel_label, ax, (column, ylabel, semilogy) in zip(
        ("(a)", "(b)", "(c)"),
        axes,
        specifications,
    ):
        full_values = full[column].to_numpy()
        eff_values = effective[column].to_numpy()
        if column in ("energy_total", "ergotropy"):
            full_values = full_values / omega_b
            eff_values = eff_values / omega_b
        if semilogy:
            ax.semilogy(
                tau,
                np.maximum(full_values, 1e-16),
                label="Full",
            )
            ax.semilogy(
                tau,
                np.maximum(eff_values, 1e-16),
                linestyle="--",
                label="Effective",
            )
        else:
            ax.plot(tau, full_values, label="Full")
            ax.plot(
                tau,
                eff_values,
                linestyle="--",
                label="Effective",
            )
        ax.set_xlabel(r"$\kappa_0 t$")
        ax.set_ylabel(ylabel)
        _axis_common(ax)
        _panel_label(ax, panel_label)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.01),
        fontsize=MAIN_TEXT_FONT_SIZE,
    )
    fig.subplots_adjust(
        left=0.08,
        right=0.995,
        bottom=0.22,
        top=0.82,
        wspace=0.38,
    )
    save_figure(
        fig,
        output.figures / f"{stem_prefix}_comparison.pdf",
        apply_tight_layout=False,
    )

    merged = pd.DataFrame(
        {
            "t": full["t"].to_numpy(),
            "kappa0_t": tau,
            "energy_full": full["energy_total"],
            "energy_effective": effective["energy_total"],
            "ergotropy_full": full["ergotropy"],
            "ergotropy_effective": effective["ergotropy"],
            "snr_full": full["snr"],
            "snr_effective": effective["snr"],
        }
    )
    for observable in ("energy", "ergotropy", "snr"):
        merged[f"{observable}_relative_error"] = (
            np.abs(
                merged[f"{observable}_full"]
                - merged[f"{observable}_effective"]
            )
            / np.maximum(
                np.abs(merged[f"{observable}_full"]),
                1e-14,
            )
        )
    save_dataframe(
        merged,
        output.data / f"{stem_prefix}_comparison.csv",
    )


# =============================================================================
# Optional saturable-gain calculation with QuTiP
# =============================================================================

@dataclass(frozen=True)
class SaturableParams:
    linear_reference: TwoModeParams
    P_sigma: float = 0.08
    gamma_sigma: float = 0.02
    gamma_phi: float = 0.0
    Delta_sigma: float = 0.0
    cutoff_a: int = 18
    cutoff_b: int = 18
    omega_b: float = 1.0

    @property
    def gamma_parallel(self) -> float:
        return self.P_sigma + self.gamma_sigma

    @property
    def gamma_perp(self) -> float:
        return 0.5 * self.gamma_parallel + self.gamma_phi

    @property
    def w0(self) -> float:
        return (self.P_sigma - self.gamma_sigma) / self.gamma_parallel

    @property
    def g_sigma(self) -> float:
        if self.w0 <= 0.0:
            raise ValueError("The emitter must be inverted: P_sigma > gamma_sigma.")
        target = self.linear_reference.kappa_a_up
        return float(
            np.sqrt(
                target
                * (self.gamma_perp ** 2 + self.Delta_sigma ** 2)
                / (2.0 * self.gamma_perp * self.w0)
            )
        )

    @property
    def n_sat(self) -> float:
        return float(
            self.gamma_parallel
            * (self.gamma_perp ** 2 + self.Delta_sigma ** 2)
            / (4.0 * self.g_sigma ** 2 * self.gamma_perp)
        )


def _exact_ergotropy_from_density_matrix(rho_b, omega_b: float) -> float:
    eigenvalues = np.linalg.eigvalsh(rho_b.full())
    eigenvalues = np.sort(np.real(eigenvalues))[::-1]
    energies = omega_b * np.arange(eigenvalues.size, dtype=float)
    passive_energy = float(np.dot(eigenvalues, energies))
    number_operator = np.diag(np.arange(eigenvalues.size, dtype=float))
    actual_energy = float(
        omega_b * np.real(np.trace(rho_b.full() @ number_operator))
    )
    return actual_energy - passive_energy


def simulate_saturable_gain(
    params: SaturableParams,
    times: np.ndarray,
) -> pd.DataFrame:
    try:
        import qutip as qt
    except ImportError as exc:
        raise RuntimeError(
            "The saturation calculation requires QuTiP. "
            "Install it with: pip install qutip"
        ) from exc

    reference = params.linear_reference
    if abs(reference.coherent_J) > 1e-14:
        raise ValueError("The saturation check is defined for the dissipative model J=0.")
    na_cut = params.cutoff_a
    nb_cut = params.cutoff_b
    identity_a = qt.qeye(na_cut)
    identity_b = qt.qeye(nb_cut)
    identity_s = qt.qeye(2)

    a = qt.tensor(qt.destroy(na_cut), identity_b, identity_s)
    b = qt.tensor(identity_a, qt.destroy(nb_cut), identity_s)

    # Define the emitter basis explicitly to avoid convention ambiguity:
    # |g> = basis(2,0), |e> = basis(2,1),
    # sigma_- = |g><e|, sigma_z = |e><e| - |g><g|.
    ground = qt.basis(2, 0)
    excited = qt.basis(2, 1)
    sigma_minus_tls = ground * excited.dag()
    sigma_plus_tls = sigma_minus_tls.dag()
    sigma_z_tls = excited * excited.dag() - ground * ground.dag()
    sigma_minus = qt.tensor(identity_a, identity_b, sigma_minus_tls)
    sigma_plus = qt.tensor(identity_a, identity_b, sigma_plus_tls)
    sigma_z = qt.tensor(identity_a, identity_b, sigma_z_tls)

    hamiltonian = (
        reference.delta * a.dag() * a
        - reference.delta * b.dag() * b
        + params.Delta_sigma * sigma_plus * sigma_minus
        + params.g_sigma * (a.dag() * sigma_minus + a * sigma_plus)
        + reference.drive * a.dag()
        + np.conjugate(reference.drive) * a
    )
    collective = (
        np.cos(reference.theta) * a
        + np.exp(1j * reference.phi) * np.sin(reference.theta) * b
    )
    collapse = [
        np.sqrt(reference.kappa_a_down) * a,
        np.sqrt(reference.kappa_b) * b,
        np.sqrt(reference.Gamma_col) * collective,
        np.sqrt(params.P_sigma) * sigma_plus,
        np.sqrt(params.gamma_sigma) * sigma_minus,
    ]
    if params.gamma_phi > 0.0:
        collapse.append(np.sqrt(params.gamma_phi / 2.0) * sigma_z)

    excited_population = params.P_sigma / (params.P_sigma + params.gamma_sigma)
    rho_tls = (
        excited_population * excited * excited.dag()
        + (1.0 - excited_population) * ground * ground.dag()
    )
    rho0 = qt.tensor(
        qt.fock_dm(na_cut, 0),
        qt.fock_dm(nb_cut, 0),
        rho_tls,
    )

    options = {"nsteps": 200000, "atol": 1e-10, "rtol": 1e-8, "store_states": True}
    solution = qt.mesolve(
        hamiltonian,
        rho0,
        times,
        collapse,
        e_ops=[],
        options=options,
    )

    rows = []
    x_b = (b + b.dag()) / np.sqrt(2.0)
    p_b = (b - b.dag()) / (1j * np.sqrt(2.0))
    for time, state in zip(times, solution.states):
        rho_b = state.ptrace(1)
        mean_b = qt.expect(b, state)
        n_b = float(np.real(qt.expect(b.dag() * b, state)))
        x_mean = float(np.real(qt.expect(x_b, state)))
        p_mean = float(np.real(qt.expect(p_b, state)))
        var_x = float(np.real(qt.expect(x_b * x_b, state)) - x_mean ** 2)
        var_p = float(np.real(qt.expect(p_b * p_b, state)) - p_mean ** 2)
        ergotropy = _exact_ergotropy_from_density_matrix(rho_b, params.omega_b)
        energy = params.omega_b * n_b
        rows.append(
            {
                "t": float(time),
                "n_total": n_b,
                "energy_total": energy,
                "ergotropy": ergotropy,
                "ergotropy_fraction": ergotropy / energy if energy > 1e-15 else np.nan,
                "snr": abs(mean_b) ** 2 / max(0.5 * (var_x + var_p), 1e-300),
                "var_x": var_x,
                "var_p": var_p,
            }
        )
    return pd.DataFrame(rows)


# =============================================================================
# Paper parameter choices
# =============================================================================

@dataclass(frozen=True)
class PaperParameters:
    kappa_a_down: float = 0.01
    kappa_b: float = 0.01
    drive: complex = 0.1 + 0.0j
    omega_b: float = 1.0

    # Moderate-composition matched benchmark.  The value of Gamma_col is chosen
    # so that |g| remains exactly 0.032 while theta is moved away from pi/2.
    matched_offdiagonal_strength: float = PAPER_MATCHED_G
    Gamma_col: float = PAPER_GAMMA_COL
    theta: float = PAPER_THETA
    phi: float = 0.0

    # Matched-growth comparison
    target_growth_margin: float = 5.0e-4

    def dissipative_base(self) -> TwoModeParams:
        return TwoModeParams(
            kappa_a_down=self.kappa_a_down,
            kappa_a_up=0.0,
            kappa_b=self.kappa_b,
            delta=0.0,
            drive=self.drive,
            coherent_J=0.0,
            Gamma_col=self.Gamma_col,
            theta=self.theta,
            phi=self.phi,
            omega_b=self.omega_b,
        )

    def matched_pair(self) -> tuple[TwoModeParams, TwoModeParams]:
        dissipative_base = self.dissipative_base()
        dissipative_pump = pump_for_target_margin(
            dissipative_base,
            self.target_growth_margin,
            lower=0.0,
            upper=0.2,
        )
        dissipative = replace(
            dissipative_base,
            kappa_a_up=dissipative_pump,
        )
        coherent_base = TwoModeParams(
            kappa_a_down=self.kappa_a_down,
            kappa_a_up=0.0,
            kappa_b=self.kappa_b,
            delta=0.0,
            drive=self.drive,
            coherent_J=abs(dissipative.dissipative_g),
            Gamma_col=0.0,
            theta=np.pi / 4.0,
            phi=0.0,
            omega_b=self.omega_b,
        )
        coherent_pump = pump_for_target_margin(
            coherent_base,
            self.target_growth_margin,
            lower=0.0,
            upper=2.0,
        )
        coherent = replace(coherent_base, kappa_a_up=coherent_pump)
        return coherent, dissipative


# =============================================================================
# Validation
# =============================================================================

def run_validation(output: OutputPaths) -> dict[str, Any]:
    print("\nRunning validation suite...")
    paper = PaperParameters()

    # 1. Correct complex-J quadrature drift.
    complex_params = TwoModeParams(
        kappa_a_down=0.03,
        kappa_a_up=0.004,
        kappa_b=0.02,
        delta=0.17,
        drive=0.08 + 0.03j,
        coherent_J=0.11 + 0.07j,
        Gamma_col=0.45,
        theta=0.91,
        phi=0.37,
    )
    amat, dmat, source = two_mode_gaussian_matrices(complex_params)
    complex_eigenvalues = np.linalg.eigvals(two_mode_complex_drift(complex_params))
    quadrature_eigenvalues = np.linalg.eigvals(amat)
    expected = np.concatenate([complex_eigenvalues, np.conjugate(complex_eigenvalues)])
    spectral_error = float(
        np.max(
            np.abs(
                np.sort_complex(quadrature_eigenvalues)
                - np.sort_complex(expected)
            )
        )
    )

    # 2. Old-to-normalized equivalence.
    legacy_gamma = 0.4
    legacy_pa = 0.04 + 0.0j
    legacy_pb = 4.0 + 0.0j
    scale = np.sqrt(abs(legacy_pa) ** 2 + abs(legacy_pb) ** 2)
    normalized = TwoModeParams(
        kappa_a_down=0.01,
        kappa_a_up=0.011,
        kappa_b=0.01,
        delta=0.03,
        drive=0.1,
        coherent_J=0.0,
        Gamma_col=legacy_gamma * scale ** 2,
        theta=np.arctan2(abs(legacy_pb), abs(legacy_pa)),
        phi=np.angle(legacy_pb) - np.angle(legacy_pa),
    )
    normalized_drift = two_mode_complex_drift(normalized)
    gamma_a_legacy = 0.5 * (
        normalized.kappa_a_down
        - normalized.kappa_a_up
        + legacy_gamma * abs(legacy_pa) ** 2
    )
    gamma_b_legacy = 0.5 * (
        normalized.kappa_b
        + legacy_gamma * abs(legacy_pb) ** 2
    )
    g_legacy = 0.5 * legacy_gamma * np.conjugate(legacy_pa) * legacy_pb
    legacy_drift = np.array(
        [
            [-gamma_a_legacy - 1j * normalized.delta, -g_legacy],
            [-np.conjugate(g_legacy), -gamma_b_legacy + 1j * normalized.delta],
        ],
        dtype=complex,
    )
    normalization_error = float(np.max(np.abs(normalized_drift - legacy_drift)))

    # 3. Analytical threshold.
    threshold_base = replace(paper.dissipative_base(), delta=0.0)
    analytical_threshold = dissipative_critical_pump_resonance(threshold_base)
    numerical_threshold = critical_pump_numerical(
        threshold_base, lower=0.0, upper=0.2
    )
    threshold_error = abs(analytical_threshold - numerical_threshold)

    # 4. Phase invariance.
    phases = np.linspace(0.0, 2.0 * np.pi, 9)
    spectra = []
    margins = []
    for phase in phases:
        p = replace(normalized, phi=float(phase))
        spectra.append(np.sort_complex(np.linalg.eigvals(two_mode_complex_drift(p))))
        margins.append(two_mode_stability_margin(p))
    phase_spectral_error = float(
        max(np.max(np.abs(spectrum - spectra[0])) for spectrum in spectra)
    )
    phase_margin_error = float(np.max(np.abs(np.asarray(margins) - margins[0])))

    # 5. Ergotropy identity and anomalous moment.
    times = np.linspace(0.0, 500.0, 1001)
    test_result = simulate_two_mode(normalized, times)
    battery = test_result["b"]
    ergotropy_identity_error = float(
        np.max(np.abs(battery["ergotropy"] - battery["coherent_energy"]))
    )
    decomposition_error = float(
        np.max(
            np.abs(
                battery["energy_total"]
                - battery["ergotropy"]
                - battery["passive_energy"]
            )
        )
    )
    anomalous_error = float(
        np.max(
            np.hypot(
                battery["anomalous_real"],
                battery["anomalous_imag"],
            )
        )
    )

    # 6. Effective Kossakowski positivity.
    mediator_params = ThreeModeParams()
    _, _, _, mediator_metadata = effective_two_mode_from_mediator(mediator_params)
    minimum_keff_eigenvalue = float(
        np.min(mediator_metadata["K_eff_eigenvalues"])
    )

    summary = {
        "complex_J_spectral_error": spectral_error,
        "old_to_normalized_drift_error": normalization_error,
        "threshold_analytical": analytical_threshold,
        "threshold_numerical": numerical_threshold,
        "threshold_error": threshold_error,
        "phi_spectral_invariance_error": phase_spectral_error,
        "phi_margin_invariance_error": phase_margin_error,
        "ergotropy_equals_coherent_energy_error": ergotropy_identity_error,
        "energy_decomposition_error": decomposition_error,
        "maximum_centered_anomalous_moment": anomalous_error,
        "minimum_effective_kossakowski_eigenvalue": minimum_keff_eigenvalue,
    }
    save_json(summary, output.summaries / "validation.json")

    tolerances = {
        "complex_J_spectral_error": 1e-10,
        "old_to_normalized_drift_error": 1e-12,
        "threshold_error": 1e-10,
        "phi_spectral_invariance_error": 1e-10,
        "phi_margin_invariance_error": 1e-12,
        "ergotropy_equals_coherent_energy_error": 1e-9,
        "energy_decomposition_error": 1e-9,
        "maximum_centered_anomalous_moment": 1e-9,
    }
    failed = [
        key for key, tolerance in tolerances.items()
        if abs(summary[key]) > tolerance
    ]
    if minimum_keff_eigenvalue < -1e-10:
        failed.append("minimum_effective_kossakowski_eigenvalue")

    print(json.dumps(_json_ready(summary), indent=2))
    if failed:
        raise AssertionError(f"Validation failed: {failed}")
    print("All validation tests passed.")
    return summary


# =============================================================================
# Figure recipes
# =============================================================================

def recipe_stability(output: OutputPaths) -> None:
    paper = PaperParameters()
    coherent, dissipative = paper.matched_pair()

    paper_secondary = PaperParameters(
        theta=PAPER_THETA_SECONDARY,
        Gamma_col=PAPER_GAMMA_COL_SECONDARY,
    )
    _, dissipative_secondary = paper_secondary.matched_pair()

    # Exact theta=pi/2 endpoint.  The fixed-|g| construction cannot be used at
    # this endpoint because sin(theta) cos(theta)=0.  We therefore retain the
    # finite Gamma_col of the theta=1.30 panel and rotate the collective jump to
    # z=b.  The endpoint has g=0 and is used only as a stability reference.
    paper_endpoint = PaperParameters(
        theta=PAPER_THETA_ENDPOINT,
        Gamma_col=PAPER_GAMMA_COL_ENDPOINT,
    )
    _, dissipative_endpoint = paper_endpoint.matched_pair()

    delta_values = np.linspace(0.0, 0.12, 181)

    coherent_landscape = _compute_stability_landscape(
        replace(coherent, kappa_a_up=0.0),
        np.linspace(
            0.0,
            max(0.08, 1.25 * coherent.kappa_a_up),
            181,
        ),
        delta_values,
        show_ep=False,
    )
    dissipative_landscape = _compute_stability_landscape(
        replace(dissipative, kappa_a_up=0.0),
        np.linspace(
            0.0,
            max(0.03, 1.5 * dissipative.kappa_a_up),
            181,
        ),
        delta_values,
        show_ep=True,
    )
    dissipative_landscape_secondary = _compute_stability_landscape(
        replace(dissipative_secondary, kappa_a_up=0.0),
        np.linspace(
            0.0,
            max(0.03, 1.5 * dissipative_secondary.kappa_a_up),
            181,
        ),
        delta_values,
        show_ep=True,
    )
    dissipative_landscape_endpoint = _compute_stability_landscape(
        replace(dissipative_endpoint, kappa_a_up=0.0),
        np.linspace(
            0.0,
            max(0.03, 1.5 * dissipative_endpoint.kappa_a_up),
            181,
        ),
        delta_values,
        show_ep=False,
    )

    # Save the numerical landscapes without generating the old standalone
    # stability PDFs.  Remove stale copies if the user reuses an output folder.
    for obsolete_name in (
        "stability_coherent.pdf",
        "stability_dissipative.pdf",
        "stability_dissipative_theta_1p30.pdf",
    ):
        obsolete_path = output.figures / obsolete_name
        if obsolete_path.exists():
            obsolete_path.unlink()

    _save_stability_data(
        coherent_landscape,
        output,
        "stability_coherent",
    )
    _save_stability_data(
        dissipative_landscape,
        output,
        "stability_dissipative_theta_1p15",
    )
    _save_stability_data(
        dissipative_landscape_secondary,
        output,
        "stability_dissipative_theta_1p30",
    )
    _save_stability_data(
        dissipative_landscape_endpoint,
        output,
        "stability_dissipative_theta_pi_over_2",
    )

    plot_main_figure_1(
        coherent_landscape,
        dissipative_landscape,
        dissipative_landscape_secondary,
        dissipative_landscape_endpoint,
        output,
    )

    save_json(
        {
            "units": unit_metadata(),
            "coherent": asdict(coherent),
            "coherent_dimensionless": normalized_two_mode_parameters(
                coherent
            ),
            "dissipative_theta_1p15": asdict(dissipative),
            "dissipative_theta_1p15_dimensionless": normalized_two_mode_parameters(
                dissipative
            ),
            "dissipative_theta_1p30": asdict(dissipative_secondary),
            "dissipative_theta_1p30_dimensionless": normalized_two_mode_parameters(
                dissipative_secondary
            ),
            "dissipative_theta_pi_over_2": asdict(dissipative_endpoint),
            "dissipative_theta_pi_over_2_dimensionless": normalized_two_mode_parameters(
                dissipative_endpoint
            ),
            "theta_pi_over_2_endpoint_note": (
                "At theta=pi/2, g=(Gamma_col/2) sin(theta) cos(theta)=0; "
                "the endpoint is a stability reference, not a matched charging point."
            ),
            "coherent_margin_over_kappa0": (
                two_mode_stability_margin(coherent) / REFERENCE_RATE
            ),
            "dissipative_theta_1p15_margin_over_kappa0": (
                two_mode_stability_margin(dissipative) / REFERENCE_RATE
            ),
            "dissipative_theta_1p30_margin_over_kappa0": (
                two_mode_stability_margin(dissipative_secondary) / REFERENCE_RATE
            ),
            "dissipative_theta_pi_over_2_margin_over_kappa0": (
                two_mode_stability_margin(dissipative_endpoint) / REFERENCE_RATE
            ),
        },
        output.summaries / "stability_parameters.json",
    )


def plot_normalized_scan_composite(
    theta_delta: pd.DataFrame,
    theta_gamma: pd.DataFrame,
    output: OutputPaths,
) -> None:
    theta1 = np.sort(theta_delta["theta"].unique())
    delta = np.sort(theta_delta["delta_over_kappa0"].unique())
    matrix1 = (
        theta_delta.pivot(
            index="delta_over_kappa0",
            columns="theta",
            values="kappa_a_up_crit_over_kappa0",
        )
        .reindex(index=delta, columns=theta1)
        .to_numpy()
    )

    theta2 = np.sort(theta_gamma["theta"].unique())
    gamma = np.sort(theta_gamma["Gamma_col_over_kappa0"].unique())
    matrix2 = (
        theta_gamma.pivot(
            index="Gamma_col_over_kappa0",
            columns="theta",
            values="kappa_a_up_crit_over_kappa0",
        )
        .reindex(index=gamma, columns=theta2)
        .to_numpy()
    )

    finite_values = np.concatenate(
        [matrix1[np.isfinite(matrix1)], matrix2[np.isfinite(matrix2)]]
    )
    vmin = float(np.min(finite_values))
    vmax_raw = float(np.max(finite_values))
    # Clip only the very top end of the scale so that the broad variation at
    # larger theta remains visible instead of being washed out by a few high
    # threshold values.
    vmax = float(np.quantile(finite_values, 0.8))
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmax_raw
    colorbar_extend = "max" if vmax < vmax_raw - 1.0e-12 else "neither"

    figure_fontsize = MAIN_TEXT_FONT_SIZE + 1.0
    line_theta = 1.15

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(DOUBLE_COLUMN_WIDTH - 1.5, 1.75),
        constrained_layout=True,
    )
    image1 = axes[0].imshow(
        matrix1,
        origin="lower",
        aspect="auto",
        extent=(theta1[0], theta1[-1], delta[0], delta[-1]),
        cmap=NPG_SEQUENTIAL,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        rasterized=True,
    )
    image2 = axes[1].imshow(
        matrix2,
        origin="lower",
        aspect="auto",
        extent=(theta2[0], theta2[-1], gamma[0], gamma[-1]),
        cmap=NPG_SEQUENTIAL,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        rasterized=True,
    )
    axes[0].set_xlabel(r"$\theta$", fontsize=figure_fontsize)
    axes[0].set_ylabel(r"$\delta/\kappa_0$", fontsize=figure_fontsize)
    axes[1].set_xlabel(r"$\theta$", fontsize=figure_fontsize)
    axes[1].set_ylabel(r"$\Gamma_{\rm col}/\kappa_0$", fontsize=figure_fontsize)
    for label, ax in zip(("(a)", "(b)"), axes):
        _axis_common(ax)
        ax.tick_params(labelsize=figure_fontsize)
        # White underlay plus black dotted guide keeps theta=1.15 visible over
        # both dark and light parts of the colormap.
        ax.axvline(line_theta, color="white", linestyle=":", linewidth=2.2, zorder=4)
        ax.axvline(line_theta, color="black", linestyle=":", linewidth=1.1, zorder=5)
        _panel_label(ax, label)
        if ax.texts:
            ax.texts[-1].set_fontsize(figure_fontsize)

    colorbar = fig.colorbar(
        image2,
        ax=axes,
        pad=0.02,
        shrink=0.95,
        extend=colorbar_extend,
    )
    colorbar.set_label(
        r"$\kappa_{a,\uparrow}^{\rm crit,diss}/\kappa_0$",
        fontsize=figure_fontsize,
    )
    colorbar.ax.tick_params(labelsize=figure_fontsize)
    _style_colorbar(colorbar)
    colorbar.ax.tick_params(labelsize=figure_fontsize)
    colorbar.ax.yaxis.label.set_size(figure_fontsize)
    save_figure(
        fig,
        output.figures / "FigS1_normalized_threshold_scans.pdf",
        apply_tight_layout=False,
    )


def recipe_normalized_scans(output: OutputPaths) -> None:
    paper = PaperParameters()
    base = replace(paper.dissipative_base(), kappa_a_up=0.0)
    theta_values = np.linspace(0.08, np.pi / 2.0 - 0.01, 151)
    delta_values = np.linspace(0.0, 0.15, 121)
    gamma_values = np.linspace(0.05, 8.0, 151)
    theta_delta = plot_critical_pump_theta_delta(
        base,
        theta_values,
        delta_values,
        output,
        "critical_pump_theta_delta",
        pump_upper=0.5,
    )
    theta_gamma = plot_critical_pump_theta_gamma(
        base,
        theta_values,
        gamma_values,
        output,
        "critical_pump_theta_Gamma_col",
        pump_upper=0.5,
    )
    plot_normalized_scan_composite(
        theta_delta,
        theta_gamma,
        output,
    )


def plot_matched_growth_main_figure(
    coherent_result: dict[str, Any],
    dissipative_result: dict[str, Any],
    output: OutputPaths,
) -> None:
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(SINGLE_COLUMN_WIDTH, 3.45),
    )
    axes = np.asarray(axes)

    # Thin only the coherent curves in panels (a) and (c), where resolving the
    # oscillations benefits from a lighter stroke.  Keep the dissipative curves in
    # panels (b) and (d) at their original thickness.
    fig2_linewidth_left = 0.80
    fig2_linewidth_right = 1.50
    fig2_inset_linewidth_left = 0.75
    fig2_inset_linewidth_right = 1.25
    # Time window chosen to show the early ergotropy-fraction oscillations.
    fig2_inset_time_end = 40.0  # kappa_0 * t

    # The main energy panels now show only total energy and ergotropy.  The
    # ergotropy fraction is displayed directly in an inset so that Figs. 2 and 3
    # use a consistent set of quality metrics, as requested by the collaborator.
    energy_styles = (
        ("energy_total", r"$E_b$", "-", NPG_COLORS[0]),
        ("ergotropy", r"$\mathcal{W}_b$", "--", NPG_COLORS[1]),
    )
    for column_index, result in enumerate(
        (coherent_result, dissipative_result)
    ):
        battery = result["b"]
        omega_b = float(result["params"].omega_b)
        tau = normalized_time(battery["t"].to_numpy())
        for column, _, linestyle, color in energy_styles:
            axes[0, column_index].plot(
                tau,
                battery[column].to_numpy() / omega_b,
                linestyle=linestyle,
                color=color,
                linewidth=(
                    fig2_linewidth_left
                    if column_index == 0
                    else fig2_linewidth_right
                ),
            )

        inset = axes[0, column_index].inset_axes(
            [0.27, 0.325, 0.46, 0.35]
        )
        inset.plot(
            tau,
            battery["ergotropy_fraction"].to_numpy(),
            color=NPG_COLORS[2],
            linewidth=(
                fig2_inset_linewidth_left
                if column_index == 0
                else fig2_inset_linewidth_right
            ),
        )
        # Give the dissipative inset in panel (b) a little extra headroom so
        # its upper oscillations do not visually merge with the inset frame.
        inset.set_ylim(-0.03, 1.08 if column_index == 1 else 1.03)
        inset.set_yticks([0.0, 1.0])
        inset.set_xlim(0.0, fig2_inset_time_end)
        inset.set_xticks([0.0, fig2_inset_time_end])
        inset.set_ylabel(
            r"$\eta_{\mathcal{W}}$",
            fontsize=MAIN_TEXT_FONT_SIZE,
            labelpad=0.0,
        )
        inset.set_xlabel(
            r"$\kappa_0 t$",
            fontsize=MAIN_TEXT_FONT_SIZE,
            labelpad=0.0,
        )
        inset.tick_params(
            direction="in",
            top=True,
            right=True,
            labelsize=MAIN_TEXT_FONT_SIZE,
            width=0.8,
            length=2.5,
        )

        axes[1, column_index].semilogy(
            tau,
            np.maximum(battery["snr"], 1e-16),
            color=(
                NPG_COLORS[0]
                if column_index == 0
                else NPG_COLORS[1]
            ),
            linewidth=(
                fig2_linewidth_left
                if column_index == 0
                else fig2_linewidth_right
            ),
        )

    # New phase-locking diagnostics in the centers of the SNR panels.
    # v_a=<a> and v_b=<b>; the plotted relative phase is defined only when
    # both coherent amplitudes are nonzero.
    coherent_relative_phase = _add_fig2_phase_inset(
        axes[1, 0],
        coherent_result,
        color=NPG_COLORS[0],
        linewidth=1.15,
    )
    dissipative_relative_phase = _add_fig2_phase_inset(
        axes[1, 1],
        dissipative_result,
        color=NPG_COLORS[1],
        linewidth=1.50,
    )

    axes[0, 0].set_ylabel(r"Energy$/\omega_b$")

    # Panel (b): force the scientific multiplier to be rendered as
    # ``\times 10^6`` and keep it on the right, as in the original code.
    panel_b_formatter = ScalarFormatter(useMathText=True)
    panel_b_formatter.set_scientific(True)
    panel_b_formatter.set_powerlimits((6, 6))
    axes[0, 1].yaxis.set_major_formatter(panel_b_formatter)
    axes[0, 1].yaxis.set_offset_position("right")
    axes[1, 0].set_ylabel(r"$\mathrm{SNR}_b$")
    axes[1, 0].set_xlabel(r"$\kappa_0 t$")
    axes[1, 1].set_xlabel(r"$\kappa_0 t$")

    for label, ax in zip(("(a)", "(b)", "(c)", "(d)"), axes.flat):
        _axis_common(ax)
        _panel_label(ax, label)

    legend_handles = [
        Line2D(
            [0], [0], color=color, linestyle=linestyle,
            linewidth=fig2_linewidth_left, label=label
        )
        for _, label, linestyle, color in energy_styles
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.970),
        fontsize=MAIN_TEXT_FONT_SIZE,
    )
    fig.subplots_adjust(
        left=0.18,
        right=0.98,
        bottom=0.13,
        top=0.87,
        wspace=0.34,
        hspace=0.22,
    )
    save_figure(
        fig,
        output.figures / "Fig2_matched_growth_single_column.pdf",
        apply_tight_layout=False,
    )

    merged = pd.DataFrame(
        {
            "t": coherent_result["b"]["t"],
            "kappa0_t": normalized_time(
                coherent_result["b"]["t"].to_numpy()
            ),
            "coherent_energy_over_omega_b": (
                coherent_result["b"]["energy_total"]
                / coherent_result["params"].omega_b
            ),
            "coherent_ergotropy_over_omega_b": (
                coherent_result["b"]["ergotropy"]
                / coherent_result["params"].omega_b
            ),
            "coherent_ergotropy_fraction": (
                coherent_result["b"]["ergotropy_fraction"]
            ),
            "coherent_passive_energy_over_omega_b": (
                coherent_result["b"]["passive_energy"]
                / coherent_result["params"].omega_b
            ),
            "coherent_snr": coherent_result["b"]["snr"],
            "coherent_relative_phase_rad": coherent_relative_phase,
            "dissipative_relative_phase_rad": dissipative_relative_phase,
            "dissipative_energy_over_omega_b": (
                dissipative_result["b"]["energy_total"]
                / dissipative_result["params"].omega_b
            ),
            "dissipative_ergotropy_over_omega_b": (
                dissipative_result["b"]["ergotropy"]
                / dissipative_result["params"].omega_b
            ),
            "dissipative_ergotropy_fraction": (
                dissipative_result["b"]["ergotropy_fraction"]
            ),
            "dissipative_passive_energy_over_omega_b": (
                dissipative_result["b"]["passive_energy"]
                / dissipative_result["params"].omega_b
            ),
            "dissipative_snr": dissipative_result["b"]["snr"],
        }
    )
    save_dataframe(
        merged,
        output.data / "Fig2_matched_growth_data.csv",
    )


def recipe_dynamics(output: OutputPaths) -> None:
    paper = PaperParameters()
    coherent, dissipative = paper.matched_pair()
    maximum_target = 100.0
    coherent_result = simulate_until_target(coherent, maximum_target)
    dissipative_result = simulate_until_target(dissipative, maximum_target)

    save_json(
        {
            "units": unit_metadata(),
            "coherent": asdict(coherent),
            "coherent_dimensionless": normalized_two_mode_parameters(coherent),
            "dissipative": asdict(dissipative),
            "dissipative_dimensionless": normalized_two_mode_parameters(dissipative),
            "matched_growth_margin": paper.target_growth_margin,
            "matched_growth_margin_over_kappa0": (
                paper.target_growth_margin / REFERENCE_RATE
            ),
            "benchmark_theta": paper.theta,
            "matched_abs_g_over_kappa0": (
                abs(dissipative.dissipative_g) / REFERENCE_RATE
            ),
        },
        output.summaries / "Fig2_full_master_equation_parameters.json",
    )

    print("\nFull matched-growth master-equation parameters")
    print(json.dumps(_json_ready({
        "coherent": normalized_two_mode_parameters(coherent),
        "dissipative": normalized_two_mode_parameters(dissipative),
        "lambda_max_over_kappa0": (
            paper.target_growth_margin / REFERENCE_RATE
        ),
    }), indent=2))

    plot_matched_growth_main_figure(
        coherent_result,
        dissipative_result,
        output,
    )
    plot_matched_growth_complex_ratio_figure(
        coherent_result,
        dissipative_result,
        output,
    )

    for label, result in (
        ("coherent", coherent_result),
        ("dissipative", dissipative_result),
    ):
        battery = result["b"]
        final = battery.iloc[-1]
        print(
            f"{label}: margin/kappa0="
            f"{result['margin'] / REFERENCE_RATE:+.8e}, "
            f"final n_b={final['n_total']:.8e}, "
            f"final W_b/E_b={final['ergotropy_fraction']:.8e}, "
            f"final SNR={final['snr']:.8e}"
        )


def plot_matched_growth_complex_ratio_figure(
    coherent_result: dict[str, Any],
    dissipative_result: dict[str, Any],
    output: OutputPaths,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(SINGLE_COLUMN_WIDTH, 3.45))
    axes = np.asarray(axes)

    fig2_linewidth_left = 0.80
    fig2_linewidth_right = 1.50
    fig2_inset_linewidth_left = 0.75
    fig2_inset_linewidth_right = 1.25

    energy_styles = (
        ("energy_total", r"$E_b$", "-", NPG_COLORS[0]),
        ("ergotropy", r"$\mathcal{W}_b$", "--", NPG_COLORS[1]),
    )
    for column_index, result in enumerate((coherent_result, dissipative_result)):
        battery = result["b"]
        omega_b = float(result["params"].omega_b)
        tau = normalized_time(battery["t"].to_numpy())
        for column, _, linestyle, color in energy_styles:
            axes[0, column_index].plot(
                tau,
                battery[column].to_numpy() / omega_b,
                linestyle=linestyle,
                color=color,
                linewidth=(fig2_linewidth_left if column_index == 0 else fig2_linewidth_right),
            )

        inset = axes[0, column_index].inset_axes([0.27, 0.325, 0.46, 0.35])
        inset.plot(
            tau,
            battery["ergotropy_fraction"].to_numpy(),
            color=NPG_COLORS[2],
            linewidth=(fig2_inset_linewidth_left if column_index == 0 else fig2_inset_linewidth_right),
        )
        inset.set_ylim(-0.03, 1.08 if column_index == 1 else 1.03)
        inset.set_yticks([0.0, 1.0])
        inset.set_xticks([])
        inset.set_title(r"$\eta_{\mathcal{W}}$", fontsize=MAIN_TEXT_FONT_SIZE, pad=0.0)
        inset.tick_params(direction="in", top=True, right=True, labelsize=MAIN_TEXT_FONT_SIZE, width=0.8, length=2.5)

    coherent_ratio = complex_amplitude_ratio(coherent_result["a"], coherent_result["b"])
    dissipative_ratio = complex_amplitude_ratio(dissipative_result["a"], dissipative_result["b"])

    axes[1, 0].plot(
        coherent_ratio.real,
        coherent_ratio.imag,
        color=NPG_COLORS[0],
        linewidth=1.10,
    )
    axes[1, 1].plot(
        dissipative_ratio.real,
        dissipative_ratio.imag,
        color=NPG_COLORS[1],
        linewidth=1.50,
    )
    for ax, ratio in ((axes[1,0], coherent_ratio), (axes[1,1], dissipative_ratio)):
        _set_complex_plane_limits(ax, ratio)
        ax.axhline(0.0, color="0.82", linewidth=0.7, zorder=0)
        ax.axvline(0.0, color="0.82", linewidth=0.7, zorder=0)

    axes[0, 0].set_ylabel(r"Energy$/\omega_b$")
    panel_b_formatter = ScalarFormatter(useMathText=True)
    panel_b_formatter.set_scientific(True)
    panel_b_formatter.set_powerlimits((6, 6))
    axes[0, 1].yaxis.set_major_formatter(panel_b_formatter)
    axes[0, 1].yaxis.set_offset_position("right")

    axes[1, 0].set_ylabel(r"$\mathrm{Im}[\langle b\rangle/\langle a\rangle]$")
    axes[1, 0].set_xlabel(r"$\mathrm{Re}[\langle b\rangle/\langle a\rangle]$")
    axes[1, 1].set_xlabel(r"$\mathrm{Re}[\langle b\rangle/\langle a\rangle]$")

    for label, ax in zip(("(a)", "(b)", "(c)", "(d)"), axes.flat):
        _axis_common(ax)
        _panel_label(ax, label)

    legend_handles = [
        Line2D([0], [0], color=color, linestyle=linestyle, linewidth=fig2_linewidth_left, label=label)
        for _, label, linestyle, color in energy_styles
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.970),
        fontsize=MAIN_TEXT_FONT_SIZE,
    )
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.13, top=0.87, wspace=0.34, hspace=0.22)
    save_figure(
        fig,
        output.figures / "Fig2_matched_growth_complex_ratio_single_column.pdf",
        apply_tight_layout=False,
    )

    ratio_df = pd.DataFrame({
        "t": coherent_result["b"]["t"],
        "kappa0_t": normalized_time(coherent_result["b"]["t"].to_numpy()),
        "coherent_ratio_real": coherent_ratio.real,
        "coherent_ratio_imag": coherent_ratio.imag,
        "dissipative_ratio_real": dissipative_ratio.real,
        "dissipative_ratio_imag": dissipative_ratio.imag,
    })
    ratio_df.to_csv(output.data / "Fig2_matched_growth_complex_ratio_data.csv", index=False)




# =============================================================================
# Additive phase-locking diagnostic (no existing figure is modified)
# =============================================================================

def normalized_intermode_coherence(
    charger: pd.DataFrame,
    battery: pd.DataFrame,
) -> np.ndarray:
    """Bounded complex phase-coherence diagnostic built from first moments.

    chi(t) = <b(t)> <a(t)>* / (|<a(t)>|^2 + |<b(t)>|^2).

    For two nonzero first moments, arg chi = arg<b> - arg<a>, and |chi|<=1/2.
    Unlike the ratio <b>/<a>, chi does not diverge when <a> crosses zero.
    At times when *both* first moments vanish, the phase is undefined, and
    the diagnostic is recorded as NaN rather than assigning a spurious phase.
    """
    v_a = (
        charger["amplitude_real"].to_numpy(dtype=float)
        + 1j * charger["amplitude_imag"].to_numpy(dtype=float)
    )
    v_b = (
        battery["amplitude_real"].to_numpy(dtype=float)
        + 1j * battery["amplitude_imag"].to_numpy(dtype=float)
    )
    if v_a.shape != v_b.shape:
        raise ValueError("Charger and battery have different time-grid shapes.")
    denominator = np.abs(v_a) ** 2 + np.abs(v_b) ** 2
    coherence = np.full(v_a.shape, np.nan + 1j * np.nan, dtype=complex)
    valid = denominator > 1.0e-28
    coherence[valid] = v_b[valid] * np.conjugate(v_a[valid]) / denominator[valid]
    if np.nanmax(np.abs(coherence)) > 0.500000001:
        raise AssertionError("The coherence exceeds its exact |chi|<=1/2 bound.")
    return coherence


def dominant_mode_coherence(params: TwoModeParams) -> tuple[complex, complex]:
    """Coherence of a unique fastest-growing first-moment right eigenmode."""
    eigvals, eigvecs = np.linalg.eig(two_mode_complex_drift(params))
    index = int(np.argmax(eigvals.real))
    others = [j for j in range(len(eigvals)) if j != index]
    if others and max(float(eigvals[j].real) for j in others) >= float(eigvals[index].real) - 1e-11:
        raise ValueError("No unique dominant-growth mode; a fixed point is not certified.")
    v_a, v_b = eigvecs[:, index]
    norm = float(abs(v_a) ** 2 + abs(v_b) ** 2)
    if norm <= 0.0:
        raise ArithmeticError("Dominant eigenvector has zero norm.")
    chi_infty = v_b * np.conjugate(v_a) / norm
    return complex(chi_infty), complex(eigvals[index])


def propagated_driven_first_moments(
    params: TwoModeParams,
    times: np.ndarray,
    initial: np.ndarray,
) -> np.ndarray:
    """Solve dot v=M v+(-i E,0)^T by its two-mode spectral decomposition.

    This computes only first moments, since covariance noise does not enter
    their equation. The externally driven dynamics and all rates are exactly
    those of the paper; only the initial coherent amplitudes are varied.
    """
    matrix = two_mode_complex_drift(params)
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    inverse = np.linalg.inv(eigenvectors)
    initial = np.asarray(initial, dtype=complex)
    times = np.asarray(times, dtype=float)
    if initial.shape != (2,):
        raise ValueError("A two-component initial coherent-amplitude vector is required.")
    source = np.array([-1j * params.drive, 0.0j], dtype=complex)
    modal_initial = inverse @ initial
    modal_source = inverse @ source
    exponents = eigenvalues[:, None] * times[None, :]
    expfactors = np.exp(exponents)
    propagator_integral = np.empty_like(expfactors)
    for i, ev in enumerate(eigenvalues):
        if abs(ev) < 1.0e-14:
            propagator_integral[i, :] = times
        else:
            propagator_integral[i, :] = np.expm1(exponents[i]) / ev
    modes = (
        expfactors * modal_initial[:, None]
        + propagator_integral * modal_source[:, None]
    )
    return eigenvectors @ modes


def projective_complex_coherence(trajectory: np.ndarray) -> np.ndarray:
    """The bounded order parameter chi from a (2,N) amplitude trajectory."""
    if trajectory.ndim != 2 or trajectory.shape[0] != 2:
        raise ValueError("The amplitude trajectory must have shape (2,N).")
    a, b = trajectory
    denom = np.abs(a) ** 2 + np.abs(b) ** 2
    chi = np.full(a.shape, np.nan + 1j * np.nan, dtype=complex)
    mask = denom > 1.0e-28
    chi[mask] = b[mask] * np.conjugate(a[mask]) / denom[mask]
    if np.nanmax(abs(chi)) > 0.500000001:
        raise AssertionError("The exact normalized-coherence bound was violated.")
    return chi


def plot_phase_locking_fixed_point_diagnostic(output: OutputPaths) -> None:
    """Four initial phases reveal the dissipative fixed-point attractor.

    This is an additional standalone figure. No existing figure function,
    manuscript parameter, established output file, or dataset is altered.
    The original charging figures continue to start from zero first moments.
    This diagnostic instead varies only the initial coherent displacements.
    """
    paper = PaperParameters()
    coherent, dissipative = paper.matched_pair()
    time = np.linspace(0.0, 20.0 / REFERENCE_RATE, 4001)
    tau = normalized_time(time)
    chi_infty, dominant_ev = dominant_mode_coherence(dissipative)

    # These four displacements have equal initial occupations in the two modes,
    # and sample all four quadrants of the initial relative phase.
    seed_amplitude = 5.0
    phases = (0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi)
    names = ("0", "pi_over_2", "pi", "3pi_over_2")
    colors = (NPG_COLORS[0], NPG_COLORS[3], NPG_COLORS[2], NPG_COLORS[8])
    legend_labels = (r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$")

    coherent_curves: list[np.ndarray] = []
    dissipative_curves: list[np.ndarray] = []
    for phi0 in phases:
        initial = np.array(
            [seed_amplitude, seed_amplitude * np.exp(1j * phi0)],
            dtype=complex,
        )
        a_coherent = propagated_driven_first_moments(coherent, time, initial)
        a_dissipative = propagated_driven_first_moments(dissipative, time, initial)
        coherent_curves.append(projective_complex_coherence(a_coherent))
        dissipative_curves.append(projective_complex_coherence(a_dissipative))

    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COLUMN_WIDTH, 4.65))
    axes = np.asarray(axes)
    fig.patch.set_facecolor("white")

    for j, (phi0, color) in enumerate(zip(phases, colors)):
        chi_c = coherent_curves[j]
        chi_d = dissipative_curves[j]
        early_c = tau <= 4.0
        early_d = tau <= 8.0
        for ax, chi, early in (
            (axes[0, 0], chi_c, early_c),
            (axes[0, 1], chi_d, early_d),
        ):
            ax.plot(
                chi.real[early], chi.imag[early],
                color=color, linewidth=1.15, alpha=0.90,
                solid_capstyle="round", zorder=3,
            )
            # Open circle denotes the initial seed, not a time-series marker.
            ax.plot(
                [chi[0].real], [chi[0].imag], marker="o", markersize=4.0,
                markerfacecolor="white", markeredgecolor=color,
                markeredgewidth=1.0, linestyle="None", zorder=6,
            )
        axes[1, 0].plot(
            tau, chi_c.imag, color=color, lw=1.15, alpha=0.92,
        )
        axes[1, 1].semilogy(
            tau, np.maximum(np.abs(chi_d - chi_infty), 1e-14),
            color=color, lw=1.3, alpha=0.96,
        )

    for ax, label, scheme in (
        (axes[0, 0], "(a)", "Coherent"),
        (axes[0, 1], "(b)", "Dissipative"),
    ):
        ax.axhline(0.0, color="0.80", linewidth=0.70, zorder=0)
        ax.axvline(0.0, color="0.80", linewidth=0.70, zorder=0)
        ax.set_xlim(-0.56, 0.56)
        ax.set_ylim(-0.56, 0.56)
        ax.set_xticks([-0.5, 0.0, 0.5])
        ax.set_yticks([-0.5, 0.0, 0.5])
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(r"$\operatorname{Re}\chi$")
        ax.set_ylabel(r"$\operatorname{Im}\chi$")
        ax.text(0.96, 0.95, scheme, transform=ax.transAxes,
                ha="right", va="top", fontsize=MAIN_TEXT_FONT_SIZE)
        _axis_common(ax)
        _panel_label(ax, label)

    axes[0, 1].plot(
        [chi_infty.real], [chi_infty.imag], marker="*", markersize=11,
        markerfacecolor=NPG_COLORS[2], markeredgecolor="black",
        markeredgewidth=0.8, linestyle="None", zorder=10,
    )

    axes[1, 0].axhline(0.0, color="0.8", linewidth=0.75, zorder=0)
    axes[1, 0].set_xlim(0.0, 4.0)
    axes[1, 0].set_ylim(-0.55, 0.55)
    axes[1, 0].set_yticks([-0.5, 0.0, 0.5])
    axes[1, 0].set_ylabel(r"$\operatorname{Im}\chi(t)$")
    axes[1, 0].set_xlabel(r"$\kappa_0 t$")
    _axis_common(axes[1, 0])
    _panel_label(axes[1, 0], "(c)")

    axes[1, 1].set_xlim(0.0, 20.0)
    axes[1, 1].set_ylim(2.0e-4, 1.0)
    axes[1, 1].set_xlabel(r"$\kappa_0 t$")
    axes[1, 1].set_ylabel(r"$|\chi(t)-\chi_\infty|$")
    _axis_common(axes[1, 1])
    _panel_label(axes[1, 1], "(d)")

    # Make all four plotting areas geometrically proportional.
    # Preserve equal data scales in the complex-plane panels (a) and (b).
    # Panels (c) and (d) retain their original data scales.
    axes[1, 0].set_box_aspect(1)
    axes[1, 1].set_box_aspect(1)
    handles = [Line2D([0], [0], color=c, linewidth=1.55, label=label)
        for c, label in zip(colors, legend_labels)]
    fig.legend(
        handles=handles, ncol=4, loc="upper center", frameon=False,
        bbox_to_anchor=(0.5, 0.996), fontsize=MAIN_TEXT_FONT_SIZE,
        handlelength=1.4, columnspacing=1.5, handletextpad=0.3,
    )
    fig.subplots_adjust(
        left=0.13, right=0.985, bottom=0.12, top=0.89,
        wspace=0.1, hspace=0.31,
    )
    pdf_path = output.figures / "FigS_phase_locking_fixed_point.pdf"
    save_figure(fig, pdf_path, apply_tight_layout=False)

    columns: dict[str, Any] = {"t": time, "kappa0_t": tau}
    for name, chi_c, chi_d in zip(names, coherent_curves, dissipative_curves):
        columns[f"coherent_chi_real_psi_{name}"] = chi_c.real
        columns[f"coherent_chi_imag_psi_{name}"] = chi_c.imag
        columns[f"dissipative_chi_real_psi_{name}"] = chi_d.real
        columns[f"dissipative_chi_imag_psi_{name}"] = chi_d.imag
        columns[f"dissipative_error_psi_{name}"] = abs(chi_d - chi_infty)
    save_dataframe(
        pd.DataFrame(columns),
        output.data / "phase_locking_fixed_point.csv",
    )
    save_json({
        "observable": "chi = <b> conjugate(<a>) / (|<a>|^2+|<b>|^2)",
        "argument_when_defined": "arg chi = arg<b> - arg<a>",
        "bound": "|chi| <= 0.5",
        "initial_states": "coherent displacements, nonvacuum diagnostic only",
        "initial_a_amplitude": seed_amplitude,
        "initial_b_amplitudes": [seed_amplitude * np.exp(1j * p) for p in phases],
        "initial_relative_phases": list(phases),
        "drive_unchanged": True,
        "coherent_behavior": "persistent relative oscillations",
        "dissipative_behavior": "all initial phases converge to the same dominant eigenmode",
        "dissipative_chi_infty": chi_infty,
        "dissipative_dominant_eigenvalue_over_kappa0": dominant_ev / REFERENCE_RATE,
        "dissipative_final_errors": [float(abs(ch[-1] - chi_infty)) for ch in dissipative_curves],
        "matched_growth_parameters": {
            "coherent": normalized_two_mode_parameters(coherent),
            "dissipative": normalized_two_mode_parameters(dissipative),
        },
    }, output.summaries / "phase_locking_fixed_point.json")
    print(
        "Dissipative coherence fixed point: "
        f"chi_inf={chi_infty.real:+.9f}{chi_infty.imag:+.9f}j; "
        "all four driven coherent initial phases converge."
    )


def recipe_phase_locking(output: OutputPaths) -> None:
    """Generate only the additional phase-locking figure and its data."""
    plot_phase_locking_fixed_point_diagnostic(output)


def recipe_finite_capacity(output: OutputPaths) -> None:
    paper = PaperParameters()
    coherent, dissipative = paper.matched_pair()
    # Denser physical time sampling yields reliable local peak heights.
    coherent_result = simulate_until_target(coherent, 100.0, points=20001)
    dissipative_result = simulate_until_target(dissipative, 100.0, points=10001)
    targets, knees = dense_finite_capacity_targets(coherent_result)
    first_knee = float(knees[0])
    print(
        "First coherent first-passage jump at "
        f"n_star={first_knee:.6f}; "
        f"all record-high peak capacities: {np.round(knees, 4)}"
    )
    metrics = plot_finite_capacity_comparison(
        coherent_result,
        dissipative_result,
        targets,
        output,
        "finite_capacity_matched_growth",
        first_coherent_knee=first_knee,
    )
    plot_active_energy_accounting(
        metrics,
        output,
    )
    plot_gross_input_endmatter(
        metrics,
        coherent,
        dissipative,
        output,
        first_coherent_knee=first_knee,
    )
    save_json(
        {
            "units": unit_metadata(),
            "coherent": asdict(coherent),
            "coherent_dimensionless": normalized_two_mode_parameters(
                coherent
            ),
            "dissipative": asdict(dissipative),
            "dissipative_dimensionless": normalized_two_mode_parameters(
                dissipative
            ),
            "target_growth_margin": paper.target_growth_margin,
            "target_growth_margin_over_kappa0": (
                paper.target_growth_margin / REFERENCE_RATE
            ),
            "charging_efficiency_definition": (
                "eta_ch,*^net = W_b(t_ch) / E_active^net(t_ch), with "
                "E_active^net = integral[P_drive + P_gain] dt"
            ),
            "gross_input_endmatter_definition": (
                "eta_ch,*^gross = W_b(t_ch) / [E_gain(t_ch) + E_in^coh(t_ch)], "
                "E_in^coh = omega_drive |E|^2 t_ch / kappa_ext"
            ),
            "gross_input_default_kappa_ext_over_kappa_a_down": (
                GROSS_INPUT_EXTERNAL_COUPLING_FRACTION
            ),
            "power_symbol": "P_{W,*}",
            "power_name": "ergotropic power",
        },
        output.summaries / "finite_capacity_parameters.json",
    )




def recipe_gross_input(output: OutputPaths) -> pd.DataFrame:
    """Generate only the End-Matter gross-input comparison and its data."""
    paper = PaperParameters()
    coherent, dissipative = paper.matched_pair()
    coherent_result = simulate_until_target(coherent, 100.0, points=20001)
    dissipative_result = simulate_until_target(dissipative, 100.0, points=10001)
    targets, knees = dense_finite_capacity_targets(coherent_result)
    first_knee = float(knees[0])
    coherent_table = finite_capacity_table(
        coherent_result,
        targets,
        "coherent",
    )
    dissipative_table = finite_capacity_table(
        dissipative_result,
        targets,
        "dissipative",
    )
    metrics = pd.concat(
        [coherent_table, dissipative_table],
        ignore_index=True,
    )
    save_dataframe(
        metrics,
        output.data / "gross_input_base_finite_capacity_metrics.csv",
    )
    return plot_gross_input_endmatter(
        metrics,
        coherent,
        dissipative,
        output,
        first_coherent_knee=first_knee,
    )


def plot_theta_eta_snr_battery_fraction(
    dataframe: pd.DataFrame,
    output: OutputPaths,
    n_star: float = 100.0,
    benchmark_theta: float = PAPER_THETA,
) -> None:
    """Two-panel fixed-growth theta scan with distinct, physical colorbars.

    Panel (a) is eta_W,* with color p_b=n_b/(n_a+n_b) at first passage;
    panel (b) is SNR_b,* with color Gamma_col/kappa0 (log normalization).
    Both panels mark the matched operating point theta=1.15 with a dotted
    vertical line, as requested by Lei.
    """
    required = (
        "theta", "etaW_at_n100", "snr_at_n100",
        "p_b_at_n100", "Gamma_col_over_kappa0",
    )
    missing = [key for key in required if key not in dataframe.columns]
    if missing:
        raise ValueError(f"Missing theta-scan columns: {missing}")

    ordered = dataframe.sort_values("theta").reset_index(drop=True)
    theta = ordered["theta"].to_numpy(dtype=float)
    eta = ordered["etaW_at_n100"].to_numpy(dtype=float)
    snr = ordered["snr_at_n100"].to_numpy(dtype=float)
    p_b = ordered["p_b_at_n100"].to_numpy(dtype=float)
    gamma = ordered["Gamma_col_over_kappa0"].to_numpy(dtype=float)

    if theta.size < 2:
        raise ValueError("At least two theta values are required.")
    if not np.all(np.isfinite(theta) & np.isfinite(eta) & np.isfinite(snr)
                  & np.isfinite(p_b) & np.isfinite(gamma)):
        raise ValueError("Nonfinite theta, quality, SNR, p_b, or Gamma_col data.")
    if np.any(snr <= 0.0) or np.any(gamma <= 0.0):
        raise ValueError("SNR and collective decay must be strictly positive.")
    if np.any((p_b < 0.0) | (p_b > 1.0)):
        raise ValueError("The battery fraction must lie in [0, 1].")

    p_min, p_max = float(p_b.min()), float(p_b.max())
    if np.isclose(p_min, p_max, atol=1.0e-14, rtol=0.0):
        p_min, p_max = max(0, p_min - 0.005), min(1, p_max + 0.005)
    norms = (
        Normalize(vmin=p_min, vmax=p_max),
        LogNorm(vmin=float(gamma.min()), vmax=float(gamma.max())),
    )
    # Keep the battery-fraction colors unchanged in (a). Encode the
    # collective decay in (b) using a distinct, perceptually uniform palette.
    battery_cmap = NPG_SEQUENTIAL
    gamma_cmap = mpl.colormaps["plasma"]
    fig, axes = plt.subplots(
        1, 2,
        figsize=(DOUBLE_COLUMN_WIDTH, 2.0),
        constrained_layout=True,
    )
    theta_axis_font_size = 12.0
    theta_tick_font_size = 11.0
    for i, (ax, y, colors, norm, ylabel, label) in enumerate((
        (axes[0], eta, p_b, norms[0], r"$\eta_{\mathcal{W},\star}$", "(a)"),
        (axes[1], snr, gamma, norms[1], r"$\mathrm{SNR}_{b,\star}$", "(b)"),
    )):
        cmap = battery_cmap if i == 0 else gamma_cmap
        xy = np.column_stack((theta, y))
        segments = np.stack((xy[:-1], xy[1:]), axis=1)
        curve = LineCollection(
            segments, cmap=cmap, norm=norm,
            linewidths=2.8, zorder=2,
        )
        # For LogNorm, use a geometric mean of adjacent strictly positive
        # gamma values so that midpoint colors are consistent in log space.
        if i == 1:
            colors_segment = np.sqrt(colors[:-1] * colors[1:])
        else:
            colors_segment = 0.5 * (colors[:-1] + colors[1:])
        curve.set_array(colors_segment)
        ax.add_collection(curve)
        ax.set_xlim(float(theta[0]), float(theta[-1]))
        ax.set_xlabel(r"$\theta$")
        ax.set_ylabel(ylabel)
        if i == 1:
            ax.set_yscale("log")
        else:
            y_span = max(float(y.max() - y.min()), 1.0e-6)
            ax.set_ylim(
                max(0.0, float(y.min()) - 0.08 * y_span),
                min(1.02, float(y.max()) + 0.08 * y_span),
            )
        ax.margins(y=0.09)
        ax.axvline(
            benchmark_theta,
            color="#202020", linestyle=":",
            linewidth=1.15, zorder=6,
        )
        _axis_common(ax)
        _panel_label(ax, label)
        # Enlarge lettering only in this supplemental figure.
        ax.xaxis.label.set_fontsize(theta_axis_font_size+4)
        ax.yaxis.label.set_fontsize(theta_axis_font_size+4)
        ax.tick_params(which="both", labelsize=theta_tick_font_size+4)
        ax.texts[-1].set_fontsize(theta_axis_font_size+4)
        scalar = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        scalar.set_array(colors)
        colorbar = fig.colorbar(
            scalar, ax=ax,
            fraction=0.057, pad=0.025, shrink=0.96,
        )
        colorbar.set_label(
            r"$p_b=n_b/(n_a+n_b)$" if i == 0
            else r"$\Gamma_{\rm col}/\kappa_0$"
        )
        if i == 1:
            possible_ticks = [15.0, 30.0, 100.0, 300.0]
            ticks = [x for x in possible_ticks if gamma.min() <= x <= gamma.max()]
            colorbar.set_ticks(ticks)
            colorbar.set_ticklabels([f"{value:g}" for value in ticks])
        _style_colorbar(colorbar)
        colorbar.ax.tick_params(labelsize=theta_tick_font_size+4)
        colorbar.ax.yaxis.label.set_fontsize(theta_axis_font_size+4)

    path = output.figures / "FigS_theta_eta_snr_battery_fraction.pdf"
    save_figure(fig, path, apply_tight_layout=False)
    benchmark = ordered.iloc[int(np.argmin(abs(theta - benchmark_theta)))]
    if not np.isclose(float(benchmark["theta"]), benchmark_theta, atol=1.0e-11):
        raise ValueError("Theta scan must include the benchmark angle exactly.")
    save_json(
        {
            "figure": path.name,
            "n_star": float(n_star),
            "first_passage_benchmark_theta": float(benchmark_theta),
            "benchmark_p_b": float(benchmark["p_b_at_n100"]),
            "benchmark_Gamma_col_over_kappa0": float(
                benchmark["Gamma_col_over_kappa0"]
            ),
            "panel_a_color": "p_b=n_b(t_ch)/(n_a(t_ch)+n_b(t_ch))",
            "panel_b_color": "Gamma_col/kappa0, logarithmic color normalization",
            "panel_a_colormap": "npg_sequential",
            "panel_b_colormap": "plasma",
            "panel_a_color_range": [p_min, p_max],
            "panel_b_color_range": [float(gamma.min()), float(gamma.max())],
            "scan_convention": "fixed |g| and matched maximal growth rate",
        },
        output.summaries / "FigS_theta_eta_snr_battery_fraction.json",
    )



def recipe_theta_tradeoff(output: OutputPaths) -> pd.DataFrame:
    """
    Diagnostic scan supporting the moderate-theta benchmark choice.

    The off-diagonal dissipative strength |g| and the target growth margin are
    held fixed while theta is varied.  Gamma_col and kappa_a_up are recalculated
    at each point.  The scan is deliberately saved outside Figs_New so that it
    does not lengthen the paper or SM unless explicitly needed.
    """
    paper = PaperParameters()
    theta_values = np.linspace(0.75, 1.55, 25)
    target = 100.0
    rows: list[dict[str, float]] = []

    for theta in theta_values:
        gamma_col = (
            2.0
            * paper.matched_offdiagonal_strength
            / (np.sin(theta) * np.cos(theta))
        )
        base = replace(
            paper.dissipative_base(),
            kappa_a_up=0.0,
            Gamma_col=float(gamma_col),
            theta=float(theta),
        )
        pump = pump_for_target_margin(
            base,
            paper.target_growth_margin,
            lower=0.0,
            upper=0.2,
        )
        params = replace(base, kappa_a_up=pump)
        result = simulate_until_target(
            params,
            target,
            points=3001,
        )
        crossing_time, index, weight = first_crossing(
            result["b"]["t"].to_numpy(),
            result["b"]["n_total"].to_numpy(),
            target,
        )
        battery_row = interpolate_dataframe_row(
            result["b"], index, weight
        )
        charger_row = interpolate_dataframe_row(
            result["a"], index, weight
        )

        drift = two_mode_complex_drift(params)
        eigenvalues, eigenvectors = np.linalg.eig(drift)
        dominant = int(np.argmax(np.real(eigenvalues)))
        vector = eigenvectors[:, dominant]
        battery_fraction = float(
            abs(vector[1]) ** 2
            / np.sum(np.abs(vector) ** 2)
        )
        modal_ratio = float(
            abs(vector[0]) ** 2
            / max(abs(vector[1]) ** 2, 1.0e-300)
        )

        rows.append(
            {
                "theta": float(theta),
                "Gamma_col": float(gamma_col),
                "Gamma_col_over_kappa0": float(
                    gamma_col / REFERENCE_RATE
                ),
                "kappa_a_up": float(pump),
                "kappa_a_up_over_kappa0": float(
                    pump / REFERENCE_RATE
                ),
                "g_abs_over_kappa0": float(
                    abs(params.dissipative_g) / REFERENCE_RATE
                ),
                "kappa0_t_ch_n100": float(
                    crossing_time * REFERENCE_RATE
                ),
                "etaW_at_n100": float(
                    battery_row["ergotropy_fraction"]
                ),
                "snr_at_n100": float(battery_row["snr"]),
                "p_b_at_n100": float(
                    battery_row["n_total"]
                    / max(
                        battery_row["n_total"] + charger_row["n_total"],
                        1.0e-300,
                    )
                ),
                "n_a_over_n_b_at_n100": float(
                    charger_row["n_total"] / target
                ),
                "dominant_mode_battery_fraction": battery_fraction,
                "dominant_mode_n_a_over_n_b": modal_ratio,
            }
        )

    dataframe = pd.DataFrame(rows)
    save_dataframe(
        dataframe,
        output.data / "theta_tradeoff_performance.csv",
    )
    plot_theta_eta_snr_battery_fraction(
        dataframe,
        output,
        n_star=target,
        benchmark_theta=paper.theta,
    )

    fig, axes = plt.subplots(2, 2, figsize=(SINGLE_COLUMN_WIDTH, 3.45), sharex=True)
    axes = np.asarray(axes)
    axes[0, 0].plot(
        dataframe["theta"],
        dataframe["Gamma_col_over_kappa0"],
    )
    axes[0, 1].semilogy(
        dataframe["theta"],
        dataframe["dominant_mode_n_a_over_n_b"],
    )
    axes[1, 0].plot(
        dataframe["theta"],
        dataframe["etaW_at_n100"],
    )
    axes[1, 1].semilogy(
        dataframe["theta"],
        dataframe["snr_at_n100"],
    )
    axes[0, 0].set_ylabel(r"$\Gamma_{\rm col}/\kappa_0$")
    axes[0, 1].set_ylabel(r"mode $n_a/n_b$")
    axes[1, 0].set_ylabel(r"$\eta_{\mathcal{W},\star}$")
    axes[1, 1].set_ylabel(r"$\mathrm{SNR}_{b,\star}$")
    axes[1, 0].set_xlabel(r"$\theta$")
    axes[1, 1].set_xlabel(r"$\theta$")
    for label, ax in zip(("(a)", "(b)", "(c)", "(d)"), axes.flat):
        ax.axvline(
            paper.theta,
            color="blue",
            linestyle=":",
            linewidth=1.3,
        )
        _axis_common(ax)
        _panel_label(ax, label)
    fig.subplots_adjust(
        left=0.12,
        right=0.98,
        bottom=0.14,
        top=0.98,
        wspace=0.6,
        hspace=0.22,
    )
    save_figure(
        fig,
        output.diagnostics / "theta_tradeoff_diagnostic.pdf",
        apply_tight_layout=False,
    )

    selected_index = int(
        np.argmin(np.abs(dataframe["theta"].to_numpy() - paper.theta))
    )
    selected = dataframe.iloc[selected_index].to_dict()
    save_json(
        {
            "selected_benchmark": selected,
            "design_reason": (
                "moderate theta with fixed |g| and matched growth; "
                "retains high eta_W and SNR while greatly increasing "
                "battery participation and reducing Gamma_col"
            ),
        },
        output.summaries / "theta_tradeoff_selected.json",
    )
    print(
        "\nAt the Fig. 2/3 operating point theta="
        f"{paper.theta:.2f}, physical battery fraction "
        f"p_b={selected['p_b_at_n100']:.9f}; "
        f"Gamma_col/kappa0={selected['Gamma_col_over_kappa0']:.6f}"
    )
    print("\nModerate-theta benchmark diagnostic")
    print(json.dumps(_json_ready(selected), indent=2))
    return dataframe

def plot_detuning_supplement(
    results: dict[float, dict[str, Any]],
    output: OutputPaths,
) -> None:
    detunings = tuple(results.keys())

    fig_energy, axes_energy = plt.subplots(
        2,
        2,
        figsize=(DOUBLE_COLUMN_WIDTH-2, 4.45),
    )
    fig_snr, axes_snr = plt.subplots(
        2,
        2,
        figsize=(DOUBLE_COLUMN_WIDTH-2, 4.45),
    )
    energy_styles = (
        ("energy_total", r"$E_b$", "-", NPG_COLORS[0]),
        ("ergotropy", r"$\mathcal{W}_b$", "--", NPG_COLORS[1]),
        ("passive_energy", r"$E_b^{\rm pas}$", ":", NPG_COLORS[2]),
    )

    for index, (detuning, result) in enumerate(results.items()):
        ax_e = axes_energy.flat[index]
        ax_s = axes_snr.flat[index]
        battery = result["b"]
        omega_b = result["params"].omega_b
        tau = normalized_time(battery["t"].to_numpy())
        for column, _, linestyle, color in energy_styles:
            ax_e.plot(
                tau,
                battery[column] / omega_b,
                linestyle=linestyle,
                color=color,
            )
        ax_s.semilogy(
            tau,
            np.maximum(battery["snr"], 1e-16),
            color=NPG_COLORS[0],
        )
        for ax in (ax_e, ax_s):
            ax.set_xlabel(r"$\kappa_0 t$")
            _axis_common(ax)
            _panel_label(ax, f"({chr(97 + index)})")
            ax.text(
                0.8,
                0.8,
                rf"$\delta/\kappa_0={detuning / REFERENCE_RATE:g}$",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=MAIN_TEXT_FONT_SIZE,
            )
        ax_e.set_ylabel(r"Energy$/\omega_b$")
        ax_s.set_ylabel(r"$\mathrm{SNR}_b$")

    energy_handles = [
        Line2D([0], [0], color=color, linestyle=linestyle, label=label)
        for _, label, linestyle, color in energy_styles
    ]
    fig_energy.legend(
        energy_handles,
        [handle.get_label() for handle in energy_handles],
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
        fontsize=MAIN_TEXT_FONT_SIZE,
    )
    for fig in (fig_energy, fig_snr):
        fig.subplots_adjust(
            left=0.09,
            right=0.99,
            bottom=0.10,
            top=0.90,
            wspace=0.4,
            hspace=0.28,
        )

    save_figure(
        fig_energy,
        output.figures / "FigS2_detuning_energy.pdf",
        apply_tight_layout=False,
    )
    save_figure(
        fig_snr,
        output.figures / "FigS3_detuning_snr.pdf",
        apply_tight_layout=False,
    )


def recipe_detuning(output: OutputPaths) -> None:
    paper = PaperParameters()
    base = paper.dissipative_base()
    base_pump = pump_for_target_margin(
        base,
        paper.target_growth_margin,
        lower=0.0,
        upper=0.2,
    )
    detunings = (0.0, 0.01, 0.05, 0.10)
    summary_rows = []
    results: dict[float, dict[str, Any]] = {}
    for detuning in detunings:
        params = replace(
            base,
            delta=detuning,
            kappa_a_up=base_pump,
        )
        margin = two_mode_stability_margin(params)
        times = np.linspace(0.0, 3000.0, 4001)
        result = simulate_two_mode(params, times)
        results[detuning] = result
        summary_rows.append(
            {
                "delta": detuning,
                "delta_over_kappa0": detuning / REFERENCE_RATE,
                "stability_margin": margin,
                "stability_margin_over_kappa0": margin / REFERENCE_RATE,
                "maximum_snr": float(np.nanmax(result["b"]["snr"])),
                "final_snr": float(result["b"]["snr"].iloc[-1]),
                "final_ergotropy_fraction": float(
                    result["b"]["ergotropy_fraction"].iloc[-1]
                ),
            }
        )
    plot_detuning_supplement(results, output)
    save_dataframe(
        pd.DataFrame(summary_rows),
        output.data / "detuning_summary.csv",
    )


def recipe_modal_diagnostics(output: OutputPaths) -> None:
    paper = PaperParameters()
    coherent, dissipative = paper.matched_pair()
    frames = []
    for label, params in (("coherent", coherent), ("dissipative", dissipative)):
        amat, dmat, source = two_mode_gaussian_matrices(params)
        frame = modal_source_noise_diagnostics(amat, dmat, source)
        frame.insert(0, "case", label)
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    save_dataframe(result, output.data / "modal_source_noise_diagnostics.csv")
    print(result.to_string(index=False))


def recipe_mediator(output: OutputPaths) -> None:
    base = ThreeModeParams(
        kappa_a_down=0.01,
        kappa_a_up=0.011,
        kappa_b=0.01,
        kappa_c=1.5,
        delta=0.0,
        Delta_c=0.0,
        drive=0.1,
        Gamma_ac=1.0,
        Gamma_bc=1.0,
        theta_a=0.35,
        theta_b=0.45,
        phi_a=0.0,
        phi_b=0.0,
    )
    linewidths = (0.8, 1.5, 3.0)
    summary = []
    for figure_index, linewidth in enumerate(linewidths, start=6):
        params = replace(base, kappa_c=linewidth)
        times = np.linspace(0.0, 800.0, 2501)
        comparison = simulate_three_mode_full_and_effective(
            params,
            times,
        )
        stem = (
            f"FigS{figure_index}_mediator_"
            f"kappa_c_over_kappa0_{linewidth / REFERENCE_RATE:.0f}"
        )
        plot_full_effective_comparison(
            comparison,
            output,
            stem,
        )
        metadata = comparison["effective"]["metadata"]
        full = comparison["full"]["b"]
        effective = comparison["effective"]["b"]
        summary.append(
            {
                "kappa_c": linewidth,
                "kappa_c_over_kappa0": linewidth / REFERENCE_RATE,
                "min_K_eff_eigenvalue": float(
                    np.min(metadata["K_eff_eigenvalues"])
                ),
                "max_energy_relative_error": float(
                    np.max(
                        np.abs(
                            full["energy_total"]
                            - effective["energy_total"]
                        )
                        / np.maximum(
                            np.abs(full["energy_total"]),
                            1e-14,
                        )
                    )
                ),
                "max_ergotropy_relative_error": float(
                    np.max(
                        np.abs(
                            full["ergotropy"]
                            - effective["ergotropy"]
                        )
                        / np.maximum(
                            np.abs(full["ergotropy"]),
                            1e-14,
                        )
                    )
                ),
            }
        )
    save_dataframe(
        pd.DataFrame(summary),
        output.data / "mediator_convergence_summary.csv",
    )


def recipe_saturation(output: OutputPaths) -> None:
    raise RuntimeError(
        "The converged nonlinear calculation is intentionally maintained in "
        "the separate one-click file "
        "'saturable_gain_low_energy_normalized.py'.  The old unconverged "
        "high-energy saturation recipe has been disabled."
    )


# =============================================================================
# Command line and Jupyter execution
# =============================================================================

# Choose what should run when this code is executed inside Jupyter.
#
# Available choices:
#   "validate"
#   "stability"
#   "normalized-scans"
#   "dynamics"
#   "phase-locking"
#   "finite-capacity"
#   "gross-input"
#   "theta-tradeoff"
#   "detuning"
#   "modal-diagnostics"
#   "mediator"
#   "saturation"
#   "all-linear"
NOTEBOOK_COMMAND = "all-linear"
NOTEBOOK_OUTPUT = "paper_output_normalized"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate all numerical data and individual PDF panels "
            "for the paper."
        )
    )
    parser.add_argument(
        "command",
        choices=(
            "validate",
            "stability",
            "normalized-scans",
            "dynamics",
            "phase-locking",
            "finite-capacity",
            "gross-input",
            "theta-tradeoff",
            "detuning",
            "modal-diagnostics",
            "mediator",
            "saturation",
            "all-linear",
        ),
    )
    parser.add_argument(
        "--output",
        default="paper_output",
        help=(
            "Output directory containing figures/, data/, "
            "and summaries/."
        ),
    )
    return parser


def run_selected_command(command: str, output: OutputPaths) -> None:
    commands = {
        "validate": lambda: run_validation(output),
        "stability": lambda: recipe_stability(output),
        "normalized-scans": lambda: recipe_normalized_scans(output),
        "dynamics": lambda: recipe_dynamics(output),
        "phase-locking": lambda: recipe_phase_locking(output),
        "finite-capacity": lambda: recipe_finite_capacity(output),
        "gross-input": lambda: recipe_gross_input(output),
        "theta-tradeoff": lambda: recipe_theta_tradeoff(output),
        "detuning": lambda: recipe_detuning(output),
        "modal-diagnostics": lambda: recipe_modal_diagnostics(output),
        "mediator": lambda: recipe_mediator(output),
        "saturation": lambda: recipe_saturation(output),
    }

    if command == "all-linear":
        run_validation(output)
        recipe_stability(output)
        recipe_normalized_scans(output)
        recipe_dynamics(output)
        recipe_phase_locking(output)
        recipe_finite_capacity(output)
        recipe_theta_tradeoff(output)
        recipe_detuning(output)
        recipe_modal_diagnostics(output)
        recipe_mediator(output)
    else:
        commands[command]()


def main(argv: Sequence[str] | None = None) -> None:
    configure_plot_style()

    # In Jupyter, argv is passed explicitly below so that the kernel's
    # internal "-f <kernel.json>" arguments are never parsed as the required
    # positional command.
    args = build_parser().parse_args(argv)
    output = OutputPaths.create(args.output)
    run_selected_command(args.command, output)


if __name__ == "__main__":
    import sys

    if "ipykernel" in sys.modules:
        # Jupyter / IPython execution.
        main(
            [
                NOTEBOOK_COMMAND,
                "--output",
                NOTEBOOK_OUTPUT,
            ]
        )
    else:
        # Normal command-line execution.
        main()