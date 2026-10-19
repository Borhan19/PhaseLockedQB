#!/usr/bin/env python3
"""
One-click exact nonlinear saturable-gain validation.

Open this file in Jupyter, Spyder, VS Code, or another Python editor and click
Run. No command-line arguments or separate user-defined module are required.

The full master equation is

    dot rho = -i[H_sat,rho]
              + kappa_a_down D[a] rho
              + kappa_b D[b] rho
              + Gamma_col D[cos(theta) a + exp(i phi) sin(theta) b] rho
              + P_sigma D[sigma_+] rho
              + gamma_sigma D[sigma_-] rho
              + gamma_sigma_phi/2 D[sigma_z] rho,

with

    H_sat = delta a^dagger a - delta b^dagger b
            + Delta_sigma sigma_+ sigma_-
            + g_sigma (a^dagger sigma_- + a sigma_+)
            + E a^dagger + E* a.

The default low-energy benchmark uses

    kappa_a_down = kappa_b = 0.05,
    Gamma_col = 2,
    theta = arctan(2),
    delta = 0,
    E = 0.02,
    P_sigma = 0.16,
    gamma_sigma = 0.03,

and a requested positive small-signal growth margin of 0.002. The TLS coupling
is solved from the exact linearized charger-battery-TLS drift, rather than only
from the adiabatic pump-rate approximation.

The code performs:

1. exact QuTiP evolution of two oscillators and a pumped two-level gain medium;
2. exact battery ergotropy from the reduced battery density matrix;
3. comparison with a linear-Gaussian reference having the same growth margin;
4. cutoff convergence of charger occupation, battery occupation, ergotropy,
   ergotropy fraction, and SNR;
5. independent charger and battery edge-population checks;
6. long-time stationarity tests for charger occupation, battery occupation,
   and ergotropy;
7. dense early-time sampling and coarser late-time sampling;
8. PDF-only manuscript figures and CSV/JSON numerical output.

The code is single-threaded by default. Every manuscript axis uses the global reference rate kappa_0=0.01.
"""


from __future__ import annotations

# Set these before NumPy/SciPy import.  This avoids hidden BLAS oversubscription
# on a cluster and makes cutoff runs easier to compare.
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import json
import math
import sys
import warnings
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib import font_manager
from matplotlib.lines import Line2D
from scipy.optimize import brentq
from scipy.integrate import solve_ivp
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


# =============================================================================
# Manuscript units and physical figure dimensions
# =============================================================================

# The nonlinear benchmark uses the same global reference rate as every other
# figure in the manuscript.  Numerical propagation remains in the original
# rates; every manuscript-facing time/rate is reported as:
#
#     tau = kappa_0 t,    tilde{x} = x / kappa_0.
REFERENCE_RATE = 1.0e-2
MAIN_TEXT_FONT_SIZE = 10.0
SINGLE_COLUMN_WIDTH = 3.37
DOUBLE_COLUMN_WIDTH = 7.00

# The revised main-paper benchmark uses a moderate collective-mode composition.
# The total collective rate is chosen so that the off-diagonal dissipative drift
# remains |g| = 0.032, matching the coherent benchmark, while theta is moved well
# away from pi/2 to give the battery substantial participation in the quasi-dark
# mode.
PAPER_MATCHED_G = 0.032
PAPER_THETA = 1.15
PAPER_GAMMA_COL = (
    2.0 * PAPER_MATCHED_G
    / (math.sin(PAPER_THETA) * math.cos(PAPER_THETA))
)


def normalized_rate(value: Any) -> Any:
    return np.asarray(value) / REFERENCE_RATE


def normalized_time(value: Any) -> Any:
    return REFERENCE_RATE * np.asarray(value)


def unit_metadata() -> dict[str, Any]:
    return {
        "reference_rate_symbol": "kappa_0",
        "reference_rate": REFERENCE_RATE,
        "dimensionless_time": "kappa_0 * t",
        "dimensionless_rate": "rate / kappa_0",
        "main_text_font_size_pt": MAIN_TEXT_FONT_SIZE,
        "single_column_width_in": SINGLE_COLUMN_WIDTH,
        "double_column_width_in": DOUBLE_COLUMN_WIDTH,
    }


def normalized_bosonic_parameters(
    bosonic: "BosonicParams",
) -> dict[str, Any]:
    return {
        "kappa_a_down_over_kappa0": (
            bosonic.kappa_a_down / REFERENCE_RATE
        ),
        "kappa_b_over_kappa0": bosonic.kappa_b / REFERENCE_RATE,
        "delta_over_kappa0": bosonic.delta / REFERENCE_RATE,
        "drive_over_kappa0": complex(
            bosonic.drive / REFERENCE_RATE
        ),
        "Gamma_col_over_kappa0": (
            bosonic.Gamma_col / REFERENCE_RATE
        ),
        "theta": bosonic.theta,
        "phi": bosonic.phi,
        "target_growth_margin_over_kappa0": (
            bosonic.target_growth_margin / REFERENCE_RATE
        ),
    }


def normalized_tls_parameters(tls: "TLSParams") -> dict[str, Any]:
    return {
        "P_sigma_over_kappa0": tls.P_sigma / REFERENCE_RATE,
        "gamma_sigma_over_kappa0": (
            tls.gamma_sigma / REFERENCE_RATE
        ),
        "gamma_sigma_phi_over_kappa0": (
            tls.gamma_sigma_phi / REFERENCE_RATE
        ),
        "Delta_sigma_over_kappa0": (
            tls.Delta_sigma / REFERENCE_RATE
        ),
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


def _choose_serif_font() -> str:
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
            "figure.max_open_warning": 0,
        }
    )
    print(f"Plot font: {chosen}")
    print(
        f"Manuscript font size: {MAIN_TEXT_FONT_SIZE:g} pt; "
        f"reference rate kappa_0={REFERENCE_RATE:g}"
    )
    return chosen


# =============================================================================
# Data classes and profiles
# =============================================================================

@dataclass(frozen=True)
class BosonicParams:
    kappa_a_down: float
    kappa_b: float
    delta: float
    drive: complex
    Gamma_col: float
    theta: float
    phi: float
    omega_b: float
    target_growth_margin: float

    def validate(self) -> None:
        if self.kappa_a_down < 0.0 or self.kappa_b < 0.0:
            raise ValueError("Local loss rates must be nonnegative.")
        if self.Gamma_col < 0.0:
            raise ValueError("Gamma_col must be nonnegative.")
        if self.omega_b <= 0.0:
            raise ValueError("omega_b must be positive.")
        if not (0.0 <= self.theta <= np.pi / 2.0):
            raise ValueError("theta must lie in [0, pi/2].")


@dataclass(frozen=True)
class TLSParams:
    P_sigma: float
    gamma_sigma: float
    gamma_sigma_phi: float
    Delta_sigma: float

    def validate(self) -> None:
        if self.P_sigma <= self.gamma_sigma:
            raise ValueError(
                "Population inversion requires P_sigma > gamma_sigma."
            )
        if min(
            self.P_sigma,
            self.gamma_sigma,
            self.gamma_sigma_phi,
        ) < 0.0:
            raise ValueError("TLS rates must be nonnegative.")

    @property
    def gamma_parallel(self) -> float:
        return self.P_sigma + self.gamma_sigma

    @property
    def gamma_perp(self) -> float:
        return 0.5 * self.gamma_parallel + self.gamma_sigma_phi

    @property
    def w0(self) -> float:
        return (
            self.P_sigma - self.gamma_sigma
        ) / (
            self.P_sigma + self.gamma_sigma
        )

    @property
    def excited_population_unsaturated(self) -> float:
        return self.P_sigma / (self.P_sigma + self.gamma_sigma)


@dataclass(frozen=True)
class DerivedGain:
    kappa_a_up_critical: float
    kappa_a_up_linear: float
    relative_pump_excess: float
    g_sigma_threshold: float
    g_sigma: float
    g_sigma_adiabatic: float
    relative_tls_coupling_excess: float
    effective_pump_from_g: float
    n_sat: float
    linear_margin: float
    tls_linear_margin: float
    linear_eigenvalues: tuple[complex, complex]
    tls_linear_eigenvalues: tuple[complex, complex, complex]


@dataclass(frozen=True)
class Numerics:
    t_end: float = 4000.0
    convergence_points: int = 161
    final_points: int = 501
    solver_method: str = "bdf"
    atol: float = 1.0e-9
    rtol: float = 1.0e-7
    nsteps: int = 200000
    edge_levels: int = 2
    tail_fraction: float = 0.15
    target_occupation: float = 0.5
    early_fraction_n_sat: float = 0.10

    # Cutoff-convergence tolerances.
    curve_rel_n_a_tol: float = 3.0e-3
    curve_rel_n_b_tol: float = 3.0e-3
    curve_rel_ergotropy_tol: float = 5.0e-3
    curve_abs_eta_tol: float = 3.0e-3
    curve_log_snr_tol: float = 8.0e-3
    edge_population_tol: float = 1.0e-7
    required_consecutive_passes: int = 2

    # Long-time saturation tests.
    tail_relative_drift_tol: float = 1.0e-2
    tail_coefficient_variation_tol: float = 1.0e-2

    def validate(self) -> None:
        if self.t_end <= 0.0:
            raise ValueError("t_end must be positive.")
        if self.convergence_points < 21 or self.final_points < 21:
            raise ValueError("Use at least 21 time points.")
        if self.edge_levels < 1:
            raise ValueError("edge_levels must be at least one.")
        if not (0.0 < self.tail_fraction < 0.5):
            raise ValueError("tail_fraction must lie in (0,0.5).")
        if self.target_occupation <= 0.0:
            raise ValueError("target_occupation must be positive.")


@dataclass(frozen=True)
class OutputPaths:
    root: Path
    figures: Path
    data: Path
    summaries: Path

    @classmethod
    def create(cls, root: str | Path) -> "OutputPaths":
        root_path = Path(root).expanduser().resolve()
        figures = root_path / "Figs_New"
        data = root_path / "data"
        summaries = root_path / "summaries"
        for directory in (root_path, figures, data, summaries):
            directory.mkdir(parents=True, exist_ok=True)
        return cls(root_path, figures, data, summaries)


def benchmark_profile(name: str) -> tuple[BosonicParams, TLSParams]:
    """
    Return a fully specified benchmark.

    low_energy:
        Representative nonlinear robustness benchmark.  The local losses are
        increased, the coherent seed is reduced, and the gain is only modestly
        above threshold.  This keeps the physical saturation occupation small
        enough for reliable exact Fock-space convergence.

    balanced:
        Normalized representation of the legacy Gamma=0.4, p_a=1, p_b=2
        point.

    paper:
        Revised moderate-composition main-paper point.  It keeps |g|=0.032 and
        the same small-signal growth margin as the matched linear benchmark, but
        uses theta=1.15 so the quasi-dark mode has substantial battery support.
        This profile is used for the many-emitter proof-of-principle calculation.
    """
    if name == "low_energy":
        bosonic = BosonicParams(
            kappa_a_down=0.05,
            kappa_b=0.05,
            delta=0.0,
            drive=0.02 + 0.0j,
            Gamma_col=2.0,
            theta=math.atan2(2.0, 1.0),
            phi=0.0,
            omega_b=1.0,
            target_growth_margin=2.0e-3,
        )
        tls = TLSParams(
            P_sigma=0.16,
            gamma_sigma=0.03,
            gamma_sigma_phi=0.0,
            Delta_sigma=0.0,
        )
        return bosonic, tls

    common = dict(
        kappa_a_down=0.01,
        kappa_b=0.01,
        delta=0.0,
        drive=0.1 + 0.0j,
        phi=0.0,
        omega_b=1.0,
        target_growth_margin=5.0e-4,
    )

    if name == "balanced":
        bosonic = BosonicParams(
            **common,
            Gamma_col=0.4 * (1.0 ** 2 + 2.0 ** 2),
            theta=math.atan2(2.0, 1.0),
        )
        tls = TLSParams(
            P_sigma=0.18,
            gamma_sigma=0.02,
            gamma_sigma_phi=0.0,
            Delta_sigma=0.0,
        )
    elif name == "paper":
        bosonic = BosonicParams(
            **common,
            Gamma_col=PAPER_GAMMA_COL,
            theta=PAPER_THETA,
        )
        tls = TLSParams(
            P_sigma=0.18,
            gamma_sigma=0.02,
            gamma_sigma_phi=0.0,
            Delta_sigma=0.0,
        )
    else:
        raise ValueError(f"Unknown profile: {name}")
    return bosonic, tls


# =============================================================================
# Serialization
# =============================================================================

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
    path.write_text(
        json.dumps(_json_ready(payload), indent=2),
        encoding="utf-8",
    )
    print(f"Wrote: {path}")


def save_dataframe(dataframe: pd.DataFrame, path: Path) -> None:
    dataframe.to_csv(path, index=False)
    print(f"Wrote: {path}")


def save_figure(
    fig: mpl.figure.Figure,
    path: Path,
    apply_tight_layout: bool = True,
) -> None:
    if path.suffix.lower() != ".pdf":
        raise ValueError("Manuscript figures must be saved as PDF.")
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


def _axis_common(ax: mpl.axes.Axes) -> None:
    ax.minorticks_on()
    ax.tick_params(which="both", direction="in", top=True, right=True)


# =============================================================================
# Linear first-moment threshold and gain matching
# =============================================================================

def linear_complex_drift(
    bosonic: BosonicParams,
    kappa_a_up: float,
) -> np.ndarray:
    bosonic.validate()
    c = np.cos(bosonic.theta)
    s = np.sin(bosonic.theta)
    g = (
        0.5
        * bosonic.Gamma_col
        * np.exp(1j * bosonic.phi)
        * s
        * c
    )
    gamma_a = 0.5 * (
        bosonic.kappa_a_down
        - kappa_a_up
        + bosonic.Gamma_col * c ** 2
    )
    gamma_b = 0.5 * (
        bosonic.kappa_b
        + bosonic.Gamma_col * s ** 2
    )
    return np.array(
        [
            [-gamma_a - 1j * bosonic.delta, -g],
            [-np.conjugate(g), -gamma_b + 1j * bosonic.delta],
        ],
        dtype=complex,
    )


def linear_margin(
    bosonic: BosonicParams,
    kappa_a_up: float,
) -> float:
    return float(
        np.max(
            np.real(
                np.linalg.eigvals(
                    linear_complex_drift(bosonic, kappa_a_up)
                )
            )
        )
    )


def solve_linear_pump_for_margin(
    bosonic: BosonicParams,
    target_margin: float,
    upper_initial: float = 0.05,
) -> float:
    """Solve for the phenomenological pump producing a requested margin."""

    def objective(kappa_up: float) -> float:
        return linear_margin(bosonic, kappa_up) - target_margin

    lower = 0.0
    upper = upper_initial
    f_lower = objective(lower)
    f_upper = objective(upper)
    for _ in range(40):
        if abs(f_lower) < 1.0e-15:
            return lower
        if f_lower * f_upper < 0.0:
            return float(
                brentq(
                    objective,
                    lower,
                    upper,
                    xtol=1.0e-13,
                    rtol=1.0e-12,
                )
            )
        upper *= 2.0
        f_upper = objective(upper)
    raise RuntimeError(
        "Could not bracket the linear pump required for the requested margin."
    )


def solve_linear_pump(
    bosonic: BosonicParams,
    upper_initial: float = 0.05,
) -> float:
    return solve_linear_pump_for_margin(
        bosonic,
        bosonic.target_growth_margin,
        upper_initial,
    )


def tls_linearized_drift(
    bosonic: BosonicParams,
    tls: TLSParams,
    g_sigma: float,
) -> np.ndarray:
    """
    Linearized first-moment drift of (a,b,sigma_-) about the unsaturated TLS.

    The inversion is fixed at w0 only for this small-signal stability analysis.
    The exact master-equation simulation retains the full nonlinear TLS.
    """
    c = np.cos(bosonic.theta)
    s = np.sin(bosonic.theta)
    dissipative_coupling = (
        0.5
        * bosonic.Gamma_col
        * np.exp(1j * bosonic.phi)
        * s
        * c
    )
    gamma_a_passive = 0.5 * (
        bosonic.kappa_a_down
        + bosonic.Gamma_col * c ** 2
    )
    gamma_b = 0.5 * (
        bosonic.kappa_b
        + bosonic.Gamma_col * s ** 2
    )
    return np.array(
        [
            [
                -gamma_a_passive - 1j * bosonic.delta,
                -dissipative_coupling,
                -1j * g_sigma,
            ],
            [
                -np.conjugate(dissipative_coupling),
                -gamma_b + 1j * bosonic.delta,
                0.0,
            ],
            [
                1j * g_sigma * tls.w0,
                0.0,
                -tls.gamma_perp - 1j * tls.Delta_sigma,
            ],
        ],
        dtype=complex,
    )


def tls_small_signal_margin(
    bosonic: BosonicParams,
    tls: TLSParams,
    g_sigma: float,
) -> float:
    return float(
        np.max(
            np.real(
                np.linalg.eigvals(
                    tls_linearized_drift(
                        bosonic,
                        tls,
                        g_sigma,
                    )
                )
            )
        )
    )


def solve_tls_coupling_for_margin(
    bosonic: BosonicParams,
    tls: TLSParams,
    target_margin: float,
    upper_initial: float,
) -> float:
    """
    Match the exact TLS small-signal growth margin, not only its adiabatic
    pump-rate approximation.
    """

    def objective(g_sigma: float) -> float:
        return (
            tls_small_signal_margin(
                bosonic,
                tls,
                g_sigma,
            )
            - target_margin
        )

    lower = 0.0
    upper = max(float(upper_initial), 1.0e-8)
    f_lower = objective(lower)
    f_upper = objective(upper)
    for _ in range(40):
        if abs(f_lower) < 1.0e-15:
            return lower
        if f_lower * f_upper < 0.0:
            return float(
                brentq(
                    objective,
                    lower,
                    upper,
                    xtol=1.0e-13,
                    rtol=1.0e-12,
                )
            )
        upper *= 2.0
        f_upper = objective(upper)
    raise RuntimeError(
        "Could not bracket the TLS coupling required for the requested margin."
    )


def derive_gain(
    bosonic: BosonicParams,
    tls: TLSParams,
) -> DerivedGain:
    bosonic.validate()
    tls.validate()

    kappa_critical = solve_linear_pump_for_margin(
        bosonic,
        target_margin=0.0,
    )
    kappa_up = solve_linear_pump(bosonic)

    denominator = 2.0 * tls.gamma_perp * tls.w0
    if denominator <= 0.0:
        raise ValueError("The unsaturated gain denominator must be positive.")

    g_adiabatic = math.sqrt(
        kappa_up
        * (
            tls.gamma_perp ** 2
            + tls.Delta_sigma ** 2
        )
        / denominator
    )

    g_threshold = solve_tls_coupling_for_margin(
        bosonic,
        tls,
        target_margin=0.0,
        upper_initial=g_adiabatic,
    )
    g_sigma = solve_tls_coupling_for_margin(
        bosonic,
        tls,
        target_margin=bosonic.target_growth_margin,
        upper_initial=g_adiabatic,
    )

    effective_pump = (
        2.0
        * g_sigma ** 2
        * tls.gamma_perp
        * tls.w0
        / (
            tls.gamma_perp ** 2
            + tls.Delta_sigma ** 2
        )
    )
    n_sat = (
        tls.gamma_parallel
        * (
            tls.gamma_perp ** 2
            + tls.Delta_sigma ** 2
        )
        / (
            4.0
            * g_sigma ** 2
            * tls.gamma_perp
        )
    )

    linear_matrix = linear_complex_drift(
        bosonic,
        kappa_up,
    )
    tls_matrix = tls_linearized_drift(
        bosonic,
        tls,
        g_sigma,
    )
    linear_eigenvalues = tuple(
        complex(value)
        for value in np.linalg.eigvals(linear_matrix)
    )
    tls_eigenvalues = tuple(
        complex(value)
        for value in np.linalg.eigvals(tls_matrix)
    )

    return DerivedGain(
        kappa_a_up_critical=kappa_critical,
        kappa_a_up_linear=kappa_up,
        relative_pump_excess=(
            kappa_up / kappa_critical - 1.0
        ),
        g_sigma_threshold=g_threshold,
        g_sigma=g_sigma,
        g_sigma_adiabatic=g_adiabatic,
        relative_tls_coupling_excess=(
            g_sigma / g_threshold - 1.0
        ),
        effective_pump_from_g=effective_pump,
        n_sat=n_sat,
        linear_margin=linear_margin(
            bosonic,
            kappa_up,
        ),
        tls_linear_margin=tls_small_signal_margin(
            bosonic,
            tls,
            g_sigma,
        ),
        linear_eigenvalues=linear_eigenvalues,
        tls_linear_eigenvalues=tls_eigenvalues,
    )


# =============================================================================
# Exact linear-Gaussian reference
# =============================================================================

def symplectic_form(n_modes: int) -> np.ndarray:
    return np.kron(
        np.eye(n_modes),
        np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=float),
    )


def annihilation_lindblad_vector(
    coefficients: Sequence[complex],
) -> np.ndarray:
    coefficients = np.asarray(coefficients, dtype=complex)
    vector = np.zeros(2 * len(coefficients), dtype=complex)
    for index, coefficient in enumerate(coefficients):
        vector[2 * index] = coefficient / np.sqrt(2.0)
        vector[2 * index + 1] = 1j * coefficient / np.sqrt(2.0)
    return vector


def creation_lindblad_vector(
    coefficients: Sequence[complex],
) -> np.ndarray:
    coefficients = np.asarray(coefficients, dtype=complex)
    vector = np.zeros(2 * len(coefficients), dtype=complex)
    for index, coefficient in enumerate(coefficients):
        vector[2 * index] = coefficient / np.sqrt(2.0)
        vector[2 * index + 1] = -1j * coefficient / np.sqrt(2.0)
    return vector


def linear_gaussian_matrices(
    bosonic: BosonicParams,
    kappa_a_up: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    omega = symplectic_form(2)

    # H = delta a^dagger a - delta b^dagger b
    h_r = np.diag(
        [
            bosonic.delta,
            bosonic.delta,
            -bosonic.delta,
            -bosonic.delta,
        ]
    ).astype(float)

    dvec = np.array(
        [
            np.sqrt(2.0) * np.real(bosonic.drive),
            np.sqrt(2.0) * np.imag(bosonic.drive),
            0.0,
            0.0,
        ],
        dtype=float,
    )

    vectors: list[np.ndarray] = [
        np.sqrt(bosonic.kappa_a_down)
        * annihilation_lindblad_vector([1.0, 0.0]),
        np.sqrt(kappa_a_up)
        * creation_lindblad_vector([1.0, 0.0]),
        np.sqrt(bosonic.kappa_b)
        * annihilation_lindblad_vector([0.0, 1.0]),
        np.sqrt(bosonic.Gamma_col)
        * annihilation_lindblad_vector(
            [
                np.cos(bosonic.theta),
                np.exp(1j * bosonic.phi) * np.sin(bosonic.theta),
            ]
        ),
    ]

    ymat = np.zeros((4, 4), dtype=complex)
    for vector in vectors:
        ymat += np.outer(vector, vector.conj())

    amat = omega @ h_r - omega @ np.imag(ymat)
    dmat = omega @ np.real(ymat) @ omega.T
    source = omega @ dvec
    return (
        np.real_if_close(amat).astype(float),
        np.real_if_close(0.5 * (dmat + dmat.T)).astype(float),
        source,
    )


def propagate_linear_gaussian(
    bosonic: BosonicParams,
    derived: DerivedGain,
    times: np.ndarray,
) -> pd.DataFrame:
    amat, dmat, source = linear_gaussian_matrices(
        bosonic,
        derived.kappa_a_up_linear,
    )
    dimension = 4
    covariance_generator = (
        np.kron(np.eye(dimension), amat)
        + np.kron(amat, np.eye(dimension))
    )
    total_dimension = dimension + dimension ** 2 + 1
    generator = np.zeros(
        (total_dimension, total_dimension),
        dtype=float,
    )
    generator[:dimension, :dimension] = amat
    generator[:dimension, -1] = source

    start = dimension
    stop = dimension + dimension ** 2
    generator[start:stop, start:stop] = covariance_generator
    generator[start:stop, -1] = dmat.reshape(-1, order="F")

    initial = np.zeros(total_dimension, dtype=float)
    initial[start:stop] = (
        0.5 * np.eye(dimension)
    ).reshape(-1, order="F")
    initial[-1] = 1.0

    shifted = np.asarray(times, dtype=float) - float(times[0])
    generator_sparse = csr_matrix(generator)
    time_steps = np.diff(shifted)

    if (
        shifted.size <= 2
        or np.allclose(
            time_steps,
            time_steps[0],
            rtol=1.0e-11,
            atol=1.0e-14,
        )
    ):
        solution = expm_multiply(
            generator_sparse,
            initial,
            start=float(shifted[0]),
            stop=float(shifted[-1]),
            num=times.size,
            endpoint=True,
        )
    else:
        # Exact piecewise propagation on an arbitrary output grid.  This lets
        # us resolve the early small-signal transient densely without wasting
        # hundreds of points in the stationary late-time tail.
        solution = np.empty(
            (times.size, initial.size),
            dtype=float,
        )
        solution[0] = initial
        state = initial.copy()
        for index, time_step in enumerate(time_steps, start=1):
            state = expm_multiply(
                generator_sparse * float(time_step),
                state,
            )
            solution[index] = np.real_if_close(state)
    means = solution[:, :dimension]
    covariances = solution[:, start:stop].reshape(
        times.size,
        dimension,
        dimension,
        order="F",
    )
    covariances = 0.5 * (
        covariances + np.swapaxes(covariances, 1, 2)
    )

    rows = {"t": times}
    for mode, name in ((0, "a"), (1, "b")):
        index = 2 * mode
        xmean = means[:, index]
        pmean = means[:, index + 1]
        beta = (xmean + 1j * pmean) / np.sqrt(2.0)
        var_x = covariances[:, index, index]
        var_p = covariances[:, index + 1, index + 1]
        n_total = 0.5 * (
            var_x
            + var_p
            + xmean ** 2
            + pmean ** 2
            - 1.0
        )
        n_coherent = np.abs(beta) ** 2
        rows[f"n_{name}"] = n_total
        rows[f"mean_{name}_real"] = beta.real
        rows[f"mean_{name}_imag"] = beta.imag
        rows[f"n_{name}_coherent"] = n_coherent
        if name == "b":
            denominator = np.maximum(
                n_total - n_coherent + 0.5,
                1.0e-300,
            )
            rows["energy_b"] = bosonic.omega_b * n_total
            rows["ergotropy_b"] = (
                bosonic.omega_b * n_coherent
            )
            rows["passive_energy_b"] = (
                rows["energy_b"] - rows["ergotropy_b"]
            )
            rows["ergotropy_fraction_b"] = np.divide(
                rows["ergotropy_b"],
                rows["energy_b"],
                out=np.full_like(n_total, np.nan),
                where=np.abs(rows["energy_b"]) > 1.0e-15,
            )
            rows["snr_b"] = n_coherent / denominator
    return pd.DataFrame(rows)


# =============================================================================
# Symmetric pumped-emitter ensemble: large-N proof of principle
# =============================================================================

def emitter_ensemble_mean_field_rhs(
    time: float,
    state: np.ndarray,
    bosonic: BosonicParams,
    tls: TLSParams,
    emitter_count: int,
    g_per_emitter: float,
) -> np.ndarray:
    """
    Maxwell-Bloch equations for N identical incoherently pumped emitters.

    The symmetric mean-field variables are the charger amplitude alpha, battery
    amplitude beta, the polarization s=<sigma_-> of one representative emitter,
    and the inversion w=<sigma_z> of one representative emitter.  Every emitter
    couples identically to the charger.
    """
    del time
    alpha = state[0] + 1j * state[1]
    beta = state[2] + 1j * state[3]
    polarization = state[4] + 1j * state[5]
    inversion = float(state[6])

    c = np.cos(bosonic.theta)
    s = np.sin(bosonic.theta)
    g_diss = (
        0.5
        * bosonic.Gamma_col
        * np.exp(1j * bosonic.phi)
        * s
        * c
    )
    gamma_a_passive = 0.5 * (
        bosonic.kappa_a_down
        + bosonic.Gamma_col * c ** 2
    )
    gamma_b = 0.5 * (
        bosonic.kappa_b
        + bosonic.Gamma_col * s ** 2
    )

    d_alpha = (
        (-gamma_a_passive - 1j * bosonic.delta) * alpha
        - g_diss * beta
        - 1j * emitter_count * g_per_emitter * polarization
        - 1j * bosonic.drive
    )
    d_beta = (
        (-gamma_b + 1j * bosonic.delta) * beta
        - np.conjugate(g_diss) * alpha
    )
    d_polarization = (
        (-tls.gamma_perp - 1j * tls.Delta_sigma) * polarization
        + 1j * g_per_emitter * alpha * inversion
    )
    d_inversion = (
        -tls.gamma_parallel * (inversion - tls.w0)
        + 2j
        * g_per_emitter
        * (
            np.conjugate(alpha) * polarization
            - alpha * np.conjugate(polarization)
        )
    )

    return np.array(
        [
            np.real(d_alpha),
            np.imag(d_alpha),
            np.real(d_beta),
            np.imag(d_beta),
            np.real(d_polarization),
            np.imag(d_polarization),
            np.real(d_inversion),
        ],
        dtype=float,
    )


def simulate_emitter_ensemble_mean_field(
    bosonic: BosonicParams,
    tls: TLSParams,
    derived_single_emitter: DerivedGain,
    emitter_count: int,
    times: np.ndarray,
) -> pd.DataFrame:
    """
    Simulate a symmetric ensemble while keeping the small-signal gain fixed.

    The coupling of each emitter is scaled as g_1/sqrt(N), so that

        N * g_N^2 = g_1^2.

    Therefore the exact linearized charger-battery-polarization spectrum is the
    same as in the N=1 matched model, whereas the finite inversion reservoir and
    hence the saturation scale grow with N.  This isolates the finite-gain
    capacity effect from a trivial increase of the small-signal growth rate.
    """
    if emitter_count < 1:
        raise ValueError("emitter_count must be at least one.")
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or times.size < 2:
        raise ValueError("times must be a one-dimensional array with >=2 entries.")

    g_per_emitter = (
        derived_single_emitter.g_sigma / math.sqrt(emitter_count)
    )
    initial = np.array(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, tls.w0],
        dtype=float,
    )
    solution = solve_ivp(
        emitter_ensemble_mean_field_rhs,
        (float(times[0]), float(times[-1])),
        initial,
        t_eval=times,
        args=(
            bosonic,
            tls,
            emitter_count,
            g_per_emitter,
        ),
        method="DOP853",
        rtol=1.0e-9,
        atol=1.0e-11,
    )
    if not solution.success:
        raise RuntimeError(
            "Emitter-ensemble mean-field integration failed: "
            + solution.message
        )

    alpha = solution.y[0] + 1j * solution.y[1]
    beta = solution.y[2] + 1j * solution.y[3]
    polarization = solution.y[4] + 1j * solution.y[5]
    inversion = solution.y[6]

    dataframe = pd.DataFrame(
        {
            "t": solution.t,
            "kappa0_t": normalized_time(solution.t),
            "emitter_count": emitter_count,
            "g_per_emitter": g_per_emitter,
            "sqrtN_g_per_emitter": (
                math.sqrt(emitter_count) * g_per_emitter
            ),
            "alpha_real": np.real(alpha),
            "alpha_imag": np.imag(alpha),
            "beta_real": np.real(beta),
            "beta_imag": np.imag(beta),
            "polarization_real": np.real(polarization),
            "polarization_imag": np.imag(polarization),
            "inversion_per_emitter": inversion,
            "n_a_mean_field": np.abs(alpha) ** 2,
            "n_b_mean_field": np.abs(beta) ** 2,
        }
    )
    return dataframe


def _interpolated_crossing_time(
    times: np.ndarray,
    values: np.ndarray,
    target: float,
) -> float:
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    indices = np.where(values >= target)[0]
    if indices.size == 0:
        return float("nan")
    index = int(indices[0])
    if index == 0:
        return float(times[0])
    t0, t1 = times[index - 1], times[index]
    y0, y1 = values[index - 1], values[index]
    if abs(y1 - y0) < 1.0e-15:
        return float(t1)
    weight = (target - y0) / (y1 - y0)
    return float(t0 + weight * (t1 - t0))


def run_emitter_ensemble_proof(
    bosonic: BosonicParams,
    tls: TLSParams,
    derived: DerivedGain,
    emitter_counts: Sequence[int],
    t_end: float,
    points: int,
    target_occupation: float,
    output: OutputPaths,
) -> tuple[dict[int, pd.DataFrame], pd.DataFrame]:
    """Run and save the large-N saturation-capacity proof of principle."""
    times = np.linspace(0.0, float(t_end), int(points))
    results: dict[int, pd.DataFrame] = {}
    rows: list[dict[str, float | int | bool]] = []

    for emitter_count in emitter_counts:
        count = int(emitter_count)
        dataframe = simulate_emitter_ensemble_mean_field(
            bosonic,
            tls,
            derived,
            count,
            times,
        )
        results[count] = dataframe
        save_dataframe(
            dataframe,
            output.data / f"emitter_ensemble_N_{count}.csv",
        )

        tail_start = max(0, int(0.9 * len(dataframe)))
        plateau = float(
            dataframe["n_b_mean_field"].iloc[tail_start:].mean()
        )
        crossing_time = _interpolated_crossing_time(
            dataframe["t"].to_numpy(),
            dataframe["n_b_mean_field"].to_numpy(),
            target_occupation,
        )
        rows.append(
            {
                "emitter_count": count,
                "g_per_emitter": float(
                    derived.g_sigma / math.sqrt(count)
                ),
                "sqrtN_g_per_emitter": float(derived.g_sigma),
                "small_signal_margin": float(derived.tls_linear_margin),
                "small_signal_margin_over_kappa0": float(
                    derived.tls_linear_margin / REFERENCE_RATE
                ),
                "plateau_n_b_mean_field": plateau,
                "final_n_a_mean_field": float(
                    dataframe["n_a_mean_field"].iloc[-1]
                ),
                "final_inversion_per_emitter": float(
                    dataframe["inversion_per_emitter"].iloc[-1]
                ),
                "target_occupation": float(target_occupation),
                "target_reached": bool(np.isfinite(crossing_time)),
                "target_crossing_time": crossing_time,
                "target_crossing_kappa0_t": (
                    float(crossing_time * REFERENCE_RATE)
                    if np.isfinite(crossing_time)
                    else float("nan")
                ),
            }
        )

    summary = pd.DataFrame(rows)
    save_dataframe(
        summary,
        output.data / "emitter_ensemble_scaling.csv",
    )
    save_json(
        {
            "units": unit_metadata(),
            "model": "symmetric Maxwell-Bloch mean-field ensemble",
            "coupling_scaling": "g_N = g_1/sqrt(N)",
            "purpose": (
                "proof that the finite gain reservoir can sustain the same "
                "small-signal instability to much larger battery occupation"
            ),
            "bosonic": asdict(bosonic),
            "bosonic_dimensionless": normalized_bosonic_parameters(bosonic),
            "tls": asdict(tls),
            "tls_dimensionless": normalized_tls_parameters(tls),
            "derived_single_emitter": asdict(derived),
            "emitter_counts": [int(value) for value in emitter_counts],
            "target_occupation": float(target_occupation),
            "rows": summary.to_dict(orient="records"),
        },
        output.summaries / "emitter_ensemble_proof.json",
    )
    return results, summary


def plot_emitter_ensemble_proof(
    results: dict[int, pd.DataFrame],
    summary: pd.DataFrame,
    target_occupation: float,
    output: OutputPaths,
) -> None:
    """Compact two-panel proof-of-principle ensemble figure."""
    fig, axes = plt.subplots(1, 2, figsize=(SINGLE_COLUMN_WIDTH, 1.75))
    counts = sorted(results)
    for count in counts:
        dataframe = results[count]
        axes[0].semilogy(
            dataframe["kappa0_t"],
            np.maximum(dataframe["n_b_mean_field"], 1.0e-8),
            label=rf"$N_\sigma={count}$",
        )
    axes[0].axhline(
        target_occupation,
        color="black",
        linestyle=":",
        linewidth=1.1,
    )
    axes[0].set_xlabel(r"$\kappa_0 t$")
    axes[0].set_ylabel(r"$|\langle b\rangle|^2$")
    axes[0].legend(frameon=False, ncol=1)

    axes[1].plot(
        summary["emitter_count"],
        summary["plateau_n_b_mean_field"],
        marker="o",
    )
    axes[1].axhline(
        target_occupation,
        color="black",
        linestyle=":",
        linewidth=1.1,
    )
    axes[1].set_xscale("log")
    axes[1].set_xlabel(r"$N_\sigma$")
    axes[1].set_ylabel(r"stationary $|\langle b\rangle|^2$")

    for label, ax in zip(("(a)", "(b)"), axes):
        _axis_common(ax)
        _panel_label(ax, label)
    fig.subplots_adjust(
        left=0.10,
        right=0.98,
        bottom=0.20,
        top=0.98,
        wspace=0.5,
    )
    diagnostic_directory = output.root / "diagnostics"
    diagnostic_directory.mkdir(parents=True, exist_ok=True)
    save_figure(
        fig,
        diagnostic_directory / "emitter_ensemble_proof.pdf",
        apply_tight_layout=False,
    )


def plot_revised_end_matter_figure(
    exact: pd.DataFrame,
    linear: pd.DataFrame,
    ensemble_results: dict[int, pd.DataFrame],
    ensemble_summary: pd.DataFrame,
    target_occupation: float,
    output: OutputPaths,
) -> None:
    """
    Revised compact End Matter figure.

    Panels (a,b) retain the exact cutoff-converged single-emitter regularization
    as the minimal microscopic saturation model.  Panels (c,d) show the separate
    main-benchmark ensemble proof that the finite-gain capacity increases strongly
    with emitter number while the small-signal growth rate is held fixed.
    """
    tau_exact = normalized_time(exact["t"].to_numpy())
    tau_linear = normalized_time(linear["t"].to_numpy())
    floor = 1.0e-8

    # Split the original four panels into (a,d) and (b,c).
    fig_ad, axes_ad = plt.subplots(
        1, 2, figsize=(SINGLE_COLUMN_WIDTH, 1.85)
    )
    fig_bc, axes_bc = plt.subplots(
        1, 2, figsize=(SINGLE_COLUMN_WIDTH, 1.85)
    )
    # Preserve every original axes[index] plotting command and panel label.
    axes = np.empty((2, 2), dtype=object)
    axes[0, 0], axes[1, 1] = axes_ad
    axes[0, 1], axes[1, 0] = axes_bc

    axes[0, 0].semilogy(
        tau_linear,
        np.maximum(linear["n_b_coherent"], floor),
        linestyle="--",
        label="Linear",
    )
    axes[0, 0].semilogy(
        tau_exact,
        np.maximum(exact["n_b_coherent"], floor),
        label="1 emitter",
    )
    axes[0, 0].set_ylabel(r"$|\langle b\rangle|^2$")
    axes[0, 0].set_xlabel(r"$\kappa_0t$")
    axes[0, 0].legend(frameon=False)

    axes[0, 1].plot(
        tau_exact,
        exact["tls_inversion"],
    )
    axes[0, 1].axhline(
        exact["tls_inversion"].iloc[0],
        linestyle="--",
        color=NPG_COLORS[1],
        linewidth=1.2,
    )
    axes[0, 1].set_ylabel(r"$\langle\sigma_z\rangle$")
    axes[0, 1].set_xlabel(r"$\kappa_0t$")

    for count in sorted(ensemble_results):
        dataframe = ensemble_results[count]
        axes[1, 0].semilogy(
            dataframe["kappa0_t"],
            np.maximum(dataframe["n_b_mean_field"], floor),
            label=rf"${count}$",
        )
    axes[1, 0].axhline(
        target_occupation,
        color="black",
        linestyle=":",
        linewidth=1.0,
    )
    axes[1, 0].set_xlabel(r"$\kappa_0t$")
    axes[1, 0].set_ylabel(r"$|\langle b\rangle|^2$")
    axes[1, 0].legend(
        frameon=False,
        ncol=1,
        loc="lower left",
        title=r"$N_\sigma$",
    )

    axes[1, 1].plot(
        ensemble_summary["emitter_count"],
        ensemble_summary["plateau_n_b_mean_field"],
        marker="o",
    )
    axes[1, 1].axhline(
        target_occupation,
        color="black",
        linestyle=":",
        linewidth=1.0,
    )
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_xlabel(r"$N_\sigma$")
    axes[1, 1].set_ylabel(r"stationary $|\langle b\rangle|^2$")

    for label, ax in zip(("(a)", "(a)", "(b)", "(b)"), axes.flat):
        _axis_common(ax)
        _panel_label(ax, label)
    for fig, filename in (
        (fig_ad, "Fig4_saturable_gain_end_matter.pdf"),
        (fig_bc, "Fig4_saturable_gain_end_matter_bc.pdf"),
    ):
        fig.subplots_adjust(
            left=0.19,
            right=0.98,
            bottom=0.24,
            top=0.98,
            wspace=0.50,
        )
        save_figure(
            fig,
            output.figures / filename,
            apply_tight_layout=False,
        )


# =============================================================================
# Exact QuTiP model
# =============================================================================

def _import_qutip():
    try:
        import qutip as qt
    except ImportError as exc:
        raise RuntimeError(
            "QuTiP is required for the nonlinear calculation. "
            "Install the supplied requirements file first."
        ) from exc
    major = int(str(qt.__version__).split(".")[0])
    if major < 5:
        raise RuntimeError(
            f"QuTiP >= 5 is required; found {qt.__version__}."
        )
    return qt


def _top_level_projector(qt, dimension: int, levels: int):
    levels = min(levels, dimension)
    projector = 0
    for index in range(dimension - levels, dimension):
        ket = qt.basis(dimension, index)
        projector = projector + ket * ket.dag()
    return projector


def _exact_ergotropy_callback(
    battery_subsystem_index: int,
    omega_b: float,
) -> Callable[[float, Any], float]:
    def callback(_time: float, state) -> float:
        rho_b = state.ptrace(battery_subsystem_index)
        matrix = np.asarray(rho_b.full(), dtype=complex)
        matrix = 0.5 * (matrix + matrix.conj().T)

        probabilities = np.linalg.eigvalsh(matrix)
        probabilities = np.real(probabilities)
        probabilities[np.abs(probabilities) < 1.0e-13] = 0.0
        if np.min(probabilities) < -1.0e-8:
            warnings.warn(
                "Reduced battery state has a noticeable negative eigenvalue: "
                f"{np.min(probabilities):.3e}"
            )
        probabilities = np.clip(probabilities, 0.0, None)
        trace = float(np.sum(probabilities))
        if trace <= 0.0:
            return np.nan
        probabilities /= trace

        levels = np.arange(matrix.shape[0], dtype=float)
        actual_energy = omega_b * float(
            np.real(
                np.dot(
                    levels,
                    np.diag(matrix),
                )
            )
        )
        passive_energy = omega_b * float(
            np.dot(
                np.sort(probabilities)[::-1],
                levels,
            )
        )
        ergotropy = actual_energy - passive_energy
        if ergotropy < 0.0 and abs(ergotropy) < 1.0e-10:
            ergotropy = 0.0
        return float(ergotropy)

    return callback


def build_exact_model(
    bosonic: BosonicParams,
    tls: TLSParams,
    derived: DerivedGain,
    cutoff_a: int,
    cutoff_b: int,
    edge_levels: int,
):
    qt = _import_qutip()

    if cutoff_a < 3 or cutoff_b < 3:
        raise ValueError("Each bosonic cutoff must be at least three.")

    identity_a = qt.qeye(cutoff_a)
    identity_b = qt.qeye(cutoff_b)
    identity_tls = qt.qeye(2)

    a = qt.tensor(
        qt.destroy(cutoff_a),
        identity_b,
        identity_tls,
    )
    b = qt.tensor(
        identity_a,
        qt.destroy(cutoff_b),
        identity_tls,
    )

    # Explicit emitter convention:
    # |g> = |0>, |e> = |1>,
    # sigma_- = |g><e|,
    # sigma_z = |e><e| - |g><g|.
    ground = qt.basis(2, 0)
    excited = qt.basis(2, 1)
    sigma_minus_tls = ground * excited.dag()
    sigma_plus_tls = sigma_minus_tls.dag()
    sigma_z_tls = (
        excited * excited.dag()
        - ground * ground.dag()
    )
    projector_excited_tls = excited * excited.dag()

    sigma_minus = qt.tensor(
        identity_a,
        identity_b,
        sigma_minus_tls,
    )
    sigma_plus = sigma_minus.dag()
    sigma_z = qt.tensor(
        identity_a,
        identity_b,
        sigma_z_tls,
    )
    projector_excited = qt.tensor(
        identity_a,
        identity_b,
        projector_excited_tls,
    )

    hamiltonian = (
        bosonic.delta * a.dag() * a
        - bosonic.delta * b.dag() * b
        + tls.Delta_sigma * projector_excited
        + derived.g_sigma
        * (
            a.dag() * sigma_minus
            + a * sigma_plus
        )
        + bosonic.drive * a.dag()
        + np.conjugate(bosonic.drive) * a
    )

    collective = (
        np.cos(bosonic.theta) * a
        + np.exp(1j * bosonic.phi)
        * np.sin(bosonic.theta)
        * b
    )

    collapse_operators = [
        np.sqrt(bosonic.kappa_a_down) * a,
        np.sqrt(bosonic.kappa_b) * b,
        np.sqrt(bosonic.Gamma_col) * collective,
        np.sqrt(tls.P_sigma) * sigma_plus,
        np.sqrt(tls.gamma_sigma) * sigma_minus,
    ]
    if tls.gamma_sigma_phi > 0.0:
        collapse_operators.append(
            np.sqrt(tls.gamma_sigma_phi / 2.0)
            * sigma_z
        )

    p_excited = tls.excited_population_unsaturated
    rho_tls = (
        p_excited * excited * excited.dag()
        + (1.0 - p_excited)
        * ground * ground.dag()
    )
    rho0 = qt.tensor(
        qt.fock_dm(cutoff_a, 0),
        qt.fock_dm(cutoff_b, 0),
        rho_tls,
    )

    edge_a_local = _top_level_projector(
        qt,
        cutoff_a,
        edge_levels,
    )
    edge_b_local = _top_level_projector(
        qt,
        cutoff_b,
        edge_levels,
    )
    edge_a = qt.tensor(
        edge_a_local,
        identity_b,
        identity_tls,
    )
    edge_b = qt.tensor(
        identity_a,
        edge_b_local,
        identity_tls,
    )

    e_ops = {
        "n_a": a.dag() * a,
        "n_b": b.dag() * b,
        "mean_a": a,
        "mean_b": b,
        "tls_excited": projector_excited,
        "tls_inversion": sigma_z,
        "edge_a": edge_a,
        "edge_b": edge_b,
        "trace": lambda _t, state: float(np.real(state.tr())),
        "ergotropy_b": _exact_ergotropy_callback(
            battery_subsystem_index=1,
            omega_b=bosonic.omega_b,
        ),
    }

    return {
        "qt": qt,
        "H": hamiltonian,
        "rho0": rho0,
        "c_ops": collapse_operators,
        "e_ops": e_ops,
        "e_op_keys": list(e_ops.keys()),
        "dimension": cutoff_a * cutoff_b * 2,
    }


def _extract_expectation(result, key: str, key_order: Sequence[str]):
    if hasattr(result, "e_data"):
        e_data = getattr(result, "e_data")
        if isinstance(e_data, dict) and key in e_data:
            return np.asarray(e_data[key])
    expect = getattr(result, "expect", None)
    if isinstance(expect, dict) and key in expect:
        return np.asarray(expect[key])
    if isinstance(expect, (list, tuple)):
        return np.asarray(expect[key_order.index(key)])
    raise RuntimeError(
        f"Could not extract expectation value {key!r} "
        "from the QuTiP result object."
    )


def simulate_exact(
    bosonic: BosonicParams,
    tls: TLSParams,
    derived: DerivedGain,
    numerics: Numerics,
    cutoff_a: int,
    cutoff_b: int,
    times: np.ndarray,
    progress_bar: str | None,
) -> pd.DataFrame:
    model = build_exact_model(
        bosonic,
        tls,
        derived,
        cutoff_a,
        cutoff_b,
        numerics.edge_levels,
    )
    qt = model["qt"]

    options = {
        "method": numerics.solver_method,
        "atol": numerics.atol,
        "rtol": numerics.rtol,
        "nsteps": numerics.nsteps,
        "store_states": False,
        "store_final_state": True,
        "normalize_output": True,
        "progress_bar": progress_bar,
    }

    print(
        f"Exact run: N_a={cutoff_a}, N_b={cutoff_b}, "
        f"Hilbert dimension={model['dimension']}, "
        f"time points={times.size}"
    )
    result = qt.mesolve(
        model["H"],
        model["rho0"],
        times,
        model["c_ops"],
        e_ops=model["e_ops"],
        options=options,
    )

    key_order = model["e_op_keys"]
    values = {
        key: _extract_expectation(result, key, key_order)
        for key in key_order
    }

    n_a = np.real(values["n_a"]).astype(float)
    n_b = np.real(values["n_b"]).astype(float)
    mean_a = np.asarray(values["mean_a"], dtype=complex)
    mean_b = np.asarray(values["mean_b"], dtype=complex)
    ergotropy = np.real(values["ergotropy_b"]).astype(float)
    energy = bosonic.omega_b * n_b
    passive_energy = energy - ergotropy
    coherent_b = np.abs(mean_b) ** 2
    incoherent_b = n_b - coherent_b
    snr = coherent_b / np.maximum(
        incoherent_b + 0.5,
        1.0e-300,
    )
    eta = np.divide(
        ergotropy,
        energy,
        out=np.full_like(energy, np.nan),
        where=np.abs(energy) > 1.0e-15,
    )

    dataframe = pd.DataFrame(
        {
            "t": times,
            "cutoff_a": cutoff_a,
            "cutoff_b": cutoff_b,
            "hilbert_dimension": model["dimension"],
            "n_a": n_a,
            "n_b": n_b,
            "mean_a_real": mean_a.real,
            "mean_a_imag": mean_a.imag,
            "mean_b_real": mean_b.real,
            "mean_b_imag": mean_b.imag,
            "n_b_coherent": coherent_b,
            "n_b_incoherent": incoherent_b,
            "energy_b": energy,
            "ergotropy_b": ergotropy,
            "passive_energy_b": passive_energy,
            "ergotropy_fraction_b": eta,
            "snr_b": snr,
            "tls_excited": np.real(
                values["tls_excited"]
            ).astype(float),
            "tls_inversion": np.real(
                values["tls_inversion"]
            ).astype(float),
            "edge_a": np.real(values["edge_a"]).astype(float),
            "edge_b": np.real(values["edge_b"]).astype(float),
            "trace": np.real(values["trace"]).astype(float),
        }
    )
    return dataframe


# =============================================================================
# Diagnostics
# =============================================================================

def tail_slice(length: int, fraction: float) -> slice:
    start = max(0, int(math.floor((1.0 - fraction) * length)))
    return slice(start, length)


def tail_statistics(
    dataframe: pd.DataFrame,
    column: str,
    fraction: float,
) -> dict[str, float]:
    segment = dataframe.iloc[tail_slice(len(dataframe), fraction)]
    x = segment["t"].to_numpy(dtype=float)
    y = segment[column].to_numpy(dtype=float)
    mean = float(np.mean(y))
    std = float(np.std(y))
    if x.size >= 2:
        slope = float(np.polyfit(x, y, 1)[0])
        relative_drift = (
            abs(slope)
            * (x[-1] - x[0])
            / max(abs(mean), 1.0e-12)
        )
    else:
        slope = np.nan
        relative_drift = np.nan
    coefficient_variation = std / max(abs(mean), 1.0e-12)
    return {
        "tail_mean": mean,
        "tail_std": std,
        "tail_slope": slope,
        "tail_relative_drift": relative_drift,
        "tail_coefficient_variation": coefficient_variation,
    }


def _curve_relative_error(
    previous: np.ndarray,
    current: np.ndarray,
    scale_floor: float = 1.0e-12,
) -> float:
    """
    Supremum-norm curve error normalized by the actual signal amplitude.

    The old implementation used a floor of one, which falsely made relative
    errors look tiny whenever the physical occupation was much smaller than
    one.  Here both curves contribute to the normalization and the floor is
    only a numerical zero guard.
    """
    previous = np.asarray(previous, dtype=float)
    current = np.asarray(current, dtype=float)
    scale = max(
        float(np.nanmax(np.abs(previous))),
        float(np.nanmax(np.abs(current))),
        scale_floor,
    )
    return float(
        np.nanmax(np.abs(current - previous)) / scale
    )


def _curve_absolute_error(
    previous: np.ndarray,
    current: np.ndarray,
) -> float:
    return float(
        np.nanmax(np.abs(current - previous))
    )


def compare_cutoffs(
    previous: pd.DataFrame,
    current: pd.DataFrame,
    numerics: Numerics,
) -> dict[str, float | bool]:
    if not np.allclose(
        previous["t"].to_numpy(),
        current["t"].to_numpy(),
    ):
        raise ValueError("Cutoff runs must share the same time grid.")

    rel_n_a = _curve_relative_error(
        previous["n_a"].to_numpy(),
        current["n_a"].to_numpy(),
    )
    rel_n_b = _curve_relative_error(
        previous["n_b"].to_numpy(),
        current["n_b"].to_numpy(),
    )
    rel_ergotropy = _curve_relative_error(
        previous["ergotropy_b"].to_numpy(),
        current["ergotropy_b"].to_numpy(),
    )

    energy_previous = np.abs(
        previous["energy_b"].to_numpy()
    )
    energy_current = np.abs(
        current["energy_b"].to_numpy()
    )
    energy_scale = max(
        float(np.nanmax(energy_previous)),
        float(np.nanmax(energy_current)),
        1.0e-12,
    )
    quality_mask = np.maximum(
        energy_previous,
        energy_current,
    ) >= 1.0e-6 * energy_scale

    eta_previous = np.nan_to_num(
        previous["ergotropy_fraction_b"].to_numpy(),
        nan=0.0,
    )
    eta_current = np.nan_to_num(
        current["ergotropy_fraction_b"].to_numpy(),
        nan=0.0,
    )
    abs_eta = _curve_absolute_error(
        eta_previous[quality_mask],
        eta_current[quality_mask],
    )

    log_snr_previous = np.log10(
        1.0 + np.maximum(previous["snr_b"].to_numpy(), 0.0)
    )
    log_snr_current = np.log10(
        1.0 + np.maximum(current["snr_b"].to_numpy(), 0.0)
    )
    log_snr_error = _curve_absolute_error(
        log_snr_previous[quality_mask],
        log_snr_current[quality_mask],
    )

    edge_a = float(np.nanmax(current["edge_a"]))
    edge_b = float(np.nanmax(current["edge_b"]))
    trace_error = float(
        np.nanmax(np.abs(current["trace"] - 1.0))
    )

    passed = (
        rel_n_a <= numerics.curve_rel_n_a_tol
        and rel_n_b <= numerics.curve_rel_n_b_tol
        and rel_ergotropy <= numerics.curve_rel_ergotropy_tol
        and abs_eta <= numerics.curve_abs_eta_tol
        and log_snr_error <= numerics.curve_log_snr_tol
        and edge_a <= numerics.edge_population_tol
        and edge_b <= numerics.edge_population_tol
        and trace_error <= 1.0e-7
    )

    return {
        "curve_rel_n_a": rel_n_a,
        "curve_rel_n_b": rel_n_b,
        "curve_rel_ergotropy": rel_ergotropy,
        "curve_abs_eta": abs_eta,
        "curve_log10_1p_snr": log_snr_error,
        "maximum_edge_a": edge_a,
        "maximum_edge_b": edge_b,
        "maximum_trace_error": trace_error,
        "pair_pass": bool(passed),
    }


def first_crossing(
    times: np.ndarray,
    values: np.ndarray,
    target: float,
) -> tuple[bool, float]:
    indices = np.where(values >= target)[0]
    if not indices.size:
        return False, np.nan
    index = int(indices[0])
    if index == 0:
        return True, float(times[0])
    t0, t1 = times[index - 1], times[index]
    y0, y1 = values[index - 1], values[index]
    if abs(y1 - y0) < 1.0e-15:
        return True, float(t1)
    weight = (target - y0) / (y1 - y0)
    return True, float(t0 + weight * (t1 - t0))


def early_time_agreement(
    exact: pd.DataFrame,
    linear: pd.DataFrame,
    n_sat: float,
    fraction: float,
) -> dict[str, float | int]:
    mask = (
        linear["n_a"].to_numpy()
        <= fraction * n_sat
    )
    # Remove the trivial t=0 point.
    mask &= linear["t"].to_numpy() > 0.0
    indices = np.where(mask)[0]
    if indices.size < 5:
        return {
            "number_of_points": int(indices.size),
            "maximum_time": np.nan,
            "relative_rms_n_a": np.nan,
            "relative_rms_n_b": np.nan,
            "relative_rms_mean_a": np.nan,
            "relative_rms_mean_b": np.nan,
        }

    def relative_rms(
        exact_values: np.ndarray,
        reference_values: np.ndarray,
    ) -> float:
        return float(
            np.sqrt(
                np.mean(
                    np.abs(
                        exact_values - reference_values
                    ) ** 2
                )
            )
            / max(
                float(
                    np.max(
                        np.abs(reference_values)
                    )
                ),
                1.0e-12,
            )
        )

    exact_n_a = exact["n_a"].to_numpy()[indices]
    linear_n_a = linear["n_a"].to_numpy()[indices]
    exact_n_b = exact["n_b"].to_numpy()[indices]
    linear_n_b = linear["n_b"].to_numpy()[indices]

    exact_mean_a = (
        exact["mean_a_real"].to_numpy()[indices]
        + 1j * exact["mean_a_imag"].to_numpy()[indices]
    )
    linear_mean_a = (
        linear["mean_a_real"].to_numpy()[indices]
        + 1j * linear["mean_a_imag"].to_numpy()[indices]
    )
    exact_mean_b = (
        exact["mean_b_real"].to_numpy()[indices]
        + 1j * exact["mean_b_imag"].to_numpy()[indices]
    )
    linear_mean_b = (
        linear["mean_b_real"].to_numpy()[indices]
        + 1j * linear["mean_b_imag"].to_numpy()[indices]
    )

    return {
        "number_of_points": int(indices.size),
        "maximum_time": float(
            linear["t"].to_numpy()[indices[-1]]
        ),
        "relative_rms_n_a": relative_rms(
            exact_n_a,
            linear_n_a,
        ),
        "relative_rms_n_b": relative_rms(
            exact_n_b,
            linear_n_b,
        ),
        "relative_rms_mean_a": relative_rms(
            exact_mean_a,
            linear_mean_a,
        ),
        "relative_rms_mean_b": relative_rms(
            exact_mean_b,
            linear_mean_b,
        ),
    }


def make_time_grid(
    t_end: float,
    number_of_points: int,
    early_end: float | None = None,
    early_point_fraction: float = 0.40,
) -> np.ndarray:
    """
    Construct a monotonic grid with dense early-time resolution and a coarser
    late-time tail.

    The early interval resolves the TLS transient and the onset of gain
    saturation.  The late interval retains enough points for stationary-tail
    diagnostics without making exact density-matrix evolution unnecessarily
    expensive.
    """
    if number_of_points < 3:
        raise ValueError("number_of_points must be at least three.")
    if t_end <= 0.0:
        raise ValueError("t_end must be positive.")

    if early_end is None:
        early_end = min(100.0, 0.10 * t_end)
    early_end = float(np.clip(early_end, 0.0, t_end))

    if early_end <= 0.0 or early_end >= t_end:
        return np.linspace(
            0.0,
            t_end,
            number_of_points,
        )

    early_count = int(
        round(early_point_fraction * number_of_points)
    )
    early_count = min(
        max(early_count, 20),
        number_of_points - 2,
    )
    late_count = number_of_points - early_count

    early = np.linspace(
        0.0,
        early_end,
        early_count,
        endpoint=False,
    )
    late = np.linspace(
        early_end,
        t_end,
        late_count,
        endpoint=True,
    )
    grid = np.concatenate((early, late))

    if grid.size != number_of_points:
        raise RuntimeError("Internal time-grid size error.")
    if np.any(np.diff(grid) <= 0.0):
        raise RuntimeError("Time grid is not strictly increasing.")
    return grid


# =============================================================================
# Cutoff parsing and convergence driver
# =============================================================================

def parse_cutoffs(specification: str) -> list[tuple[int, int]]:
    """
    Accepted forms:
        10,14,18,22
        10x12,14x16,18x20
    """
    pairs: list[tuple[int, int]] = []
    for item in specification.split(","):
        token = item.strip().lower()
        if not token:
            continue
        if "x" in token:
            left, right = token.split("x", maxsplit=1)
            pair = (int(left), int(right))
        else:
            value = int(token)
            pair = (value, value)
        if min(pair) < 3:
            raise ValueError("Cutoffs must be at least three.")
        pairs.append(pair)

    if len(pairs) < 2:
        raise ValueError(
            "At least two cutoff pairs are required for convergence."
        )
    for previous, current in zip(pairs[:-1], pairs[1:]):
        if (
            current[0] < previous[0]
            or current[1] < previous[1]
        ):
            raise ValueError(
                "Cutoff pairs must increase monotonically."
            )
    return pairs


def run_cutoff_convergence(
    bosonic: BosonicParams,
    tls: TLSParams,
    derived: DerivedGain,
    numerics: Numerics,
    cutoffs: Sequence[tuple[int, int]],
    output: OutputPaths,
    progress_bar: str | None,
    stop_after_convergence: bool = False,
) -> tuple[pd.DataFrame, tuple[int, int] | None]:
    times = make_time_grid(
        numerics.t_end,
        numerics.convergence_points,
    )

    rows: list[dict[str, Any]] = []
    previous: pd.DataFrame | None = None
    consecutive_passes = 0
    selected: tuple[int, int] | None = None

    for cutoff_a, cutoff_b in cutoffs:
        dataframe = simulate_exact(
            bosonic,
            tls,
            derived,
            numerics,
            cutoff_a,
            cutoff_b,
            times,
            progress_bar,
        )
        save_dataframe(
            dataframe,
            output.data
            / f"saturation_cutoff_Na_{cutoff_a}_Nb_{cutoff_b}.csv",
        )

        n_a_tail = tail_statistics(
            dataframe,
            "n_a",
            numerics.tail_fraction,
        )
        n_b_tail = tail_statistics(
            dataframe,
            "n_b",
            numerics.tail_fraction,
        )
        w_tail = tail_statistics(
            dataframe,
            "ergotropy_b",
            numerics.tail_fraction,
        )
        eta_tail = tail_statistics(
            dataframe,
            "ergotropy_fraction_b",
            numerics.tail_fraction,
        )
        snr_tail = tail_statistics(
            dataframe,
            "snr_b",
            numerics.tail_fraction,
        )

        row: dict[str, Any] = {
            "cutoff_a": cutoff_a,
            "cutoff_b": cutoff_b,
            "hilbert_dimension": cutoff_a * cutoff_b * 2,
            "maximum_edge_a": float(
                np.max(dataframe["edge_a"])
            ),
            "maximum_edge_b": float(
                np.max(dataframe["edge_b"])
            ),
            "maximum_trace_error": float(
                np.max(np.abs(dataframe["trace"] - 1.0))
            ),
            "tail_mean_n_a": n_a_tail["tail_mean"],
            "tail_mean_n_b": n_b_tail["tail_mean"],
            "tail_mean_ergotropy": w_tail["tail_mean"],
            "tail_mean_eta": eta_tail["tail_mean"],
            "tail_mean_snr": snr_tail["tail_mean"],
            "tail_relative_drift_n_a": n_a_tail[
                "tail_relative_drift"
            ],
            "tail_relative_drift_n_b": n_b_tail[
                "tail_relative_drift"
            ],
            "tail_cv_n_a": n_a_tail[
                "tail_coefficient_variation"
            ],
            "tail_cv_n_b": n_b_tail[
                "tail_coefficient_variation"
            ],
            "pair_pass": False,
            "consecutive_passes": 0,
        }

        if previous is not None:
            comparison = compare_cutoffs(
                previous,
                dataframe,
                numerics,
            )
            row.update(comparison)
            if bool(comparison["pair_pass"]):
                consecutive_passes += 1
            else:
                consecutive_passes = 0
            row["consecutive_passes"] = consecutive_passes

            if (
                consecutive_passes
                >= numerics.required_consecutive_passes
            ):
                selected = (cutoff_a, cutoff_b)
            else:
                selected = None

        rows.append(row)
        previous = dataframe

        if selected is not None and stop_after_convergence:
            print(
                "Required consecutive cutoff checks passed. "
                f"Stopping the scan at N_a={selected[0]}, "
                f"N_b={selected[1]}."
            )
            break

    summary = pd.DataFrame(rows)
    save_dataframe(
        summary,
        output.data / "saturation_cutoff_convergence.csv",
    )
    save_json(
        {
            "selected_cutoff": selected,
            "required_consecutive_passes": (
                numerics.required_consecutive_passes
            ),
            "stop_after_convergence": stop_after_convergence,
            "converged": selected is not None,
        },
        output.summaries
        / "saturation_cutoff_selection.json",
    )
    return summary, selected


# =============================================================================
# Plotting
# =============================================================================

def plot_cutoff_summary(
    summary: pd.DataFrame,
    output: OutputPaths,
) -> None:
    labels = [
        rf"${int(a)}\times{int(b)}$"
        for a, b in zip(
            summary["cutoff_a"],
            summary["cutoff_b"],
        )
    ]
    x = np.arange(len(labels))
    columns = (
        ("curve_rel_n_a", r"$n_a$"),
        ("curve_rel_n_b", r"$n_b$"),
        ("curve_rel_ergotropy", r"$\mathcal{W}_b$"),
        ("curve_abs_eta", r"$\eta_{\mathcal{W}}$"),
        (
            "curve_log10_1p_snr",
            r"$\log_{10}(1+\mathrm{SNR}_b)$",
        ),
    )

    # Individual supporting panels.
    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_WIDTH, 2.50))
    ax.plot(
        x,
        summary["tail_mean_n_a"],
        marker="o",
        label=r"$\overline{n}_a$",
    )
    ax.plot(
        x,
        summary["tail_mean_n_b"],
        marker="s",
        linestyle="--",
        label=r"$\overline{n}_b$",
    )
    ax.plot(
        x,
        summary["tail_mean_ergotropy"],
        marker="^",
        linestyle=":",
        label=r"$\overline{\mathcal{W}}_b/\omega_b$",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_xlabel(r"Cutoffs $N_a\times N_b$")
    ax.set_ylabel(r"Tail-averaged value")
    _axis_common(ax)
    ax.legend(frameon=False)
    save_figure(
        fig,
        output.figures / "saturation_cutoff_tail_values.pdf",
    )

    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_WIDTH, 2.50))
    for column, label in columns:
        if column in summary:
            values = summary[column].to_numpy(dtype=float)
            ax.semilogy(
                x,
                np.maximum(values, 1.0e-16),
                marker="o",
                label=label,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_xlabel(r"Cutoffs $N_a\times N_b$")
    ax.set_ylabel(r"Successive-cutoff error")
    _axis_common(ax)
    ax.legend(frameon=False, ncol=2)
    save_figure(
        fig,
        output.figures / "saturation_cutoff_curve_errors.pdf",
    )

    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_WIDTH, 2.50))
    ax.semilogy(
        x,
        np.maximum(summary["maximum_edge_a"], 1.0e-18),
        marker="o",
        label="Charger edge",
    )
    ax.semilogy(
        x,
        np.maximum(summary["maximum_edge_b"], 1.0e-18),
        marker="s",
        linestyle="--",
        label="Battery edge",
    )
    ax.axhline(
        1.0e-7,
        linestyle=":",
        color=NPG_COLORS[2],
        linewidth=1.2,
        label="Tolerance",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_xlabel(r"Cutoffs $N_a\times N_b$")
    ax.set_ylabel(r"Maximum edge population")
    _axis_common(ax)
    ax.legend(frameon=False)
    save_figure(
        fig,
        output.figures / "saturation_cutoff_edge_population.pdf",
    )

    # Directly usable three-panel Supplementary figure.
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(DOUBLE_COLUMN_WIDTH, 2.25),
    )
    axes[0].plot(
        x,
        summary["tail_mean_n_a"],
        marker="o",
        label=r"$\overline{n}_a$",
    )
    axes[0].plot(
        x,
        summary["tail_mean_n_b"],
        marker="s",
        linestyle="--",
        label=r"$\overline{n}_b$",
    )
    axes[0].plot(
        x,
        summary["tail_mean_ergotropy"],
        marker="^",
        linestyle=":",
        label=r"$\overline{\mathcal{W}}_b/\omega_b$",
    )
    for column, label in columns:
        if column in summary:
            values = summary[column].to_numpy(dtype=float)
            axes[1].semilogy(
                x,
                np.maximum(values, 1.0e-16),
                marker="o",
                label=label,
            )
    axes[2].semilogy(
        x,
        np.maximum(summary["maximum_edge_a"], 1.0e-18),
        marker="o",
        label="Charger edge",
    )
    axes[2].semilogy(
        x,
        np.maximum(summary["maximum_edge_b"], 1.0e-18),
        marker="s",
        linestyle="--",
        label="Battery edge",
    )
    axes[2].axhline(
        1.0e-7,
        linestyle=":",
        color=NPG_COLORS[2],
        linewidth=1.2,
        label="Tolerance",
    )
    ylabels = (
        "Tail-averaged value",
        "Successive-cutoff error",
        "Maximum edge population",
    )
    for index, ax in enumerate(axes):
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_xlabel(r"$N_a\times N_b$")
        ax.set_ylabel(ylabels[index])
        _axis_common(ax)
        _panel_label(ax, f"({chr(97 + index)})")
        ax.legend(frameon=False, ncol=1)
    fig.subplots_adjust(
        left=0.075,
        right=0.995,
        bottom=0.25,
        top=0.98,
        wspace=0.42,
    )
    save_figure(
        fig,
        output.figures / "FigS5_saturation_cutoff_convergence.pdf",
        apply_tight_layout=False,
    )


def _target_for_plot(
    exact: pd.DataFrame,
    requested_target: float,
    tail_fraction: float,
) -> tuple[float, bool, float]:
    reached, crossing = first_crossing(
        exact["t"].to_numpy(),
        exact["n_b"].to_numpy(),
        requested_target,
    )
    if reached:
        return requested_target, True, crossing

    plateau = tail_statistics(
        exact,
        "n_b",
        tail_fraction,
    )["tail_mean"]
    fallback = max(0.5 * plateau, 1.0e-12)
    fallback_reached, fallback_crossing = first_crossing(
        exact["t"].to_numpy(),
        exact["n_b"].to_numpy(),
        fallback,
    )
    if not fallback_reached:
        return fallback, False, np.nan
    return fallback, False, fallback_crossing


def plot_final_results(
    exact: pd.DataFrame,
    linear: pd.DataFrame,
    numerics: Numerics,
    n_sat: float,
    output: OutputPaths,
) -> dict[str, Any]:
    target_used, requested_reached, crossing_time = _target_for_plot(
        exact,
        numerics.target_occupation,
        numerics.tail_fraction,
    )

    t = exact["t"].to_numpy()
    tau = normalized_time(t)
    crossing_tau = (
        crossing_time * REFERENCE_RATE
        if np.isfinite(crossing_time)
        else np.nan
    )
    floor = 1.0e-10
    omega_b = 1.0

    amplitude_exact = np.hypot(
        exact["mean_b_real"],
        exact["mean_b_imag"],
    )
    amplitude_linear = np.hypot(
        linear["mean_b_real"],
        linear["mean_b_imag"],
    )
    early_mask = t <= 10.0
    non_displacement = (
        exact["ergotropy_b"] / omega_b
        - exact["n_b_coherent"]
    )

    # Individual supporting figures.
    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_WIDTH, 2.50))
    ax.semilogy(
        tau,
        np.maximum(linear["n_a"], floor),
        linestyle="--",
        label="Linear gain",
    )
    ax.semilogy(
        tau,
        np.maximum(exact["n_a"], floor),
        label="Saturable gain",
    )
    ax.set_xlabel(r"$\kappa_0 t$")
    ax.set_ylabel(r"$\langle n_a\rangle$")
    _axis_common(ax)
    ax.legend(frameon=False)
    save_figure(
        fig,
        output.figures / "saturation_charger_occupation.pdf",
    )

    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_WIDTH, 2.50))
    ax.semilogy(
        tau,
        np.maximum(linear["n_b"], floor),
        linestyle="--",
        label="Linear gain",
    )
    ax.semilogy(
        tau,
        np.maximum(exact["n_b"], floor),
        label="Saturable gain",
    )
    if np.isfinite(crossing_time):
        ax.axvline(
            crossing_tau,
            color=NPG_COLORS[2],
            linestyle=":",
            linewidth=1.2,
        )
        ax.axhline(
            target_used,
            color=NPG_COLORS[2],
            linestyle=":",
            linewidth=1.0,
        )
    ax.set_xlabel(r"$\kappa_0 t$")
    ax.set_ylabel(r"$\langle n_b\rangle$")
    _axis_common(ax)
    ax.legend(frameon=False)
    save_figure(
        fig,
        output.figures / "saturation_occupation.pdf",
    )

    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_WIDTH, 2.50))
    ax.plot(
        tau[early_mask],
        amplitude_linear[early_mask],
        linestyle="--",
        label="Linear gain",
    )
    ax.plot(
        tau[early_mask],
        amplitude_exact[early_mask],
        label="Saturable gain",
    )
    ax.set_xlabel(r"$\kappa_0 t$")
    ax.set_ylabel(r"$|\langle b\rangle|$")
    _axis_common(ax)
    ax.legend(frameon=False)
    save_figure(
        fig,
        output.figures / "saturation_early_time_battery_amplitude.pdf",
    )

    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_WIDTH, 2.50))
    ax.plot(
        tau,
        exact["energy_b"] / omega_b,
        label=r"$E_b$",
    )
    ax.plot(
        tau,
        exact["ergotropy_b"] / omega_b,
        linestyle="--",
        label=r"$\mathcal{W}_b$",
    )
    ax.plot(
        tau,
        exact["passive_energy_b"] / omega_b,
        linestyle=":",
        label=r"$E_b^{\rm pas}$",
    )
    if np.isfinite(crossing_time):
        ax.axvline(
            crossing_tau,
            color=NPG_COLORS[2],
            linestyle=":",
            linewidth=1.2,
        )
    ax.set_xlabel(r"$\kappa_0 t$")
    ax.set_ylabel(r"Energy$/\omega_b$")
    _axis_common(ax)
    ax.legend(frameon=False)
    save_figure(
        fig,
        output.figures / "saturation_energy_ergotropy.pdf",
    )

    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_WIDTH, 2.50))
    ax.plot(tau, exact["ergotropy_fraction_b"])
    if np.isfinite(crossing_time):
        ax.axvline(
            crossing_tau,
            color=NPG_COLORS[2],
            linestyle=":",
            linewidth=1.2,
        )
    ax.set_xlabel(r"$\kappa_0 t$")
    ax.set_ylabel(r"$\eta_{\mathcal{W}}$")
    ax.set_ylim(-0.02, 1.02)
    _axis_common(ax)
    save_figure(
        fig,
        output.figures / "saturation_ergotropy_fraction.pdf",
    )

    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_WIDTH, 2.50))
    ax.semilogy(
        tau,
        np.maximum(linear["snr_b"], 1.0e-16),
        linestyle="--",
        label="Linear gain",
    )
    ax.semilogy(
        tau,
        np.maximum(exact["snr_b"], 1.0e-16),
        label="Saturable gain",
    )
    if np.isfinite(crossing_time):
        ax.axvline(
            crossing_tau,
            color=NPG_COLORS[2],
            linestyle=":",
            linewidth=1.2,
        )
    ax.set_xlabel(r"$\kappa_0 t$")
    ax.set_ylabel(r"$\mathrm{SNR}_b$")
    _axis_common(ax)
    ax.legend(frameon=False)
    save_figure(
        fig,
        output.figures / "saturation_snr.pdf",
    )

    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_WIDTH, 2.50))
    ax.plot(
        tau,
        exact["tls_inversion"],
        label=r"$\langle\sigma_z\rangle$",
    )
    ax.axhline(
        exact["tls_inversion"].iloc[0],
        linestyle="--",
        color=NPG_COLORS[1],
        linewidth=1.2,
        label=r"$w_0$",
    )
    ax.set_xlabel(r"$\kappa_0 t$")
    ax.set_ylabel(r"Emitter inversion")
    _axis_common(ax)
    ax.legend(frameon=False)
    save_figure(
        fig,
        output.figures / "saturation_emitter_inversion.pdf",
    )

    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_WIDTH, 2.50))
    ax.plot(tau, non_displacement)
    ax.set_xlabel(r"$\kappa_0 t$")
    ax.set_ylabel(
        r"$\mathcal{W}_b/\omega_b-|\langle b\rangle|^2$"
    )
    _axis_common(ax)
    save_figure(
        fig,
        output.figures / "saturation_non_displacement_ergotropy.pdf",
    )

    # Single-column End Matter figure: panels (a,d) and (b,c) separately.
    fig_ad, axes_ad = plt.subplots(1, 2, figsize=(SINGLE_COLUMN_WIDTH, 1.85))
    fig_bc, axes_bc = plt.subplots(1, 2, figsize=(SINGLE_COLUMN_WIDTH, 1.85))
    # Keep every original panel's plotted data, styling, and label unchanged.
    axes = np.empty((2, 2), dtype=object)
    axes[0, 0], axes[1, 1] = axes_ad
    axes[0, 1], axes[1, 0] = axes_bc
    axes[0, 0].plot(
        tau[early_mask],
        amplitude_linear[early_mask],
        linestyle="--",
        label="Linear gain",
    )
    axes[0, 0].plot(
        tau[early_mask],
        amplitude_exact[early_mask],
        label="Saturable gain",
    )
    axes[0, 1].semilogy(
        tau,
        np.maximum(linear["n_b"], floor),
        linestyle="--",
    )
    axes[0, 1].semilogy(
        tau,
        np.maximum(exact["n_b"], floor),
    )
    if np.isfinite(crossing_time):
        axes[0, 1].axvline(
            crossing_tau,
            color=NPG_COLORS[2],
            linestyle=":",
            linewidth=1.2,
        )
        axes[0, 1].axhline(
            target_used,
            color=NPG_COLORS[2],
            linestyle=":",
            linewidth=1.0,
        )
    axes[1, 0].plot(tau, exact["tls_inversion"])
    axes[1, 0].axhline(
        exact["tls_inversion"].iloc[0],
        linestyle="--",
        color=NPG_COLORS[1],
        linewidth=1.2,
    )
    axes[1, 1].plot(
        tau,
        exact["energy_b"] / omega_b,
        label=r"$E_b$",
    )
    axes[1, 1].plot(
        tau,
        exact["ergotropy_b"] / omega_b,
        linestyle="--",
        label=r"$\mathcal{W}_b$",
    )
    axes[1, 1].plot(
        tau,
        exact["passive_energy_b"] / omega_b,
        linestyle=":",
        label=r"$E_b^{\rm pas}$",
    )
    axes[0, 0].set_ylabel(r"$|\langle b\rangle|$")
    axes[0, 1].set_ylabel(r"$|\langle b\rangle|^2$")
    axes[1, 0].set_ylabel(r"$\langle\sigma_z\rangle$")
    axes[1, 1].set_ylabel(r"Energy$/\omega_b$")
    axes[1, 0].set_xlabel(r"$\kappa_0 t$")
    axes[1, 1].set_xlabel(r"$\kappa_0 t$")
    for label, ax in zip(("(a)", "(a)", "(b)", "(b)"), axes.flat):
        _axis_common(ax)
        _panel_label(ax, label)
    linear_handles = [
        Line2D(
            [0],
            [0],
            color=NPG_COLORS[0],
            linestyle="--",
            label="Linear gain",
        ),
        Line2D(
            [0],
            [0],
            color=NPG_COLORS[1],
            linestyle="-",
            label="Saturable gain",
        ),
    ]
    for fig, filename in (
        (fig_ad, "Fig4_saturable_gain_end_matter.pdf"),
        (fig_bc, "Fig4_saturable_gain_end_matter_bc.pdf"),
    ):
        fig.legend(
            handles=linear_handles,
            loc="upper center",
            ncol=2,
            frameon=False,
            bbox_to_anchor=(0.5, 0.995),
            fontsize=MAIN_TEXT_FONT_SIZE,
        )
        fig.subplots_adjust(
            left=0.18,
            right=0.98,
            bottom=0.21,
            top=0.88,
            wspace=0.47,
        )
        save_figure(
            fig,
            output.figures / filename,
            apply_tight_layout=False,
        )

    # Full-width Supplementary dynamics figure.
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(DOUBLE_COLUMN_WIDTH, 4.15),
    )
    axes = np.asarray(axes)
    axes[0, 0].plot(
        tau[early_mask],
        amplitude_linear[early_mask],
        linestyle="--",
        label="Linear gain",
    )
    axes[0, 0].plot(
        tau[early_mask],
        amplitude_exact[early_mask],
        label="Saturable gain",
    )
    axes[0, 1].semilogy(
        tau,
        np.maximum(linear["n_a"], floor),
        linestyle="--",
    )
    axes[0, 1].semilogy(
        tau,
        np.maximum(exact["n_a"], floor),
    )
    axes[0, 2].semilogy(
        tau,
        np.maximum(linear["n_b"], floor),
        linestyle="--",
    )
    axes[0, 2].semilogy(
        tau,
        np.maximum(exact["n_b"], floor),
    )
    axes[1, 0].plot(
        tau,
        exact["energy_b"] / omega_b,
        label=r"$E_b$",
    )
    axes[1, 0].plot(
        tau,
        exact["ergotropy_b"] / omega_b,
        linestyle="--",
        label=r"$\mathcal{W}_b$",
    )
    axes[1, 0].plot(
        tau,
        exact["passive_energy_b"] / omega_b,
        linestyle=":",
        label=r"$E_b^{\rm pas}$",
    )
    axes[1, 1].plot(tau, exact["ergotropy_fraction_b"])
    axes[1, 2].semilogy(
        tau,
        np.maximum(linear["snr_b"], 1.0e-16),
        linestyle="--",
    )
    axes[1, 2].semilogy(
        tau,
        np.maximum(exact["snr_b"], 1.0e-16),
    )
    ylabels = (
        r"$|\langle b\rangle|$",
        r"$|\langle a\rangle|^2$",
        r"$|\langle b\rangle|^2$",
        r"Energy$/\omega_b$",
        r"$\eta_{\mathcal{W}}$",
        r"$\mathrm{SNR}_b$",
    )
    for index, ax in enumerate(axes.flat):
        ax.set_xlabel(r"$\kappa_0 t$")
        ax.set_ylabel(ylabels[index])
        _axis_common(ax)
        _panel_label(ax, f"({chr(97 + index)})")
    axes[0, 0].legend(frameon=False)
    axes[1, 0].legend(frameon=False)
    fig.subplots_adjust(
        left=0.08,
        right=0.995,
        bottom=0.12,
        top=0.98,
        wspace=0.38,
        hspace=0.30,
    )
    save_figure(
        fig,
        output.figures / "FigS4_saturable_gain_dynamics.pdf",
        apply_tight_layout=False,
    )

    return {
        "requested_target": numerics.target_occupation,
        "requested_target_reached": requested_reached,
        "target_used_for_marker": target_used,
        "crossing_time": crossing_time,
        "kappa0_crossing_time": crossing_tau,
    }


# =============================================================================
# Final production run
# =============================================================================

def run_final(
    bosonic: BosonicParams,
    tls: TLSParams,
    derived: DerivedGain,
    numerics: Numerics,
    cutoff: tuple[int, int],
    output: OutputPaths,
    progress_bar: str | None,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    times = make_time_grid(
        numerics.t_end,
        numerics.final_points,
    )
    exact = simulate_exact(
        bosonic,
        tls,
        derived,
        numerics,
        cutoff[0],
        cutoff[1],
        times,
        progress_bar,
    )
    linear = propagate_linear_gaussian(
        bosonic,
        derived,
        times,
    )

    exact = exact.copy()
    linear = linear.copy()
    exact["kappa0_t"] = normalized_time(exact["t"].to_numpy())
    linear["kappa0_t"] = normalized_time(linear["t"].to_numpy())
    exact["energy_b_over_omega_b"] = (
        exact["energy_b"] / bosonic.omega_b
    )
    exact["ergotropy_b_over_omega_b"] = (
        exact["ergotropy_b"] / bosonic.omega_b
    )
    exact["passive_energy_b_over_omega_b"] = (
        exact["passive_energy_b"] / bosonic.omega_b
    )
    linear["energy_b_over_omega_b"] = (
        linear["energy_b"] / bosonic.omega_b
    )
    linear["ergotropy_b_over_omega_b"] = (
        linear["ergotropy_b"] / bosonic.omega_b
    )
    linear["passive_energy_b_over_omega_b"] = (
        linear["passive_energy_b"] / bosonic.omega_b
    )

    save_dataframe(
        exact,
        output.data / "saturation_exact_timeseries.csv",
    )
    save_dataframe(
        linear,
        output.data / "saturation_linear_reference.csv",
    )

    target_summary = plot_final_results(
        exact,
        linear,
        numerics,
        derived.n_sat,
        output,
    )

    early = early_time_agreement(
        exact,
        linear,
        derived.n_sat,
        numerics.early_fraction_n_sat,
    )
    n_a_tail = tail_statistics(
        exact,
        "n_a",
        numerics.tail_fraction,
    )
    n_b_tail = tail_statistics(
        exact,
        "n_b",
        numerics.tail_fraction,
    )
    w_tail = tail_statistics(
        exact,
        "ergotropy_b",
        numerics.tail_fraction,
    )
    eta_tail = tail_statistics(
        exact,
        "ergotropy_fraction_b",
        numerics.tail_fraction,
    )
    snr_tail = tail_statistics(
        exact,
        "snr_b",
        numerics.tail_fraction,
    )

    steady_pass = all(
        statistics["tail_relative_drift"]
        <= numerics.tail_relative_drift_tol
        and statistics["tail_coefficient_variation"]
        <= numerics.tail_coefficient_variation_tol
        for statistics in (
            n_a_tail,
            n_b_tail,
            w_tail,
        )
    )
    edge_pass = (
        float(np.max(exact["edge_a"]))
        <= numerics.edge_population_tol
        and float(np.max(exact["edge_b"]))
        <= numerics.edge_population_tol
    )

    summary = {
        "units": unit_metadata(),
        "cutoff_a": cutoff[0],
        "cutoff_b": cutoff[1],
        "hilbert_dimension": cutoff[0] * cutoff[1] * 2,
        "early_time_agreement": early,
        "tail_n_a": n_a_tail,
        "tail_n_b": n_b_tail,
        "tail_ergotropy": w_tail,
        "tail_ergotropy_fraction": eta_tail,
        "tail_snr": snr_tail,
        "maximum_edge_a": float(np.max(exact["edge_a"])),
        "maximum_edge_b": float(np.max(exact["edge_b"])),
        "maximum_trace_error": float(
            np.max(np.abs(exact["trace"] - 1.0))
        ),
        "steady_state_pass": bool(steady_pass),
        "edge_population_pass": bool(edge_pass),
        **target_summary,
    }
    save_json(
        summary,
        output.summaries / "saturation_final_summary.json",
    )
    print(json.dumps(_json_ready(summary), indent=2))
    return summary, exact, linear


# =============================================================================
# Command-line interface
# =============================================================================

def _complex_from_string(value: str) -> complex:
    return complex(value.replace("i", "j"))


def apply_overrides(
    bosonic: BosonicParams,
    tls: TLSParams,
    args: argparse.Namespace,
) -> tuple[BosonicParams, TLSParams]:
    bosonic_changes: dict[str, Any] = {}
    tls_changes: dict[str, Any] = {}

    mapping_bosonic = {
        "kappa_a_down": args.kappa_a_down,
        "kappa_b": args.kappa_b,
        "delta": args.delta,
        "drive": args.drive,
        "Gamma_col": args.Gamma_col,
        "theta": args.theta,
        "phi": args.phi,
        "target_growth_margin": args.target_growth_margin,
    }
    for key, value in mapping_bosonic.items():
        if value is not None:
            bosonic_changes[key] = value

    mapping_tls = {
        "P_sigma": args.P_sigma,
        "gamma_sigma": args.gamma_sigma,
        "gamma_sigma_phi": args.gamma_sigma_phi,
        "Delta_sigma": args.Delta_sigma,
    }
    for key, value in mapping_tls.items():
        if value is not None:
            tls_changes[key] = value

    return (
        replace(bosonic, **bosonic_changes),
        replace(tls, **tls_changes),
    )


def add_physical_arguments(
    parser: argparse.ArgumentParser,
    default_profile: str = "balanced",
) -> None:
    parser.add_argument(
        "--profile",
        choices=("low_energy", "balanced", "paper"),
        default=default_profile,
    )
    parser.add_argument("--kappa-a-down", type=float)
    parser.add_argument("--kappa-b", type=float)
    parser.add_argument("--delta", type=float)
    parser.add_argument("--drive", type=_complex_from_string)
    parser.add_argument("--Gamma-col", type=float)
    parser.add_argument("--theta", type=float)
    parser.add_argument("--phi", type=float)
    parser.add_argument("--target-growth-margin", type=float)
    parser.add_argument("--P-sigma", type=float)
    parser.add_argument("--gamma-sigma", type=float)
    parser.add_argument("--gamma-sigma-phi", type=float)
    parser.add_argument("--Delta-sigma", type=float)


def add_numerical_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--t-end", type=float, default=4000.0)
    parser.add_argument(
        "--convergence-points",
        type=int,
        default=161,
    )
    parser.add_argument(
        "--final-points",
        type=int,
        default=501,
    )
    parser.add_argument(
        "--target",
        type=float,
        default=0.5,
        help="Requested battery occupation for the crossing marker.",
    )
    parser.add_argument(
        "--edge-levels",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--required-consecutive-passes",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--progress",
        choices=("none", "text", "enhanced"),
        default="text",
    )
    parser.add_argument(
        "--output",
        default="saturation_output",
    )


def numerics_from_args(args: argparse.Namespace) -> Numerics:
    return Numerics(
        t_end=args.t_end,
        convergence_points=args.convergence_points,
        final_points=args.final_points,
        target_occupation=args.target,
        edge_levels=args.edge_levels,
        required_consecutive_passes=(
            args.required_consecutive_passes
        ),
    )


def progress_from_args(args: argparse.Namespace) -> str | None:
    return None if args.progress == "none" else args.progress


def print_derived(
    profile: str,
    bosonic: BosonicParams,
    tls: TLSParams,
    derived: DerivedGain,
) -> None:
    payload = {
        "profile": profile,
        "bosonic": asdict(bosonic),
        "tls": asdict(tls),
        "derived": asdict(derived),
        "checks": {
            "small_signal_gain": (
                2.0
                * derived.g_sigma ** 2
                * tls.gamma_perp
                * tls.w0
                / (
                    tls.gamma_perp ** 2
                    + tls.Delta_sigma ** 2
                )
            ),
            "target_margin": bosonic.target_growth_margin,
        },
    }
    print(json.dumps(_json_ready(payload), indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Exact saturable-gain simulation and cutoff convergence."
        )
    )
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    derive_parser = subparsers.add_parser(
        "derive",
        help="Print matched linear-pump and TLS parameters.",
    )
    add_physical_arguments(derive_parser)

    ensemble_parser = subparsers.add_parser(
        "ensemble",
        help=(
            "Run the symmetric pumped-emitter ensemble proof of principle "
            "without requiring QuTiP."
        ),
    )
    add_physical_arguments(ensemble_parser, default_profile="paper")
    ensemble_parser.add_argument(
        "--emitter-counts",
        default="1,10,30,100",
    )
    ensemble_parser.add_argument(
        "--ensemble-t-end",
        type=float,
        default=3000.0,
    )
    ensemble_parser.add_argument(
        "--ensemble-points",
        type=int,
        default=1201,
    )
    ensemble_parser.add_argument(
        "--ensemble-target",
        type=float,
        default=100.0,
    )
    ensemble_parser.add_argument(
        "--output",
        default="saturation_output_low_energy_normalized",
    )

    convergence_parser = subparsers.add_parser(
        "convergence",
        help="Run only the cutoff-convergence sequence.",
    )
    add_physical_arguments(convergence_parser)
    add_numerical_arguments(convergence_parser)
    convergence_parser.add_argument(
        "--cutoffs",
        default="10,14,18,22,26",
    )

    final_parser = subparsers.add_parser(
        "final",
        help="Run a final dense calculation at one cutoff.",
    )
    add_physical_arguments(final_parser)
    add_numerical_arguments(final_parser)
    final_parser.add_argument(
        "--cutoff",
        default="22x22",
    )

    all_parser = subparsers.add_parser(
        "all",
        help=(
            "Run convergence, choose a converged cutoff, "
            "and generate final figures."
        ),
    )
    add_physical_arguments(all_parser)
    add_numerical_arguments(all_parser)
    all_parser.add_argument(
        "--cutoffs",
        default="10,14,18,22,26",
    )
    all_parser.add_argument(
        "--allow-unconverged",
        action="store_true",
        help=(
            "Use the largest cutoff even if the convergence "
            "criteria are not satisfied."
        ),
    )
    return parser



def parse_emitter_counts(specification: str) -> tuple[int, ...]:
    values = tuple(
        int(part.strip())
        for part in specification.split(",")
        if part.strip()
    )
    if not values or any(value < 1 for value in values):
        raise ValueError("Emitter counts must be positive integers.")
    return values

def _parse_single_cutoff(specification: str) -> tuple[int, int]:
    pairs = parse_cutoffs(
        f"{specification},{specification}"
    )
    return pairs[0]


def main(argv: Sequence[str] | None = None) -> None:
    configure_plot_style()
    parser = build_parser()
    args = parser.parse_args(argv)

    bosonic, tls = benchmark_profile(args.profile)
    bosonic, tls = apply_overrides(bosonic, tls, args)
    bosonic.validate()
    tls.validate()
    derived = derive_gain(bosonic, tls)
    print_derived(args.profile, bosonic, tls, derived)

    if args.command == "derive":
        return

    if args.command == "ensemble":
        output = OutputPaths.create(args.output)
        emitter_counts = parse_emitter_counts(args.emitter_counts)
        results, summary = run_emitter_ensemble_proof(
            bosonic=bosonic,
            tls=tls,
            derived=derived,
            emitter_counts=emitter_counts,
            t_end=args.ensemble_t_end,
            points=args.ensemble_points,
            target_occupation=args.ensemble_target,
            output=output,
        )
        plot_emitter_ensemble_proof(
            results,
            summary,
            args.ensemble_target,
            output,
        )
        exact_path = output.data / "saturation_exact_timeseries.csv"
        linear_path = output.data / "saturation_linear_reference.csv"
        if exact_path.exists() and linear_path.exists():
            plot_revised_end_matter_figure(
                pd.read_csv(exact_path),
                pd.read_csv(linear_path),
                results,
                summary,
                args.ensemble_target,
                output,
            )
        print("\nEmitter-ensemble proof completed.")
        print(summary.to_string(index=False))
        return

    numerics = numerics_from_args(args)
    numerics.validate()
    output = OutputPaths.create(args.output)
    progress_bar = progress_from_args(args)

    save_json(
        {
            "units": unit_metadata(),
            "profile": args.profile,
            "bosonic": asdict(bosonic),
            "bosonic_dimensionless": normalized_bosonic_parameters(bosonic),
            "tls": asdict(tls),
            "tls_dimensionless": normalized_tls_parameters(tls),
            "derived": asdict(derived),
            "numerics": asdict(numerics),
            "thread_environment": {
                key: os.environ.get(key)
                for key in (
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            },
        },
        output.summaries / "saturation_run_configuration.json",
    )

    if args.command == "convergence":
        cutoffs = parse_cutoffs(args.cutoffs)
        summary, selected = run_cutoff_convergence(
            bosonic,
            tls,
            derived,
            numerics,
            cutoffs,
            output,
            progress_bar,
        )
        plot_cutoff_summary(summary, output)
        print(f"Selected cutoff: {selected}")
        return

    if args.command == "final":
        cutoff = _parse_single_cutoff(args.cutoff)
        run_final(
            bosonic,
            tls,
            derived,
            numerics,
            cutoff,
            output,
            progress_bar,
        )
        return

    if args.command == "all":
        cutoffs = parse_cutoffs(args.cutoffs)
        summary, selected = run_cutoff_convergence(
            bosonic,
            tls,
            derived,
            numerics,
            cutoffs,
            output,
            progress_bar,
        )
        plot_cutoff_summary(summary, output)

        if selected is None:
            if not args.allow_unconverged:
                raise RuntimeError(
                    "No cutoff passed the requested convergence criteria. "
                    "Increase the cutoff sequence or use "
                    "--allow-unconverged only for diagnosis."
                )
            selected = cutoffs[-1]
            warnings.warn(
                "Using the largest cutoff despite failed convergence."
            )

        final_summary, _, _ = run_final(
            bosonic,
            tls,
            derived,
            numerics,
            selected,
            output,
            progress_bar,
        )
        if not final_summary["steady_state_pass"]:
            warnings.warn(
                "The final time window has not reached a sufficiently "
                "stationary plateau. Increase --t-end."
            )
        if not final_summary["edge_population_pass"]:
            warnings.warn(
                "The selected cutoff has excessive edge population. "
                "Increase both oscillator cutoffs."
            )
        return

    raise RuntimeError(f"Unhandled command: {args.command}")


# =============================================================================
# ONE-CLICK SETTINGS
# =============================================================================
#
# Open this file and click Run.  No command-line arguments, separate module,
# or additional notebook cell is required.
#
# This low-energy benchmark remains in the broken small-signal regime but uses
# stronger local losses, a weaker coherent seed, and a modest distance above
# threshold so that nonlinear saturation is accessible with small Fock cutoffs.
#
ONE_CLICK_PROFILE = "low_energy"

ONE_CLICK_CUTOFFS = [
    (6, 6),
    (8, 8),
    (10, 10),
    (12, 10),
    (14, 10),
    (16, 10),
    (18, 10),
]

ONE_CLICK_T_END = 2000.0
ONE_CLICK_CONVERGENCE_POINTS = 121
ONE_CLICK_FINAL_POINTS = 401
ONE_CLICK_TARGET_OCCUPATION = 0.25
ONE_CLICK_OUTPUT_DIRECTORY = "saturation_output_low_energy_normalized"
ONE_CLICK_PROGRESS_BAR = "text"

# Reuse the already converged single-emitter CSV data when present.  This makes
# the new many-emitter proof inexpensive to regenerate after a manuscript edit.
ONE_CLICK_REUSE_EXISTING_EXACT = True

# Large-N proof-of-principle settings.  The main-paper moderate-theta profile is
# used here, and g_per_emitter is scaled as 1/sqrt(N) so the small-signal growth
# rate remains fixed while the saturation capacity grows.
ONE_CLICK_ENSEMBLE_COUNTS = (1, 10, 30, 100)
ONE_CLICK_ENSEMBLE_T_END = 3000.0
ONE_CLICK_ENSEMBLE_POINTS = 1201
ONE_CLICK_ENSEMBLE_TARGET = 100.0

# The battery is expected to remain less occupied than the charger, so N_b is
# first raised to 10 and then held fixed while N_a is increased.  Both edge
# populations are checked at every step.  Two successive cutoff increments must
# pass all curve, edge-population, and trace checks.
ONE_CLICK_REQUIRED_CONSECUTIVE_PASSES = 2
ONE_CLICK_STOP_AFTER_CONVERGENCE = True

# Keep False for all manuscript calculations.  When False, the program refuses
# to create final production figures if cutoff convergence was not established.
ONE_CLICK_CONTINUE_IF_UNCONVERGED = False


def run_one_click() -> None:
    configure_plot_style()

    # ------------------------------------------------------------------
    # Part I: exact single-emitter minimal saturation model.
    # ------------------------------------------------------------------
    bosonic, tls = benchmark_profile(ONE_CLICK_PROFILE)
    bosonic.validate()
    tls.validate()
    derived = derive_gain(bosonic, tls)

    numerics = Numerics(
        t_end=ONE_CLICK_T_END,
        convergence_points=ONE_CLICK_CONVERGENCE_POINTS,
        final_points=ONE_CLICK_FINAL_POINTS,
        target_occupation=ONE_CLICK_TARGET_OCCUPATION,
        early_fraction_n_sat=0.20,
        required_consecutive_passes=(
            ONE_CLICK_REQUIRED_CONSECUTIVE_PASSES
        ),
    )
    numerics.validate()
    output = OutputPaths.create(ONE_CLICK_OUTPUT_DIRECTORY)

    exact_path = output.data / "saturation_exact_timeseries.csv"
    linear_path = output.data / "saturation_linear_reference.csv"
    reuse_exact = (
        ONE_CLICK_REUSE_EXISTING_EXACT
        and exact_path.exists()
        and linear_path.exists()
    )

    print()
    print("=" * 76)
    print("NONLINEAR SATURABLE-GAIN VALIDATION")
    print("=" * 76)
    print("Part I: exact single-emitter minimal saturation model")
    print(f"Output directory: {output.root}")
    print(f"Reuse existing exact data: {reuse_exact}")
    print()

    save_json(
        {
            "units": unit_metadata(),
            "profile": ONE_CLICK_PROFILE,
            "bosonic": asdict(bosonic),
            "bosonic_dimensionless": normalized_bosonic_parameters(bosonic),
            "tls": asdict(tls),
            "tls_dimensionless": normalized_tls_parameters(tls),
            "derived": asdict(derived),
            "numerics": asdict(numerics),
            "cutoffs": ONE_CLICK_CUTOFFS,
            "reuse_existing_exact": ONE_CLICK_REUSE_EXISTING_EXACT,
            "ensemble_counts": ONE_CLICK_ENSEMBLE_COUNTS,
            "ensemble_target": ONE_CLICK_ENSEMBLE_TARGET,
        },
        output.summaries / "saturation_run_configuration.json",
    )

    if reuse_exact:
        exact_final = pd.read_csv(exact_path)
        linear_final = pd.read_csv(linear_path)
        print("Loaded existing cutoff-converged single-emitter CSV data.")
    else:
        convergence_summary, selected_cutoff = run_cutoff_convergence(
            bosonic=bosonic,
            tls=tls,
            derived=derived,
            numerics=numerics,
            cutoffs=ONE_CLICK_CUTOFFS,
            output=output,
            progress_bar=ONE_CLICK_PROGRESS_BAR,
            stop_after_convergence=(
                ONE_CLICK_STOP_AFTER_CONVERGENCE
            ),
        )
        plot_cutoff_summary(convergence_summary, output)

        if selected_cutoff is None:
            message = (
                "No cutoff passed all convergence criteria.  Extend "
                "ONE_CLICK_CUTOFFS and run again."
            )
            if not ONE_CLICK_CONTINUE_IF_UNCONVERGED:
                raise RuntimeError(message)
            warnings.warn(message)
            selected_cutoff = ONE_CLICK_CUTOFFS[-1]

        print(f"Selected production cutoff: {selected_cutoff}")
        final_summary, exact_final, linear_final = run_final(
            bosonic=bosonic,
            tls=tls,
            derived=derived,
            numerics=numerics,
            cutoff=selected_cutoff,
            output=output,
            progress_bar=ONE_CLICK_PROGRESS_BAR,
        )
        if not final_summary["steady_state_pass"]:
            warnings.warn(
                "The exact single-emitter calculation has not reached a "
                "sufficiently stationary tail."
            )
        if not final_summary["edge_population_pass"]:
            warnings.warn(
                "The exact single-emitter calculation has excessive Fock-edge "
                "population."
            )

    # ------------------------------------------------------------------
    # Part II: 1--100 emitter capacity scaling at the revised main benchmark.
    # ------------------------------------------------------------------
    paper_bosonic, paper_tls = benchmark_profile("paper")
    paper_derived = derive_gain(paper_bosonic, paper_tls)
    print()
    print("Part II: symmetric pumped-emitter ensemble proof of principle")
    print(
        f"  theta = {paper_bosonic.theta:.8g}, "
        f"Gamma_col/kappa0 = "
        f"{paper_bosonic.Gamma_col / REFERENCE_RATE:.8g}"
    )
    print(
        f"  matched small-signal margin/kappa0 = "
        f"{paper_derived.tls_linear_margin / REFERENCE_RATE:.8g}"
    )
    print(
        "  scaling: g_N = g_1/sqrt(N), so the small-signal spectrum "
        "is fixed while the finite gain reservoir grows with N."
    )

    ensemble_results, ensemble_summary = run_emitter_ensemble_proof(
        bosonic=paper_bosonic,
        tls=paper_tls,
        derived=paper_derived,
        emitter_counts=ONE_CLICK_ENSEMBLE_COUNTS,
        t_end=ONE_CLICK_ENSEMBLE_T_END,
        points=ONE_CLICK_ENSEMBLE_POINTS,
        target_occupation=ONE_CLICK_ENSEMBLE_TARGET,
        output=output,
    )
    plot_emitter_ensemble_proof(
        ensemble_results,
        ensemble_summary,
        ONE_CLICK_ENSEMBLE_TARGET,
        output,
    )
    plot_revised_end_matter_figure(
        exact_final,
        linear_final,
        ensemble_results,
        ensemble_summary,
        ONE_CLICK_ENSEMBLE_TARGET,
        output,
    )

    print()
    print(ensemble_summary.to_string(index=False))
    print()
    print("=" * 76)
    print("RUN FINISHED")
    print("=" * 76)
    print(f"Figures:   {output.figures}")
    print(f"CSV data:  {output.data}")
    print(f"Summaries: {output.summaries}")
    print("=" * 76)


# Clicking Run in Jupyter/Spyder or executing without command-line arguments
# starts the complete one-click calculation. Explicit CLI arguments retain the
# diagnostic subcommands such as `derive`, `convergence`, and `final`.
if __name__ == "__main__":
    if "ipykernel" in sys.modules or len(sys.argv) == 1:
        run_one_click()
    else:
        main()