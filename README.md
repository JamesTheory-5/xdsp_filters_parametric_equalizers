# xdsp_filters_parametric_equalizers
```python
# xdsp_parametric_eq.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from xdsp_rbj_biquads import (
    rbj_biquad_design,
    biquad_block,
    BiquadState,
)


# ================================================================
# Band definition
# ================================================================

@dataclass
class EQBand:
    """
    One parametric EQ band using RBJ biquad modes.

    mode:
        "lowpass", "highpass",
        "bandpass", "notch",
        "peak",
        "lowshelf", "highshelf"
        (plus any extra modes you add that rbj_biquad_design supports)

    gain_db:
        Used by peak / shelf / some modes; ignored by pure HPF/LPF/notch unless desired.

    slope:
        For shelving filters (RBJ S parameter). Default = 1.0 (max monotonic).
    """
    mode: str
    f0: float
    Q: float = 0.707
    gain_db: float = 0.0
    slope: float = 1.0
    enabled: bool = True

    def design(self, fs: float) -> BiquadState:
        coeffs = rbj_biquad_design(
            mode=self.mode,
            f0=self.f0,
            fs=fs,
            Q=self.Q,
            gain_db=self.gain_db,
            slope=self.slope,
        )
        # Initialize history to zero; include metadata for debugging/introspection.
        return {
            **coeffs,
            "mode": self.mode,
            "fs": fs,
            "f0": self.f0,
            "Q": self.Q,
            "gain_db": self.gain_db,
            "slope": self.slope,
            "x1": 0.0,
            "x2": 0.0,
            "y1": 0.0,
            "y2": 0.0,
        }


# ================================================================
# Parametric EQ: cascade of RBJ biquads
# ================================================================

class ParametricEQ:
    """
    Multi-band parametric EQ built from cascaded RBJ-style biquads.

    Usage:
        bands = [
            EQBand("highpass", f0=40.0,  Q=0.707),
            EQBand("peak",     f0=200.0, Q=2.0,  gain_db=3.0),
            EQBand("peak",     f0=2000., Q=1.0,  gain_db=-4.0),
            EQBand("highshelf",f0=8000., Q=0.707, gain_db=2.0),
        ]
        eq = ParametricEQ(fs=48000, bands=bands)

        y = eq.process_block(x)

    Implementation:
        Filters are applied in series (Direct Form 1 style per band).
    """

    def __init__(self, fs: float, bands: Optional[List[EQBand]] = None):
        if fs <= 0:
            raise ValueError("Sampling rate fs must be positive.")

        self.fs = float(fs)
        self.bands: List[EQBand] = bands[:] if bands is not None else []
        self.states: List[BiquadState] = []
        self._design_all()

    # ------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------

    def _design_all(self):
        """(Re)design all biquad states from current band configs."""
        self.states = []
        for band in self.bands:
            if band.enabled:
                self.states.append(band.design(self.fs))

    def _redesign_band(self, index: int):
        """Redesign a single band (keeping order)."""
        band = self.bands[index]
        if band.enabled:
            self.states[index] = band.design(self.fs)
        else:
            # Disabled band: zero-gain passthrough not needed in cascade; easiest is rebuild all.
            self._design_all()

    # ------------------------------------------------------------
    # Public API: band management
    # ------------------------------------------------------------

    def add_band(self, band: EQBand):
        """Append a new band at the end of the cascade."""
        self.bands.append(band)
        if band.enabled:
            self.states.append(band.design(self.fs))

    def insert_band(self, index: int, band: EQBand):
        """Insert a band at a given index in the cascade."""
        self.bands.insert(index, band)
        # easiest: rebuild to keep states aligned to bands
        self._design_all()

    def remove_band(self, index: int):
        """Remove a band by index."""
        del self.bands[index]
        self._design_all()

    def set_band(
        self,
        index: int,
        *,
        mode: Optional[str] = None,
        f0: Optional[float] = None,
        Q: Optional[float] = None,
        gain_db: Optional[float] = None,
        slope: Optional[float] = None,
        enabled: Optional[bool] = None,
    ):
        """
        Update band parameters (in-place) and redesign that band.
        Only provided keywords are modified.
        """
        band = self.bands[index]

        if mode is not None:
            band.mode = mode
        if f0 is not None:
            band.f0 = f0
        if Q is not None:
            band.Q = Q
        if gain_db is not None:
            band.gain_db = gain_db
        if slope is not None:
            band.slope = slope
        if enabled is not None:
            band.enabled = enabled

        self._design_all()

    def clear_bands(self):
        """Remove all bands."""
        self.bands.clear()
        self.states.clear()

    # ------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------

    def reset_state(self):
        """Reset all filter histories to zero (keeping current coefficients)."""
        for st in self.states:
            st["x1"] = st["x2"] = 0.0
            st["y1"] = st["y2"] = 0.0

    def process_block(self, x: np.ndarray) -> np.ndarray:
        """
        Process a block of samples through all enabled bands (cascade).
        """
        # Work on a local variable so we don't mutate caller's array.
        y = x.astype(np.float64, copy=True)

        for i, st in enumerate(self.states):
            y, self.states[i] = biquad_block(st, y)

        return y

    def process_tick(self, x: float) -> float:
        """
        Process a single sample through the cascade.
        """
        y = x
        for st in self.states:
            # inline tick to avoid import cycle:
            b0, b1, b2 = st["b0"], st["b1"], st["b2"]
            a1, a2 = st["a1"], st["a2"]
            x1, x2 = st["x1"], st["x2"]
            y1, y2 = st["y1"], st["y2"]

            yn = b0 * y + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2

            st["x2"], st["x1"] = x1, y
            st["y2"], st["y1"] = y1, yn

            y = yn

        return y

    # ------------------------------------------------------------
    # Analysis: combined frequency response
    # ------------------------------------------------------------

    def frequency_response(
        self,
        n_points: int = 2048,
        f_min: float = 20.0,
        f_max: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the magnitude response (in dB) of the entire EQ cascade.

        Returns:
            f: frequency axis (Hz)
            mag_db: magnitude response (dB)
        """
        if f_max is None:
            f_max = self.fs / 2.0

        # Frequency grid (log-spaced in [f_min, f_max])
        f = np.logspace(np.log10(f_min), np.log10(f_max), n_points)
        w = 2.0 * np.pi * f / self.fs
        z = np.exp(1j * w)

        H_total = np.ones_like(z, dtype=np.complex128)

        for st in self.states:
            b0, b1, b2 = st["b0"], st["b1"], st["b2"]
            a1, a2 = st["a1"], st["a2"]
            # H(z) = (b0 + b1 z^-1 + b2 z^-2) / (1 + a1 z^-1 + a2 z^-2)
            z_inv = 1.0 / z
            num = b0 + b1 * z_inv + b2 * (z_inv**2)
            den = 1.0 + a1 * z_inv + a2 * (z_inv**2)
            H_total *= num / den

        mag_db = 20.0 * np.log10(np.maximum(np.abs(H_total), 1e-9))
        return f, mag_db

    def plot_response(
        self,
        n_points: int = 2048,
        f_min: float = 20.0,
        f_max: Optional[float] = None,
        show_individual: bool = False,
    ):
        """
        Plot the overall frequency response of the EQ.
        Optionally plot each band’s individual response for inspection.
        """
        if f_max is None:
            f_max = self.fs / 2.0

        f, mag_db = self.frequency_response(n_points, f_min, f_max)

        plt.figure(figsize=(9, 4))
        plt.plot(f, mag_db, label="Total EQ", linewidth=2.0)

        if show_individual:
            # Plot per-band responses with light lines
            w = 2.0 * np.pi * f / self.fs
            z = np.exp(1j * w)
            for band, st in zip(self.bands, self.states):
                if not band.enabled:
                    continue
                b0, b1, b2 = st["b0"], st["b1"], st["b2"]
                a1, a2 = st["a1"], st["a2"]
                z_inv = 1.0 / z
                H = (b0 + b1 * z_inv + b2 * (z_inv**2)) / (
                    1.0 + a1 * z_inv + a2 * (z_inv**2)
                )
                mag = 20.0 * np.log10(np.maximum(np.abs(H), 1e-9))
                plt.plot(f, mag, linestyle="--", alpha=0.4, label=f"{band.mode} @ {band.f0:g} Hz")

        plt.xscale("log")
        plt.xlim(f_min, f_max)
        plt.ylim(-24, 24)
        plt.grid(True, which="both", alpha=0.25)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude (dB)")
        plt.title("Parametric EQ Frequency Response")
        plt.legend(loc="best", fontsize=8)
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------

    def describe(self) -> List[Dict]:
        """
        Return a list of bands as plain dicts for easy printing / debugging.
        """
        return [asdict(band) for band in self.bands]

import numpy as np
from xdsp_parametric_eq import ParametricEQ, EQBand

fs = 48000

bands = [
    EQBand("highpass",  f0=30.0,   Q=0.707),
    EQBand("peak",      f0=80.0,   Q=1.0,   gain_db=3.0),
    EQBand("peak",      f0=400.0,  Q=1.0,   gain_db=-2.5),
    EQBand("peak",      f0=2500.0, Q=1.2,   gain_db=4.0),
    EQBand("highshelf", f0=8000.0, Q=0.7,   gain_db=3.0),
]

eq = ParametricEQ(fs=fs, bands=bands)

# Test on white noise
N = fs
x = np.random.randn(N)
y = eq.process_block(x)

# Visualize response
eq.plot_response(show_individual=True)

```
