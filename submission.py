"""Causal spectral EEG BCI classifier for Neural Interfaces 2026 Track 02.

Science: fixed 4-s, 41-channel, 120-Hz EEG epochs supplied by the
competition. Extract non-learned mu/beta log-bandpower and within-epoch
relative power; infer a trained multinomial linear decision rule.

No hidden-label lookup, subject IDs, external network, or test-set fitting.
All model coefficients are shipped explicitly in weights.npz.
"""
from pathlib import Path
import numpy as np
from benchmark_utils.base_solver import CompetSolver

BANDS_HZ = ((4., 8.), (8., 12.), (12., 16.),
            (16., 22.), (22., 30.), (30., 40.))


def extract_bci_features(signals, sfreq=120.0):
    """Map a batch of EEG epochs (B, channels, samples) to stable features.

    The transform is deterministic and contains no fitted statistics.
    Each epoch is baseline-centered in time; spectral power is aggregated
    over neurophysiological bands with a Hann taper. We include both
    absolute log power and channel-normalized log power, which mitigates
    uniform electrode gain differences between participants.
    """
    x = np.asarray(signals, dtype=np.float64)
    if x.ndim != 3 or x.shape[-1] < 64:
        raise ValueError(f"Expected EEG (B,C,T>=64); received {x.shape}")
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = np.clip(x, -500.0, 500.0)
    x = x - x.mean(axis=-1, keepdims=True)
    window = np.hanning(x.shape[-1])
    spectrum = np.fft.rfft(x * window, axis=-1)
    power = (spectrum.real ** 2 + spectrum.imag ** 2)
    freqs = np.fft.rfftfreq(x.shape[-1], 1.0 / sfreq)
    logpower = np.stack([
        np.log1p(power[..., (freqs >= lo) & (freqs < hi)].mean(axis=-1))
        for lo, hi in BANDS_HZ
    ], axis=-1)
    # Spatial log-ratios to channel-wide band references reduce
    # subject-specific amplifier and impedance effects.
    relative = logpower - logpower.mean(axis=-2, keepdims=True)
    spectral = np.concatenate(
        (logpower.reshape(x.shape[0], -1),
         relative.reshape(x.shape[0], -1)), axis=1
    )
    return spectral.astype(np.float32, copy=False)


class SpectralBCIModel:
    """Fixed learned linear classifier with NumPy inference and CPU support."""

    def __init__(self, weights, expected_channels, expected_samples, sfreq):
        self.center = np.asarray(weights["center"], np.float64)
        self.scale = np.asarray(weights["scale"], np.float64)
        self.coef = np.asarray(weights["coef"], np.float64)
        self.intercept = np.asarray(weights["intercept"], np.float64)
        self.classes = np.asarray(weights["classes"], np.int64)
        self.expected_channels = int(expected_channels)
        self.expected_samples = int(expected_samples)
        self.sfreq = float(sfreq)
        if len(self.center) != self.coef.shape[1]:
            raise ValueError("Feature dimension does not match trained weights")
        if not np.isfinite(self.coef).all():
            raise ValueError("Weights are not finite")

    def predict(self, X):
        """Return one integer class ID for each independently observed epoch."""
        if hasattr(X, "detach"):
            X = X.detach().cpu().numpy()
        X = np.asarray(X)
        if (X.ndim != 3 or X.shape[1] != self.expected_channels
                or X.shape[2] != self.expected_samples):
            raise ValueError(
                f"Incorrect epoch shape {X.shape}; expected "
                f"(B,{self.expected_channels},{self.expected_samples})"
            )
        f = extract_bci_features(X, sfreq=self.sfreq).astype(np.float64)
        z = (f - self.center) / np.maximum(self.scale, 1e-5)
        logits = z @ self.coef.T + self.intercept
        return self.classes[np.argmax(logits, axis=1)].astype(np.int64)


class Solver(CompetSolver):
    """Benchopt model adapter matching the official inference-only contract."""

    name = "Agnuxo-Spectral-BCI"

    def load_model(self, meta):
        """Load frozen coefficients from the submitted ZIP, never retrain."""
        path = Path(meta["submission_dir"]) / "weights.npz"
        if not path.is_file():
            raise FileNotFoundError(
                "Trained weights.npz missing from BCI submission directory"
            )
        with np.load(path, allow_pickle=False) as z:
            weights = {key: z[key].copy() for key in z.files}
        return SpectralBCIModel(
            weights,
            expected_channels=meta["n_chans"],
            expected_samples=meta["n_times"],
            sfreq=meta["sfreq"],
        )
