#!/usr/bin/env python3
"""The native band filters, mirrored in Python (scope.md 2026-10-08 "sharper
band filters"). EasyEffects' Shader Bands plugin (shader_bands.cpp,
design_butterworth_bandpass) and this file implement the SAME design from the
SAME config, so the renderer knows each band's exact delay and frequency
response:
  * the renderer shifts every band by its group delay (delay_frames), so
    all bands line up with what you hear again;
  * the GPU cleanup deconvolves with the exact per-band leak (leak_matrix).

Design: one Butterworth bandpass per band, -3 dB at +-1/2 semitone around the
band centre 27.5 * 2^(b/12) Hz, order from [filters] orders (one value per
octave, 10 octaves) in ~/.config/easyeffects/shader_bands.conf. Analog
Butterworth lowpass prototype -> lowpass-to-bandpass -> bilinear transform with
prewarped edges -> N biquads, each with zeros at z = +1 and z = -1, normalised
to 0 dB at the centre.
"""
import configparser
import os

import numpy as np

N_BANDS = 120
FS_DEFAULT = 48000.0
CONF = os.path.expanduser("~/.config/easyeffects/shader_bands.conf")
DEFAULT_ORDERS = (2, 2, 2, 3, 3, 4, 4, 4, 4, 4)
R = 2.0 ** (1.0 / 12.0)


def read_orders(path=CONF):
    cp = configparser.ConfigParser()
    try:
        cp.read(path)
        raw = cp.get("filters", "orders", fallback=None)
    except configparser.Error:
        raw = None
    if not raw:
        return DEFAULT_ORDERS
    vals = [int(v) for v in raw.replace(" ", "").split(",") if v]
    vals = [min(max(v, 1), 4) for v in vals]
    return tuple((vals + [vals[-1]] * 10)[:10])


def band_order(b, orders):
    return orders[min(b // 12, len(orders) - 1)]


def centre_hz(b):
    return 27.5 * 2.0 ** (b / 12.0)


def design(b, order, fs=FS_DEFAULT):
    """(order, 6) biquads [b0, b1, b2, 1, a1, a2] for band b, or None above Nyquist."""
    f0 = centre_hz(b)
    flo, fhi = f0 / np.sqrt(R), f0 * np.sqrt(R)
    if fhi >= 0.49 * fs:
        return None
    wlo = 2 * fs * np.tan(np.pi * flo / fs)       # prewarped analog edges
    whi = 2 * fs * np.tan(np.pi * fhi / fs)
    w0 = np.sqrt(wlo * whi)
    bw = whi - wlo
    sos = []
    for k in range(order):                         # all N prototype poles (left half plane)
        theta = np.pi * (2 * k + order + 1) / (2 * order)
        p = np.exp(1j * theta)
        # lowpass -> bandpass: s^2 - p*bw*s + w0^2 = 0; of the 2N bandpass poles keep the N with Im > 0
        disc = np.sqrt((p * bw) ** 2 - 4 * w0 ** 2 + 0j)
        for s in ((p * bw + disc) / 2, (p * bw - disc) / 2):
            if s.imag <= 0:
                continue
            z = (2 * fs + s) / (2 * fs - s)        # bilinear
            sos.append([1.0, 0.0, -1.0, 1.0, -2 * z.real, abs(z) ** 2])
    sos = np.array(sos[:order])
    # normalise to 0 dB at the band centre
    w = 2 * np.pi * f0 / fs
    zc = np.exp(-1j * w)
    h = np.prod([(s[0] + s[1] * zc + s[2] * zc ** 2) / (s[3] + s[4] * zc + s[5] * zc ** 2) for s in sos])
    sos[0, :3] /= abs(h)
    return sos


def response_pow(sos, f, fs=FS_DEFAULT):
    zc = np.exp(-1j * 2 * np.pi * np.asarray(f, float) / fs)
    h = np.ones_like(zc)
    for s in sos:
        h = h * (s[0] + s[1] * zc + s[2] * zc ** 2) / (s[3] + s[4] * zc + s[5] * zc ** 2)
    return np.abs(h) ** 2


def group_delay_s(sos, f, fs=FS_DEFAULT, df=0.01):
    """Group delay at f (seconds), from the phase slope."""
    def phase(ff):
        zc = np.exp(-1j * 2 * np.pi * ff / fs)
        h = 1.0 + 0j
        for s in sos:
            h *= (s[0] + s[1] * zc + s[2] * zc ** 2) / (s[3] + s[4] * zc + s[5] * zc ** 2)
        return np.angle(h)
    dphi = np.angle(np.exp(1j * (phase(f + df) - phase(f - df))))
    return -dphi / (2 * np.pi * 2 * df)


def all_designs(orders=None, fs=FS_DEFAULT):
    orders = orders or read_orders()
    return [design(b, band_order(b, orders), fs) for b in range(N_BANDS)]


def delays_s(orders=None, fs=FS_DEFAULT):
    return np.array([0.0 if d is None else group_delay_s(d, centre_hz(b), fs)
                     for b, d in enumerate(all_designs(orders, fs))])


def delay_frames(rate_hz, orders=None, fs=FS_DEFAULT):
    """Per-band shift in published frames: band b's value for the frame you HEAR
    at t is the one published delay_b later."""
    return np.round(delays_s(orders, fs) * rate_hz).astype(int)


def leak_matrix(n_used=108, k=12, orders=None, fs=FS_DEFAULT):
    """(n_used, 2k+1): power band b passes for a tone at note b+j (j = -k..k)."""
    ds = all_designs(orders, fs)
    m = np.zeros((n_used, 2 * k + 1))
    for b in range(n_used):
        if ds[b] is None:
            continue
        m[b] = response_pow(ds[b], [centre_hz(b + j) for j in range(-k, k + 1)], fs)
    return m


if __name__ == "__main__":
    from scipy.signal import butter, sosfreqz
    orders = read_orders()
    print("orders per octave:", orders)
    worst = 0.0
    for b in (4, 12, 24, 40, 57, 70, 90, 105):
        for n in (1, 2, 3, 4):
            ours = design(b, n)
            f0 = centre_hz(b)
            ref = butter(n, [f0 / np.sqrt(R), f0 * np.sqrt(R)], btype="bandpass", fs=FS_DEFAULT, output="sos")
            fs_ = f0 * 2.0 ** (np.linspace(-3, 3, 61) / 12)
            a = 10 * np.log10(response_pow(ours, fs_) + 1e-30)
            _, h = sosfreqz(ref, worN=fs_, fs=FS_DEFAULT)
            r = 20 * np.log10(np.abs(h) + 1e-30)
            worst = max(worst, np.max(np.abs(a - r)))
    print(f"max |ours - scipy.butter| over 8 bands x orders 1-4, +-3 semitones: {worst:.4f} dB")
    d = delays_s(orders)
    print("group delay ms, bands 0/12/24/36/48/60/72/84/96:", np.round(d[[0, 12, 24, 36, 48, 60, 72, 84, 96]] * 1000, 1))
