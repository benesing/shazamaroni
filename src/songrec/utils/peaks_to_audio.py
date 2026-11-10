#!/usr/bin/env python3
# -*- encoding: Utf-8 -*-
from argparse import ArgumentParser
from csv import DictReader
from math import exp, log, sqrt
import sys

from numpy import arange, clip, hanning, pi, round as npround, sin, zeros
from numpy import exp as npexp
from pydub import AudioSegment

from songrec.fingerprinting.algorithm import SignatureGenerator
from songrec.fingerprinting.signature_format import DecodedMessage
from songrec.utils.fingerprint_to_csv import read_uri

"""
    Synthesize audio whose Shazam peak analysis reproduces a given set of
    peaks (from a peak CSV or a .sig fingerprint).

    Each target peak becomes one Gaussian-enveloped sine burst, centered on
    the FFT window of its fft_pass_number. The audio is then analyzed with
    the real SignatureGenerator, and each burst's timing, frequency and
    amplitude are corrected until the detected peaks match exactly
    (same band, fft_pass_number, corrected_peak_frequency_bin and
    peak_magnitude).

    Sample usage:
        uv run songrec-peaks-to-audio minimal/my_sharona_012_min.csv \
            -o minimal/my_sharona_012_min.wav
"""

SAMPLE_RATE = 16000
CHUNK = 128
WINDOW = 2048
HZ_PER_BIN64 = SAMPLE_RATE / 2 / 1024 / 64
MAG_SCALE = 1477.3
MAG_OFFSET = 6144


class Target:
    def __init__(self, band, fft_pass_number, bin64, magnitude):

        self.band = band
        self.fft_pass_number = fft_pass_number
        self.bin64 = bin64
        self.magnitude = magnitude

    def key(self):

        return (
            self.band,
            self.fft_pass_number,
            self.bin64,
            self.magnitude,
        )


class Burst:
    def __init__(self, target, sigma):

        self.target = target
        self.sigma = sigma

        # The FFT for pass N covers samples [128 (N + 1) - 2048, 128 (N + 1))
        self.center = CHUNK * (target.fft_pass_number + 1) - WINDOW // 2

        # Aim for the middle of the integer bin / magnitude, since the
        # detector truncates both with int()
        self.bin64 = target.bin64 + 0.5

        power = exp((target.magnitude + 0.5 - MAG_OFFSET) / MAG_SCALE)
        n = arange(WINDOW) - WINDOW // 2
        window_gain = (hanning(WINDOW + 2)[1:-1] * gaussian(n, sigma)).sum()
        self.amplitude = 2 * sqrt(power * (1 << 17)) / window_gain

    def render(self, out):

        half = int(4 * self.sigma)
        n = arange(-half, half + 1)
        start = self.center - half

        lo = max(0, start)
        hi = min(len(out), start + len(n))
        if lo >= hi:
            return

        n = n[lo - start : hi - start]
        freq = self.bin64 * HZ_PER_BIN64
        out[lo:hi] += (
            self.amplitude
            * gaussian(n, self.sigma)
            * sin(2 * pi * freq * (self.center + n) / SAMPLE_RATE)
        )


def gaussian(n, sigma):

    return npexp(-0.5 * (n / sigma) ** 2)


def read_targets(source):

    if source.endswith('.csv'):
        with open(source) as f:
            return [
                Target(
                    int(row['band']),
                    int(row['fft_pass_number']),
                    int(row['corrected_peak_frequency_bin']),
                    int(row['peak_magnitude']),
                )
                for row in DictReader(f)
            ], None

    message = DecodedMessage.decode_from_uri(read_uri(source))
    targets = [
        Target(
            int(band),
            peak.fft_pass_number,
            peak.corrected_peak_frequency_bin,
            peak.peak_magnitude,
        )
        for band, peaks in message.frequency_band_to_sound_peaks.items()
        for peak in peaks
    ]
    return targets, message.number_samples / message.sample_rate_hz


def render(bursts, num_samples):

    out = zeros(num_samples)
    for burst in bursts:
        burst.render(out)

    peak = abs(out).max()
    if peak > 32767:
        print(
            'Warning: clipping (peak %d), lower the amplitudes' % peak,
            file=sys.stderr,
        )

    return clip(npround(out), -32768, 32767).astype('int16')


def analyze(samples):

    generator = SignatureGenerator()
    generator.feed_input([int(s) for s in samples])
    generator.MAX_TIME_SECONDS = float('inf')
    signature = generator.get_next_signature()

    return [
        Target(
            int(band),
            peak.fft_pass_number,
            peak.corrected_peak_frequency_bin,
            peak.peak_magnitude,
        )
        for band, peaks in signature.frequency_band_to_sound_peaks.items()
        for peak in peaks
    ]


def nearest(target, detected):

    candidates = [
        d
        for d in detected
        if d.band == target.band
        and abs(d.fft_pass_number - target.fft_pass_number) <= 4
        and abs(d.bin64 - target.bin64) <= 3 * 64
    ]
    if not candidates:
        return None

    return min(
        candidates,
        key=lambda d: (
            abs(d.fft_pass_number - target.fft_pass_number),
            abs(d.bin64 - target.bin64),
        ),
    )


def correct(burst, found):

    target = burst.target

    burst.center -= CHUNK * (found.fft_pass_number - target.fft_pass_number)
    burst.bin64 -= found.bin64 - target.bin64

    # magnitude = 1477.3 * ln(power), and power scales with amplitude^2
    burst.amplitude *= exp(
        -(found.magnitude - target.magnitude) / MAG_SCALE / 2
    )


def main():
    args = ArgumentParser(
        description='Synthesize audio whose Shazam peak analysis '
        + 'reproduces the peaks in a peak CSV or .sig fingerprint.'
    )
    args.add_argument(
        'peaks',
        help='A peak CSV (from songrec-fingerprint-to-csv) or a .sig file.',
    )
    args.add_argument(
        '-o', '--output', required=True, help='Audio file to write.'
    )
    args.add_argument(
        '-d',
        '--duration',
        type=float,
        default=None,
        help='Length in seconds (default: the .sig length, or 12 for CSV).',
    )
    args.add_argument(
        '--sigma',
        type=float,
        default=256.0,
        help='Gaussian envelope width in samples (default: 256).',
    )
    args.add_argument(
        '--iterations',
        type=int,
        default=20,
        help='Maximum correction passes (default: 20).',
    )
    args = args.parse_args()

    targets, sig_duration = read_targets(args.peaks)
    duration = args.duration or sig_duration or 12.0
    num_samples = int(duration * SAMPLE_RATE)

    bursts = [Burst(t, args.sigma) for t in targets]
    target_keys = {t.key() for t in targets}

    for iteration in range(1, args.iterations + 1):
        samples = render(bursts, num_samples)
        detected = analyze(samples)
        detected_keys = {d.key() for d in detected}

        exact = 0
        missing = 0
        for burst in bursts:
            if burst.target.key() in detected_keys:
                exact += 1
                continue

            found = nearest(burst.target, detected)
            if found is None:
                missing += 1
            else:
                correct(burst, found)

        extra = len(detected_keys - target_keys)
        print(
            'Pass %d: %d/%d exact, %d missing, %d extra peaks'
            % (iteration, exact, len(targets), missing, extra),
            file=sys.stderr,
        )

        if exact == len(targets):
            break

    AudioSegment(
        data=samples.tobytes(),
        sample_width=2,
        frame_rate=SAMPLE_RATE,
        channels=1,
    ).export(args.output, format=args.output.rsplit('.', 1)[-1])

    for burst in bursts:
        if burst.target.key() not in detected_keys:
            t = burst.target
            print(
                '  not reproduced: band %d pass %d bin %d mag %d'
                % (t.band, t.fft_pass_number, t.bin64, t.magnitude),
                file=sys.stderr,
            )
    for d in detected:
        if d.key() not in target_keys:
            print(
                '  extra: band %d pass %d bin %d (%.1f Hz) mag %d'
                % (
                    d.band,
                    d.fft_pass_number,
                    d.bin64,
                    d.bin64 * HZ_PER_BIN64,
                    d.magnitude,
                ),
                file=sys.stderr,
            )

    print(
        'Wrote %s (%.2fs, %d Hz)' % (args.output, duration, SAMPLE_RATE),
        file=sys.stderr,
    )


if __name__ == '__main__':
    main()
