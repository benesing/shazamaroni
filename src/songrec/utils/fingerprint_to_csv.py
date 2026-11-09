#!/usr/bin/env python3
# -*- encoding: Utf-8 -*-
from argparse import ArgumentParser
from csv import writer
from os.path import isfile
import sys

from songrec.fingerprinting.signature_format import DecodedMessage

"""
    Flatten a Shazam signature into one CSV row per constellation peak.

    Sample usage:
        uv run songrec-fingerprint-to-csv sigs/my_sharona_bbr60.sig \
            -o sigs/my_sharona_bbr60.csv
"""

BAND_HZ = {
    0: '250-520',
    1: '520-1450',
    2: '1450-3500',
    3: '3500-5500',
}

COLUMNS = [
    'band',
    'band_hz',
    'fft_pass_number',
    'seconds',
    'corrected_peak_frequency_bin',
    'frequency_hz',
    'peak_magnitude',
    'amplitude_pcm',
]


def read_uri(source: str) -> str:

    if isfile(source):
        with open(source) as f:
            return f.read().strip()

    return source.strip()


def peak_rows(message: DecodedMessage):

    for band, peaks in sorted(message.frequency_band_to_sound_peaks.items()):
        for peak in peaks:
            yield [
                int(band),
                BAND_HZ.get(int(band), ''),
                peak.fft_pass_number,
                round(peak.get_seconds(), 6),
                peak.corrected_peak_frequency_bin,
                round(peak.get_frequency_hz(), 3),
                peak.peak_magnitude,
                round(peak.get_amplitude_pcm(), 6),
            ]


def main():
    args = ArgumentParser(
        description='Convert a Shazam fingerprint (data URI or .sig file) '
        + 'into a CSV with one row per constellation peak.'
    )
    args.add_argument(
        'fingerprint',
        help='A .sig file containing a data URI, or the data URI itself.',
    )
    args.add_argument(
        '-o', '--output', help='CSV file to write (default: stdout).'
    )
    args.add_argument(
        '--sort',
        choices=['band', 'time'],
        default='band',
        help='Row order: grouped by band (default) or by time across bands.',
    )
    args = args.parse_args()

    message = DecodedMessage.decode_from_uri(read_uri(args.fingerprint))

    rows = list(peak_rows(message))
    if args.sort == 'time':
        rows.sort(key=lambda r: (r[2], r[0], r[4]))

    out = open(args.output, 'w', newline='') if args.output else sys.stdout
    try:
        w = writer(out)
        w.writerow(COLUMNS)
        w.writerows(rows)
    finally:
        if args.output:
            out.close()

    print(
        '%d peaks, %d samples (%.2fs) at %d Hz'
        % (
            len(rows),
            message.number_samples,
            message.number_samples / message.sample_rate_hz,
            message.sample_rate_hz,
        ),
        file=sys.stderr,
    )


if __name__ == '__main__':
    main()
