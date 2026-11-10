#!/usr/bin/env python3
# -*- encoding: Utf-8 -*-
from argparse import ArgumentParser
from csv import writer
import sys

from pydub import AudioSegment

from songrec.fingerprinting.algorithm import SignatureGenerator
from songrec.utils.fingerprint_to_csv import COLUMNS, peak_rows

"""
    Run the Shazam peak detector over a whole audio file (or a range of
    it) in a single pass, and save every constellation peak as CSV, in
    the same format as songrec-fingerprint-to-csv.

    fft_pass_number / seconds are absolute positions in the file, so they
    line up with the "offset" returned in a Shazam match:
        reference_seconds = query_seconds + match_offset

    Sample usage:
        uv run songrec-audio-file-to-peaks audio/01_my_sharona.wav \
            -o sigs/my_sharona_full.csv
"""

SAMPLE_RATE = 16000
CHUNK = 128


def main():
    args = ArgumentParser(
        description='Save every constellation peak found in an audio '
        + 'file as CSV, one row per peak.'
    )
    args.add_argument('input_file', help='The .WAV or .MP3 file to scan.')
    args.add_argument(
        '-o', '--output', help='CSV file to write (default: stdout).'
    )
    args.add_argument(
        '-s',
        '--start',
        type=float,
        default=0.0,
        help='Offset in seconds to start scanning from (default: 0).',
    )
    args.add_argument(
        '-d',
        '--duration',
        type=float,
        default=None,
        help='Seconds to scan (default: to the end of the file).',
    )
    args.add_argument(
        '--sort',
        choices=['band', 'time'],
        default='band',
        help='Row order: grouped by band (default) or by time across bands.',
    )
    args = args.parse_args()

    audio = AudioSegment.from_file(args.input_file)
    audio = audio.set_sample_width(2)
    audio = audio.set_frame_rate(SAMPLE_RATE)
    audio = audio.set_channels(1)

    if args.start < 0 or args.start >= audio.duration_seconds:
        sys.stderr.write(
            'Error: --start must be between 0 and %g seconds\n'
            % audio.duration_seconds
        )
        sys.exit(1)

    # Start on a 128-sample boundary so fft_pass_number stays an exact
    # absolute index into the file.
    start_sample = int(SAMPLE_RATE * args.start) // CHUNK * CHUNK

    samples = audio.get_array_of_samples()
    end_sample = len(samples)
    if args.duration is not None:
        end_sample = min(
            end_sample, start_sample + int(SAMPLE_RATE * args.duration)
        )

    generator = SignatureGenerator()
    generator.feed_input(list(samples[start_sample:end_sample]))
    generator.MAX_TIME_SECONDS = float('inf')

    signature = generator.get_next_signature()

    # Shift peaks to absolute positions in the file.
    start_pass = start_sample // CHUNK
    for peaks in signature.frequency_band_to_sound_peaks.values():
        for peak in peaks:
            peak.fft_pass_number += start_pass

    rows = list(peak_rows(signature))
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
        '%d peaks over %.2fs (from %.2fs)'
        % (
            len(rows),
            signature.number_samples / SAMPLE_RATE,
            start_sample / SAMPLE_RATE,
        ),
        file=sys.stderr,
    )


if __name__ == '__main__':
    main()
