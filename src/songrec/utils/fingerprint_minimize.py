#!/usr/bin/env python3
# -*- encoding: Utf-8 -*-
from argparse import ArgumentParser
from csv import writer
from json import dump
from time import sleep, time
import sys

from songrec.fingerprinting.communication import recognize_song_from_signature
from songrec.fingerprinting.signature_format import DecodedMessage
from songrec.utils.fingerprint_to_csv import COLUMNS, peak_rows, read_uri

"""
    Find a (locally) smallest subset of a fingerprint's peaks that Shazam
    still recognizes as the same track, by re-querying with peaks removed.

    Phase 1: keep only the N strongest peaks, binary search on N.
    Phase 2: delta-debugging style removal of chunks of peaks (halves,
             quarters, ... down to single peaks), keeping every removal
             that still matches. The result is 1-minimal: removing any
             single remaining peak breaks the match.

    Every query is logged to <prefix>.log.json; the final peaks are saved
    as <prefix>.sig and <prefix>.csv.

    Sample usage:
        uv run songrec-fingerprint-minimize sigs/my_sharona_012.sig \
            -o minimal/my_sharona_012_min
"""


class Minimizer:
    def __init__(self, message, key, delay, max_queries, retries):

        self.message = message
        self.key = key
        self.delay = delay
        self.max_queries = max_queries
        self.retries = retries

        # Peak ids are (band, index within band in the original message)
        self.all_ids = [
            (band, i)
            for band, peaks in sorted(
                message.frequency_band_to_sound_peaks.items()
            )
            for i in range(len(peaks))
        ]

        self.cache = {}
        self.log = []
        self.last_query_time = 0.0

    def peak(self, peak_id):

        band, i = peak_id
        return self.message.frequency_band_to_sound_peaks[band][i]

    def build(self, ids) -> DecodedMessage:

        out = DecodedMessage()
        out.sample_rate_hz = self.message.sample_rate_hz
        out.number_samples = self.message.number_samples
        out.frequency_band_to_sound_peaks = {}

        for band, i in sorted(ids):
            out.frequency_band_to_sound_peaks.setdefault(band, []).append(
                self.message.frequency_band_to_sound_peaks[band][i]
            )

        return out

    def query(self, signature: DecodedMessage) -> dict:

        for attempt in range(self.retries + 1):
            wait = self.delay - (time() - self.last_query_time)
            if wait > 0:
                sleep(wait)
            self.last_query_time = time()

            try:
                response = recognize_song_from_signature(signature)
                if 'matches' in response:
                    return response
                error = 'unexpected response: %s' % str(response)[:200]
            except Exception as e:
                error = repr(e)

            backoff = self.delay * (2 ** (attempt + 1))
            print(
                '  query failed (%s), retrying in %.0fs' % (error, backoff),
                file=sys.stderr,
            )
            sleep(backoff)

        raise RuntimeError('Giving up after %d retries' % self.retries)

    def matches(self, ids, phase) -> bool:

        cache_key = frozenset(ids)
        if cache_key in self.cache:
            return self.cache[cache_key]

        if len(self.log) >= self.max_queries:
            raise QueryBudgetExceeded()

        response = self.query(self.build(ids))
        track_key = response.get('track', {}).get('key')
        ok = track_key == self.key and bool(response['matches'])

        self.cache[cache_key] = ok
        self.log.append(
            {
                'query': len(self.log) + 1,
                'phase': phase,
                'num_peaks': len(ids),
                'matched': ok,
                'track_key': track_key,
                'title': response.get('track', {}).get('title'),
                'matches': response['matches'],
            }
        )
        print(
            '  #%d %s: %d peaks -> %s'
            % (
                len(self.log),
                phase,
                len(ids),
                'MATCH' if ok else 'no match',
            ),
            file=sys.stderr,
        )
        return ok

    def strongest_search(self, ids):

        by_strength = sorted(
            ids, key=lambda p: self.peak(p).peak_magnitude, reverse=True
        )

        lo, hi = 1, len(by_strength)  # hi is known to match
        while lo < hi:
            mid = (lo + hi) // 2
            if self.matches(by_strength[:mid], 'strongest'):
                hi = mid
            else:
                lo = mid + 1

        return by_strength[:hi]

    def ddmin(self, ids):

        # Order by time so chunks are contiguous stretches of the clip
        current = sorted(
            ids, key=lambda p: (self.peak(p).fft_pass_number, p)
        )
        chunk = max(1, len(current) // 2)

        while True:
            removed_any = False
            start = 0
            while start < len(current):
                candidate = current[:start] + current[start + chunk :]
                if candidate and self.matches(candidate, 'chunk%d' % chunk):
                    current = candidate
                    removed_any = True
                else:
                    start += chunk

            if chunk == 1 and not removed_any:
                return current

            if not removed_any:
                chunk = max(1, chunk // 2)


class QueryBudgetExceeded(Exception):
    pass


def write_csv(path, message):

    with open(path, 'w', newline='') as f:
        w = writer(f)
        w.writerow(COLUMNS)
        w.writerows(peak_rows(message))


def main():
    args = ArgumentParser(
        description='Find a smallest subset of peaks in a Shazam '
        + 'fingerprint that still matches the same track.'
    )
    args.add_argument(
        'fingerprint',
        help='A .sig file containing a data URI, or the data URI itself.',
    )
    args.add_argument(
        '-o',
        '--output-prefix',
        required=True,
        help='Writes <prefix>.sig, <prefix>.csv and <prefix>.log.json.',
    )
    args.add_argument(
        '--key',
        help='Shazam track key to require (default: whatever the full '
        + 'fingerprint matches).',
    )
    args.add_argument(
        '--delay',
        type=float,
        default=2.0,
        help='Minimum seconds between queries (default: 2).',
    )
    args.add_argument(
        '--max-queries',
        type=int,
        default=300,
        help='Stop after this many queries (default: 300).',
    )
    args.add_argument(
        '--retries',
        type=int,
        default=4,
        help='Retries per query on errors (default: 4).',
    )
    args.add_argument(
        '--skip-strongest',
        action='store_true',
        help='Skip the strongest-N binary search phase.',
    )
    args = args.parse_args()

    message = DecodedMessage.decode_from_uri(read_uri(args.fingerprint))

    m = Minimizer(
        message, args.key, args.delay, args.max_queries, args.retries
    )
    print('%d peaks in input' % len(m.all_ids), file=sys.stderr)

    if m.key is None:
        response = m.query(message)
        m.key = response.get('track', {}).get('key')
        if not m.key:
            print('Full fingerprint does not match anything', file=sys.stderr)
            sys.exit(1)
        print(
            'Target: %s - %s (key %s)'
            % (
                response['track'].get('title'),
                response['track'].get('subtitle'),
                m.key,
            ),
            file=sys.stderr,
        )
    m.cache[frozenset(m.all_ids)] = True

    best = m.all_ids
    try:
        if not args.skip_strongest:
            best = m.strongest_search(best)
            print('Strongest-N phase: %d peaks' % len(best), file=sys.stderr)
        best = m.ddmin(best)
        complete = True
    except QueryBudgetExceeded:
        # Fall back to the smallest set that was confirmed to match
        best = min(
            (set(ids) for ids, ok in m.cache.items() if ok), key=len
        )
        complete = False
        print('Query budget exhausted', file=sys.stderr)

    result = m.build(best)

    with open(args.output_prefix + '.sig', 'w') as f:
        f.write(result.encode_to_uri() + '\n')
    write_csv(args.output_prefix + '.csv', result)
    with open(args.output_prefix + '.log.json', 'w') as f:
        dump(
            {
                'input_peaks': len(m.all_ids),
                'result_peaks': len(best),
                'one_minimal': complete,
                'track_key': m.key,
                'queries': m.log,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        '%d -> %d peaks (%s) in %d queries'
        % (
            len(m.all_ids),
            len(best),
            '1-minimal' if complete else 'NOT minimal, budget hit',
            len(m.log),
        ),
        file=sys.stderr,
    )


if __name__ == '__main__':
    main()
