"""
Fetch hourly ETH/USDC OHLCV data from Binance (free, no API key).
Outputs price_feed.npy and volume_feed.npy ready for AMMEnvironment.
"""

import time
import json
import numpy as np
import urllib.request
import urllib.error
from datetime import datetime

SYMBOL     = "ETHUSDC"
INTERVAL   = "1h"
START_MS   = 1620000000000
ENDPOINT   = "https://api.binance.com/api/v3/klines"
LIMIT      = 1000
OUT_PRICES = "price_feed.npy"
OUT_VOLUME = "volume_feed.npy"


def fetch_klines(start_ms, end_ms):
    url = (f"{ENDPOINT}?symbol={SYMBOL}&interval={INTERVAL}"
           f"&startTime={start_ms}&endTime={end_ms}&limit={LIMIT}")
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def fetch_all():
    records = []
    start = START_MS
    end_ms = int(time.time() * 1000)
    page = 1

    while start < end_ms:
        dt = datetime.utcfromtimestamp(start / 1000).strftime("%Y-%m-%d")
        print(f"  Page {page}: {dt}", end=" ... ", flush=True)
        try:
            batch = fetch_klines(start, end_ms)
        except urllib.error.HTTPError as e:
            print(f"HTTP {e.code} -- retrying")
            time.sleep(3)
            continue
        except Exception as e:
            print(f"Error: {e} -- retrying")
            time.sleep(3)
            continue

        if not batch:
            break

        records.extend(batch)
        print(f"{len(batch)} candles")

        last_open = batch[-1][0]
        if last_open <= start or len(batch) < LIMIT:
            break
        start = last_open + 3600000
        page += 1
        time.sleep(0.1)

    return records


def main():
    print("Fetching Binance ETH/USDC hourly data...")
    print(f"Symbol   : {SYMBOL}")
    print(f"Interval : {INTERVAL}")
    print()

    records = fetch_all()

    if not records:
        print("No data returned.")
        return

    open_times   = [r[0]         for r in records]
    close_prices = [float(r[4])  for r in records]
    volumes      = [float(r[5])  for r in records]
    open_prices  = [float(r[1])  for r in records]
    high_prices  = [float(r[2])  for r in records]
    low_prices   = [float(r[3])  for r in records]

    price_feed = np.array(close_prices, dtype=np.float64)
    vol_feed   = np.array(volumes,      dtype=np.float64)
    ts_arr     = np.array(open_times,   dtype=np.int64)

    np.save(OUT_PRICES, price_feed)
    np.save(OUT_VOLUME, vol_feed)

    np.savez(
        "binance_eth_usdc_hourly.npz",
        open_times=ts_arr,
        open=np.array(open_prices),
        high=np.array(high_prices),
        low=np.array(low_prices),
        close=price_feed,
        volume=vol_feed,
    )

    d0 = datetime.utcfromtimestamp(ts_arr[0]  / 1000).date()
    d1 = datetime.utcfromtimestamp(ts_arr[-1] / 1000).date()
    print(f"Done. {len(records)} hourly candles.")
    print(f"  Date range : {d0}  ->  {d1}")
    print(f"  Price range: ${price_feed.min():.2f}  ->  ${price_feed.max():.2f}")
    print(f"  Avg volume : {vol_feed.mean():.2f} ETH/hr")
    print()
    print(f"Saved:")
    print(f"  {OUT_PRICES}")
    print(f"  {OUT_VOLUME}")
    print(f"  binance_eth_usdc_hourly.npz")


if __name__ == "__main__":
    main()
