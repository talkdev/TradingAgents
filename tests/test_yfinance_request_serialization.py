"""Yahoo calls share a crumb and must not overlap between graph workers."""

from __future__ import annotations

import threading
import time

from tradingagents.dataflows.stockstats_utils import yf_retry


def test_yfinance_requests_are_serialized():
    active = 0
    peak = 0
    guard = threading.Lock()

    def request():
        nonlocal active, peak
        with guard:
            active += 1
            peak = max(peak, active)
        time.sleep(0.01)
        with guard:
            active -= 1

    threads = [threading.Thread(target=lambda: yf_retry(request)) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert peak == 1
