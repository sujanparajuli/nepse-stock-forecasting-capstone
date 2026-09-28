import unittest

import numpy as np
import pandas as pd

from forecast import FEATURES, make_features


class FeatureTests(unittest.TestCase):
    def test_future_prices_cannot_change_past_features(self):
        n = 60
        close = np.arange(n, dtype=float) + 100
        prices = pd.DataFrame({
            "published_date": pd.bdate_range("2025-01-01", periods=n),
            "open": close - 1,
            "high": close + 2,
            "low": close - 2,
            "close": close,
            "traded_quantity": np.arange(n) + 1000,
        })
        before = make_features(prices)
        prices.loc[40:, "close"] *= 10
        after = make_features(prices)
        pd.testing.assert_frame_equal(before.loc[:39, FEATURES], after.loc[:39, FEATURES])


if __name__ == "__main__":
    unittest.main()
