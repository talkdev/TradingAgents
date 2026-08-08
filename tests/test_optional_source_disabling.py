"""Explicitly disabled optional sources must be silent and network-free."""

from __future__ import annotations

import copy
from unittest.mock import patch

import pytest

import tradingagents.default_config as default_config
from tradingagents.dataflows import config as config_module
from tradingagents.dataflows import interface
from tradingagents.dataflows.config import set_config


@pytest.mark.unit
@pytest.mark.parametrize("category,method", [
    ("macro_data", "get_macro_indicators"),
    ("prediction_markets", "get_prediction_markets"),
])
def test_disabled_optional_vendor_returns_sentinel_without_call(category, method):
    config_module._config = copy.deepcopy(default_config.DEFAULT_CONFIG)
    set_config({"data_vendors": {category: "disabled"}})
    with patch.dict(interface.VENDOR_METHODS, {method: {}}, clear=False):
        result = interface.route_to_vendor(method, "test", "2026-01-01")
    assert "DATA_UNAVAILABLE" in result
    assert "disabled by configuration" in result
