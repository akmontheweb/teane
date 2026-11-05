"""The harness must report what the provider actually charges.

For weeks it did not. Six lumina sessions whose per-call debug dumps survive
recorded **$6.38** against a real **$16.34** — a 2.56x undercount — and the
error had three independent causes, all of them here:

1. **The rate table was invented.** ``deepseek:deepseek-v4-pro`` carried
   0.435 / 0.87 / 0.003625 (miss / output / hit) against a published
   1.32 / 3.96 / 0.044 peak and 0.66 / 1.98 / 0.022 off-peak. Not a stale
   snapshot of either column: the output/input ratio was 2.0 where
   DeepSeek's is 3.0, and the cache discount was 120x where DeepSeek's is
   30x. The worst-priced rate was the cache-hit one, which is the rate that
   applies to ~82% of this harness's tokens.

2. **Time-of-day pricing was not modelled at all.** DeepSeek bills peak
   (01:00-04:00 and 06:00-10:00 UTC, Mon-Fri) at double off-peak. On a
   machine in IST (UTC+5:30) those windows are 06:30-09:30 and 11:30-15:30
   local, so a working day straddles them: session 01a18134 took 273 of its
   466 calls at peak and was underpriced 3.13x.

3. **Attempts the provider billed but we never read were booked at $0.**
   ``retry_with_backoff`` returns only the final attempt's response. A read
   timeout does not stop the server generating, so with max_retries=5 one
   dispatch can be billed six times and recorded once.

A fourth defect was reporting rather than money: ``TokenUsage.input_tokens``
carried whatever the provider said, and the wire formats disagree. Anthropic
reports it net of cache reads; DeepSeek and OpenAI report a total that
includes them. ``SessionMetrics.cache_hit_rate`` divides by
``cached + tokens_in`` and so double-counted the hits for DeepSeek: a real
79.1% hit rate was displayed as 44.2%, under a gauge documented to mean "a
prefix is breaking" below 50%.

The consequence that matters operationally is the cap. Underpricing 2.56x
means ``hard_cap_usd = 10`` permitted roughly $25 of real spend, which is
how the account reached a 402 Payment Required without any run reporting it
was short of budget.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import pathlib

import pytest

from harness import gateway as gw
from harness.gateway import (
    AnthropicProvider,
    DeepSeekProvider,
    Gateway,
    GatewayConfig,
    ModelSpec,
    NodeRole,
    OpenAIProvider,
    TokenUsage,
)

REPO = pathlib.Path(__file__).resolve().parent.parent

#: DeepSeek's published figures, PEAK column, USD per 1M tokens, from
#: api-docs.deepseek.com/quick_start/pricing. Off-peak is exactly half.
OFFICIAL = {
    "deepseek-v4-pro": {"miss": 1.32, "hit": 0.044, "out": 3.96},
    "deepseek-flash": {"miss": 0.30, "hit": 0.006, "out": 1.20},
}
#: Peak hours are 01:00-04:00 and 06:00-10:00 UTC, Monday through Friday.
PEAK_WINDOWS = ("01:00-04:00", "06:00-10:00")
PEAK_WEEKDAYS = (0, 1, 2, 3, 4)


def _pro_spec(**over) -> ModelSpec:
    kw = dict(
        provider="deepseek", model_id="deepseek-v4-pro", context_window=1_000_000,
        input_cost_per_1m=1.32, output_cost_per_1m=3.96,
        cached_input_cost_per_1m=0.044,
        off_peak_multiplier=0.5,
        peak_windows_utc=PEAK_WINDOWS, peak_weekdays=PEAK_WEEKDAYS,
    )
    kw.update(over)
    return ModelSpec(**kw)


# ---------------------------------------------------------------------------
# 1. The published rates, pinned
# ---------------------------------------------------------------------------

class TestTheRateTableMatchesThePublishedOne:
    """A wrong rate is silent: every total stays plausible and is wrong by a
    constant factor. Pinning the numbers is the only thing that catches it."""

    @pytest.fixture
    def catalogue(self):
        return json.loads((REPO / "harness" / "model_prices.json").read_text())

    @pytest.fixture
    def config(self):
        return json.loads((REPO / "config" / "config.json").read_text())["models"]

    def test_v4_pro_carries_the_published_peak_rates(self, catalogue):
        e = catalogue["deepseek:deepseek-v4-pro"]
        o = OFFICIAL["deepseek-v4-pro"]
        assert e["input_cost_per_1m"] == o["miss"]
        assert e["output_cost_per_1m"] == o["out"]
        assert e["cached_input_cost_per_1m"] == o["hit"]

    def test_the_active_config_agrees_with_the_catalogue(self, config):
        """config.json entries MERGE OVER the catalogue, so a stale rate here
        silently wins. This is the file that was actually serving 0.435."""
        e = config["deepseek:deepseek-v4-pro"]
        o = OFFICIAL["deepseek-v4-pro"]
        assert e["input_cost_per_1m"] == o["miss"]
        assert e["output_cost_per_1m"] == o["out"]
        assert e["cached_input_cost_per_1m"] == o["hit"]

    def test_output_costs_three_times_input_not_twice(self, catalogue):
        """The shape, not just the scale. The old table's output/input ratio
        was 2.0; DeepSeek's is 3.0, so no uniform discount could explain it
        and 'it is a promotional rate' was not an available excuse."""
        e = catalogue["deepseek:deepseek-v4-pro"]
        assert e["output_cost_per_1m"] / e["input_cost_per_1m"] == pytest.approx(3.0)

    def test_the_cache_discount_is_thirty_fold_not_one_hundred_and_twenty(
        self, catalogue,
    ):
        """The old 120x discount under-priced the one rate that covers most
        of this harness's tokens."""
        e = catalogue["deepseek:deepseek-v4-pro"]
        ratio = e["input_cost_per_1m"] / e["cached_input_cost_per_1m"]
        assert ratio == pytest.approx(30.0)

    @pytest.mark.parametrize("key", [
        "deepseek:deepseek-v4-pro", "deepseek:deepseek-v4-flash",
        "deepseek:deepseek-chat", "deepseek:deepseek-reasoner",
        "deepseek:deepseek-flash",
    ])
    def test_every_deepseek_entry_declares_the_clock(self, catalogue, key):
        """A deepseek model without the windows is priced flat, i.e. at half
        rate for every peak call."""
        e = catalogue[key]
        assert e["off_peak_multiplier"] == 0.5
        assert tuple(e["peak_windows_utc"]) == PEAK_WINDOWS
        assert tuple(e["peak_weekdays"]) == PEAK_WEEKDAYS

    @pytest.mark.parametrize("key", [
        "deepseek:deepseek-v4-flash", "deepseek:deepseek-chat",
        "deepseek:deepseek-reasoner",
    ])
    def test_retired_aliases_are_priced_as_what_actually_serves_them(
        self, catalogue, key,
    ):
        """DeepSeek still accepts these names but serves and bills them from
        V4.1-Flash. Pricing them from the V3-era table underpriced every call
        made through them -- deepseek-reasoner was carrying 0.55/2.19."""
        e = catalogue[key]
        o = OFFICIAL["deepseek-flash"]
        assert (e["input_cost_per_1m"], e["output_cost_per_1m"],
                e["cached_input_cost_per_1m"]) == (o["miss"], o["out"], o["hit"])

    def test_the_catalogue_loads_the_clock_fields_into_the_spec(self):
        """JSON lists have to survive the loader as tuples on ModelSpec."""
        gw.load_model_prices(override=True)
        spec = gw._MODEL_REGISTRY["deepseek:deepseek-v4-pro"]
        assert spec.off_peak_multiplier == 0.5
        assert spec.peak_windows_utc == PEAK_WINDOWS
        assert spec.peak_weekdays == PEAK_WEEKDAYS


# ---------------------------------------------------------------------------
# 2. Which hour is peak
# ---------------------------------------------------------------------------

class TestPeakWindows:
    SPEC = _pro_spec()

    @pytest.mark.parametrize("hour,peak", [
        (0, False),            # before the first window
        (1, True), (3, True),  # 01:00-04:00 inclusive of start
        (4, False),            # end is exclusive
        (5, False),            # the gap between the windows
        (6, True), (9, True),  # 06:00-10:00
        (10, False),           # end exclusive -- run 14 started at 10:41
        (23, False),
    ])
    def test_weekday_hours(self, hour, peak):
        when = dt.datetime(2026, 11, 4, hour, 30, tzinfo=dt.UTC)  # Wednesday
        assert self.SPEC.is_peak(when) is peak

    def test_the_window_boundaries_are_start_inclusive_end_exclusive(self):
        assert self.SPEC.is_peak(dt.datetime(2026, 11, 4, 1, 0, tzinfo=dt.UTC))
        assert not self.SPEC.is_peak(dt.datetime(2026, 11, 4, 4, 0, tzinfo=dt.UTC))
        assert not self.SPEC.is_peak(dt.datetime(2026, 11, 4, 0, 59, tzinfo=dt.UTC))

    @pytest.mark.parametrize("day", [7, 8])  # Sat, Sun
    def test_weekends_are_off_peak_around_the_clock(self, day):
        for hour in (1, 3, 7, 9):
            assert not self.SPEC.is_peak(
                dt.datetime(2026, 11, day, hour, 30, tzinfo=dt.UTC)
            )

    def test_a_naive_datetime_is_read_as_utc_not_local(self):
        """A 5.5-hour offset is more than enough to move a call between rate
        columns, so a naive datetime must not be silently localised."""
        naive = dt.datetime(2026, 11, 4, 7, 0)       # inside 06:00-10:00 UTC
        aware = dt.datetime(2026, 11, 4, 7, 0, tzinfo=dt.UTC)
        assert self.SPEC.is_peak(naive) == self.SPEC.is_peak(aware) is True

    def test_an_aware_datetime_in_another_zone_is_converted(self):
        ist = dt.timezone(dt.timedelta(hours=5, minutes=30))
        # 12:30 IST == 07:00 UTC == peak. This is the case that made IST
        # afternoons expensive without anyone noticing.
        assert self.SPEC.is_peak(dt.datetime(2026, 11, 4, 12, 30, tzinfo=ist))

    def test_a_window_that_wraps_midnight_is_honoured(self):
        spec = _pro_spec(peak_windows_utc=("22:00-02:00",))
        assert spec.is_peak(dt.datetime(2026, 11, 4, 23, 0, tzinfo=dt.UTC))
        assert spec.is_peak(dt.datetime(2026, 11, 4, 1, 0, tzinfo=dt.UTC))
        assert not spec.is_peak(dt.datetime(2026, 11, 4, 12, 0, tzinfo=dt.UTC))

    def test_a_malformed_window_is_skipped_not_raised(self, caplog):
        """A typo in an operator's config must not kill a paid dispatch."""
        spec = _pro_spec(peak_windows_utc=("not-a-window", "06:00-10:00"))
        assert spec.is_peak(dt.datetime(2026, 11, 4, 7, 0, tzinfo=dt.UTC))
        assert not spec.is_peak(dt.datetime(2026, 11, 4, 12, 0, tzinfo=dt.UTC))

    def test_a_model_with_no_windows_is_never_peak(self):
        """Flat-priced providers must be untouched by all of this."""
        spec = ModelSpec(
            provider="anthropic", model_id="x", context_window=1000,
            input_cost_per_1m=3.0, output_cost_per_1m=15.0,
        )
        assert not spec.is_peak(dt.datetime(2026, 11, 4, 7, 0, tzinfo=dt.UTC))
        assert spec.rates_at().input_per_1m == 3.0


class TestRatesAt:
    def test_peak_returns_the_declared_rates(self):
        r = _pro_spec().rates_at(dt.datetime(2026, 11, 4, 7, 0, tzinfo=dt.UTC))
        assert (r.input_per_1m, r.cached_input_per_1m, r.output_per_1m) == (
            1.32, 0.044, 3.96)
        assert r.is_peak

    def test_off_peak_halves_every_rate(self):
        r = _pro_spec().rates_at(dt.datetime(2026, 11, 4, 12, 0, tzinfo=dt.UTC))
        assert (r.input_per_1m, r.cached_input_per_1m, r.output_per_1m) == (
            0.66, 0.022, 1.98)
        assert not r.is_peak

    def test_off_peak_matches_the_published_off_peak_column(self):
        """Not merely "half of something" -- half of the peak column has to
        land on the numbers DeepSeek publishes as off-peak."""
        r = _pro_spec().rates_at(dt.datetime(2026, 11, 4, 12, 0, tzinfo=dt.UTC))
        assert r.input_per_1m == pytest.approx(0.66)
        assert r.cached_input_per_1m == pytest.approx(0.022)
        assert r.output_per_1m == pytest.approx(1.98)

    def test_cache_creation_defaults_to_one_and_a_quarter_input(self):
        spec = ModelSpec(
            provider="anthropic", model_id="x", context_window=1000,
            input_cost_per_1m=3.0, output_cost_per_1m=15.0,
            cached_input_cost_per_1m=0.3,
        )
        assert spec.rates_at().cache_creation_per_1m == pytest.approx(3.75)

    def test_an_explicit_creation_rate_wins(self):
        spec = ModelSpec(
            provider="anthropic", model_id="x", context_window=1000,
            input_cost_per_1m=3.0, output_cost_per_1m=15.0,
            cache_creation_cost_per_1m=9.0,
        )
        assert spec.rates_at().cache_creation_per_1m == 9.0


# ---------------------------------------------------------------------------
# 3. One canonical token partition across providers
# ---------------------------------------------------------------------------

class TestTheCanonicalPartition:
    """``input_tokens`` means the same thing whatever served the call."""

    def test_deepseek_splits_its_inclusive_prompt_total(self):
        u = DeepSeekProvider(_pro_spec(), api_key="k").extract_usage({"usage": {
            "prompt_tokens": 82415, "completion_tokens": 17606,
            "prompt_tokens_details": {"cached_tokens": 27776},
        }})
        assert u.input_tokens == 54639   # misses only
        assert u.cached_tokens == 27776
        assert u.prompt_tokens == 82415  # the wire's own total, recovered

    def test_deepseek_prefers_its_explicit_hit_miss_pair(self):
        """DeepSeek reports the split twice. The explicit pair is
        authoritative; the OpenAI-compat field is a fallback."""
        u = DeepSeekProvider(_pro_spec(), api_key="k").extract_usage({"usage": {
            "prompt_tokens": 1000, "completion_tokens": 10,
            "prompt_cache_hit_tokens": 700, "prompt_cache_miss_tokens": 300,
            "prompt_tokens_details": {"cached_tokens": 999},
        }})
        assert (u.input_tokens, u.cached_tokens) == (300, 700)

    def test_deepseek_with_no_cache_block_is_all_misses(self):
        u = DeepSeekProvider(_pro_spec(), api_key="k").extract_usage(
            {"usage": {"prompt_tokens": 500, "completion_tokens": 5}})
        assert (u.input_tokens, u.cached_tokens, u.prompt_tokens) == (500, 0, 500)

    def test_openai_splits_its_inclusive_prompt_total(self):
        spec = ModelSpec(
            provider="openai", model_id="gpt", context_window=1000,
            input_cost_per_1m=10.0, output_cost_per_1m=30.0,
            cached_input_cost_per_1m=2.5,
        )
        u = OpenAIProvider(spec).extract_usage({"usage": {
            "prompt_tokens": 1_000_000, "completion_tokens": 0,
            "prompt_tokens_details": {"cached_tokens": 800_000},
        }})
        assert (u.input_tokens, u.cached_tokens, u.prompt_tokens) == (
            200_000, 800_000, 1_000_000)

    def test_anthropic_is_left_alone_because_it_already_excludes(self):
        """Anthropic's input_tokens is already net of cache reads. Applying
        the same subtraction here would have removed them twice."""
        spec = ModelSpec(
            provider="anthropic", model_id="c", context_window=1000,
            input_cost_per_1m=3.0, output_cost_per_1m=15.0,
            cached_input_cost_per_1m=0.3,
        )
        u = AnthropicProvider(spec).extract_usage({"usage": {
            "input_tokens": 200, "output_tokens": 50,
            "cache_read_input_tokens": 800,
            "cache_creation_input_tokens": 100,
        }})
        assert u.input_tokens == 200
        assert u.cached_tokens == 800
        assert u.cache_creation_tokens == 100
        assert u.prompt_tokens == 1100  # all three, nothing double-counted

    def test_prompt_tokens_is_the_sum_of_the_three_input_fields(self):
        u = TokenUsage(input_tokens=1, cached_tokens=2, cache_creation_tokens=4)
        assert u.prompt_tokens == 7

    def test_to_dict_exposes_both_numbers(self):
        d = TokenUsage(input_tokens=10, cached_tokens=90).to_dict()
        assert d["input_tokens"] == 10 and d["prompt_tokens"] == 100


# ---------------------------------------------------------------------------
# 4. What a call costs
# ---------------------------------------------------------------------------

class TestWhatACallCosts:
    #: Run 14 (01a1a622) call 1: the planning dispatch, 2026-11-04 10:41 UTC.
    #: The harness recorded $0.039186 for it.
    RUN14_RAW = {"usage": {
        "prompt_tokens": 82415, "completion_tokens": 17606,
        "prompt_tokens_details": {"cached_tokens": 27776},
    }}
    RECORDED_AT_THE_TIME = 0.039186

    def _usage(self):
        return DeepSeekProvider(_pro_spec(), api_key="k").extract_usage(
            self.RUN14_RAW)

    def test_the_run_14_call_reprices_off_peak(self):
        """10:41 UTC is outside 06:00-10:00, so this call really was
        off-peak -- and still cost 1.83x what was booked."""
        p = DeepSeekProvider(_pro_spec(), api_key="k")
        cost = p.compute_cost(
            self._usage(), at=dt.datetime(2026, 11, 4, 10, 41, tzinfo=dt.UTC))
        assert cost == pytest.approx(0.071533, abs=1e-6)
        assert cost / self.RECORDED_AT_THE_TIME == pytest.approx(1.83, abs=0.01)

    def test_the_same_call_an_hour_earlier_costs_double(self):
        p = DeepSeekProvider(_pro_spec(), api_key="k")
        u = self._usage()
        peak = p.compute_cost(u, at=dt.datetime(2026, 11, 4, 9, 41, tzinfo=dt.UTC))
        off = p.compute_cost(u, at=dt.datetime(2026, 11, 4, 10, 41, tzinfo=dt.UTC))
        assert peak / off == pytest.approx(2.0)
        assert peak / self.RECORDED_AT_THE_TIME == pytest.approx(3.65, abs=0.01)

    def test_each_term_is_priced_at_its_own_rate(self):
        """Spelled out so a regression says WHICH rate moved."""
        u = self._usage()
        r = _pro_spec().rates_at(dt.datetime(2026, 11, 4, 10, 41, tzinfo=dt.UTC))
        expected = (
            (54639 / 1e6) * 0.66      # misses
            + (27776 / 1e6) * 0.022   # hits
            + (17606 / 1e6) * 1.98    # output
        )
        assert r.price(u) == pytest.approx(expected)

    def test_cache_hits_are_not_also_billed_at_the_full_rate(self):
        """The partition's whole point. If extract_usage ever hands the
        inclusive total through, this doubles the input term."""
        p = DeepSeekProvider(_pro_spec(), api_key="k")
        u = self._usage()
        at = dt.datetime(2026, 11, 4, 12, 0, tzinfo=dt.UTC)
        all_miss = TokenUsage(
            input_tokens=u.prompt_tokens, output_tokens=u.output_tokens)
        assert p.compute_cost(u, at=at) < p.compute_cost(all_miss, at=at)

    def test_reasoning_heavy_output_is_billed_as_output(self):
        """completion_tokens includes the reasoning trace on every provider,
        so a thinking call costs output rates for the thinking."""
        p = DeepSeekProvider(_pro_spec(), api_key="k")
        quiet = p.extract_usage({"usage": {
            "prompt_tokens": 1000, "completion_tokens": 100}})
        chatty = p.extract_usage({"usage": {
            "prompt_tokens": 1000, "completion_tokens": 40_000}})
        at = dt.datetime(2026, 11, 4, 12, 0, tzinfo=dt.UTC)
        assert p.compute_cost(chatty, at=at) > 50 * p.compute_cost(quiet, at=at)

    def test_local_inference_stays_free(self):
        spec = ModelSpec(
            provider="ollama", model_id="llama", context_window=8192,
            input_cost_per_1m=0.0, output_cost_per_1m=0.0,
        )
        from harness.gateway import OllamaProvider
        assert OllamaProvider(spec).compute_cost(
            TokenUsage(input_tokens=10**6, output_tokens=10**6)) == 0.0


# ---------------------------------------------------------------------------
# 5. Priced at send time, not return time
# ---------------------------------------------------------------------------

class _FakeHTTPResponse:
    def __init__(self, payload):
        self._payload = payload
        self.status_code = 200
        self.headers = {"content-type": "application/json"}
        self.text = json.dumps(payload)

    def json(self):
        return self._payload

    def raise_for_status(self):
        return None


class _ClockedClient:
    """Answers one POST, having advanced a fake clock past a rate boundary."""

    def __init__(self, payload, advance_to, clock):
        self._payload = payload
        self._advance_to = advance_to
        self._clock = clock

    async def post(self, *_a, **_kw):
        self._clock["now"] = self._advance_to
        return _FakeHTTPResponse(self._payload)


class TestPricingUsesTheRequestTime:
    def test_a_call_that_crosses_the_boundary_keeps_its_start_rate(
        self, monkeypatch,
    ):
        """A planning dispatch can run for minutes. One that starts at 09:58
        peak and returns at 10:03 off-peak must stay billed at peak, or the
        charge depends on how slow the provider happened to be."""
        started = dt.datetime(2026, 11, 4, 9, 58, tzinfo=dt.UTC)
        returned = dt.datetime(2026, 11, 4, 10, 3, tzinfo=dt.UTC)
        clock = {"now": started}

        class _DT(dt.datetime):
            @classmethod
            def now(cls, tz=None):
                return clock["now"]

        monkeypatch.setattr(gw._dt, "datetime", _DT)

        payload = {
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1_000_000, "completion_tokens": 0},
        }
        spec = _pro_spec()
        prov = DeepSeekProvider(spec, api_key="k")
        prov._client = _ClockedClient(payload, returned, clock)

        resp = asyncio.run(prov.chat_completion([{"role": "user", "content": "x"}]))
        # 1M misses at the PEAK rate. Billed at return time it would be 0.66.
        assert resp.usage.cost_usd == pytest.approx(1.32)


# ---------------------------------------------------------------------------
# 6. The cache-hit rate the operator reads
# ---------------------------------------------------------------------------

class TestTheReportedCacheHitRate:
    """Run 14's real hit rate was 79.1%; the harness displayed 44.2%.

    The formula was never wrong. It divides by ``cached + tokens_in`` and
    needs those disjoint, which they are under the canonical partition and
    were not under DeepSeek's wire shape. Changing the formula would have
    broken Anthropic, so the fix went into the providers.
    """

    #: Run 14 totals from the surviving per-call dumps.
    PROMPT_TOTAL = 3_871_718
    CACHED = 3_063_040

    def test_the_rate_reflects_the_real_hit_fraction(self):
        from harness.metrics import SessionMetrics
        m = SessionMetrics(
            session_id="01a1a622",
            tokens_in=self.PROMPT_TOTAL - self.CACHED,  # canonical: misses
            cached_tokens=self.CACHED,
        )
        assert m.cache_hit_rate() == pytest.approx(0.791, abs=0.001)

    def test_feeding_it_an_inclusive_total_is_what_produced_44_percent(self):
        """Pins the arithmetic of the original defect, so the number in the
        docstring is checkable rather than asserted."""
        from harness.metrics import SessionMetrics
        wrong = SessionMetrics(
            session_id="x",
            tokens_in=self.PROMPT_TOTAL,  # the old, inclusive meaning
            cached_tokens=self.CACHED,
        )
        assert wrong.cache_hit_rate() == pytest.approx(0.442, abs=0.001)

    def test_prompt_tokens_recovers_the_total(self):
        from harness.metrics import SessionMetrics
        m = SessionMetrics(
            session_id="x",
            tokens_in=self.PROMPT_TOTAL - self.CACHED,
            cached_tokens=self.CACHED,
        )
        assert m.prompt_tokens == self.PROMPT_TOTAL

    def test_the_rate_survives_the_provider_round_trip(self):
        """End to end: wire shape in, correct gauge out. This is the join
        the old code got wrong, and neither half alone would catch it."""
        from harness.metrics import SessionMetrics
        u = DeepSeekProvider(_pro_spec(), api_key="k").extract_usage({"usage": {
            "prompt_tokens": 100_000, "completion_tokens": 10,
            "prompt_tokens_details": {"cached_tokens": 80_000},
        }})
        m = SessionMetrics(
            session_id="x", tokens_in=u.input_tokens, cached_tokens=u.cached_tokens)
        assert m.cache_hit_rate() == pytest.approx(0.80)

    def test_anthropic_still_reports_correctly(self):
        from harness.metrics import SessionMetrics
        spec = ModelSpec(
            provider="anthropic", model_id="c", context_window=1000,
            input_cost_per_1m=3.0, output_cost_per_1m=15.0,
        )
        u = AnthropicProvider(spec).extract_usage({"usage": {
            "input_tokens": 20_000, "output_tokens": 10,
            "cache_read_input_tokens": 80_000,
        }})
        m = SessionMetrics(
            session_id="x", tokens_in=u.input_tokens, cached_tokens=u.cached_tokens)
        assert m.cache_hit_rate() == pytest.approx(0.80)


# ---------------------------------------------------------------------------
# 7. Attempts we were billed for but never read
# ---------------------------------------------------------------------------

def _gateway(**cfg_over) -> Gateway:
    kw = dict(hard_cap_usd=10.0)
    kw.update(cfg_over)
    g = Gateway(GatewayConfig(**kw))
    g.config.retry_charge = {
        "enabled": True, "charge_read_timeouts": True,
        "charge_server_errors": False, "fallback_output_tokens": 2000,
    }
    return g


def _status_error(code: int) -> gw.httpx.HTTPStatusError:
    req = gw.httpx.Request("POST", "https://api.deepseek.com/v1/chat/completions")
    resp = gw.httpx.Response(code, request=req)
    return gw.httpx.HTTPStatusError("boom", request=req, response=resp)


class TestWhichFailuresWereBilled:
    """How far the request got, not how it looked to us."""

    @pytest.mark.parametrize("exc,billed", [
        (gw.httpx.ReadTimeout("read"), "read_timeout"),
        (gw.httpx.ConnectTimeout("connect"), ""),
        (gw.httpx.WriteTimeout("write"), ""),
        (gw.httpx.PoolTimeout("pool"), ""),
        (gw.httpx.ConnectError("refused"), ""),
        (_status_error(429), ""),
        (_status_error(402), ""),
        (_status_error(401), ""),
    ])
    def test_classification(self, exc, billed):
        assert Gateway._provider_billed_despite_failure(
            exc, charge_read_timeouts=True, charge_server_errors=False,
        ) == billed

    def test_connect_timeout_is_not_mistaken_for_a_read_timeout(self):
        """Both derive from TimeoutException and are siblings, not
        subclasses -- the distinction is the whole classification."""
        assert isinstance(gw.httpx.ConnectTimeout("x"), gw.httpx.TimeoutException)
        assert not isinstance(gw.httpx.ConnectTimeout("x"), gw.httpx.ReadTimeout)

    def test_server_errors_are_opt_in(self):
        exc = _status_error(503)
        assert Gateway._provider_billed_despite_failure(
            exc, charge_read_timeouts=True, charge_server_errors=False) == ""
        assert Gateway._provider_billed_despite_failure(
            exc, charge_read_timeouts=True, charge_server_errors=True
        ) == "server_error"


class TestChargingAnUnreadAttempt:
    def _charge(self, g, *, exc=None, prompt=100_000):
        gw.load_model_prices(override=True)
        return g._charge_unread_attempt(
            exc=exc or gw.httpx.ReadTimeout("read"),
            role=NodeRole.PLANNING,
            model_key="deepseek:deepseek-v4-pro",
            spec=_pro_spec(),
            est_prompt_tokens=prompt,
            sent_at=dt.datetime(2026, 11, 4, 12, 0, tzinfo=dt.UTC),
        )

    def test_a_read_timeout_is_charged(self):
        g = _gateway()
        assert self._charge(g) > 0.0

    def test_a_connect_error_is_not(self):
        g = _gateway()
        assert self._charge(g, exc=gw.httpx.ConnectError("refused")) == 0.0

    def test_the_charge_becomes_debt(self):
        g = _gateway()
        charged = self._charge(g)
        assert g._unread_attempt_debt_usd == pytest.approx(charged)

    def test_draining_the_debt_clears_it(self):
        g = _gateway()
        self._charge(g)
        first = g._drain_unread_attempt_debt()
        assert first > 0 and g._drain_unread_attempt_debt() == 0.0

    def test_it_lands_in_the_session_totals(self):
        g = _gateway()
        charged = self._charge(g)
        assert g.session_tracker["total_cost_usd"] == pytest.approx(charged)

    def test_the_estimate_is_marked_as_an_estimate(self):
        """An estimate that cannot be told from a measurement is worse than
        no estimate: every downstream total would read as provider-reported."""
        g = _gateway()
        self._charge(g)
        per_model = g.session_tracker["per_model"]["deepseek-v4-pro"]
        assert per_model["cost_usd"] > 0
        assert TokenUsage(cost_is_estimated=True).to_dict()["cost_is_estimated"]

    def test_a_cold_start_assumes_no_cache_hits(self):
        """Not pessimism -- the first call of a session has nothing to hit."""
        g = _gateway()
        self._charge(g, prompt=1_000_000)
        stage = g.session_tracker["per_stage"]["planning"]
        assert stage["cached_tokens"] == 0
        assert stage["input_tokens"] == 1_000_000

    def test_it_uses_the_role_s_own_observed_hit_ratio(self):
        """Once the role has real calls, the estimate tracks the workload
        instead of a constant written when the code was authored."""
        g = _gateway()
        gw.load_model_prices(override=True)
        g.aggregate_tokens(
            g.session_tracker,
            TokenUsage(input_tokens=20_000, cached_tokens=80_000,
                       output_tokens=5_000, model_name="deepseek-v4-pro"),
            role=NodeRole.PLANNING,
        )
        self._charge(g, prompt=100_000)
        stage = g.session_tracker["per_stage"]["planning"]
        # 80% of the 100k estimate assumed cached, on top of the real call.
        assert stage["cached_tokens"] == 80_000 + 80_000
        assert stage["input_tokens"] == 20_000 + 20_000

    def test_it_uses_the_role_s_own_mean_output(self):
        g = _gateway()
        g.aggregate_tokens(
            g.session_tracker,
            TokenUsage(input_tokens=1_000, output_tokens=7_777,
                       model_name="deepseek-v4-pro"),
            role=NodeRole.PLANNING,
        )
        self._charge(g)
        stage = g.session_tracker["per_stage"]["planning"]
        assert stage["output_tokens"] == 7_777 + 7_777

    def test_the_cold_start_output_estimate_is_configurable(self):
        g = _gateway()
        g.config.retry_charge["fallback_output_tokens"] = 123
        self._charge(g)
        assert g.session_tracker["per_stage"]["planning"]["output_tokens"] == 123

    def test_disabling_it_charges_nothing(self):
        g = _gateway()
        g.config.retry_charge["enabled"] = False
        assert self._charge(g) == 0.0
        assert g._unread_attempt_debt_usd == 0.0

    def test_a_free_model_produces_no_debt(self):
        """A $0 charge must not create a debt entry that rounds to nothing."""
        g = _gateway()
        free = ModelSpec(
            provider="ollama", model_id="llama", context_window=8192,
            input_cost_per_1m=0.0, output_cost_per_1m=0.0,
        )
        assert g._charge_unread_attempt(
            exc=gw.httpx.ReadTimeout("read"), role=NodeRole.PLANNING,
            model_key="ollama:llama", spec=free, est_prompt_tokens=100_000,
        ) == 0.0
        assert g._unread_attempt_debt_usd == 0.0

    def test_the_charge_is_priced_at_the_attempt_s_own_hour(self):
        g_peak, g_off = _gateway(), _gateway()
        gw.load_model_prices(override=True)
        common = dict(
            exc=gw.httpx.ReadTimeout("read"), role=NodeRole.PLANNING,
            model_key="deepseek:deepseek-v4-pro", spec=_pro_spec(),
            est_prompt_tokens=1_000_000,
        )
        peak = g_peak._charge_unread_attempt(
            **common, sent_at=dt.datetime(2026, 11, 4, 7, 0, tzinfo=dt.UTC))
        off = g_off._charge_unread_attempt(
            **common, sent_at=dt.datetime(2026, 11, 4, 12, 0, tzinfo=dt.UTC))
        assert peak / off == pytest.approx(2.0)

    def test_several_lost_attempts_accumulate(self):
        """The case that makes this matter: max_retries=5 means one dispatch
        can be billed six times and recorded once."""
        g = _gateway()
        charges = [self._charge(g) for _ in range(6)]
        assert g._unread_attempt_debt_usd == pytest.approx(sum(charges))

    def test_the_estimate_uses_the_same_rate_resolution_as_billing(self):
        """Two code paths pricing the same tokens differently is how the
        pre-flight gate and the charge drifted in the first place."""
        spec = _pro_spec()
        u = TokenUsage(input_tokens=1_000, cached_tokens=9_000, output_tokens=500)
        at = dt.datetime(2026, 11, 4, 12, 0, tzinfo=dt.UTC)
        assert Gateway.compute_cost_for(spec, u, at=at) == pytest.approx(
            DeepSeekProvider(spec, api_key="k").compute_cost(u, at=at))


# ---------------------------------------------------------------------------
# 8. The config surface
# ---------------------------------------------------------------------------

class TestTheConfigSurface:
    def test_retry_charge_is_a_known_key(self):
        """An unregistered key is silently dropped by the validator, which
        is how a typoed cap can no-op without a word."""
        from harness.cli import _KNOWN_NESTED_KEYS
        assert "retry_charge" in _KNOWN_NESTED_KEYS["token_budget"]

    def test_retry_charge_is_type_checked(self):
        from harness.cli import _TYPE_SCHEMA
        assert _TYPE_SCHEMA["token_budget.retry_charge"] == (dict,)

    def test_the_shipped_config_sets_it(self):
        cfg = json.loads((REPO / "config" / "config.json").read_text())
        rc = cfg["token_budget"]["retry_charge"]
        assert rc["enabled"] is True
        assert rc["charge_read_timeouts"] is True
        assert rc["charge_server_errors"] is False
        assert isinstance(rc["fallback_output_tokens"], int)

    def test_the_loader_threads_it_through(self):
        from harness.gateway import create_gateway_from_config
        g = create_gateway_from_config({"token_budget": {
            "hard_cap_usd": 5.0,
            "retry_charge": {"enabled": False, "fallback_output_tokens": 42},
        }})
        assert g.config.retry_charge["enabled"] is False
        assert g.config.retry_charge["fallback_output_tokens"] == 42

    def test_absent_config_keeps_the_documented_defaults(self):
        from harness.gateway import create_gateway_from_config
        g = create_gateway_from_config({"token_budget": {"hard_cap_usd": 5.0}})
        assert g.config.retry_charge["enabled"] is True
        assert g.config.retry_charge["charge_read_timeouts"] is True
        assert g.config.retry_charge["charge_server_errors"] is False

    def test_a_nonsense_fallback_is_clamped_not_honoured(self):
        from harness.gateway import create_gateway_from_config
        g = create_gateway_from_config({"token_budget": {
            "hard_cap_usd": 5.0,
            "retry_charge": {"fallback_output_tokens": -5},
        }})
        assert g.config.retry_charge["fallback_output_tokens"] >= 0

    def test_every_new_key_is_documented(self):
        """config_help.md is the operator's only description of these."""
        help_text = (REPO / "config" / "config_help.md").read_text()
        for key in (
            "retry_charge", "charge_read_timeouts", "charge_server_errors",
            "fallback_output_tokens", "off_peak_multiplier",
            "peak_windows_utc", "peak_weekdays",
        ):
            assert key in help_text, f"{key} is not documented"


# ---------------------------------------------------------------------------
# 9. What the cap really permitted
# ---------------------------------------------------------------------------

class TestTheCapWasNotBinding:
    """The operational consequence, stated as arithmetic.

    Underpricing does not merely misreport: every budget gate is a
    comparison against the same wrong number, so the cap silently permits
    more real spend than it names. ``hard_cap_usd`` was 10.
    """

    OLD = {"miss": 0.435, "hit": 0.003625, "out": 0.87}
    #: Session totals from the six surviving dump sets.
    MISS, HIT, OUT = 9_589_937, 44_896_128, 2_349_293

    def _price(self, rates):
        return (self.MISS / 1e6) * rates["miss"] + (self.HIT / 1e6) * rates["hit"] \
            + (self.OUT / 1e6) * rates["out"]

    def test_the_old_table_reproduces_the_recorded_total(self):
        assert self._price(self.OLD) == pytest.approx(6.38, abs=0.01)

    def test_the_published_off_peak_table_alone_nearly_doubles_it(self):
        """Pricing every one of those calls at the CHEAPER published column
        already costs $11.97 -- so the undercount is not a peak-hour
        artefact, it is the table."""
        off = {"miss": 0.66, "hit": 0.022, "out": 1.98}
        assert self._price(off) == pytest.approx(11.97, abs=0.01)
        assert self._price(off) / self._price(self.OLD) == pytest.approx(
            1.88, abs=0.01)

    def test_peak_hours_take_it_the_rest_of_the_way_to_the_real_figure(self):
        """308 of the 886 calls landed in a peak window. Repricing each call
        at its own hour gives $16.34 against the $6.38 recorded -- 2.56x,
        which is what makes a $10 cap permit roughly $25."""
        off, peak = self._price({"miss": 0.66, "hit": 0.022, "out": 1.98}), None
        peak = self._price({"miss": 1.32, "hit": 0.044, "out": 3.96})
        assert off < 16.34 < peak, (
            "the real mixed total must sit between the all-off-peak and "
            "all-peak bounds"
        )
        assert 16.34 / self._price(self.OLD) == pytest.approx(2.56, abs=0.01)
        assert 16.34 / 10.0 > 1.0, "a $10 cap did not bind"

    def test_the_worst_priced_rate_covered_most_of_the_tokens(self):
        """Why the undercount was so large: the rate that was 6x wrong is
        the one that applies to 82% of the harness's input tokens."""
        assert self.HIT / (self.HIT + self.MISS) > 0.8
        assert 0.022 / self.OLD["hit"] == pytest.approx(6.07, abs=0.01)


# ---------------------------------------------------------------------------
# 10. End to end through dispatch
# ---------------------------------------------------------------------------

def _priced_model(key: str = "cost:test") -> str:
    """A registered model with real rates and no peak concept."""
    gw.register_model(key, ModelSpec(
        provider="stub", model_id="cost-test", context_window=128_000,
        input_cost_per_1m=1.32, output_cost_per_1m=3.96,
        cached_input_cost_per_1m=0.044, api_base_url="", api_key="x",
    ))
    return key


class _ScriptedProvider:
    """Yields each scripted item: an LLMResponse to return, or an exception
    to raise. Mirrors only the BaseLLM surface dispatch actually touches."""

    def __init__(self, script, spec: ModelSpec):
        self._script = list(script)
        self._idx = 0
        self.calls = 0
        self.spec = spec
        self.api_key = "x"

    async def chat_completion(self, **_kw):
        self.calls += 1
        item = self._script[min(self._idx, len(self._script) - 1)]
        self._idx += 1
        if isinstance(item, BaseException):
            raise item
        return item

    async def close(self):
        return None


def _dispatch_gateway(script) -> tuple[Gateway, _ScriptedProvider]:
    key = _priced_model()
    spec = gw._MODEL_REGISTRY[key]
    prov = _ScriptedProvider(script, spec)
    g = Gateway(GatewayConfig(
        planning_primary=key, patching_primary=key, repair_primary=key,
        hard_cap_usd=10.0,
        # One retry, no sleeping: this is about accounting, not backoff.
        max_retries=1, base_delay=0.0, retry_max_delay_seconds=0.0,
    ))
    g.config.retry_charge = {
        "enabled": True, "charge_read_timeouts": True,
        "charge_server_errors": False, "fallback_output_tokens": 1000,
    }
    g._providers[key] = prov  # type: ignore[index]
    return g, prov


def _ok(**over) -> gw.LLMResponse:
    """A scripted success, priced the way a real provider prices it.

    chat_completion is where ``usage.cost_usd`` is set, so a stub that skips
    it would hand the gateway a free call and this whole class would assert
    against zeros.
    """
    u = TokenUsage(input_tokens=1000, output_tokens=500,
                   cached_tokens=9000, model_name="cost-test")
    for k, v in over.items():
        setattr(u, k, v)
    # _ok() is evaluated while building the script, i.e. before
    # _dispatch_gateway registers the model, so register it here too.
    u.cost_usd = Gateway.compute_cost_for(gw._MODEL_REGISTRY[_priced_model()], u)
    return gw.LLMResponse(content="done", usage=u, model="cost-test")


MESSAGES = [
    {"role": "system", "content": "sys"},
    {"role": "user", "content": "go"},
]


class TestDispatchAccounting:
    def test_a_normal_dispatch_charges_the_priced_cost(self):
        g, prov = _dispatch_gateway([_ok()])
        resp, left = asyncio.run(g.dispatch(
            messages=MESSAGES, role=NodeRole.PLANNING, budget_remaining_usd=10.0))
        expected = (1000 / 1e6) * 1.32 + (9000 / 1e6) * 0.044 + (500 / 1e6) * 3.96
        assert resp.usage.cost_usd == pytest.approx(expected)
        assert 10.0 - left == pytest.approx(expected)
        assert prov.calls == 1

    def test_a_lost_attempt_then_a_success_charges_both(self):
        """The core leak. retry_with_backoff hands back only the final
        response, so the timed-out attempt used to cost nothing."""
        g, prov = _dispatch_gateway([gw.httpx.ReadTimeout("read"), _ok()])
        resp, left = asyncio.run(g.dispatch(
            messages=MESSAGES, role=NodeRole.PLANNING, budget_remaining_usd=10.0))
        assert prov.calls == 2
        measured = resp.usage.cost_usd
        # The lost attempt is NOT in this dispatch's returned budget -- it is
        # carried as debt -- but it is already in the session total.
        assert g._unread_attempt_debt_usd > 0.0
        assert g.session_tracker["total_cost_usd"] > measured
        assert 10.0 - left == pytest.approx(measured)

    def test_the_debt_narrows_the_next_dispatch(self):
        """A charge that only appears in a report does not protect the cap.
        It has to come out of the budget a later call is checked against."""
        g, _prov = _dispatch_gateway([gw.httpx.ReadTimeout("read"), _ok()])
        _resp, left = asyncio.run(g.dispatch(
            messages=MESSAGES, role=NodeRole.PLANNING, budget_remaining_usd=10.0))
        debt = g._unread_attempt_debt_usd
        assert debt > 0.0

        g._providers["cost:test"]._idx = 1  # next call succeeds
        _r2, left2 = asyncio.run(g.dispatch(
            messages=MESSAGES, role=NodeRole.PLANNING, budget_remaining_usd=left))
        # The second dispatch paid its own cost AND the carried debt.
        assert (left - left2) > _r2.usage.cost_usd
        assert (left - left2) == pytest.approx(_r2.usage.cost_usd + debt)
        assert g._unread_attempt_debt_usd == 0.0

    def test_a_connect_error_then_a_success_charges_only_the_success(self):
        g, prov = _dispatch_gateway([gw.httpx.ConnectError("refused"), _ok()])
        resp, left = asyncio.run(g.dispatch(
            messages=MESSAGES, role=NodeRole.PLANNING, budget_remaining_usd=10.0))
        assert prov.calls == 2
        assert g._unread_attempt_debt_usd == 0.0
        assert 10.0 - left == pytest.approx(resp.usage.cost_usd)

    def test_an_exhausted_dispatch_still_leaves_a_debt(self):
        """The worst case under the old code: every attempt timed out, the
        dispatch raised, and the whole thing was recorded as free."""
        g, prov = _dispatch_gateway([gw.httpx.ReadTimeout("read")])
        # A read timeout surviving its retries is classified as "model
        # silent" and routed to the fallback path, which raises this when no
        # fallback is configured.
        with pytest.raises(gw.EmptyLLMResponseError):
            asyncio.run(g.dispatch(
                messages=MESSAGES, role=NodeRole.PLANNING,
                budget_remaining_usd=10.0))
        assert prov.calls == 2  # max_retries=1 -> 2 attempts
        assert g._unread_attempt_debt_usd > 0.0
        assert g.session_tracker["total_cost_usd"] > 0.0

    def test_the_debt_survives_a_dispatch_refused_for_lack_of_budget(self):
        """Draining before the exhaustion guard would make the debt vanish
        exactly when the cap is already breached."""
        g, _prov = _dispatch_gateway([gw.httpx.ReadTimeout("read")])
        with pytest.raises(gw.EmptyLLMResponseError):
            asyncio.run(g.dispatch(
                messages=MESSAGES, role=NodeRole.PLANNING,
                budget_remaining_usd=10.0))
        debt = g._unread_attempt_debt_usd
        assert debt > 0.0
        with pytest.raises(RuntimeError, match="GUARDRAIL EXHAUSTED"):
            asyncio.run(g.dispatch(
                messages=MESSAGES, role=NodeRole.PLANNING,
                budget_remaining_usd=0.0))
        assert g._unread_attempt_debt_usd == pytest.approx(debt)

    def test_the_session_tracker_keeps_the_partition_disjoint(self):
        g, _prov = _dispatch_gateway([_ok()])
        asyncio.run(g.dispatch(
            messages=MESSAGES, role=NodeRole.PLANNING, budget_remaining_usd=10.0))
        t = g.session_tracker
        assert t["total_input_tokens"] == 1000      # misses
        assert t["total_cached_tokens"] == 9000     # hits
        from harness.metrics import SessionMetrics
        m = SessionMetrics(
            session_id="x", tokens_in=t["total_input_tokens"],
            cached_tokens=t["total_cached_tokens"])
        assert m.cache_hit_rate() == pytest.approx(0.9)
