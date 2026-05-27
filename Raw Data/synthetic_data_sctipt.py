#!/usr/bin/env python3
"""Generate small synthetic CSVs to smoke-test the Netflix causal pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PLAN_PRICES = {
    "Standard with Ads": (6.99, 7.99),
    "Standard": (15.49, 17.99),
    "Premium": (22.99, 24.99),
}


def dateint(date: pd.Timestamp) -> int:
    return int(pd.Timestamp(date).strftime("%Y%m%d"))


def first_monthly_cycle_on_or_after(anchor: pd.Timestamp, start: pd.Timestamp) -> pd.Timestamp:
    date = pd.Timestamp(anchor)
    while date < start:
        date = date + pd.DateOffset(months=1)
    return date


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/toy"))
    parser.add_argument("--n-members", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    rollout_start = pd.Timestamp("2025-02-01")
    announced = pd.Timestamp("2025-01-21")
    member_ids = [f"M{20000000 + i}" for i in range(args.n_members)]
    plans = rng.choice(list(PLAN_PRICES), size=args.n_members, p=[0.28, 0.52, 0.20])
    billing_anchor_days = rng.integers(1, 29, size=args.n_members)
    signup_dates = pd.Timestamp("2022-01-01") + pd.to_timedelta(
        rng.integers(0, 1000, size=args.n_members), unit="D"
    )

    dim_member = pd.DataFrame(
        {
            "member_id": member_ids,
            "signup_dateint": [dateint(d) for d in signup_dates],
            "signup_country": "US",
            "signup_region": "UCAN",
            "signup_plan_name": plans,
            "acquisition_channel": rng.choice(["organic", "paid_search", "paid_social"], size=args.n_members),
            "payment_partner": rng.choice(["direct_card", "apple_iap", "google_play"], size=args.n_members),
            "signup_device": rng.choice(["web", "ios", "android", "roku", "smart_tv"], size=args.n_members),
            "birth_year_bucket": rng.choice(["1970s", "1980s", "1990s+"], size=args.n_members),
            "email_domain": rng.choice(["gmail.com", "yahoo.com", "outlook.com"], size=args.n_members),
            "is_household_primary": True,
        }
    )

    plan_rows = []
    plan_id_lookup = {}
    plan_id = 1
    for plan_name, (old_price, new_price) in PLAN_PRICES.items():
        old_id = f"PL{plan_id:04d}"
        plan_id += 1
        new_id = f"PL{plan_id:04d}"
        plan_id += 1
        plan_id_lookup[(plan_name, "old")] = old_id
        plan_id_lookup[(plan_name, "new")] = new_id
        plan_rows.extend(
            [
                {
                    "plan_id": old_id,
                    "plan_name": plan_name,
                    "country": "US",
                    "price_local": old_price,
                    "currency": "USD",
                    "price_usd": old_price,
                    "max_concurrent_streams": 2 if plan_name != "Premium" else 4,
                    "max_resolution": "1080p" if plan_name != "Premium" else "4K",
                    "effective_from_dateint": 20240101,
                    "effective_to_dateint": 20250120,
                    "is_current": False,
                },
                {
                    "plan_id": new_id,
                    "plan_name": plan_name,
                    "country": "US",
                    "price_local": new_price,
                    "currency": "USD",
                    "price_usd": new_price,
                    "max_concurrent_streams": 2 if plan_name != "Premium" else 4,
                    "max_resolution": "1080p" if plan_name != "Premium" else "4K",
                    "effective_from_dateint": 20250121,
                    "effective_to_dateint": 20991231,
                    "is_current": True,
                },
            ]
        )
    dim_plan = pd.DataFrame(plan_rows)

    campaign = pd.DataFrame(
        [
            {
                "country": "US",
                "plan_name": plan_name,
                "old_price_local": old,
                "new_price_local": new,
                "currency": "USD",
                "announced_dateint": 20250121,
                "new_member_effective_dateint": 20250121,
                "existing_member_rollout_start_dateint": 20250201,
                "existing_member_rollout_end_dateint": 20250430,
                "notification_channel": "email_and_in_app",
            }
            for plan_name, (old, new) in PLAN_PRICES.items()
        ]
    )

    member_frame = pd.DataFrame(
        {
            "member_id": member_ids,
            "plan_name": plans,
            "anchor_day": billing_anchor_days,
            "baseline_risk": rng.beta(2.0, 30.0, size=args.n_members),
            "engagement": rng.gamma(2.0, 2.0, size=args.n_members),
        }
    )
    member_frame["last_pre_anchor"] = [
        pd.Timestamp(year=2025, month=1, day=int(day)) for day in member_frame["anchor_day"]
    ]
    member_frame["rollout_month_offset"] = rng.choice([1, 2, 3], size=args.n_members, p=[0.34, 0.33, 0.33])
    member_frame["scheduled_exposure"] = [
        anchor + pd.DateOffset(months=int(offset))
        for anchor, offset in zip(member_frame["last_pre_anchor"], member_frame["rollout_month_offset"])
    ]

    event_rows = []
    billing_rows = []
    playback_rows = []
    weeks = pd.date_range("2025-01-06", "2025-06-30", freq="W-MON")
    for idx, row in member_frame.iterrows():
        member_id = row["member_id"]
        plan_name = row["plan_name"]
        risk = row["baseline_risk"]
        exposure = row["scheduled_exposure"]
        cancelled_at = None
        event_rows.append(
            {
                "member_id": member_id,
                "event_ts_utc": "2023-01-01T00:00:00",
                "event_local_ts": "2023-01-01T00:00:00",
                "event_type": "signup",
                "from_plan_id": "",
                "to_plan_id": plan_id_lookup[(plan_name, "old")],
                "cancel_reason_code": "",
                "country_at_event": "US",
                "device_at_event": "web",
            }
        )
        for week in weeks:
            post = week >= week_start(exposure)
            hazard = 0.0018 + 0.018 * risk + (0.0045 if post else 0.0)
            hazard += 0.0015 if plan_name == "Premium" else 0.0
            if cancelled_at is None and rng.random() < hazard:
                cancelled_at = week + pd.Timedelta(days=int(rng.integers(0, 7)))
                event_rows.append(
                    {
                        "member_id": member_id,
                        "event_ts_utc": cancelled_at.strftime("%Y-%m-%dT12:00:00"),
                        "event_local_ts": cancelled_at.strftime("%Y-%m-%dT12:00:00"),
                        "event_type": "voluntary_cancel",
                        "from_plan_id": plan_id_lookup[(plan_name, "new" if post else "old")],
                        "to_plan_id": "",
                        "cancel_reason_code": "too_expensive" if post else "other",
                        "country_at_event": "US",
                        "device_at_event": "web",
                    }
                )
                break
            sessions = rng.poisson(max(row["engagement"], 0.2))
            for s in range(sessions):
                start = week + pd.Timedelta(days=int(rng.integers(0, 7)), hours=int(rng.integers(0, 24)))
                duration = int(rng.gamma(2.0, 1200.0))
                playback_rows.append(
                    {
                        "playback_id": f"PB{idx:06d}{week.strftime('%j')}{s:02d}",
                        "member_id": member_id,
                        "profile_id": f"{member_id}_p1",
                        "title_id": "T100000",
                        "device_type": "web",
                        "session_start_utc": start.strftime("%Y-%m-%dT%H:%M:%S"),
                        "session_end_utc": (start + pd.Timedelta(seconds=duration)).strftime("%Y-%m-%dT%H:%M:%S"),
                        "duration_seconds": duration,
                        "bytes_streamed": duration * 500000,
                        "avg_bitrate_kbps": 5000,
                        "rebuffer_count": int(rng.poisson(0.1)),
                        "startup_delay_ms": int(rng.normal(2500, 500)),
                        "completion_pct": float(np.clip(rng.normal(65, 20), 0, 100)),
                        "country_at_play": "US",
                        "network_type": "wifi",
                    }
                )

        bill_date = pd.Timestamp("2024-11-01") + pd.Timedelta(days=int(row["anchor_day"]) - 1)
        while bill_date <= pd.Timestamp("2025-06-30"):
            if cancelled_at is not None and bill_date > cancelled_at:
                break
            is_new = bill_date >= exposure
            price = PLAN_PRICES[plan_name][1 if is_new else 0]
            billing_rows.append(
                {
                    "member_id": member_id,
                    "charge_ts_utc": bill_date.strftime("%Y-%m-%dT08:00:00"),
                    "billing_period_start": bill_date.date().isoformat(),
                    "billing_period_end": (bill_date + pd.DateOffset(months=1) - pd.Timedelta(days=1)).date().isoformat(),
                    "amount_local": price,
                    "currency": "USD",
                    "amount_usd": price,
                    "plan_id": plan_id_lookup[(plan_name, "new" if is_new else "old")],
                    "payment_partner": "direct_card",
                    "status": "success",
                    "retry_attempt_num": 1,
                    "loaded_ts": bill_date.strftime("%Y-%m-%dT08:10:00"),
                }
            )
            if rng.random() < 0.01:
                billing_rows.append(
                    {
                        "member_id": member_id,
                        "charge_ts_utc": bill_date.strftime("%Y-%m-%dT07:55:00"),
                        "billing_period_start": bill_date.date().isoformat(),
                        "billing_period_end": (bill_date + pd.DateOffset(months=1) - pd.Timedelta(days=1)).date().isoformat(),
                        "amount_local": price,
                        "currency": "USD",
                        "amount_usd": price,
                        "plan_id": plan_id_lookup[(plan_name, "new" if is_new else "old")],
                        "payment_partner": "direct_card",
                        "status": "failed",
                        "retry_attempt_num": 1,
                        "loaded_ts": bill_date.strftime("%Y-%m-%dT08:00:00"),
                    }
                )
            bill_date = bill_date + pd.DateOffset(months=1)

    fact_billing = pd.DataFrame(billing_rows)
    fact_events = pd.DataFrame(event_rows)
    fact_playback = pd.DataFrame(playback_rows)

    dim_member.to_csv(out / "dim_member.csv", index=False)
    dim_plan.to_csv(out / "dim_plan.csv", index=False)
    fact_billing.to_csv(out / "fact_billing_events.csv", index=False)
    fact_events.to_csv(out / "fact_member_events.csv", index=False)
    campaign.to_csv(out / "fact_price_change_campaign.csv", index=False)
    fact_playback.to_csv(out / "fact_playback_session.csv", index=False)

    print(f"Wrote toy data to {out}")
    return 0


def week_start(value: pd.Timestamp) -> pd.Timestamp:
    value = pd.Timestamp(value)
    return value - pd.Timedelta(days=value.weekday())


if __name__ == "__main__":
    raise SystemExit(main())