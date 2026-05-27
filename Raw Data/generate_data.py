"""
Synthetic data generator for the Google Fit goal-setting study.

Study design:
  - 100 participants, ages 25-50
  - 60-day pre-goal baseline window
  - 90-day post-goal window
  - 70% of participants set a fitness goal (goal-setters)
  - 30% never set a goal (control group)

Outputs (written to data/):
  participants.csv     – one row per participant (demographics, goal info)
  daily_metrics.csv    – one row per participant per day (150 days × 100 = 15 000 rows)
  weekly_summary.csv   – aggregated weekly averages for trend analysis
"""

import os
import numpy as np
import pandas as pd
from datetime import date, timedelta

# ── Reproducibility ────────────────────────────────────────────────────────
SEED = 42
rng = np.random.default_rng(SEED)

# ── Study parameters ───────────────────────────────────────────────────────
N            = 100
PRE_DAYS     = 60
POST_DAYS    = 90
TOTAL_DAYS   = PRE_DAYS + POST_DAYS       # day offsets: -60 … +89
GOAL_SET_PCT = 0.70
AGE_MIN, AGE_MAX = 25, 50

GOAL_DATE = date(2024, 3, 1)              # anchor: day 0 = goal-setting date

os.makedirs("data", exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────
# 1. PARTICIPANTS
# ──────────────────────────────────────────────────────────────────────────
pids      = [f"P{i+1:03d}" for i in range(N)]
ages      = rng.integers(AGE_MIN, AGE_MAX + 1, N)
genders   = rng.choice(["Male", "Female"], N, p=[0.48, 0.52])

# Fitness level drives baseline activity values
fitness   = rng.choice(["Low", "Moderate", "High"], N, p=[0.30, 0.50, 0.20])

# Assign goal-setter status (70 %)
goal_mask = rng.random(N) < GOAL_SET_PCT

GOAL_TYPES = ["Daily Steps", "Active Minutes", "Calories Burned", "Distance (km)"]
goal_type  = np.where(goal_mask, rng.choice(GOAL_TYPES, N), "None")

# Goal values vary by type and fitness level
def _goal_value(gt, fl):
    if gt == "None":
        return np.nan
    targets = {
        "Daily Steps":       {"Low": 8000,  "Moderate": 10000, "High": 12000},
        "Active Minutes":    {"Low": 30,    "Moderate": 45,    "High": 60},
        "Calories Burned":   {"Low": 2000,  "Moderate": 2400,  "High": 2800},
        "Distance (km)":     {"Low": 5.0,   "Moderate": 7.0,   "High": 9.0},
    }
    base = targets[gt][fl]
    return round(base * rng.uniform(0.9, 1.1), 1)

goal_value = np.array([_goal_value(gt, fl) for gt, fl in zip(goal_type, fitness)])

# Height / weight / BMI
height_m = np.where(
    genders == "Male",
    rng.normal(1.76, 0.07, N),
    rng.normal(1.63, 0.06, N),
).clip(1.50, 1.95)

bmi_base = np.where(
    fitness == "Low",    rng.normal(27.5, 2.5, N),
    np.where(
        fitness == "Moderate", rng.normal(24.5, 2.0, N),
        rng.normal(22.0, 1.5, N)
    )
).clip(18.5, 38.0)

weight_base = (bmi_base * height_m ** 2).round(1)

participants = pd.DataFrame({
    "participant_id": pids,
    "age":            ages,
    "gender":         genders,
    "height_m":       height_m.round(2),
    "baseline_weight_kg": weight_base,
    "baseline_bmi":   bmi_base.round(1),
    "fitness_level":  fitness,
    "goal_set":       goal_mask,
    "goal_type":      goal_type,
    "goal_value":     goal_value,
})

participants.to_csv("data/participants.csv", index=False)
print(f"participants.csv  →  {len(participants)} rows")


# ──────────────────────────────────────────────────────────────────────────
# 2. DAILY METRICS
# ──────────────────────────────────────────────────────────────────────────

# Baseline values by fitness level
BASELINES = {
    "Low":      {"steps": 4800,  "active_min": 22, "calories": 1850, "hr": 77, "sleep": 6.5},
    "Moderate": {"steps": 7200,  "active_min": 38, "calories": 2200, "hr": 70, "sleep": 7.1},
    "High":     {"steps": 10000, "active_min": 58, "calories": 2600, "hr": 62, "sleep": 7.5},
}

# Improvement magnitudes for goal-setters (applied post-goal)
IMPROVEMENTS = {
    "steps":      (0.15, 0.28),   # +15–28 % of baseline
    "active_min": (0.20, 0.35),
    "calories":   (0.08, 0.16),
    "hr":         (2.0,  6.0),    # absolute BPM reduction
    "sleep":      (0.10, 0.35),   # extra hours
    "weight":     (0.8,  2.5),    # kg lost by end of post-period
}

records = []

for idx, p in participants.iterrows():
    pid      = p["participant_id"]
    fl       = p["fitness_level"]
    is_gs    = p["goal_set"]
    w0       = p["baseline_weight_kg"]
    bl       = BASELINES[fl]

    # Per-participant noise multipliers (simulate individual variability)
    ind_var = rng.normal(1.0, 0.08)          # overall activity level offset

    # Improvement amounts drawn once per participant
    impr = {k: rng.uniform(*v) for k, v in IMPROVEMENTS.items()}

    for day_offset in range(-PRE_DAYS, POST_DAYS):
        current_date = GOAL_DATE + timedelta(days=day_offset)
        is_post = day_offset >= 0
        is_weekend = current_date.weekday() >= 5

        # Gradual ramp-up over first 30 post-goal days, then plateau
        ramp = 0.0
        if is_post and is_gs:
            ramp = min(day_offset / 30.0, 1.0)
            # Slight late-period decay (habit fading) after day 60
            if day_offset > 60:
                fade = (day_offset - 60) / POST_DAYS * 0.15
                ramp = max(ramp - fade, 0.6)

        # Occasional sick / rest day (2 % chance)
        sick = rng.random() < 0.02

        # ── Steps ──
        step_boost = ramp * impr["steps"]
        base_s = bl["steps"] * ind_var * (1 + step_boost)
        if is_weekend:
            base_s *= rng.uniform(0.85, 1.15)   # weekend variability
        steps = int(rng.normal(base_s, base_s * 0.15))
        if sick:
            steps = int(steps * rng.uniform(0.15, 0.35))
        steps = max(0, steps)

        # ── Active minutes ──
        active_boost = ramp * impr["active_min"]
        active_min = int(rng.normal(bl["active_min"] * ind_var * (1 + active_boost), 8))
        if sick:
            active_min = rng.integers(0, 8)
        active_min = max(0, active_min)

        # ── Calories burned ──
        cal_boost = ramp * impr["calories"]
        calories = int(rng.normal(bl["calories"] * ind_var * (1 + cal_boost), 180))
        if sick:
            calories = int(calories * rng.uniform(0.7, 0.85))
        calories = max(1200, calories)

        # ── Distance ──
        distance_km = round(steps * 0.000762, 2)     # avg stride ~76.2 cm

        # ── Resting heart rate ──
        hr_improve = ramp * impr["hr"]
        resting_hr = int(rng.normal(bl["hr"] - hr_improve, 3.5))
        resting_hr = max(45, min(100, resting_hr))

        # ── Sleep ──
        sleep_improve = ramp * impr["sleep"]
        sleep_h = round(rng.normal(bl["sleep"] + sleep_improve, 0.55), 1)
        if sick:
            sleep_h += rng.uniform(0.5, 1.5)
        sleep_h = round(max(4.0, min(10.5, sleep_h)), 1)

        # ── Sleep quality (1–10 Likert) ──
        sq_base = {"Low": 5.5, "Moderate": 6.5, "High": 7.5}[fl]
        sleep_quality = round(
            np.clip(rng.normal(sq_base + ramp * 0.8, 1.0), 1, 10), 1
        )

        # ── Weight (slow drift) ──
        if is_post and is_gs:
            weight_loss = impr["weight"] * ramp * (day_offset / POST_DAYS)
        else:
            weight_loss = 0.0
        weight = round(w0 - weight_loss + rng.normal(0, 0.25), 1)

        # ── Goal achievement flag (goal-setters only) ──
        if is_post and is_gs:
            gt = p["goal_type"]
            gv = p["goal_value"]
            metric_map = {
                "Daily Steps":       steps,
                "Active Minutes":    active_min,
                "Calories Burned":   calories,
                "Distance (km)":     distance_km,
            }
            achieved = int(metric_map.get(gt, 0) >= gv) if not pd.isna(gv) else np.nan
        else:
            achieved = np.nan

        records.append({
            "participant_id":     pid,
            "date":               current_date.isoformat(),
            "day_offset":         day_offset,
            "phase":              "post_goal" if is_post else "pre_goal",
            "is_weekend":         int(is_weekend),
            "steps":              steps,
            "distance_km":        distance_km,
            "active_minutes":     active_min,
            "calories_burned":    calories,
            "resting_heart_rate": resting_hr,
            "sleep_hours":        sleep_h,
            "sleep_quality_1_10": sleep_quality,
            "weight_kg":          weight,
            "goal_achieved":      achieved,
        })

daily = pd.DataFrame(records)
daily.to_csv("data/daily_metrics.csv", index=False)
print(f"daily_metrics.csv →  {len(daily)} rows  ({len(daily.columns)} columns)")


# ──────────────────────────────────────────────────────────────────────────
# 3. WEEKLY SUMMARY  (for trend / visualisation)
# ──────────────────────────────────────────────────────────────────────────
# Tag each day with a study week (-8 … +12)
daily["study_week"] = (daily["day_offset"] // 7).astype(int)

agg_cols = ["steps", "distance_km", "active_minutes", "calories_burned",
            "resting_heart_rate", "sleep_hours", "sleep_quality_1_10", "weight_kg"]

weekly = (
    daily
    .groupby(["participant_id", "study_week", "phase"])[agg_cols]
    .mean()
    .round(2)
    .reset_index()
)

# Merge goal_set flag for easy grouping in analysis
weekly = weekly.merge(
    participants[["participant_id", "goal_set", "fitness_level", "age", "gender"]],
    on="participant_id",
)
weekly.to_csv("data/weekly_summary.csv", index=False)
print(f"weekly_summary.csv →  {len(weekly)} rows")


# ──────────────────────────────────────────────────────────────────────────
# 4. QUICK SANITY CHECK
# ──────────────────────────────────────────────────────────────────────────
print("\n── Sanity check ──────────────────────────────────────────────────")
print(f"Goal-setters : {goal_mask.sum()} / {N}  ({goal_mask.mean()*100:.0f} %)")
print(f"Age range    : {ages.min()} – {ages.max()}")
print(f"Date range   : {daily['date'].min()} → {daily['date'].max()}")

gs_post  = daily[(daily["phase"] == "post_goal") & daily["participant_id"].isin(participants[participants["goal_set"]]["participant_id"])]
ctrl_post = daily[(daily["phase"] == "post_goal") & daily["participant_id"].isin(participants[~participants["goal_set"]]["participant_id"])]
pre_all  = daily[daily["phase"] == "pre_goal"]

print(f"\nMean daily steps (pre-goal, all)    : {pre_all['steps'].mean():,.0f}")
print(f"Mean daily steps (post-goal, goal-setters) : {gs_post['steps'].mean():,.0f}")
print(f"Mean daily steps (post-goal, control)      : {ctrl_post['steps'].mean():,.0f}")

print(f"\nMean resting HR (pre, all)          : {pre_all['resting_heart_rate'].mean():.1f} bpm")
print(f"Mean resting HR (post, goal-setters): {gs_post['resting_heart_rate'].mean():.1f} bpm")
print(f"Mean resting HR (post, control)     : {ctrl_post['resting_heart_rate'].mean():.1f} bpm")
print("──────────────────────────────────────────────────────────────────")
