# Hierarchical Classification Rules (25 Hz, Level 1 -> Level 4)

This document is the detailed `25 Hz` profile reference for how classes are separated in the current rule-based pipeline.

- Dataset scope: HDD CAN sessions at `25 Hz` (`137` sessions observed under `data/can_data_25hz`).
- Signal schema (`(T,6)` canonical): `[pedalangle, pedalpressure, steer_angle, steer_speed, speed, yaw]`.
- Units: `speed` in `ft/s`, `yaw` in `deg/s`, `steer_angle` in `deg`, `steer_speed` in `deg/s`.
- Batch CLIs: `cli/run_level1_all_25hz.py`, `cli/run_level2_all_25hz.py`, `cli/run_level3_all_25hz.py`, `cli/run_level4_all_25hz.py`.
- Artifacts output roots: `artifacts/level{1,2,3,4}_25hz/...`.
- Rule logic is the same as the current `10 Hz` design; only sampling-rate-sensitive parameters are rescaled.

---

## How the 25 Hz thresholds were derived

### Principle 1: keep physical thresholds unchanged
The core thresholds are in physical units (`ft/s`, `deg/s`, `deg`, `ft/s^2`, pedal/brake units), so they are carried over from the `10 Hz` profile:

- Level 1 speed thresholds (`v_stopped`, `v_moving`)
- Level 2 yaw/steer thresholds
- Level 3 accel/pedal/brake/yaw-deadband thresholds
- Level 4 SHARP/SMOOTH thresholds (already in seconds and physical units)

### Principle 2: scale only sample-count parameters by sampling rate
Let `r = 25 / 10 = 2.5`.

- Persistence steps (time-based):
  - `steps_25hz = round_half_up(steps_10hz * r)`
- Centered odd median windows (preserve centered time span):
  - `window_25hz = odd_nearest((window_10hz - 1) * r + 1)`
  - This keeps `(window - 1) / fs` approximately constant while preserving odd window size.

Examples:
- windows: `5 -> 11`, `7 -> 17`
- persistence: `3 -> 8`, `8 -> 20`, `10 -> 25`, `6 -> 15`, `5 -> 13`

---

## Level 1: `STOPPED` vs `MOVING`

### Purpose
Partition the full timeline into two mutually exclusive states: `STOPPED` or `MOVING`.

### Inputs and preprocessing
- Primary signal: `speed`.
- Smoothing: centered (non-causal) rolling median:
  - `speed_smooth = median_centered(speed, median_window)` (current `25 Hz`: `median_window=11`).

### Decision rule (hysteresis + hold)
At each timestep:
- If `speed_smooth <= v_stopped` -> candidate `STOPPED` (`v_stopped=0.5`)
- If `speed_smooth >= v_moving` -> candidate `MOVING` (`v_moving=1.0`)
- If `v_stopped < speed_smooth < v_moving` -> no candidate, hold previous state (hysteresis band `(0.5, 1.0)`)

### Persistence and initialization
- State changes only after `persistence_steps` consecutive samples of the candidate state (`persistence_steps=8`).
- On first timestep:
  - If first sample is outside hysteresis band, infer directly from thresholds.
  - If first sample is inside hysteresis band, use `initial_state` (`STOPPED`).

### Values used in current 25 Hz profile
- `fs_hz=25.0`
- `median_window=11`
- `v_stopped=0.5`
- `v_moving=1.0`
- `persistence_steps=8`
- `initial_state=STOPPED`

### Why this works
- Speed is the most direct indicator of movement.
- Hysteresis avoids flicker near zero speed.
- Persistence suppresses one-off sensor spikes.
- Centered median smoothing denoises without causal lag in offline processing.
- Rule is deterministic and easy to retune by sampling rate.

---

## Level 2: MOVING-gated `STRAIGHT` vs `TURNING`

### Purpose
Classify only MOVING timesteps into `STRAIGHT` or `TURNING`, while preserving full-timeline output with explicit `STOPPED`.

### Hard gate from Level 1
- If `moving_state[t] == 0`:
  - force `turn_state[t] = STOPPED`
  - reset persistence counters
  - reset moving substate to `initial_turn_state`

### Inputs and preprocessing
- `yaw_abs_smooth = median_centered(abs(yaw), yaw_median_window)` (`yaw_median_window=11`)
- `steer_abs = abs(steer_angle)`
- `steer_speed_abs = abs(steer_speed)`
- `speed`

### Candidate generation (strict precedence order)
For each MOVING timestep:
1. If `yaw_abs_smooth >= yaw_on` -> candidate `TURNING` (`yaw_on=2.2`)
2. Else if `yaw_abs_smooth <= yaw_off` and `steer_abs <= straight_steer_max_deg` -> candidate `STRAIGHT` (`yaw_off=1.0`, `straight_steer_max_deg=10.0`)
3. Else if `steer_abs >= steer_on_deg` and `steer_speed_abs >= steer_speed_on_dps` and `speed >= assist_speed_min` -> candidate `TURNING` (steering-assist condition; `steer_on_deg=35.0`, `steer_speed_on_dps=80.0`, `assist_speed_min=2.0`)
4. Else -> no candidate, hold previous moving substate

### Persistence and block initialization
- `STRAIGHT -> TURNING` requires `persistence_on_steps` consecutive TURNING candidates (`20`).
- `TURNING -> STRAIGHT` requires `persistence_off_steps` consecutive STRAIGHT candidates (`25`).
- Every new MOVING block starts from `initial_turn_state` (`STRAIGHT`).

### Values used in current 25 Hz profile
- `fs_hz=25.0`
- `yaw_median_window=11`
- `yaw_on=2.2`, `yaw_off=1.0`
- `steer_on_deg=35.0`, `steer_speed_on_dps=80.0`, `assist_speed_min=2.0`
- `straight_steer_max_deg=10.0`
- `persistence_on_steps=20`, `persistence_off_steps=25`
- `initial_turn_state=STRAIGHT`

### Why this works
- Yaw magnitude is the primary kinematic indicator for turning.
- ON/OFF hysteresis stabilizes labels near boundary regions.
- Steering-assist catches turns when yaw is weak/delayed.
- Asymmetric persistence damps flicker while preserving maneuver blocks.
- MOVING gate enforces hierarchical consistency with Level-1.

---

## Level 3: Fine classes inside `STRAIGHT` and `TURNING`

### Purpose
Split Level-2 states into finer, non-contradictory classes on the full timeline:
- `MOVING+STRAIGHT` -> `ACCELERATING` / `DECELERATING` / `CRUISING`
- `MOVING+TURNING` -> `LEFT` / `RIGHT`
- `STOPPED` remains explicit as `[(STOPPED)]`

### Level 3A (Longitudinal): `ACCELERATING` / `DECELERATING` / `CRUISING`

Applies only when `turn_state == STRAIGHT` and `moving_state == MOVING`.

#### Inputs and preprocessing
- `speed_smooth = median_centered(speed, speed_median_window)` (`speed_median_window=17`)
- `accel_raw = d(speed_smooth)/dt` using central difference
- `accel = median_centered(accel_raw, accel_median_window)` (`accel_median_window=11`)

#### Candidate generation (strict precedence order)
1. `DECELERATING` if `accel <= -a_on` or `pedalpressure >= brake_on` (`a_on=1.2`, `brake_on=150.0`)
2. Else `ACCELERATING` if `accel >= a_on` or (`pedalangle >= pedal_on` and `pedalpressure < brake_on`) (`pedal_on=12.0`)
3. Else `CRUISING` if `abs(accel) <= a_off` and `pedalangle <= pedal_cruise_max` and `pedalpressure <= brake_cruise_max` (`a_off=0.6`, `pedal_cruise_max=8.0`, `brake_cruise_max=100.0`)
4. Else hold previous longitudinal state

#### Persistence and block initialization
- Any longitudinal class switch requires `lon_persistence_steps` consecutive candidate samples (`15`).
- Each new STRAIGHT block starts from `initial_lon_state` (`CRUISING`).

#### Values used in current 25 Hz profile
- `fs_hz=25.0`
- `speed_median_window=17`, `accel_median_window=11`
- `a_on=1.2`, `a_off=0.6`
- `pedal_on=12.0`, `brake_on=150.0`
- `pedal_cruise_max=8.0`, `brake_cruise_max=100.0`
- `lon_persistence_steps=15`
- `initial_lon_state=CRUISING`

### Level 3B (Lateral Direction): `LEFT` / `RIGHT`

Applies only when `turn_state == TURNING` and `moving_state == MOVING`.

#### Inputs and preprocessing
- `yaw_smooth = median_centered(yaw, yaw_median_window)` (`yaw_median_window=11`)

#### Candidate generation (deadband hold)
- If `yaw_smooth >= +yaw_deadband` -> candidate `RIGHT` (`yaw_deadband=0.8`)
- If `yaw_smooth <= -yaw_deadband` -> candidate `LEFT` (`yaw_deadband=0.8`)
- If `|yaw_smooth| < yaw_deadband` -> no candidate, hold previous direction

#### TURNING-onset bootstrap (if no prior direction in current turning block)
Use first available signed evidence in this order:
1. `sign(steer_angle)` if `abs(steer_angle) >= steer_angle_bootstrap_min` (`5.0`)
2. Else `sign(steer_speed)` if `abs(steer_speed) >= steer_speed_bootstrap_min` (`20.0`)
3. Else `sign(yaw_smooth)` if non-zero
4. Else fallback to `default_turn_direction` (`RIGHT`)

#### Persistence
- `LEFT <-> RIGHT` switching requires `direction_persistence_steps` consecutive opposite candidates (`13`).
- Direction counters reset when leaving TURNING.

#### Values used in current 25 Hz profile
- `yaw_median_window=11`, `yaw_deadband=0.8`
- `steer_angle_bootstrap_min=5.0`
- `steer_speed_bootstrap_min=20.0`
- `direction_persistence_steps=13`
- `default_turn_direction=RIGHT`

### Full-timeline composite mapping
- `STOPPED` -> `[(STOPPED)]`
- `MOVING+STRAIGHT+ACCELERATING` -> `[(MOVING),(STRAIGHT),(ACCELERATING)]`
- `MOVING+STRAIGHT+DECELERATING` -> `[(MOVING),(STRAIGHT),(DECELERATING)]`
- `MOVING+STRAIGHT+CRUISING` -> `[(MOVING),(STRAIGHT),(CRUISING)]`
- `MOVING+TURNING+LEFT` -> `[(MOVING),(TURNING),(LEFT)]`
- `MOVING+TURNING+RIGHT` -> `[(MOVING),(TURNING),(RIGHT)]`

### Why this works
- Motion is decomposed into orthogonal branches (longitudinal and lateral).
- Acceleration is stabilized by speed smoothing + derivative smoothing and reinforced by pedal/brake signals.
- Deadband + persistence prevents LEFT/RIGHT sign flicker near zero yaw.
- Bootstrap guarantees deterministic turn direction at onset.
- Strict gating from Levels 1 and 2 prevents contradictory labels.

---

## Level 4: TURNING subtype classification (`SHARP` / `SMOOTH`)

### Purpose
Refine Level-3 TURNING segments into:
- `SHARP LEFT`
- `SHARP RIGHT`
- `SMOOTH LEFT`
- `SMOOTH RIGHT`

Direction (`LEFT` / `RIGHT`) is inherited from Level-3. Level-4 decides only subtype (`SHARP` vs `SMOOTH`).

### Hard gate from Level 3
Level-4 runs only on Level-3 TURNING segments:
- `[(MOVING),(TURNING),(LEFT)]`
- `[(MOVING),(TURNING),(RIGHT)]`

All non-TURNING Level-3 segments are passed through unchanged.

### Core event features (event-local only)
Computed on each Level-3 TURNING segment window:
- `duration_sec`
- `peak_abs_yaw`
- `mean_abs_yaw`
- `peak_abs_steer_angle`
- `peak_abs_steer_speed`
- `speed_mean`

### SHARP vs SMOOTH rule (conservative, smooth-by-default)
`SHARP` is assigned only when event evidence is strong and corner-like.

#### SHARP hard guards
- `duration_sec >= min_sharp_duration_sec` (`2.0 s`)
- Long-duration guard:
  - if `duration_sec >= long_duration_guard_sec` (`12.0 s`)
  - require `mean_abs_yaw >= long_duration_mean_abs_yaw_min` (`12.0 deg/s`)
  - otherwise force `SMOOTH`

#### SHARP votes (4 metrics)
One vote for each condition:
- `mean_abs_yaw >= sharp_mean_abs_yaw_min` (`10.0`)
- `peak_abs_yaw >= sharp_peak_abs_yaw_min` (`20.0`)
- `peak_abs_steer_angle >= sharp_peak_abs_steer_angle_min` (`180.0`)
- `peak_abs_steer_speed >= sharp_peak_abs_steer_speed_min` (`240.0`)

Decision:
- `SHARP` if total votes `>= sharp_vote_threshold` (`3`)
- and at least one yaw vote + one steering vote (`require_yaw_vote=True`, `require_steer_vote=True`)
- otherwise `SMOOTH`

### Values used in current 25 Hz profile
- `fs_hz=25.0`
- `min_sharp_duration_sec=2.0`
- `long_duration_guard_sec=12.0`
- `long_duration_mean_abs_yaw_min=12.0`
- `sharp_mean_abs_yaw_min=10.0`
- `sharp_peak_abs_yaw_min=20.0`
- `sharp_peak_abs_steer_angle_min=180.0`
- `sharp_peak_abs_steer_speed_min=240.0`
- `sharp_vote_threshold=3`
- `require_yaw_vote=True`
- `require_steer_vote=True`

### Why this is robust
- Event-local metrics only (no context dependency).
- Smooth-by-default behavior reduces false SHARP labels.
- Multi-metric voting avoids reliance on a single noisy signal.
- Long-duration guard reduces highway-curve false positives.

### Full-timeline Level-4 composite mapping
- `[(MOVING),(TURNING),(LEFT)] + (SHARP LEFT)` -> `[(MOVING),(TURNING),(LEFT),(SHARP LEFT)]`
- `[(MOVING),(TURNING),(RIGHT)] + (SHARP RIGHT)` -> `[(MOVING),(TURNING),(RIGHT),(SHARP RIGHT)]`
- `[(MOVING),(TURNING),(LEFT)] + (SMOOTH LEFT)` -> `[(MOVING),(TURNING),(LEFT),(SMOOTH LEFT)]`
- `[(MOVING),(TURNING),(RIGHT)] + (SMOOTH RIGHT)` -> `[(MOVING),(TURNING),(RIGHT),(SMOOTH RIGHT)]`

---
