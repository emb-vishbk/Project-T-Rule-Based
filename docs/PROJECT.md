# CAN 10 Hz Dataset Context

This document describes the `data/can_data_10hz/` dataset that was prepared for downstream modeling and maneuver discovery.

## Honda Driving Dataset - Background
- Provenance: Each .npy file in data/can_data_10hz/ corresponds to one driving session from the Honda Research Institute Driving Dataset (HDD). HDD contains 137 sessions totaling ~104 hours of video, with an average session duration of ~45 minutes.
- Where/when it was collected: HDD was recorded in the San Francisco Bay Area (urban/suburban/highway driving), spanning Feb 2017 → Oct 2017, using an instrumented vehicle.
- In this workspace, we primarily use a CAN-only subset exported as one .npy per session, shaped (T,6) at 10 Hz with fixed column order and units, for unsupervised / rule-based maneuver discovery; video/labels are reserved for spot-checking and evaluation.

## What `can_data_10hz_numpy/` Contains

- Root folder: `data/can_data_10hz/`
- File format: one `.npy` file per driving session
- Total files: `137`
- Current layout is flat:
  - `data/can_data_10hz/<session_id>.npy`
  - Example: `can_data_10hz_numpy/201702271017.npy`

Each filename is the HDD session ID (`YYYYMMDDhhmm`).

## Array Contract (Very Important)

Each `.npy` file stores one 2D array:

- shape: `(T, 6)`
- dtype: `float64`
- `T`: number of 10 Hz timesteps in that session.

Column order is fixed and has no header in `.npy`:

1. `pedalangle` (percent, `%`)
2. `pedalpressure` (kPa)
3. `steer_angle` (deg)
4. `steer_speed` (deg/s)
5. `speed` (ft/s)
6. `yaw` (deg/s)

When loading `.npy`, always map columns using this exact order.


## Signal Meaning and Units

- `pedalangle (%)`
  - Accelerator pedal position as a percentage of pedal travel.
- `pedalpressure (kPa)`
  - Brake hydraulic pressure; spikes are expected during hard braking or ABS activity.
- `steer_angle (deg)`
  - Steering wheel angle (not tire-road wheel angle), so several hundred degrees is plausible.
- `steer_speed (deg/s)`
  - Time derivative of steering wheel angle; high magnitude indicates rapid steering input.
- `speed (ft/s)`
  - Vehicle speed in feet per second.
- `yaw (deg/s)`
  - Yaw rate (vehicle rotational speed around vertical axis).

## Dataset-Wide Stats (Current `can_data_10hz_round/`)

- sessions/files: `137`
- total timesteps across all sessions: `3,825,552`
- min `T`: `2,719` (session `201703061033`)
- median `T`: `21,532`
- max `T`: `98,721` (session `201702281017`)

Global channel ranges:

- `pedalangle`: `0.0` to `100.000035`
- `pedalpressure`: `0.0` to `7350.928`
- `steer_angle`: `-480.7` to `473.6`
- `steer_speed`: `-804.1` to `789.5`
- `speed`: `0.0` to `136.998`
- `yaw`: `-59.814453` to `45.507812`

## Known Data Characteristics

- Validation on `can_data_10hz` passed structure/rate/monotonicity checks.
- Value ranges were checked and found physically plausible for the given units.
- 0% NaN across 6 channels. 

## Practical Usage Notes for New Workspace

- Treat each `.npy` as one session matrix with fixed 6-channel schema.
- Do not assume equal `T` across sessions; all processing must be session-wise and robust to variable length.

# Project Approach Details
- hierarchical rule-based maneuver labeling (Levels 1–3)

- High-level idea: We process each driving session independently (one .npy = one HDD session). For each session we produce per-timestep hierarchical labels using simple, robust rules first (hysteresis + persistence), and only later consider deep learning models for finer maneuver subtypes. This creates an interpretable baseline that is complete (covers all timesteps) and stable (avoids jitter).

- Representation: Each session is a multivariate time series x with shape (T, 6) sampled at fs = 25 Hz in this column order: 
[pedalangle, pedalpressure, steer_angle, steer_speed, speed, yaw]. Units follow HDD conventions (notably speed in ft/s, yaw in deg/s).

## Level 1 — STOPPED vs MOVING (total partition)

- Goal: Partition the entire session timeline into exactly two disjoint sets whose union covers all timesteps.

- Rule: Use speed-only hysteresis with memory (and optional persistence):

If speed <= v_stopped → label STOPPED

If speed >= v_moving → label MOVING

Else (in the band v_stopped < speed < v_moving) → hold previous label
Optional persistence: require the condition to hold for N consecutive samples before switching states; while pending, keep previous label.

- Output: moving_state[t] ∈ {0,1} for all timesteps.

## Level 2 — STRAIGHT vs TURNING (on MOVING only)

- Goal: Within MOVING timesteps, separate straight driving vs turning robustly.

- Rule: Use yaw (and optionally steering) with hysteresis + memory:

TURNING if abs(yaw) >= yaw_on

STRAIGHT if abs(yaw) <= yaw_off

Else → hold previous
Optional persistence: require condition to hold for N consecutive samples before switching.

- Output: turn_state[t] ∈ {0,1} defined for MOVING timesteps (and can be set to 0 for STOPPED by convention).

## Level 3 — basic maneuver modes (rule-based baseline; Deep Learning - Modelling - optional later)

- At this level we start with rules (fast + interpretable). Later, if the results are not as desired, we may refine with deep learning methods if needed.

- Level 3A: Longitudinal mode (on MOVING & STRAIGHT)
We classify straight driving into:

ACCELERATING
DECELERATING
CRUISING

- Rule: compute smoothed acceleration a = d(speed)/dt (after smoothing speed over a short window), optionally backed by pedal/brake:

ACCELERATING if a > a_thr or pedalangle > pedal_on
DECELERATING if a < -a_thr or pedalpressure > brake_on
else CRUISING

- Output: lon_state[t] ∈ {ACCELERATING, DECELERATING, CRUISING} for MOVING & STRAIGHT timesteps.

- Level 3B: Turn direction (on MOVING & TURNING)
We classify turning into:

LEFT
RIGHT

- Rule: use sign of yaw with a small deadband + memory to avoid sign flicker:

RIGHT if yaw > yaw_deadband
LEFT if yaw < -yaw_deadband

Else → hold previous direction

- Output: lat_state[t] ∈ {LEFT, RIGHT} for MOVING & TURNING timesteps.

## Segment-level output format (after Level-1 / Level-2 / Level-3)

- Core idea

We do per-timestep labeling first, then compress into segments using run-length encoding (RLE): consecutive timesteps with identical labels become one segment. This yields an interpretable partition of the session timeline into disjoint segments whose union covers the whole session.

- Segment table schema (per session)

For each session, produce a list/table of segments with the following fields:

"segment_id": ,
"timesteps": ,
"duration": "",
"starting_time": "",
"ending_time": "",
"label_vector": [(),(),..]

This table is the primary correctness/debug artifact.

Label vector format (hierarchical tokens)

We represent segment semantics as an ordered list of tokens.

- Rules:

If a segment is STOPPED, it contains only: [(STOPPED)]
If MOVING, it always contains: [(MOVING), (STRAIGHT|TURNING), …]
If MOVING+STRAIGHT, it additionally contains one longitudinal token:
(ACCELERATING) or (DECELERATING) or (CRUISING)
If MOVING+TURNING, it additionally contains one direction token:
(LEFT) or (RIGHT)
No token should contradict another (e.g., no (STOPPED) with (MOVING)).

- Examples:

[(MOVING)(TURNING)(RIGHT)]
[(MOVING)(STRAIGHT)(ACCELERATING)]
[(STOPPED)]



## Note: 
What “after Level-1” means in practice
Even after only Level-1 is implemented, we still output segments, but label vectors are simpler:

- Level-1 segment labels:

[(STOPPED)]
[(MOVING)]

- When Level-2 is added:

[(MOVING)(STRAIGHT)]
[(MOVING)(TURNING)]

- When Level-3 is added:

[(MOVING)(STRAIGHT)(CRUISE/ACCELERATING/DECELERATING)]
[(MOVING)(TURNING)(RIGHT/LEFT)]

straight gets accelerating/decelerating/cruising

turning gets left/right

So the segment table evolves as we progress, but the schema stays constant.


## Correctness expectations

- For each session:

Segments are chronologically ordered
Segments are pairwise disjoint
Segments cover the entire session (segment_0.start_idx == 0, segment_last.end_idx == T)
Every segment has a valid label vector consistent with the current completed level(s)

# When we will use deep learning

- Rules are expected to work strongly for Levels 1–3. We will consider learned encoders + clustering/CPD for finer subtypes that depend on temporal patterns (motifs), e.g. lane changes vs gentle bends, U-turns, roundabouts, turn phases (entry/apex/exit), intensity/style, etc. The learned stage will operate within the already-separated buckets (e.g., within MOVING & TURNING segments).

Rules = robust primitive segmentation + search-space pruning
Learning = pattern discovery/composition over primitive sequences + edge cases
