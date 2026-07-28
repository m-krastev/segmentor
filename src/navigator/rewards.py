"""Small, dependency-free reward helpers for Navigator.

The helpers here deliberately make every positive signal finite:

* geodesic progress is a state-potential difference, so a round trip earns zero;
* coverage shaping is the signed change in the evaluation Dice score; and
* terminal success requires both the endpoint and meaningful path coverage.
"""


def is_path_success(reached_goal: bool, coverage: float, threshold: float) -> bool:
    """Return whether an episode solved the path-tracing task."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    bounded_coverage = min(max(float(coverage), 0.0), 1.0)
    return bool(reached_goal and bounded_coverage >= threshold)


def terminal_path_reward(
    coverage: float,
    reached_goal: bool,
    threshold: float,
    success_scale: float,
    failure_penalty: float,
) -> float:
    """Score terminal path quality with a strict success/failure separation.

    ``failure_penalty`` is chosen by the environment to cover the upper bounds
    of every positive shaping term. Consequently a failed episode cannot finish
    with positive return even after covering the entire target.
    """
    if success_scale < 0:
        raise ValueError("success_scale must be non-negative")
    if failure_penalty <= 0:
        raise ValueError("failure_penalty must be positive")
    bounded_coverage = min(max(float(coverage), 0.0), 1.0)
    if is_path_success(reached_goal, bounded_coverage, threshold):
        return float(success_scale) * bounded_coverage
    return -float(failure_penalty)


def coverage_potential_reward(
    previous_coverage: float,
    next_coverage: float,
    scale: float,
) -> float:
    """Return the signed change in the exact bounded Dice objective.

    These rewards telescope over an episode to
    ``scale * (final_coverage - initial_coverage)``. Repeats earn zero, while
    path changes that damage Dice receive the corresponding negative reward.
    """
    if scale < 0:
        raise ValueError("scale must be non-negative")
    bounded_previous = min(max(float(previous_coverage), 0.0), 1.0)
    bounded_next = min(max(float(next_coverage), 0.0), 1.0)
    return float(scale) * (bounded_next - bounded_previous)


def gdt_progress_reward(delta: float, maximum_delta: float, scale: float) -> float:
    """Return a signed, bounded reward for geodesic progress toward the goal."""
    if maximum_delta <= 0:
        raise ValueError("maximum_delta must be positive")
    if abs(delta) > maximum_delta:
        return -float(scale)
    return float(scale) * float(delta) / float(maximum_delta)


def target_recovery_potential_reward(
    previous_distance_mm: float,
    next_distance_mm: float,
    maximum_step_mm: float,
    scale: float,
) -> float:
    """Return a signed distance-to-target potential difference.

    A leave/re-enter excursion telescopes to zero. This supplies an off-target
    recovery gradient without changing which action the environment executes.
    """
    if previous_distance_mm < 0 or next_distance_mm < 0:
        raise ValueError("target distances must be non-negative")
    if maximum_step_mm <= 0:
        raise ValueError("maximum_step_mm must be positive")
    if scale < 0:
        raise ValueError("scale must be non-negative")
    return (
        float(scale)
        * (float(previous_distance_mm) - float(next_distance_mm))
        / float(maximum_step_mm)
    )


def target_distance_state_penalty(
    distance_mm: float,
    radius_mm: float,
    scale: float,
) -> float:
    """Return a bounded persistent cost for being away from the target.

    Unlike a potential difference, this remains negative while the agent moves
    tangentially or stays far from the target. ``radius_mm`` is the distance at
    which the penalty reaches its finite maximum.
    """
    if distance_mm < 0:
        raise ValueError("distance_mm must be non-negative")
    if radius_mm <= 0:
        raise ValueError("radius_mm must be positive")
    if scale < 0:
        raise ValueError("scale must be non-negative")
    return -float(scale) * min(float(distance_mm) / float(radius_mm), 1.0)


def terminal_outcome_reward(
    reached_goal: bool,
    coverage: float,
    threshold: float,
    success_bonus: float,
    failure_penalty: float,
) -> float:
    """Return a fixed success bonus or an explicit finite failure penalty."""
    if success_bonus < 0:
        raise ValueError("success_bonus must be non-negative")
    if failure_penalty < 0:
        raise ValueError("failure_penalty must be non-negative")
    if is_path_success(reached_goal, coverage, threshold):
        return float(success_bonus)
    return -float(failure_penalty)
