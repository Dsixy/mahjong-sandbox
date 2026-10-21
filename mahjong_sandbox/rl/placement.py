"""Pure placement utility, independent of raw points and episode duration."""
import math

DEFAULT_RANK_REWARDS = (1.5, .5, -.5, -1.5)


def rank_rewards(values):
    try:
        result = tuple(float(x) for x in values)
    except (TypeError, ValueError) as error:
        raise ValueError('four finite, strictly decreasing rank rewards required') from error
    if len(result) != 4 or not all(math.isfinite(x) for x in result) or not all(a > b for a, b in zip(result, result[1:])):
        raise ValueError('four finite, strictly decreasing rank rewards required')
    return result


def ranking(scores, dealer=0):
    return sorted(range(4), key=lambda p: (-int(scores[p]), (p - dealer) % 4))


def validation_objective(stats, config):
    return stats['rank_utility'] if config.reward_mode == 'rank' else stats['mean_net_score']
