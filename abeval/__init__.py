"""abeval: A/B-test statistics for LLM evals.

Error bars for a single run, paired significance tests for comparing two
runs, sample-size planning before you spend on inference, and judge-noise
reliability. Standard library only.
"""

from .ci import Interval, clustered_mean_ci, mean_ci, proportion_ci
from .compare import Comparison, mcnemar_exact, paired_compare
from .data import extract, pair, read_jsonl
from .power import mde, power_simulated, sample_size, sd_diff_from_rates
from .reliability import Reliability, judge_reliability

__version__ = "0.1.0"

__all__ = [
    "Comparison",
    "Interval",
    "Reliability",
    "__version__",
    "clustered_mean_ci",
    "extract",
    "judge_reliability",
    "mcnemar_exact",
    "mde",
    "mean_ci",
    "pair",
    "paired_compare",
    "power_simulated",
    "proportion_ci",
    "read_jsonl",
    "sample_size",
    "sd_diff_from_rates",
]
