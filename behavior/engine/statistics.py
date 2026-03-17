import numpy as np
from typing import Dict, Any


class BehavioralEngine:
    def __init__(
        self,
        baseline_data: Dict[str, Any],
        threshold: float = 1.5,
        drift_allowance: float = 0.01,
    ):
        """
        Behavioral drift engine.

        baseline_data must be a dict with shape:
        {
          "baseline": {
            "mean": [v, a, d, o, c, e, a2, n],
            "std": [...],
            "sample_size": int
          },
          "lsm_reference": [...]
        }
        """
        self.threshold = threshold
        self.drift_allowance = drift_allowance

        # 8 accumulators for increases (+) and 8 for decreases (-)
        self.cusum_pos = np.zeros(8)
        self.cusum_neg = np.zeros(8)
        self.cusum_score = 0.0  # for compatibility

        try:
            self.baseline_mean = np.array(baseline_data["baseline"]["mean"])
            self.baseline_std = np.array(baseline_data["baseline"]["std"]) + 1e-6
        except KeyError as e:
            raise ValueError(f"Invalid baseline_data structure: missing {e!r}") from e

    def update_and_check(self, current_scores):
        """
        Calculate the z-scores (drift) and update the CUSUM.
        current_scores: dictionary from extractor.get_scores()
        """
        if self.baseline_mean is None:
            return "NO_BASELINE", np.zeros(8)

        # 1. Create the current vector (8 dimensions : V, A, D, O, C, E, A, N) 
        current_vec = np.array([
            current_scores['v'], 
            current_scores['a'], 
            current_scores['d'],
            current_scores['ocean'][0], # Openness
            current_scores['ocean'][1], # Conscientiousness
            current_scores['ocean'][2], # Extraversion
            current_scores['ocean'][3], # Agreeableness
            current_scores['ocean'][4]  # Neuroticism
        ])

        # check the dimension
        if len(current_vec) != len(self.baseline_mean):
            print(f"ERROR: Dimension mismatch! Current: {len(current_vec)}, Baseline: {len(self.baseline_mean)}")
            return "ERROR", np.zeros(8)

        # 2. Calculate the Z-Score (Statistical distance in number of standard deviations)
        z_scores = (current_vec - self.baseline_mean) / self.baseline_std

        # 3. Logic bilateral CUSUM 
        self.cusum_pos = np.maximum(0, self.cusum_pos + (z_scores - self.drift_allowance))
        self.cusum_neg = np.maximum(0, self.cusum_neg + (-z_scores - self.drift_allowance))

        # For visualizer: current max drift
        self.cusum_score = np.max(self.cusum_pos) if np.max(self.cusum_pos) > np.max(self.cusum_neg) else -np.max(self.cusum_neg)

        # 4. DETECTION OF 4 STATES (Statistical Signatures)

        # 1. FATIGUE: "Neutral" Exhaustion (A- and C- / V stable)
        # Triggered when energy and structure drop, while mood remains baseline.
        if (self.cusum_neg[1] > self.threshold and self.cusum_neg[4] > self.threshold and
            self.cusum_neg[0] < self.threshold and self.cusum_pos[0] < self.threshold):
            self._reset_engine()
            return "TRIGGER_FATIGUE", z_scores

        # 2. STALLING: Loss of pleasure and structure (V- and C- / A stable)
        # Triggered when interest and logic fade, but physical energy hasn't collapsed yet.
        elif (self.cusum_neg[0] > self.threshold and self.cusum_neg[4] > self.threshold and 
            self.cusum_neg[1] < self.threshold and self.cusum_pos[1] < self.threshold):
            self._reset_engine()
            return "TRIGGER_STALLING", z_scores

        # 3. COGNITIVE OVERLOAD: Stress and loss of structure (A+ and C- / D stable or low)
        # Triggered when the brain "overheats": high tension combined with collapsing organization.
        elif (self.cusum_pos[1] > self.threshold and self.cusum_neg[4] > self.threshold):
            self._reset_engine()
            return "TRIGGER_COGNITIVE_OVERLOAD", z_scores

        # 4. REACTANCE: Defensive/Aggressive state (D- and A+ / often V-)
        # Triggered when the user feels cornered or loses control, leading to a spike in tension.
        elif (self.cusum_neg[2] > self.threshold and self.cusum_pos[1] > self.threshold):
            self._reset_engine()
            return "TRIGGER_REACTANCE", z_scores
        return "OK", z_scores

    def _reset_engine(self):
        self.cusum_pos = np.zeros(8)
        self.cusum_neg = np.zeros(8)