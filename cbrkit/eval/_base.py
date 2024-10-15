import statistics
import warnings
from dataclasses import dataclass

import ranx

# https://amenra.github.io/ranx/metrics/


@dataclass(slots=True, frozen=True)
class Base:
    qrels: ranx.Qrels
    run: ranx.Run
    metrics: list[str] = [
        "precision",
        "recall",
        "f1",
        "map",
        "ndcg",
    ]

    def compute_metrics(self) -> dict[str, float]:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            eval_results = ranx.evaluate(
                self.qrels,
                self.run,
                metrics=self.metrics,
            )

        assert isinstance(eval_results, dict)

        correctness, completeness = self.correctness_completeness()

        eval_results["correctness"] = correctness
        eval_results["completeness"] = completeness

        return eval_results

    def correctness_completeness(self, k: int | None = None) -> tuple[float, float]:
        keys = set(self.qrels.keys()).intersection(set(self.run.keys()))

        scores = [self._correctness_completeness(key, k) for key in keys]
        correctness_scores = [score[0] for score in scores]
        completeness_scores = [score[1] for score in scores]

        try:
            return statistics.mean(correctness_scores), statistics.mean(
                completeness_scores
            )
        except statistics.StatisticsError:
            return float("nan"), float("nan")

    def _correctness_completeness(
        self, key: str, k: int | None = None
    ) -> tuple[float, float]:
        qrel = self.qrels[key]
        sorted_run = sorted(self.run[key].items(), key=lambda x: x[1], reverse=True)
        run_ranking = {x[0]: i + 1 for i, x in enumerate(sorted_run[:k])}

        orders = 0
        concordances = 0
        disconcordances = 0

        correctness = 1
        completeness = 1

        for user_key_1, user_rank_1 in qrel.items():
            for user_key_2, user_rank_2 in qrel.items():
                if user_key_1 != user_key_2 and user_rank_1 > user_rank_2:
                    orders += 1

                    system_rank_1 = run_ranking.get(user_key_1)
                    system_rank_2 = run_ranking.get(user_key_2)

                    if system_rank_1 is not None and system_rank_2 is not None:
                        if system_rank_1 > system_rank_2:
                            concordances += 1
                        elif system_rank_1 < system_rank_2:
                            disconcordances += 1

        if concordances + disconcordances > 0:
            correctness = (concordances - disconcordances) / (
                concordances + disconcordances
            )
        if orders > 0:
            completeness = (concordances + disconcordances) / orders

        return correctness, completeness
