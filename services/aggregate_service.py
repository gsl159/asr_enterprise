"""ASR multi-model aggregation by pairwise CER consensus."""
from typing import Dict, List, Tuple


class AggregateService:
    @staticmethod
    def _normalize_text(text: str) -> str:
        return "".join(text.split())

    @staticmethod
    def _levenshtein_distance(a: str, b: str) -> int:
        if a == b:
            return 0
        if not a:
            return len(b)
        if not b:
            return len(a)

        prev = list(range(len(b) + 1))
        for i, ch_a in enumerate(a, start=1):
            curr = [i]
            for j, ch_b in enumerate(b, start=1):
                cost = 0 if ch_a == ch_b else 1
                curr.append(min(
                    prev[j] + 1,      # deletion
                    curr[j - 1] + 1,  # insertion
                    prev[j - 1] + cost,  # substitution
                ))
            prev = curr
        return prev[-1]

    def _cer(self, ref: str, hyp: str) -> float:
        ref_text = self._normalize_text(ref)
        hyp_text = self._normalize_text(hyp)
        if not ref_text and not hyp_text:
            return 0.0
        if not ref_text:
            return 1.0
        dist = self._levenshtein_distance(ref_text, hyp_text)
        return dist / max(1, len(ref_text))

    def _symmetric_cer(self, a: str, b: str) -> float:
        return (self._cer(a, b) + self._cer(b, a)) / 2.0

    def aggregate(self, asr_results: List[dict]) -> Tuple[str, float, Dict[str, float]]:
        valid_results = []
        for item in asr_results:
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            error = item.get("error")
            model = item.get("model", "unknown")
            if isinstance(text, str) and text.strip() and error is None:
                valid_results.append({"model": model, "text": text.strip()})

        if not valid_results:
            return "", 1.0, {}

        if len(valid_results) == 1:
            only = valid_results[0]
            return only["text"], 0.0, {only["model"]: 0.0}

        per_model_score: Dict[str, float] = {}
        for idx, candidate in enumerate(valid_results):
            candidate_text = candidate["text"]
            pair_scores = []
            for j, other in enumerate(valid_results):
                if idx == j:
                    continue
                pair_scores.append(self._symmetric_cer(candidate_text, other["text"]))

            avg_score = sum(pair_scores) / max(1, len(pair_scores))
            per_model_score[candidate["model"]] = avg_score

        best_item = min(
            valid_results,
            key=lambda item: (
                per_model_score[item["model"]],
                -len(item["text"]),
                item["model"],
            ),
        )
        best_text = best_item["text"]
        best_score = per_model_score[best_item["model"]]

        rounded_scores = {k: round(v, 6) for k, v in per_model_score.items()}
        return best_text, round(best_score, 6), rounded_scores
