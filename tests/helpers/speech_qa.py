from __future__ import annotations


class FakeSquadMetric:
    def compute(self, *, predictions, references):
        em_scores = []
        f1_scores = []
        for pred, ref in zip(predictions, references):
            pred_text = str(pred["prediction_text"]).strip().lower()
            gold_answers = [str(x).strip().lower() for x in ref["answers"]["text"]]
            exact = 100.0 if pred_text in gold_answers else 0.0
            em_scores.append(exact)
            f1_scores.append(exact)
        return {
            "exact_match": float(sum(em_scores) / max(1, len(em_scores))),
            "f1": float(sum(f1_scores) / max(1, len(f1_scores))),
        }
