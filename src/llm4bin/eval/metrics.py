"""Classification metrics for the vuln task.

Math identical to the legacy ``utils/eval_result_analyzer.get_metrics`` but
operates on already-gathered ``true_labels`` / ``pred_labels`` lists (Accelerate
handles cross-process aggregation) instead of pkl files written per rank.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List


C_LABELS = {
    "0": "good", "1": "CWE121", "2": "CWE78",
    "3": "CWE191", "4": "CWE789", "5": "CWE23",
}

JAVA_LABELS = {
    "0": "no vuln", "1": "CWE78", "2": "CWE23",
    "3": "CWE36", "4": "CWE129", "5": "CWE190",
    "6": "CWE191", "7": "CWE789", "8": "CWE134",
}


def get_labels_dict(code_type: str) -> Dict[str, str]:
    if code_type == "C":
        return C_LABELS
    if code_type == "Java":
        return JAVA_LABELS
    raise ValueError(f"unknown code_type: {code_type}")


def compute_metrics(true_labels: List[str], pred_labels: List[str], code_type: str) -> Dict:
    if len(true_labels) != len(pred_labels):
        raise ValueError(f"length mismatch: {len(true_labels)} vs {len(pred_labels)}")

    cwe_labels = get_labels_dict(code_type)
    type_nums = len(cwe_labels)

    metrics = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0, "TN": 0})
    class_counts: Dict[str, int] = defaultdict(int)

    for true, pred in zip(true_labels, pred_labels):
        class_counts[true] += 1
        for class_label in range(type_nums):
            if true == str(class_label):
                if pred == str(class_label):
                    metrics[class_label]["TP"] += 1
                else:
                    metrics[class_label]["FN"] += 1
            else:
                if pred == str(class_label):
                    metrics[class_label]["FP"] += 1
                else:
                    metrics[class_label]["TN"] += 1

    precision, recall, f1, support, acc_per_class = {}, {}, {}, {}, {}
    for c in range(type_nums):
        TP, FP, FN, TN = metrics[c]["TP"], metrics[c]["FP"], metrics[c]["FN"], metrics[c]["TN"]
        precision[c] = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall[c] = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        denom = precision[c] + recall[c]
        f1[c] = 2 * precision[c] * recall[c] / denom if denom > 0 else 0.0
        support[c] = class_counts[str(c)]
        total = TP + FP + FN + TN
        acc_per_class[c] = (TP + TN) / total if total > 0 else 0.0

    overall_accuracy = sum(metrics[c]["TP"] for c in range(type_nums)) / max(1, len(true_labels))
    macro_f1 = sum(f1.values()) / type_nums

    return {
        "code_type": code_type,
        "num_samples": len(true_labels),
        "overall_accuracy": overall_accuracy,
        "macro_f1": macro_f1,
        "per_class": {
            str(c): {
                "name": cwe_labels[str(c)],
                "precision": precision[c],
                "recall": recall[c],
                "f1": f1[c],
                "accuracy": acc_per_class[c],
                "support": support[c],
            }
            for c in range(type_nums)
        },
    }


def format_metrics(m: Dict) -> str:
    lines = []
    for c, stats in m["per_class"].items():
        lines.append(
            f"Class {c} ({stats['name']}): P={stats['precision']:.4f} "
            f"R={stats['recall']:.4f} F1={stats['f1']:.4f} "
            f"Acc={stats['accuracy']:.4f} Support={stats['support']}"
        )
    lines.append("---")
    lines.append(f"Overall Accuracy: {m['overall_accuracy']:.4f}")
    lines.append(f"Macro F1:         {m['macro_f1']:.4f}")
    lines.append(f"Total Samples:    {m['num_samples']}")
    return "\n".join(lines)
