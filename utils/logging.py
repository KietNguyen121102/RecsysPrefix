import os
import csv
import uuid
import time





def write_div_metric_dict_as_rows(csv_path, sample_id, method, k, metric_name, metric_dict,
                              n_voters=None, n_candidates=None, run_id=None, ts=None):

    fieldnames = [
        "run_id", "ts", "sample_id", "method", "k",
        "metric", "group_id", "value",
        "n_voters", "n_candidates"
    ]

    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        for group_id, value in metric_dict.items():
            writer.writerow({
                "run_id": run_id,
                "ts": ts,
                "sample_id": sample_id,
                "method": method,
                "k": k,
                "metric": metric_name,   # "coverage" or "percentage"
                "group_id": int(group_id),
                "value": float(value),
                "n_voters": n_voters,
                "n_candidates": n_candidates,
            })