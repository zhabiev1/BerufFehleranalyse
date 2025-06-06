import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any

def extract_job_name(text: str) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        raise TypeError(f"String erwartet, erhalten: {type(text)}")

    lowered = text.lower()
    if "esco role:" in lowered:
        try:
            result = lowered.split("esco role:")[1].split("\n")[0].strip()
            return result
        except IndexError:
            return ""
    return lowered.strip()

def calculate_metrics(predictions: List[Tuple[str, List[str]]]) -> Dict[str, float]:
    if not predictions:
        raise ValueError("Liste der Vorhersagen darf nicht leer sein.")

    mrr = 0.0
    recall5 = 0.0
    recall10 = 0.0
    total = len(predictions)

    for query, candidates in predictions:
        if not candidates:
            continue

        query_job = extract_job_name(query)
        candidate_jobs = [extract_job_name(c) for c in candidates]

        found_rank = None
        for rank, cand in enumerate(candidate_jobs, start=1):
            if cand == query_job:
                found_rank = rank
                break

        if found_rank:
            mrr += 1.0 / found_rank
            if found_rank <= 5:
                recall5 += 1.0
            if found_rank <= 10:
                recall10 += 1.0

    if total > 0:
        mrr /= total
        recall5 /= total
        recall10 /= total

    return {'MRR': mrr, 'Recall@5': recall5, 'Recall@10': recall10}

def cluster_transition_analysis(predictions: List[Tuple[str, List[str]]], job_to_group: Dict[str, str], job_to_cluster: Dict[str, int]) -> Dict[str, float]:
    if not job_to_group:
        raise ValueError("Wörterbuch job_to_group darf nicht leer sein.")
    if not job_to_cluster:
        raise ValueError("Wörterbuch job_to_cluster darf nicht leer sein.")

    within_group = 0
    cross_group = 0
    within_cluster = 0
    cross_cluster = 0
    error_categories = {"Korrekt": 0, "Gleiche Berufsgruppe": 0, "Gleiche semantische Berufsgruppe": 0, "Falsch": 0}
    total = len(predictions)

    for query, candidates in predictions:
        if not candidates:
            continue

        query_job = extract_job_name(query)
        top_cand = extract_job_name(candidates[0])

        q_group = job_to_group.get(query_job)
        c_group = job_to_group.get(top_cand)
        q_cluster = job_to_cluster.get(query_job)
        c_cluster = job_to_cluster.get(top_cand)

        if q_group is None or c_group is None or q_cluster is None or c_cluster is None:
            continue

        is_correct = (query_job == top_cand)
        same_group = (q_group == c_group)
        same_cluster = (q_cluster == c_cluster)

        if is_correct:
            error_categories["Korrekt"] += 1
        elif same_group:
            error_categories["Gleiche Berufsgruppe"] += 1
        elif same_cluster:
            error_categories["Gleiche semantische Berufsgruppe"] += 1
        else:
            error_categories["Falsch"] += 1

        if same_group:
            within_group += 1
        else:
            cross_group += 1

        if same_cluster:
            within_cluster += 1
        else:
            cross_cluster += 1

    if total > 0:
        result = {
            'within_group_accuracy': within_group / total,
            'cross_group_error': cross_group / total,
            'within_cluster_accuracy': within_cluster / total,
            'cross_cluster_error': cross_cluster / total,
            'error_categories': {k: v / total for k, v in error_categories.items()}
        }
    else:
        result = {
            'within_group_accuracy': 0.0,
            'cross_group_error': 0.0,
            'within_cluster_accuracy': 0.0,
            'cross_cluster_error': 0.0,
            'error_categories': {k: 0.0 for k in error_categories}
        }

    return result

def calculate_diversity(predictions: List[Tuple[str, List[str]]]) -> Dict[str, float]:
    if not predictions:
        raise ValueError("Liste der Vorhersagen darf nicht leer sein.")

    job_counts = {}
    total_count = 0

    for query, candidates in predictions:
        query_job = extract_job_name(query)
        if query_job:
            job_counts[query_job] = job_counts.get(query_job, 0) + 1
            total_count += 1
        if candidates:
            top_candidate = extract_job_name(candidates[0])
            if top_candidate:
                job_counts[top_candidate] = job_counts.get(top_candidate, 0) + 1
                total_count += 1

    if total_count == 0:
        return {"entropy": 0.0, "hhi": 0.0, "unique_titles": 0}

    freq = np.array(list(job_counts.values())) / total_count
    entropy_val = -np.sum(freq * np.log2(freq + 1e-10))
    hhi = np.sum(freq ** 2)
    unique_titles = len(job_counts)

    return {"entropy": entropy_val, "hhi": hhi, "unique_titles": unique_titles}

def calculate_average_group_changes(predictions: List[Tuple[str, List[str]]], esco_mapping: Dict[str, str]) -> Dict[str, float]:
    if not predictions:
        raise ValueError("Liste der Vorhersagen darf nicht leer sein.")
    if not esco_mapping:
        raise ValueError("ESCO-Mapping darf nicht leer sein.")

    jobs = [extract_job_name(query) for query, _ in predictions]
    esco_groups = [esco_mapping.get(job, "Unknown") for job in jobs]

    path_changes = []
    current_path_changes = 0
    for i in range(1, len(esco_groups)):
        if esco_groups[i] != esco_groups[i-1] and esco_groups[i] != "Unknown" and esco_groups[i-1] != "Unknown":
            current_path_changes += 1
        if i == len(esco_groups) - 1 or esco_groups[i] == "Unknown" or esco_groups[i-1] == "Unknown":
            if current_path_changes > 0 or i == len(esco_groups) - 1:
                path_changes.append(current_path_changes)
            current_path_changes = 0

    avg_changes = sum(path_changes) / len(path_changes) if path_changes else 0.0
    return {"average_group_changes": avg_changes}