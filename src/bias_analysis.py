from scipy.stats import entropy, chisquare
import numpy as np
from metrics import extract_job_name
from typing import List, Tuple, Dict

def get_distribution(predictions: List[Tuple[str, List[str]]]) -> Dict[str, float]:
    if not predictions:
        return {}

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
        return {}

    distribution = {job: count / total_count for job, count in job_counts.items()}
    return distribution

def calculate_bias(reference_predictions: List[Tuple[str, List[str]]], current_predictions: List[Tuple[str, List[str]]]) -> Dict[str, float]:
    if not reference_predictions:
        raise ValueError("Liste der Referenzvorhersagen darf nicht leer sein.")
    if not current_predictions:
        raise ValueError("Liste der aktuellen Vorhersagen darf nicht leer sein.")

    ref_dist = get_distribution(reference_predictions)
    curr_dist = get_distribution(current_predictions)

    if not ref_dist or not curr_dist:
        return {
            'KL_divergence': np.nan,
            'Chi2_statistic': np.nan,
            'Chi2_p_value': np.nan
        }

    common_jobs = set(ref_dist.keys()) & set(curr_dist.keys())
    if not common_jobs:
        return {
            'KL_divergence': np.inf,
            'Chi2_statistic': np.inf,
            'Chi2_p_value': 0.0
        }

    epsilon = 1e-10
    ref_values = np.array([ref_dist[job] for job in common_jobs])
    curr_values = np.array([curr_dist[job] for job in common_jobs])

    ref_values = ref_values + epsilon
    curr_values = curr_values + epsilon

    ref_values = ref_values / ref_values.sum()
    curr_values = curr_values / curr_values.sum()

    kl_div = entropy(ref_values, curr_values)

    try:
        chi2_stat, chi2_pvalue = chisquare(curr_values, f_exp=ref_values)
    except ValueError:
        chi2_stat, chi2_pvalue = np.nan, np.nan

    return {
        'KL_divergence': kl_div,
        'Chi2_statistic': chi2_stat,
        'Chi2_p_value': chi2_pvalue
    }