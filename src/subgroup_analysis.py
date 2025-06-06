import pandas as pd
from metrics import extract_job_name, calculate_metrics
from bias_analysis import calculate_bias
from sentence_transformers import SentenceTransformer, util
import torch
import pickle
import os
from typing import Set, List, Dict, Tuple

WOMEN_JOBS = {
    "child care social worker", "social worker", "domestic housekeeper", "religious education teacher",
    "nurse responsible for general care", "midwife", "medical administrative assistant", "nurse assistant",
    "elderly home manager", "beauty vocational teacher", "physiotherapist",
    "legal administrative assistant", "administrative assistant", "public administration manager",
    "sales assistant", "cashier", "retail department manager", "primary school teacher",
    "hairdresser", "accountant", "secretary", "advanced nurse practitioner", "shop assistant",
    "youth worker", "primary school teaching assistant", "specialised doctor", "specialist nurse",
    "massage therapist", "dental hygienist", "human resources assistant", "legal assistant",
    "domestic cleaner", "home care aide", "nanny", "educational counsellor",
    "early childhood educator", "kindergarten teacher", "occupational therapist", "manicurist",
    "cosmetologist", "nutritionist", "health visitor", "clerical support worker", "laundry worker",
    "florist", "caregiver"
}

MIGRANT_JOBS = {
    "building cleaner", "food production operator", "building construction worker", "hotel porter",
    "logistics analyst", "crop production worker", "truck driving instructor", "construction painter",
    "elderly home manager", "domestic cleaner", "mover", "kitchen porter", "shelf filler",
    "waiter", "bus driver", "car and van delivery driver", "butcher", "civil engineering worker",
    "road construction worker", "chef", "warehouse worker", "livestock worker", "taxi driver",
    "furniture finisher", "tile fitter", "home care aide", "materials handler",
    "rail logistics coordinator", "domestic housekeeper", "bartender", "baker", "animal care attendant",
    "plasterer", "carpenter",
    "dishwasher", "kitchen assistant", "laundry worker", "childcare worker", "nursing assistant",
    "parcel sorter", "forklift operator", "warehouse packer", "fruit picker", "greenhouse worker",
    "floor layer", "roofer", "scaffolder", "glazier", "window cleaner",
    "checkout operator", "shop assistant", "retail shelf stacker", "care assistant", "housekeeping attendant"
}

intersection = WOMEN_JOBS & MIGRANT_JOBS
MIGRANT_JOBS -= intersection

def match_jobs_sbert(job_list: Set[str], esco_jobs: List[str], threshold: float = 0.7, cache_dir: str = 'cache', batch_size: int = 32) -> Set[str]:
    if not job_list:
        raise ValueError("Liste der zuzuordnenden Berufe darf nicht leer sein.")
    if not esco_jobs:
        raise ValueError("Liste der ESCO-Berufe darf nicht leer sein.")

    model = SentenceTransformer('all-MiniLM-L6-v2')

    job_list = [j.lower() for j in job_list]
    esco_jobs_l = [e.lower() for e in esco_jobs]

    os.makedirs(cache_dir, exist_ok=True)
    job_cache_path = os.path.join(cache_dir, 'job_embs.pkl')
    esco_cache_path = os.path.join(cache_dir, 'esco_embs.pkl')

    if os.path.exists(job_cache_path):
        with open(job_cache_path, 'rb') as f:
            job_embs = pickle.load(f)
        if len(job_embs) != len(job_list):
            job_embs = model.encode(job_list, convert_to_tensor=True, batch_size=batch_size)
            with open(job_cache_path, 'wb') as f:
                pickle.dump(job_embs, f)
    else:
        job_embs = model.encode(job_list, convert_to_tensor=True, batch_size=batch_size)
        with open(job_cache_path, 'wb') as f:
            pickle.dump(job_embs, f)

    if os.path.exists(esco_cache_path):
        with open(esco_cache_path, 'rb') as f:
            esco_embs = pickle.load(f)
        if len(esco_embs) != len(esco_jobs_l):
            esco_embs = model.encode(esco_jobs_l, convert_to_tensor=True, show_progress_bar=True, batch_size=batch_size)
            with open(esco_cache_path, 'wb') as f:
                pickle.dump(esco_embs, f)
    else:
        esco_embs = model.encode(esco_jobs_l, convert_to_tensor=True, show_progress_bar=True, batch_size=batch_size)
        with open(esco_cache_path, 'wb') as f:
            pickle.dump(esco_embs, f)

    matched = set()
    for i, job in enumerate(job_list):
        scores = util.cos_sim(job_embs[i], esco_embs)[0]
        max_score, idx = torch.max(scores, dim=0)
        max_score = max_score.item()
        if max_score >= threshold:
            matched_job = esco_jobs_l[idx]
            matched.add(matched_job)
    
    return matched

def analyze_subgroups(predictions_data: Dict[str, List[Tuple[str, List[str]]]], esco_jobs: List[str], job_to_cluster: Dict[str, int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not predictions_data:
        raise ValueError("Wörterbuch der Vorhersagen darf nicht leer sein.")
    if not esco_jobs:
        raise ValueError("Liste der ESCO-Berufe darf nicht leer sein.")
    if not job_to_cluster:
        raise ValueError("Wörterbuch job_to_cluster darf nicht leer sein.")

    women_jobs = match_jobs_sbert(WOMEN_JOBS, esco_jobs)
    migrant_jobs = match_jobs_sbert(MIGRANT_JOBS, esco_jobs)

    metrics_results = []
    bias_results = []
    reference = 'decorte_esco_predictions_linear.pkl'

    if reference not in predictions_data:
        raise KeyError(f"Referenzdatei {reference} fehlt.")

    for name, preds in predictions_data.items():
        if not preds:
            continue

        overall = calculate_metrics(preds)

        w_preds = [p for p in preds if extract_job_name(p[0]) in women_jobs]
        m_preds = [p for p in preds if extract_job_name(p[0]) in migrant_jobs]

        women = calculate_metrics(w_preds) if w_preds else {'MRR': 0.0, 'Recall@5': 0.0, 'Recall@10': 0.0}
        migrant = calculate_metrics(m_preds) if m_preds else {'MRR': 0.0, 'Recall@5': 0.0, 'Recall@10': 0.0}

        metrics_results.append({
            'File': name,
            'Overall_MRR': overall['MRR'],
            'Overall_Recall@5': overall['Recall@5'],
            'Overall_Recall@10': overall['Recall@10'],
            'Women_MRR': women['MRR'],
            'Women_Recall@5': women['Recall@5'],
            'Women_Recall@10': women['Recall@10'],
            'Migrant_MRR': migrant['MRR'],
            'Migrant_Recall@5': migrant['Recall@5'],
            'Migrant_Recall@10': migrant['Recall@10']
        })

        if name != reference:
            obias = calculate_bias(predictions_data[reference], preds)
            wbias = calculate_bias(
                [(q, c) for q, c in predictions_data[reference] if extract_job_name(q) in women_jobs],
                [(q, c) for q, c in preds if extract_job_name(q) in women_jobs]
            )
            mbias = calculate_bias(
                [(q, c) for q, c in predictions_data[reference] if extract_job_name(q) in migrant_jobs],
                [(q, c) for q, c in preds if extract_job_name(q) in migrant_jobs]
            )

            bias_results.append({
                'File': name,
                'Overall_KL_divergence': obias['KL_divergence'],
                'Overall_Chi2_statistic': obias['Chi2_statistic'],
                'Overall_Chi2_p_value': obias['Chi2_p_value'],
                'Women_KL_divergence': wbias['KL_divergence'],
                'Women_Chi2_statistic': wbias['Chi2_statistic'],
                'Women_Chi2_p_value': wbias['Chi2_p_value'],
                'Migrant_KL_divergence': mbias['KL_divergence'],
                'Migrant_Chi2_statistic': mbias['Chi2_statistic'],
                'Migrant_Chi2_p_value': mbias['Chi2_p_value']
            })

    if not metrics_results:
        raise ValueError("Ergebnisse der Untergruppenanalyse sind leer.")

    metrics_df = pd.DataFrame(metrics_results)
    bias_df = pd.DataFrame(bias_results) if bias_results else pd.DataFrame()
    return metrics_df, bias_df