import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Tuple
from data_loading import load_esco_data, load_predictions
from clustering import assign_berufsgruppe, cluster_esco_industries, get_mappings
from metrics import calculate_metrics, cluster_transition_analysis, calculate_diversity, calculate_average_group_changes, extract_job_name
from subgroup_analysis import analyze_subgroups
import os
from collections import Counter

data_types = {
    'decorte_esco_predictions_linear.pkl': 'real',
    'decorte_predictions_linear.pkl': 'real',
    'karrierewege_predictions_linear.pkl': 'real',
    'karrierewege_cp_predictions_linear.pkl': 'synthetic',
    'karrierewege_occ_predictions_linear.pkl': 'synthetic'
}

def plot_metrics(metrics_df: pd.DataFrame, subgroup_metrics: pd.DataFrame):
    sns.set_style('whitegrid')
    os.makedirs('results/plots', exist_ok=True)

    metrics_df['Data_Type'] = metrics_df['File'].map(data_types)
    subgroup_metrics['Data_Type'] = subgroup_metrics['File'].map(data_types)

    plt.figure(figsize=(12, 6))
    sns.barplot(x='File', y='MRR', hue='Data_Type', data=metrics_df)
    plt.title('Allgemeine Metriken der Vorhersagequalität')
    plt.ylabel('MRR')
    plt.xlabel('Datensatz')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('results/plots/overall_metrics.png')
    plt.close()

    plt.figure(figsize=(12, 6))
    subgroup_metrics_melted = subgroup_metrics.melt(
        id_vars=['File', 'Data_Type'],
        value_vars=['Overall_MRR', 'Women_MRR', 'Migrant_MRR'],
        var_name='Group',
        value_name='MRR'
    )
    sns.barplot(x='File', y='MRR', hue='Data_Type', data=subgroup_metrics_melted)
    plt.title('MRR für Subgruppen')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('results/plots/subgroup_mrr.png')
    plt.close()

def plot_bias(subgroup_bias: pd.DataFrame):
    sns.set_style('whitegrid')
    os.makedirs('results/plots', exist_ok=True)

    subgroup_bias['Data_Type'] = subgroup_bias['File'].map(data_types)

    plt.figure(figsize=(12, 6))
    bias_melted = subgroup_bias.melt(
        id_vars=['File', 'Data_Type'],
        value_vars=['Overall_KL_divergence', 'Women_KL_divergence', 'Migrant_KL_divergence'],
        var_name='Group',
        value_name='KL_divergence'
    )
    sns.barplot(x='File', y='KL_divergence', hue='Data_Type', data=bias_melted)
    plt.title('KL-Divergenz für Subgruppen')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('results/plots/subgroup_bias.png')
    plt.close()

def plot_transitions(transitions_df: pd.DataFrame):
    sns.set_style('whitegrid')
    os.makedirs('results/plots', exist_ok=True)

    transitions_df['Data_Type'] = transitions_df['File'].map(data_types)

    plt.figure(figsize=(12, 6))
    transitions_melted = transitions_df.melt(
        id_vars=['File', 'Data_Type'],
        value_vars=['within_group_accuracy', 'cross_group_error', 'within_cluster_accuracy', 'cross_cluster_error'],
        var_name='Metric',
        value_name='Value'
    )
    sns.barplot(x='File', y='Value', hue='Data_Type', data=transitions_melted)
    plt.title('Metriken der Übergänge')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('results/plots/transitions.png')
    plt.close()

def plot_error_categories(error_categories_df: pd.DataFrame):
    sns.set_style('whitegrid')
    os.makedirs('results/plots', exist_ok=True)

    error_categories_df['Data_Type'] = error_categories_df['File'].map(data_types)

    plt.figure(figsize=(12, 6))
    error_melted = error_categories_df.melt(
        id_vars=['File', 'Data_Type'],
        value_vars=['Korrekt', 'Gleiche Berufsgruppe', 'Gleiche semantische Berufsgruppe', 'Falsch'],
        var_name='Category',
        value_name='Value'
    )
    sns.barplot(x='File', y='Value', hue='Data_Type', data=error_melted)
    plt.title('Fehlerklassifikation')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('results/plots/error_categories.png')
    plt.close()

def plot_cluster_summary(cluster_df: pd.DataFrame):
    sns.set_style('whitegrid')
    os.makedirs('results/plots', exist_ok=True)

    plt.figure(figsize=(10, 6))
    cluster_df['Cluster'] = cluster_df['Cluster'].astype(str)
    sns.barplot(x='Num_Jobs', y='Cluster', hue='Cluster', data=cluster_df, orient='h', legend=False)
    plt.title('Anzahl der Berufe in Clustern')
    plt.xlabel('Anzahl der Berufe')
    plt.ylabel('Cluster')
    plt.tight_layout()
    plt.savefig('results/plots/cluster_summary.png')
    plt.close()

def plot_job_distribution(predictions_data: Dict[str, List[Tuple[str, List[str]]]]):
    sns.set_style('whitegrid')
    os.makedirs('results/plots', exist_ok=True)

    plt.figure(figsize=(12, 6))
    top_n = 10
    plot_data = []
    for name, preds in predictions_data.items():
        job_counts = Counter(extract_job_name(pred[0]) for pred in preds if extract_job_name(pred[0]))
        top_jobs = job_counts.most_common(top_n)
        jobs, counts = zip(*top_jobs) if top_jobs else ([], [])
        plot_data.append({"file": name, "jobs": jobs, "counts": counts})

    if not plot_data:
        return

    job_labels = plot_data[0]["jobs"]
    plot_df = pd.DataFrame({"Job Title": job_labels})
    for data in plot_data:
        counts = [data["counts"][data["jobs"].index(job)] if job in data["jobs"] else 0 for job in job_labels]
        plot_df[data["file"]] = counts

    plot_df.plot(kind="bar", x="Job Title", title="Verteilung der Top-10-Berufstitel")
    plt.ylabel("Häufigkeit")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("results/plots/job_distribution.png")
    plt.close()

def plot_esco_transitions(transitions_df: pd.DataFrame):
    sns.set_style('whitegrid')
    os.makedirs('results/plots', exist_ok=True)

    transitions_df['Data_Type'] = transitions_df['File'].map(data_types)

    plt.figure(figsize=(8, 6))
    sns.barplot(x='File', y='average_group_changes', hue='Data_Type', data=transitions_df)
    plt.title('Durchschnittliche Anzahl der ESCO-Gruppenwechsel')
    plt.ylabel('Durchschnittliche Anzahl der Wechsel')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('results/plots/esco_transitions.png')
    plt.close()

def interpret_results(metrics_df: pd.DataFrame, subgroup_metrics: pd.DataFrame, subgroup_bias: pd.DataFrame):
    metrics_df['Data_Type'] = metrics_df['File'].map(data_types)
    subgroup_metrics['Data_Type'] = subgroup_metrics['File'].map(data_types)
    subgroup_bias['Data_Type'] = subgroup_bias['File'].map(data_types)

    interpretation = ["\n=== Interpretation der Ergebnisse ==="]
    
    real_metrics = metrics_df[metrics_df['Data_Type'] == 'real']
    synthetic_metrics = metrics_df[metrics_df['Data_Type'] == 'synthetic']
    
    if not real_metrics.empty and not synthetic_metrics.empty:
        real_mean_mrr = real_metrics['MRR'].mean()
        synthetic_mean_mrr = synthetic_metrics['MRR'].mean()
        interpretation.append(
            f"Reale Daten ({', '.join(real_metrics['File'].tolist())}) haben einen durchschnittlichen MRR = {real_mean_mrr:.3f}, "
            f"was niedriger ist als bei synthetischen Daten ({', '.join(synthetic_metrics['File'].tolist())}) mit MRR = {synthetic_mean_mrr:.3f}. "
            f"Dies kann auf die größere Vorhersagbarkeit synthetischer Daten aufgrund ihres künstlichen Charakters hinweisen."
        )

    for _, row in subgroup_metrics.iterrows():
        file_name = row['File']
        data_type = data_types.get(file_name, 'unknown')
        women_diff = row['Women_MRR'] - row['Overall_MRR']
        migrant_diff = row['Migrant_MRR'] - row['Overall_MRR']
        if women_diff < -0.05:
            interpretation.append(
                f"Für die Datei {file_name} ({'reale' if data_type == 'real' else 'synthetische'} Daten) "
                f"ist der MRR für Frauen ({row['Women_MRR']:.3f}) deutlich niedriger als der Gesamt-MRR ({row['Overall_MRR']:.3f}), "
                f"was auf eine Verzerrung bei den Vorhersagen für Frauenberufe hinweisen könnte."
            )
        if migrant_diff < -0.05:
            interpretation.append(
                f"Für die Datei {file_name} ({'reale' if data_type == 'real' else 'synthetische'} Daten) "
                f"ist der MRR für Migranten ({row['Migrant_MRR']:.3f}) deutlich niedriger als der Gesamt-MRR ({row['Overall_MRR']:.3f}), "
                f"was auf eine Verzerrung bei den Vorhersagen für Migrantenberufe hinweisen könnte."
            )

    for _, row in subgroup_bias.iterrows():
        file_name = row['File']
        data_type = data_types.get(file_name, 'unknown')
        if row['Women_KL_divergence'] > 1:
            interpretation.append(
                f"Für die Datei {file_name} ({'reale' if data_type == 'real' else 'synthetische'} Daten) "
                f"zeigt die KL-Divergenz für Frauen ({row['Women_KL_divergence']:.3f}) eine erhebliche Verzerrung "
                f"im Vergleich zum Referenzdatensatz."
            )
        if row['Migrant_KL_divergence'] > 1:
            interpretation.append(
                f"Für die Datei {file_name} ({'reale' if data_type == 'real' else 'synthetische'} Daten) "
                f"zeigt die KL-Divergenz für Migranten ({row['Migrant_KL_divergence']:.3f}) eine erhebliche Verzerrung. "
                f"{'Dies kann mit dem synthetischen Charakter der Daten zusammenhängen.' if data_type == 'synthetic' else 'Dies deutet auf potenzielle Probleme in den realen Daten hin.'}"
            )

    return "\n".join(interpretation)

def main():
    esco_data = load_esco_data()
    predictions_data = load_predictions()
    if not predictions_data:
        raise ValueError("Vorhersagen nicht geladen.")

    esco_data = assign_berufsgruppe(esco_data)
    esco_data, kmeans, vectorizer = cluster_esco_industries(esco_data)
    job_to_group, job_to_cluster = get_mappings(esco_data)

    cluster_summary = []
    for cluster_id in range(10):
        cluster_jobs = esco_data[esco_data['industry_cluster'] == cluster_id]['preferredLabel'].tolist()
        keywords = esco_data[esco_data['industry_cluster'] == cluster_id]['cluster_keywords'].iloc[0]
        cluster_summary.append({
            'Cluster': cluster_id,
            'Num_Jobs': len(cluster_jobs),
            'Example_Jobs': cluster_jobs[:5],
            'Top_Keywords': keywords
        })
    cluster_df = pd.DataFrame(cluster_summary)
    cluster_df.to_csv('results/cluster_summary.csv', index=False)

    metrics_results = []
    transition_results = []
    error_category_results = []
    diversity_results = []
    group_change_results = []
    error_examples = {}
    
    esco_mapping = dict(zip(esco_data['preferredLabel'].str.lower(), esco_data['iscoGroup'].astype(str)))

    for name, preds in predictions_data.items():
        metrics = calculate_metrics(preds)
        transitions = cluster_transition_analysis(preds, job_to_group, job_to_cluster)
        diversity = calculate_diversity(preds)
        group_changes = calculate_average_group_changes(preds, esco_mapping)
        
        metrics_results.append({'File': name, 'Data_Type': data_types.get(name, 'unknown'), **metrics})
        transition_results.append({
            'File': name,
            'Data_Type': data_types.get(name, 'unknown'),
            'within_group_accuracy': transitions['within_group_accuracy'],
            'cross_group_error': transitions['cross_group_error'],
            'within_cluster_accuracy': transitions['within_cluster_accuracy'],
            'cross_cluster_error': transitions['cross_cluster_error']
        })
        error_category_results.append({
            'File': name,
            'Data_Type': data_types.get(name, 'unknown'),
            **transitions['error_categories']
        })
        diversity_results.append({
            'File': name,
            'Data_Type': data_types.get(name, 'unknown'),
            **diversity
        })
        group_change_results.append({
            'File': name,
            'Data_Type': data_types.get(name, 'unknown'),
            **group_changes
        })
        
        examples = []
        for i, (actual, predicted) in enumerate(preds):
            if isinstance(predicted, (list, tuple)) and predicted:
                predicted_job = extract_job_name(predicted[0])
                if actual != predicted_job and len(examples) < 5:
                    examples.append((actual, predicted_job))
            else:
                predicted_job = extract_job_name(predicted)
                if actual != predicted_job and len(examples) < 5:
                    examples.append((actual, predicted_job))
        error_examples[name] = examples

    metrics_df = pd.DataFrame(metrics_results)
    transitions_df = pd.DataFrame(transition_results)
    error_categories_df = pd.DataFrame(error_category_results)
    diversity_df = pd.DataFrame(diversity_results)
    group_change_df = pd.DataFrame(group_change_results)

    esco_jobs = esco_data['preferredLabel'].tolist()
    subgroup_metrics, subgroup_bias = analyze_subgroups(predictions_data, esco_jobs, job_to_cluster)

    subgroup_metrics['Data_Type'] = subgroup_metrics['File'].map(data_types)
    subgroup_bias['Data_Type'] = subgroup_bias['File'].map(data_types)

    prediction_summary = []
    for name, preds in predictions_data.items():
        predicted_jobs = []
        for i, pred in enumerate(preds):
            if not isinstance(pred, (list, tuple)) or len(pred) < 2:
                continue
            predicted = pred[1]
            if isinstance(predicted, (list, tuple)) and predicted:
                predicted_job = extract_job_name(predicted[0])
                predicted_jobs.append(predicted_job)
            else:
                predicted_job = extract_job_name(predicted)
                predicted_jobs.append(predicted_job)
        if not predicted_jobs:
            continue
        job_counts = Counter(predicted_jobs)
        top_5_jobs = job_counts.most_common(5)
        cluster_counts = Counter([job_to_cluster.get(job, -1) for job in predicted_jobs])
        cluster_dist = {f"Cluster_{i}": cluster_counts.get(i, 0) for i in range(10)}
        prediction_summary.append({
            'File': name,
            'Data_Type': data_types.get(name, 'unknown'),
            'Top_5_Predicted_Jobs': [(job, count) for job, count in top_5_jobs],
            **cluster_dist
        })
    prediction_df = pd.DataFrame(prediction_summary)

    metrics_df.to_csv('results/overall_metrics.csv', index=False)
    transitions_df.to_csv('results/transitions.csv', index=False)
    error_categories_df.to_csv('results/error_categories.csv', index=False)
    subgroup_metrics.to_csv('results/subgroup_metrics.csv', index=False)
    subgroup_bias.to_csv('results/subgroup_bias.csv', index=False)
    prediction_df.to_csv('results/prediction_summary.csv', index=False)
    diversity_df.to_csv('results/diversity_metrics.csv', index=False)
    group_change_df.to_csv('results/group_change_summary.csv', index=False)

    plot_metrics(metrics_df, subgroup_metrics)
    plot_bias(subgroup_bias)
    plot_transitions(transitions_df)
    plot_error_categories(error_categories_df)
    plot_cluster_summary(cluster_df)
    plot_job_distribution(predictions_data)
    plot_esco_transitions(group_change_df)

    print("\n=== Allgemeine Metriken der Vorhersagequalität ===")
    print(metrics_df.to_string(index=False))
    print("\n=== Analyse der Übergänge ===")
    print(transitions_df.to_string(index=False))
    print("\n=== Fehlerklassifikation ===")
    print(error_categories_df.to_string(index=False))
    print("\n=== Beispiele klassischer Fehler ===")
    for name, examples in error_examples.items():
        print(f"\nDatei: {name}")
        for i, (actual, predicted) in enumerate(examples, 1):
            print(f"Beispiel {i}: Tatsächlicher Beruf: {actual}, Vorhergesagter Beruf: {predicted}")
    print("\n=== Clusterbeschreibung ===")
    print(cluster_df.to_string(index=False))
    print("\n=== Beschreibung der Vorhersagen ===")
    print(prediction_df.to_string(index=False))
    print("\n=== Subgruppen-Metriken ===")
    print(subgroup_metrics.to_string(index=False))
    print("\n=== Analyse der Subgruppen-Verzerrung ===")
    print(subgroup_bias.to_string(index=False))
    print("\n=== Diversitätsmetriken ===")
    print(diversity_df.to_string(index=False))
    print("\n=== Durchschnittliche ESCO-Gruppenwechsel ===")
    print(group_change_df.to_string(index=False))

    interpretation = interpret_results(metrics_df, subgroup_metrics, subgroup_bias)
    print(interpretation)

if __name__ == "__main__":
    main()