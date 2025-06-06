# Fehleranalyse für die Vorhersage von Karrierewegen

Dieses Projekt analysiert die Vorhersagequalität von beruflichen Karrierepfaden anhand realer und synthetischer Datensätze. Es werden semantische Cluster gebildet, Vorhersagemetriken berechnet sowie Verzerrungen (Bias) insbesondere für Frauen- und Migrant:innenberufe identifiziert und visualisiert.

## 🔍 Ziel des Projekts

- Analyse der Vorhersagegüte (MRR, Recall@k) von Berufsempfehlungsmodellen
- Untersuchung semantischer und klassifikatorischer Fehler
- Bias-Analyse mittels KL-Divergenz und Chi²-Tests
- Subgruppenanalyse (Frauen, Migrant:innen) basierend auf semantischer Ähnlichkeit
- Vergleich zwischen realen und synthetisch erzeugten Daten

## 📁 Datenstruktur

- `data/occupations_en.csv` — ESCO-Berufsdaten mit ISCO-Gruppen
- `data/predictions/*.pkl` — Modellvorhersagen im Pickle-Format
- `cache/` — Zwischenspeicherung von Embeddings und TF-IDF-Matrizen
- `results/` — Generierte Auswertungen und Visualisierungen

## ⚙️ Hauptmodule

| Datei | Beschreibung |
|-------|--------------|
| `main.py` | Führt die komplette Analysepipeline aus und erzeugt Plots sowie CSV-Dateien |
| `data_loading.py` | Lädt ESCO-Daten und Vorhersagedateien |
| `clustering.py` | Führt TF-IDF-basierte Clusteranalyse der Berufe durch |
| `metrics.py` | Berechnet MRR, Recall, Fehlerkategorien, Diversität und Clusterübergänge |
| `bias_analysis.py` | Berechnet Verzerrungsmetriken (KL, Chi²) zwischen Vorhersageverteilungen |
| `subgroup_analysis.py` | Analysiert Vorhersagen für Frauen- und Migrantenberufe mittels SBERT |

## 📊 Wichtige Metriken

- **MRR (Mean Reciprocal Rank)**
- **Recall@5 / Recall@10**
- **KL-Divergenz** zur Messung der Verteilungsunterschiede
- **Chi²-Test** zur statistischen Signifikanzprüfung
- **Fehlerklassifikation**: korrekt, gleiche Berufsgruppe, semantischer Cluster, falsch

## 📦 Benötigte Bibliotheken

```bash
pip install pandas numpy scikit-learn matplotlib seaborn sentence-transformers torch
