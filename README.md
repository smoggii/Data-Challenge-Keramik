# Keramikfragment-Klassifikator

Automatische Klassifikation römischer Keramikfragmente mittels Template-Matching.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Überblick

Dieses Projekt klassifiziert Querschnitte von Keramikfragmenten durch Vergleich mit einer Referenzbibliothek vollständiger Gefäßquerschnitte. Der Algorithmus verwendet Template-Matching mit automatischer Skalierung, Alignment und mehreren Score-Komponenten, welche ich im Nachhinein voch genauer erklären werde.

**Typische Anwendung:** Archäologische Fundstücke (Scherben) werden digitalisiert und automatisch dem passenden Gefäßtyp zugeordnet (z. B. Drag.33, Niederbieber15, Lud.SSa).

### Features

- **Template-Matching** mit schrittweisem Scale-Up (1.0x – 4.0x)
- **Pre-Scale X-Shift** für optimale horizontale Ausrichtung
- **Drei Score-Komponenten:** Kontur-Score, Kontur-Coverage, Flächen-Overlap
- **Y-gewichteter Coverage-Score** (Gefäßrand zählt mehr als unterer Bereich)
- **Visualisierung** als PDF (einzeln oder Batch)
- **Frontend-API** für einfache Integration in Web-Anwendungen

## Installation

### Voraussetzungen

- Python 3.8 oder höher
- pip

### Abhängigkeiten installieren

```bash
pip install numpy scipy scikit-learn pandas matplotlib seaborn shapely
```

## Schnellstart

### 1. Ordnerstruktur vorbereiten

```
projekt/
├── referenzen/          # SVG-Dateien vollständiger Gefäße aus dem ordner svg_files
│   ├── Drag.33.svg
│   ├── NB15.svg
│   └── Lud.SSa.svg
├── testdaten/           # SVG-Dateien der zu klassifizierenden Fragmente
│   ├── frag_001.svg
│   ├── frag_002.svg
│   └── ...
└── output/              # Dateiordner für die KLassifizierungsergebnisse
```

### 2. Skript ausführen

```bash
python keramik_svg_classifier_final.py
```

Interaktive Eingaben:
```
Referenz-Ordner:  referenzen
Test-Ordner:      testdaten
Output-Ordner:    output

Modus:
  1 = Klassifizierung aller Fragmnete (nur Ergebnis-CSV + Konfusionsmatrix)
  2 = Klassifizierung aller Fragmnete + Top-K PDFs (1 PDF pro Fragment mit visualiserten Top-K Ergebnissen, K = auswählbare Anzahl)
  3 = Batch-Visualisierung  (1 Fragment vs ALLE Referenzen mit Visualisierung und Scores)
```

### 3. Ergebnisse

- **Modus 1:** CSV-Datei mit allen Scores + Konfusionsmatrix-PNG
- **Modus 2:** Pro Fragment ein PDF mit Top-K Visualisierungen
- **Modus 3:** Ein PDF mit einem Fragment vs. allen Referenzen

## Frontend-Integration

Die API ermöglicht einfache Integration in Web-Frontends:

```python
from keramik_FINAL_v17_KC import ClassifierAPI

# Einmalig beim App-Start
api = ClassifierAPI(
    ref_folder    = "referenzen",
    test_folder   = "testdaten",
    output_folder = "output",
    top_k         = 5
)

# Dropdown befüllen
fragments = api.list_fragments()
# → {"fragments": ["frag_001.svg", ...]}

# Fragment klassifizieren
result = api.classify_fragment("frag_001.svg")
# → {"fragment_image": "<base64-PNG>",
#    "top": [{"class": "Drag.33", "score": 0.78, "overlap_image": "<base64-PNG>", ...}]}

# Alle Fragmente klassifizieren
results = api.classify_all(ground_truth_csv="ground_truth.csv")
# → {"accuracy": 0.24, "results": [...], "confusion_matrix_path": "..."}
```

**Vollständige Dokumentation:** Siehe `frontend_integration_doku.docx`

## Konfiguration

Alle Parameter befinden sich am Anfang der Datei:

```python
# Score-Gewichte (Summe = 1.0)
CONTOUR_WEIGHT    = 0.3  # Kontur-Distanz
KONTUR_COV_WEIGHT = 0.3  # Kontur-Coverage
COVERAGE_WEIGHT   = 0.4  # Flächen-Overlap

# Pre-Scale X-Shift
PRE_XSHIFT_ENABLED = True
PRE_XSHIFT_RANGE   = 30.0   # ±30px Suchbereich
PRE_XSHIFT_STEPS   = 13     # Anzahl getesteter Positionen

# Y-Gewichtung
FEATURE_D_YWEIGHT  = True
FEATURE_D_EXPONENT = 4.0    # Höher = Rand dominiert stärker

# Scale-Up
SCALE_START = 1.0
SCALE_MAX   = 4.0
SCALE_STEP  = 0.1
```

## Algorithmus-Überblick

### 1. Preprocessing
- SVG-Parsing (polyline oder path)
- Outlier-Filterung (DBSCAN)
- Zentrierung und Y-Invertierung
- Alignment am höchsten Punkt (Gefäßrand)

### 2. Pre-Scale X-Shift
- Fragment wird horizontal verschoben (±30px, 13 Positionen)
- Kriterium: KC-Score + Coverage-Score
- Bester Offset wird für gesamten Scale-Up verwendet

### 3. Scale-Up (1.0x – 4.0x)
- Fragment wird schrittweise vergrößert
- Pro Schritt: Kontur-Score + KC-Score + Coverage-Score berechnen
- Bester Gesamtscore gewinnt (kein Early Stopping)

### 4. Scoring
- **Kontur-Score (K):** Bidirektionale Punkt-zu-Punkt-Distanz
- **Kontur-Coverage (KC):** Anteil Referenzpunkte < Threshold vom Fragment
- **Coverage-Score (Cov):** Flächen-IoU im Y-Streifen, Y-gewichtet

## Ground Truth Format

Für Genauigkeitsmessung wird eine CSV-Datei benötigt:

```csv
filename,true_class
frag_001.svg,Drag.33
frag_002.svg,NB15
frag_003.svg,Drag.33
```

- **filename:** Exakter Dateiname inkl. `.svg`
- **true_class:** Klassenname = Referenz-Dateiname ohne `.svg`

## SVG-Anforderungen

- **Format:** `<polyline>` oder `<path>` Element
- **Koordinaten:** Mindestens 100 Zeichen in `points` bzw. `d` Attribut
- **Skalierung:** Referenzen sollten ähnliche Größenordnung haben (~150px Höhe)

### SVG-Skalierung

Falls neue Referenzen unterschiedliche Größen haben:

```bash
python scale_svgs.py
```

Skaliert alle SVGs proportional auf 150px Höhe. Die Referendateien im Ordner `svg_files` wurden schon mit diesem Script auf eine einheitliche Größe geändert.

## Beispiel-Ausgabe

### Modus 1 — Terminal

```
[1/41] recons_10014.svg -> Drag.33 (0.782) ✓
[2/41] recons_10017.svg -> NB15 (0.651) ✗
[3/41] recons_10021.svg -> Drag.33 (0.743) ✓
...

Accuracy: 24.39%

              precision    recall  f1-score   support
    Drag.33       0.35      0.50      0.41         8
       NB15       0.20      0.15      0.17        13
    Lud.SSa       0.18      0.22      0.20         9
    ...
```

### Modus 2 — PDF pro Fragment

Jedes PDF enthält:
- **Deckblatt:** Fragment-Name, True Class, Predicted Class, Top-5 Scores
- **Pro Treffer eine Seite:** Overlap-Visualisierung mit Score-Details auch rechter Seite

### Modus 3 — Ein PDF für ein Fragment

Ein Fragment wird gegen alle Referenzen visualisiert:
- Deckblatt mit allen Scores sortiert
- Pro Referenz eine Seite (True Class zuerst auf grünem Hintergrund)

## Entwicklungsgeschichte

Das Projekt durchlief 18 Iterationen (V1 – V18):

- **V1–V8:** Feature-Engineering (Hu-Momente, Fourier, Krümmung) → gescheitert (0–5% Accuracy)
- **V9:** Paradigmenwechsel zu Template-Matching, bidirektionaler Kontur-Score
- **V11–V13:** IoU-Einführung, Alignment-Fixes
- **V15:** Y-Streifen Coverage (nur relevanter Höhenbereich)
- **V17:** Symmetrischer Y-Streifen IoU + Y-Gewichtung (24.3% Accuracy)
- **V17+KC:** Kontur-Coverage Score (Threshold relativ zur Fragmentgröße)
- **V18:** Pre-Scale X-Shift, kein Early Stopping, Bug-Fixes

**Vollständige Entwicklungsdokumentation:** Siehe `keramik_entwicklungsdokumentation.docx`

## Dateien

- `keramik_svg_classifier_final.py` — Haupt-Skript (Klassifikator + API)
- `scale_svgs.py` — Hilfsskript zum Skalieren von SVGs
- `keramik_entwicklungsdokumentation.docx` — Entwicklungsgeschichte der Backend-Lösung

## Performance

- **Ladezeit (Referenzen):** ~2–5 Sekunden (einmalig beim Start)
- **Klassifizierung (1 Fragment):** ~10–30 Sekunden (abhängig von Anzahl Referenzen)
- **Batch (x Fragmente × 45 Referenzen):** ~10-15 Minuten oder mehrere Stunden falls alle 800 Testdaten vom Moodle-Ordner auf einmal klassifiziert werden

**Optimierung:** `SCALE_STEP` von 0.05 auf 0.1 erhöhen halbiert die Laufzeit, ist aber wegen Einbüßen der Genauigkeit der Klassifizierung nicht zu empfehlen.

## Known Issues und Zukünftige Verbesserungen


## Lizenz

MIT License

## Kontakt

Bei Fragen zum Code oder zur Methodik bitte eine Mail an florian.ebner@stud.uni-frankfurt.de 

## Zitation

Falls dieses Projekt in wissenschaftlichen Arbeiten verwendet wird:

```
Keramikfragment-Klassifikator (2026)
Template-Matching für römische Keramik
https://github.com/smoggii/Data-Challenge-Keramik
```
└── README.md                         # Documentation
