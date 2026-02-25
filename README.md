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

## Installation

### Voraussetzungen

- Python 3.8 oder höher
- pip

### Abhängigkeiten installieren

```bash
pip install numpy scipy scikit-learn pandas matplotlib seaborn shapely flask
```

Hinweis: Ich empfehle hierfür die Verwendung von Anaconda, da sich damit Python-Abhängigkeiten und virtuelle Umgebungen deutlich einfacher verwalten lassen. Insbesondere unter Windows ist das aus meiner Erfahrung die stabilste und komfortabelste Lösung für das Environment-Management. Der Download ist hier verfügbar: https://www.anaconda.com/download. Mit Anaconda lassen sich isolierte Environments unkompliziert erstellen, verwalten und reproduzierbar konfigurieren, was Versionskonflikte zuverlässig vermeidet. Es kann sein, dass die Installationsbefehle für die Abhängigkeiten bei Anaconda leicht unterschiedlich sind, da es bei mir schon mal vorgekommen ist, dass pip dort Probleme macht. Sollte das der Fall sein, empfehle ich das jeweilige package bei Anaconda zu suchen, so findet man die Installationsbefehle am Schnellsten.

## Schnellstart

### 0. Test-SVGs für die Klassifizierung vorbereiten

In `test_svg_preprocessing.py` Eingabe- und Ausgabeordner setzen. Danach:
```bash
python test_svg_preprocessing.py
```
Hilfsskript um Test-SVGs die zu Klassifizieren sind von Bestandteilen zu befreien, die für die Klassifizierung keine Rolle spielen. Dieses Skript muss vor der Klassifizierung durchgeführt werden, damit diese ohne Probleme ausgeführt werden kann.

### 1. Ordnerstruktur vorbereiten

```
projekt/
├── referenzen/          # SVG-Dateien vollständiger Gefäße aus dem ordner svg_files
│   ├── Drag.33.svg
│   ├── NB15.svg
│   └── Lud.SSa.svg
├── testdaten/           # SVG-Dateien der zu klassifizierenden Fragmente, also die Daten aus dem Moodle-Ordner welche danach mit test_svg_preprocessing.py bereinigt wurden.
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

## Frontend-Integration & Ausführung

Das Frontend ist eine Single-Page-Application (SPA), die auf **Tailwind CSS** und nativem **JavaScript** basiert. Es ermöglicht eine nahtlose Interaktion mit der KI-Engine ohne Seiten-Reloads.

### Struktur & Startvorgang

Das gesamte System wird über das Backend gesteuert:
* **Zentraler Startpunkt:** `app.py` (Flask-Server). Durch das Ausführen von `python app.py` wird sowohl die API als auch das Frontend gestartet.
* **Frontend-Datei:** `index.html`. Diese Datei enthält das HTML-Gerüst, die Styles (CSS) und die JavaScript-Logik. Sie wird vom Flask-Server automatisch unter der Root-Domain `/` ausgeliefert.
* **Keine Installation nötig:** Da das Frontend auf nativem JS und CDN-basiertem Tailwind basiert, ist keine separate Installation oder Build-Pipeline für das User-Interface erforderlich.

### Funktionsweise der Oberfläche

1. **Dynamisches Laden:** Beim Start ruft das Frontend über `/api/fragments` alle verfügbaren Dateien ab und befüllt die Seitenleiste.
2. **Asynchrone Analyse:** Ein Klick auf ein Fragment triggert den `POST`-Request an `/api/classify`. Während die KI rechnet, zeigt ein Lade-Overlay den Fortschritt an.
3. **Interaktive Visualisierung:** Die Top-5 Ergebnisse werden in einer Tabelle gerendert. Ein Klick auf eine Tabellenzeile tauscht sofort das Vorschaubild (`overlap_image`) aus.

### Endpunkte & Datenfluss

| Endpunkt | Methode | Beschreibung |
| :--- | :--- | :--- |
| `/api/fragments` | `GET` | Liefert eine Liste aller verfügbaren `.svg`-Dateien im Preprocessing-Ordner. |
| `/api/classify` | `POST` | Analysiert ein Fragment. Liefert Base64-Bilder und die Top-5-Klassen zurück. |
| `/api/download_report` | `POST` | Generiert und sendet einen PDF-Bericht (Top-5 oder Full-Scan). |

### Kern-Logik (JavaScript)

Das Frontend verarbeitet die JSON-Antwort der API und injiziert die Daten direkt in das DOM:

```batch
async function analyze(file) {
    const res = await fetch(`${API}/classify`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({filename: file})
    });

    const data = await res.json();

    // Anzeige der Bilder (Base64)
    document.getElementById('fragImg').src = 'data:image/png;base64,' + data.fragment_image;
    document.getElementById('overlapImg').src = 'data:image/png;base64,' + data.top[0].overlap_image;

    // Speicherung für CSV-Export
    lastAnalysisData = data.top;
}
```
## Konfiguration

Alle Parameter befinden sich am Anfang der Datei `keramik_svg_classifier_final.py`:

```batch
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

Für Genauigkeitsmessung wird eine CSV-Datei wie `ground_truth_template.csv` benötigt:

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

## Beispiel-Ausgabe von `keramik_svg_classifier_final.py`

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

**Vollständige Entwicklungsdokumentation:** Siehe `keramik_doku.odt`

## Dateien

- `keramik_svg_classifier_final.py` — Haupt-Skript (Klassifikator + API)
- `scale_svgs.py` — Hilfsskript zum Skalieren von SVGs
- `keramik_doku.odt` — Entwicklungsgeschichte der Backend-Lösung
- `test_svg_preprocessing.py` — Hilfsskript um Test-SVGs von Bestandteilen zu befreien die für die Klassifizierung keine Rolle spielen
- In dem Ordner `svg_files` befinden sich die Referenz-SVG Dateien
- In den Ordnern `images_Heising`, `images_Oelmann` und `images_typentafel` befinden sich die Screenshots der Literaturrefrenzen

## Performance

- **Ladezeit (Referenzen):** ~2–5 Sekunden (einmalig beim Start)
- **Klassifizierung (1 Fragment):** ~10–30 Sekunden (abhängig von Anzahl Referenzen)
- **Batch (x Fragmente × 45 Referenzen):** ~10-15 Minuten oder mehrere Stunden falls alle 800 Testdaten vom Moodle-Ordner auf einmal klassifiziert werden

**Optimierung:** `SCALE_STEP` von 0.05 auf 0.1 erhöhen halbiert die Laufzeit, ist aber wegen Einbüßen der Genauigkeit der Klassifizierung nicht zu empfehlen.


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
