import io
import base64
import numpy as np
from pathlib import Path
import xml.etree.ElementTree as ET
from sklearn.cluster import DBSCAN
import csv
from datetime import datetime
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import re
from shapely.geometry import Polygon, box

# ===========================================================================
# CONFIG
# ===========================================================================

FEATURE_A2_SYMMETRIC = True

# --- Pre-Scale X-Shift (einmalig VOR dem ersten Scale-Up-Schritt) ---
# Das Fragment wird VOR dem Scale-Up lateral verschoben und am Punkt
# mit dem besten kombinierten Coverage-Score gestartet.
# Score-Kriterium: KC-Score + Coverage-Score (beide Flaechenmetriken)
# → kein iterativer Shift pro Scale-Schritt (Feature B war zu instabil)
PRE_XSHIFT_ENABLED = True
PRE_XSHIFT_RANGE = 30.0   # px Suchbereich links UND rechts
PRE_XSHIFT_STEPS = 13     # Anzahl getesteter Positionen (inkl. 0)

FEATURE_D_YWEIGHT = True
FEATURE_D_EXPONENT = 4.0

# --- Score-Gewichte ---
# Drei Komponenten:
#   CONTOUR_WEIGHT:       Distanz-Score  1/(1 + (d_frag+d_ref)/2) * 5
#   KONTUR_COV_WEIGHT:    Anteil Ref-Konturpunkte nah am Fragment
#   COVERAGE_WEIGHT:      Flächen-Overlap Score
# Summe sollte 1.0 ergeben
CONTOUR_WEIGHT = 0.3
KONTUR_COV_WEIGHT = 0.3
COVERAGE_WEIGHT = 0.4

# --- Kontur-Coverage Parameter ---
# Threshold = Fragment-Höhe (nach Scale-Up) × KONTUR_THRESHOLD_REL
# → skaliert automatisch mit Fragmentgröße:
#   Höhe=30px, REL=0.08 → Threshold=2.4px
#   Höhe=90px, REL=0.08 → Threshold=7.2px
# 0.05 = streng  |  0.08 = Standard  |  0.15 = großzügig
# ← HIER ANPASSEN:
# AUFPASSEN!!! wird weiter unten nicht benutzt und durch einen statischen wert ersetzt
KONTUR_THRESHOLD_REL = 0.08

SCALE_START = 1.0
SCALE_MAX = 4.0
SCALE_STEP = 0.05

# ===========================================================================


class TemplateMatchingClassifier:

    def __init__(self, reference_folder, test_folder, output_folder):
        self.reference_folder = Path(reference_folder)
        self.test_folder = Path(test_folder)
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(exist_ok=True)
        self.reference_data = {}

    def parse_svg(self, svg_path):
        tree = ET.parse(svg_path)
        root = tree.getroot()
        for elem in root.iter():
            tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
            if tag == 'polyline':
                pts = elem.get('points', '')
                if len(pts) > 100:
                    c = self._parse_polyline(pts)
                    if len(c):
                        return c
        for elem in root.iter():
            tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
            if tag == 'path':
                d = elem.get('d', '')
                if len(d) > 100:
                    c = self._parse_path(d)
                    if len(c):
                        return c
        raise ValueError(f"Kein Pfad in {svg_path}")

    def _parse_polyline(self, pts):
        nums = ' '.join(pts.replace(',', ' ').split()).split()
        return np.array([[float(nums[i]), float(nums[i+1])]
                         for i in range(0, len(nums)-1, 2)])

    def _parse_path(self, path_data):
        path_data = path_data.replace(',', ' ')
        commands = re.findall(r'[MLCZmlcz][^MLCZmlcz]*', path_data)
        coords, cur, start = [], np.array([0.0, 0.0]), np.array([0.0, 0.0])
        for cmd in commands:
            t = cmd[0]
            n = [float(x) for x in re.findall(r'-?\d+\.?\d*', cmd[1:])]
            if t == 'M':
                if n:
                    cur = np.array([n[0], n[1]])
                    start = cur.copy()
                    coords.append(cur.copy())
                    for i in range(2, len(n)-1, 2):
                        cur = np.array([n[i], n[i+1]])
                        coords.append(cur.copy())
            elif t == 'm':
                cur = (cur+np.array([n[0], n[1]])
                       ) if coords else np.array([n[0], n[1]])
                start = cur.copy()
                coords.append(cur.copy())
                for i in range(2, len(n)-1, 2):
                    cur += np.array([n[i], n[i+1]])
                    coords.append(cur.copy())
            elif t == 'L':
                for i in range(0, len(n)-1, 2):
                    cur = np.array([n[i], n[i+1]])
                    coords.append(cur.copy())
            elif t == 'l':
                for i in range(0, len(n)-1, 2):
                    cur += np.array([n[i], n[i+1]])
                    coords.append(cur.copy())
            elif t == 'C':
                for i in range(0, len(n)-5, 6):
                    cur = np.array([n[i+4], n[i+5]])
                    coords.append(cur.copy())
            elif t == 'c':
                for i in range(0, len(n)-5, 6):
                    cur += np.array([n[i+4], n[i+5]])
                    coords.append(cur.copy())
            elif t in ['Z', 'z']:
                if not np.allclose(cur, start):
                    coords.append(start.copy())
                cur = start.copy()
        return np.array(coords)

    def filter_outliers(self, coords, eps=10, min_samples=10):
        if len(coords) < min_samples:
            return coords
        lbl = DBSCAN(eps=eps, min_samples=min_samples).fit(coords).labels_
        return coords[lbl >= 0]

    def center_and_invert_y(self, coords):
        c = coords - coords.mean(axis=0)
        c[:, 1] = -c[:, 1]
        return c

    def align_to_top(self, coords):
        c = coords.copy()
        c[:, 1] -= c[:, 1].max()
        return c

    def align_fragment_to_reference(self, frag, ref):
        return frag + (ref[np.argmax(ref[:, 1])] - frag[np.argmax(frag[:, 1])])

    # -----------------------------------------------------------------------
    # Kontur-Score (bidirektional, unveraendert aus V17)
    # -----------------------------------------------------------------------
    def _contour_score(self, frag, ref):
        from scipy.spatial.distance import cdist
        d_f = cdist(frag, ref).min(axis=1).mean()
        d_r = cdist(ref, frag).min(axis=1).mean()
        return (1 / (1 + (d_f + d_r) / 2)) * 5

    # -----------------------------------------------------------------------
    # NEU: Kontur-Coverage-Score
    # -----------------------------------------------------------------------
    def _kontur_coverage_score(self, frag, ref):
        """
        Anteil der Referenz-Konturpunkte die innerhalb des Schwellwerts
        vom Fragment liegen.

        Schwellwert = Fragment-Hoehe x KONTUR_THRESHOLD_REL
        Skaliert automatisch mit der Fragmentgroesse nach Scale-Up.

        Beispiel: Hoehe=60px, REL=0.08 -> Threshold=4.8px
          Ref hat 500 Punkte, 350 davon < 4.8px vom Fragment
          -> KC-Score = 350/500 = 0.70
        """
        from scipy.spatial.distance import cdist
        frag_height = frag[:, 1].max() - frag[:, 1].min()
        threshold = frag_height * KONTUR_THRESHOLD_REL
        d_ref_to_frag = cdist(ref, frag).min(axis=1)
        abgedeckt = (d_ref_to_frag < threshold).sum()
        return abgedeckt / len(ref)

    # -----------------------------------------------------------------------
    # Coverage-Score (unveraendert aus V17)
    # -----------------------------------------------------------------------
    def _coverage_score(self, frag, ref):
        try:
            ref_poly = Polygon(ref).buffer(0)
            frag_poly = Polygon(frag).buffer(0)
            if not (ref_poly.is_valid and frag_poly.is_valid):
                return 0.0

            y_min = frag[:, 1].min()
            y_max = frag[:, 1].max()
            x_min = min(ref[:, 0].min(), frag[:, 0].min()) - 1
            x_max = max(ref[:, 0].max(), frag[:, 0].max()) + 1
            streifen = box(x_min, y_min, x_max, y_max)
            ref_strip = ref_poly.intersection(streifen)

            if FEATURE_D_YWEIGHT:
                score = self._weighted_coverage(
                    frag_poly, ref_strip, frag, y_min, y_max)
            else:
                inter_area = ref_strip.intersection(frag_poly).area
                if FEATURE_A2_SYMMETRIC:
                    union = ref_strip.area + frag_poly.area - inter_area
                    score = inter_area / union if union > 0 else 0.0
                else:
                    score = inter_area / ref_strip.area if ref_strip.area > 0 else 0.0

            return float(np.clip(score, 0, 1))
        except Exception:
            return 0.0

    def _weighted_coverage(self, frag_poly, ref_strip, frag_coords, y_min, y_max):
        n_slices = 40
        y_edges = np.linspace(y_min, y_max, n_slices + 1)
        weighted_sum = 0.0
        weight_total = 0.0
        for i in range(n_slices):
            y_lo = y_edges[i]
            y_hi = y_edges[i+1]
            y_mid = (y_lo + y_hi) / 2
            w = ((y_mid - y_min) / (y_max - y_min)) ** FEATURE_D_EXPONENT
            try:
                slice_box = box(frag_coords[:, 0].min()-1, y_lo,
                                frag_coords[:, 0].max()+1, y_hi)
                ref_slice = ref_strip.intersection(slice_box)
                frag_slice = frag_poly.intersection(slice_box)
                if ref_slice.is_empty or frag_slice.is_empty:
                    slice_score = 0.0
                else:
                    inter = ref_slice.intersection(frag_slice).area
                    if FEATURE_A2_SYMMETRIC:
                        union = ref_slice.area + frag_slice.area - inter
                        slice_score = inter / union if union > 0 else 0.0
                    else:
                        slice_score = inter / ref_slice.area if ref_slice.area > 0 else 0.0
            except Exception:
                slice_score = 0.0
            weighted_sum += w * slice_score
            weight_total += w
        return weighted_sum / weight_total if weight_total > 0 else 0.0

    # -----------------------------------------------------------------------
    # Template Matching
    # -----------------------------------------------------------------------
    def template_match_scaleup(self, fragment_coords, reference_coords, verbose=False):
        ref_top = reference_coords[np.argmax(reference_coords[:, 1])]
        best_contour = 0
        best_scale = SCALE_START
        best_frag = None
        best_x_off = 0.0
        scale_hist, contour_hist = [], []

        # ── Pre-Scale X-Shift: einmalig VOR dem ersten Scale-Up ─────────
        # Testet PRE_XSHIFT_STEPS Positionen im Bereich ±PRE_XSHIFT_RANGE px.
        # Kriterium: kombinierter Coverage-Score (KC + Coverage scores).
        # Der beste Startoffset wird fuer die gesamte Scale-Up-Schleife verwendet, danach wird nicht mehr geshiftet.
        if PRE_XSHIFT_ENABLED:
            x_candidates = np.linspace(
                -PRE_XSHIFT_RANGE, PRE_XSHIFT_RANGE, PRE_XSHIFT_STEPS)
            best_pre_score = -1.0
            best_pre_off = 0.0
            frag_at_start = (fragment_coords - ref_top) * SCALE_START + ref_top
            for dx in x_candidates:
                ft = frag_at_start.copy()
                ft[:, 0] += dx
                # Beide Coverage-Metriken als gemeinsames Kriterium
                cov_score = self._coverage_score(ft, reference_coords)
                kc_score = self._kontur_coverage_score(ft, reference_coords)
                combined = kc_score + cov_score          # gleichgewichtet
                if combined > best_pre_score:
                    best_pre_score = combined
                    best_pre_off = dx
            best_x_off = best_pre_off
        # ────────────────────────────────────────────────────────────────

        # ── Vollstaendiger Scale-Up bis SCALE_MAX ───────────────────────
        # Kein Early Stopping: alle Scale-Schritte werden durchlaufen.
        # Pro Schritt wird sofort der Gesamtscore berechnet (K + KC + Cov).
        # Das Fragment mit dem besten GESAMT-Score gewinnt, nicht das mit
        # dem besten Kontur-Score an einem Zwischenpunkt.
        best_final = -1.0
        best_contour = 0.0
        best_cov = 0.0
        best_kc = 0.0
        current_scale = SCALE_START
        while current_scale <= SCALE_MAX:
            frag_scaled = (fragment_coords - ref_top) * current_scale + ref_top
            frag_scaled = frag_scaled.copy()
            frag_scaled[:, 0] += best_x_off   # konstanter Offset aus Pre-Shift

            c_score = self._contour_score(frag_scaled, reference_coords)
            cov = self._coverage_score(frag_scaled, reference_coords)
            kc = self._kontur_coverage_score(frag_scaled, reference_coords)
            total = (c_score * CONTOUR_WEIGHT
                     + kc * KONTUR_COV_WEIGHT
                     + cov * COVERAGE_WEIGHT)

            scale_hist.append(current_scale)
            contour_hist.append(c_score)

            if total > best_final:
                best_final = total
                best_contour = c_score
                best_scale = current_scale
                best_frag = frag_scaled.copy()
                best_cov = cov
                best_kc = kc

            current_scale = round(current_scale + SCALE_STEP, 10)
        # ────────────────────────────────────────────────────────────────

        coverage = best_cov
        kontur_cov = best_kc
        final = best_final

        return {
            'score':            final,
            'contour_score':    best_contour,
            'kontur_cov_score': kontur_cov,
            'coverage_score':   coverage,
            'scale':            best_scale,
            'x_offset':         best_x_off,
            'transformed':      best_frag,
            'scale_hist':       np.array(scale_hist),
            'contour_hist':     np.array(contour_hist),
        }

    # -----------------------------------------------------------------------
    # Klassifizierung
    # -----------------------------------------------------------------------
    def classify_with_ground_truth(self, ground_truth_csv=None, top_k=5):
        print(f"\n{'='*70}\nV17+KC  {self._feat_str()}\n{'='*70}")
        gt = {}
        if ground_truth_csv and Path(ground_truth_csv).exists():
            df = pd.read_csv(ground_truth_csv)
            gt = dict(zip(df['filename'], df['true_class']))
            print(f"Ground Truth: {len(gt)} Samples")
        test_files = sorted(self.test_folder.glob('*.svg'))
        print(f"Test-SVGs: {len(test_files)}\n")

        results, y_true, y_pred = [], [], []
        for idx, tp in enumerate(test_files, 1):
            print(f"[{idx}/{len(test_files)}] {tp.name}", end=' ')
            try:
                tc = self._prep(tp)
                sims = []
                for cls, rd in self.reference_data.items():
                    ta = self.align_fragment_to_reference(tc, rd['aligned'])
                    res = self.template_match_scaleup(ta, rd['aligned'])
                    sims.append((cls, res['score'], res))
                sims.sort(key=lambda x: x[1], reverse=True)
                top = sims[:top_k]
                true_cls = gt.get(tp.name)
                pred_cls = top[0][0]
                if true_cls:
                    y_true.append(true_cls)
                    y_pred.append(pred_cls)
                    print(f"-> {pred_cls} ({top[0][1]:.3f}) "
                          f"{'✓' if pred_cls==true_cls else '✗'}")
                else:
                    print(f"-> {pred_cls} ({top[0][1]:.3f})")
                results.append({'filename': tp.name, 'true_class': true_cls,
                                'predicted_class': pred_cls,
                                'confidence': top[0][1], 'top_matches': top})
            except Exception as e:
                print(f"✗ {e}")

        if y_true:
            print(f"\nAccuracy: {accuracy_score(y_true,y_pred):.2%}\n")
            print(classification_report(y_true, y_pred, zero_division=0))
            self._save_cm(y_true, y_pred)
        self._save_csv(results, top_k)
        return results

    def classify_with_batch_vis(self, vis_folder, ground_truth_csv=None, top_k=5):
        """
        Modus 2: Klassifizierung + pro Fragment ein PDF mit den Top-K Visualisierungen.
        Alle PDFs landen in vis_folder, benannt nach dem Fragment.
        """
        print(
            f"\n{'='*70}\nV17+KC  Klassifizierung + Batch-Vis  {self._feat_str()}\n{'='*70}")
        vis_path = Path(vis_folder)
        vis_path.mkdir(parents=True, exist_ok=True)

        gt = {}
        if ground_truth_csv and Path(ground_truth_csv).exists():
            df = pd.read_csv(ground_truth_csv)
            gt = dict(zip(df['filename'], df['true_class']))
            print(f"Ground Truth: {len(gt)} Samples")

        test_files = sorted(self.test_folder.glob('*.svg'))
        print(f"Test-SVGs: {len(test_files)}")
        print(f"PDFs werden gespeichert in: {vis_path}\n")

        results, y_true, y_pred = [], [], []

        for idx, tp in enumerate(test_files, 1):
            print(f"[{idx}/{len(test_files)}] {tp.name}", end=' ')
            try:
                tc = self._prep(tp)
                sims = []
                for cls, rd in self.reference_data.items():
                    ta = self.align_fragment_to_reference(tc, rd['aligned'])
                    res = self.template_match_scaleup(ta, rd['aligned'])
                    sims.append((cls, res['score'], ta, rd['aligned'], res))
                sims.sort(key=lambda x: x[1], reverse=True)

                true_cls = gt.get(tp.name)
                pred_cls = sims[0][0]
                top = [(cls, sc, res) for cls, sc, _, _, res in sims[:top_k]]

                if true_cls:
                    y_true.append(true_cls)
                    y_pred.append(pred_cls)
                    print(f"-> {pred_cls} ({sims[0][1]:.3f}) "
                          f"{'✓' if pred_cls==true_cls else '✗'}", end='')
                else:
                    print(f"-> {pred_cls} ({sims[0][1]:.3f})", end='')

                results.append({'filename': tp.name, 'true_class': true_cls,
                                'predicted_class': pred_cls,
                                'confidence': sims[0][1], 'top_matches': top})

                # PDF mit Top-K Visualisierungen erzeugen
                # True Class immer zuerst wenn vorhanden, dann Top-K nach Score
                if true_cls:
                    # True Class ans erste stellen (falls nicht schon #1)
                    sorted_sims = sorted(
                        sims,
                        key=lambda x: (
                            0, -x[1]) if x[0] == true_cls else (1, -x[1])
                    )[:top_k]
                else:
                    sorted_sims = sims[:top_k]

                pdf_path = vis_path / f"v17kc_{tp.stem}.pdf"
                with PdfPages(pdf_path) as pdf:
                    # Deckblatt
                    fig = plt.figure(figsize=(14, 5))
                    fig.patch.set_facecolor('#f8f8f8')
                    plt.axis('off')
                    plt.text(0.5, 0.88, f"Top-{top_k} Visualisierung",
                             ha='center', fontsize=20, fontweight='bold',
                             transform=fig.transFigure)
                    plt.text(0.5, 0.74, f"Fragment: {tp.name}",
                             ha='center', fontsize=14, transform=fig.transFigure)
                    plt.text(0.5, 0.62,
                             f"True Class: {true_cls or 'unbekannt'}   |   "
                             f"Predicted: {pred_cls}   "
                             f"{'✓' if pred_cls==true_cls else '✗' if true_cls else ''}",
                             ha='center', fontsize=13,
                             color='darkgreen' if pred_cls == true_cls else 'darkred' if true_cls else 'gray',
                             transform=fig.transFigure)
                    plt.text(0.5, 0.51,
                             f"Features: {self._feat_str()}   "
                             f"Gewichte: K={CONTOUR_WEIGHT} KC={KONTUR_COV_WEIGHT} Cov={COVERAGE_WEIGHT}",
                             ha='center', fontsize=9, color='#444',
                             transform=fig.transFigure)
                    plt.text(0.5, 0.40, f"Top-{top_k} Ergebnisse:",
                             ha='center', fontsize=10, fontweight='bold',
                             transform=fig.transFigure)
                    for i, (cls, sc, _, _, res) in enumerate(sorted_sims):
                        marker = " <- TRUE" if cls == true_cls else ""
                        plt.text(0.5, 0.32 - i*0.042,
                                 f"{i+1}. {cls}:  {sc:.4f}"
                                 f"  (K={res['contour_score']:.3f}"
                                 f"  KC={res['kontur_cov_score']:.3f}"
                                 f"  Cov={res['coverage_score']:.3f}){marker}",
                                 ha='center', fontsize=9,
                                 color='darkgreen' if cls == true_cls else 'black',
                                 transform=fig.transFigure)
                    pdf.savefig(fig)
                    plt.close()

                    # Eine Seite pro Top-K Treffer
                    for rank, (cls, sc, aln, ref, res) in enumerate(sorted_sims, 1):
                        self._plot_step3_page(pdf, aln, ref, res,
                                              cls, true_cls, rank)

                print(f"  -> PDF: {pdf_path.name}")

            except Exception as e:
                print(f"  ✗ {e}")

        if y_true:
            print(f"\nAccuracy: {accuracy_score(y_true,y_pred):.2%}\n")
            print(classification_report(y_true, y_pred, zero_division=0))
            self._save_cm(y_true, y_pred)
        self._save_csv(results, top_k)
        print(f"\n✓ {len(test_files)} PDFs in: {vis_path}")
        return results

    def _feat_str(self):
        a = "A2sym" if FEATURE_A2_SYMMETRIC else "A2asym"
        px = (f"PreShift=AN(±{PRE_XSHIFT_RANGE:.0f}px,{PRE_XSHIFT_STEPS}steps)"
              if PRE_XSHIFT_ENABLED else "PreShift=AUS")
        d = f"D={'AN(exp='+str(FEATURE_D_EXPONENT)+')' if FEATURE_D_YWEIGHT else 'AUS'}"
        return f"{a}  {px}  {d}  KC_rel={KONTUR_THRESHOLD_REL}"

    def _prep(self, path):
        c = self.parse_svg(path)
        c = self.filter_outliers(c)
        c = self.center_and_invert_y(c)
        c = self.align_to_top(c)
        return c

    def _save_cm(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        classes = sorted(set(y_true+y_pred))
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=classes, yticklabels=classes)
        plt.title(f'Confusion Matrix V17+KC  {self._feat_str()}')
        plt.ylabel('True')
        plt.xlabel('Predicted')
        plt.tight_layout()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        p = self.output_folder / f'confusion_matrix_{ts}.png'
        plt.savefig(p, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 {p}")

    def _save_csv(self, results, top_k):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        p = self.output_folder / f"results_{ts}.csv"
        with open(p, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            h = ['Filename', 'True', 'Predicted', 'Score']
            for i in range(1, top_k+1):
                h += [f'Top{i}_Class', f'Top{i}_Score',
                      f'Top{i}_Kontur', f'Top{i}_KonturCov', f'Top{i}_Coverage']
            w.writerow(h)
            for r in results:
                row = [r['filename'], r.get('true_class', ''),
                       r['predicted_class'], f"{r['confidence']:.4f}"]
                for cls, sc, res in r['top_matches']:
                    row += [cls, f"{sc:.4f}",
                            f"{res['contour_score']:.4f}",
                            f"{res['kontur_cov_score']:.4f}",
                            f"{res['coverage_score']:.4f}"]
                w.writerow(row)
        print(f"📄 {p}")

    # -----------------------------------------------------------------------
    # Visualisierung (unveraendert aus V17, nur Titel + Deckblatt angepasst)
    # -----------------------------------------------------------------------
    def visualize_single_match(self, test_svg, ref_class, output_pdf):
        tc = self._prep(test_svg)
        ref = self.reference_data[ref_class]['aligned']
        aln = self.align_fragment_to_reference(tc, ref)
        res = self.template_match_scaleup(aln, ref)
        with PdfPages(output_pdf) as pdf:
            self._plot_step1(pdf, tc, aln, ref)
            self._plot_step2(pdf, res)
            self._plot_step3_page(pdf, aln, ref, res,
                                  ref_class, true_class=None, rank=1)
        print(f"✓ {output_pdf}")

    def visualize_batch(self, test_svg, output_pdf, true_class=None):
        print(f"\nBatch-Vis: {Path(test_svg).name} vs alle Referenzen...")
        tc = self._prep(test_svg)
        results = []
        for cls, rd in self.reference_data.items():
            aln = self.align_fragment_to_reference(tc, rd['aligned'])
            res = self.template_match_scaleup(aln, rd['aligned'])
            results.append((cls, res['score'], aln, rd['aligned'], res))
            print(f"  {cls}: {res['score']:.4f} "
                  f"(K={res['contour_score']:.3f} "
                  f"KC={res['kontur_cov_score']:.3f} "
                  f"Cov={res['coverage_score']:.3f})")

        results.sort(key=lambda x: (
            0, -x[1]) if (true_class and x[0] == true_class) else (1, -x[1]))

        with PdfPages(output_pdf) as pdf:
            fig = plt.figure(figsize=(14, 5))
            fig.patch.set_facecolor('#f8f8f8')
            plt.axis('off')
            plt.text(0.5, 0.88, "Batch-Visualisierung V17+KC", ha='center',
                     fontsize=20, fontweight='bold', transform=fig.transFigure)
            plt.text(0.5, 0.74, f"Fragment: {Path(test_svg).name}",
                     ha='center', fontsize=14, transform=fig.transFigure)
            plt.text(0.5, 0.62, f"True Class: {true_class or 'unbekannt'}",
                     ha='center', fontsize=13,
                     color='darkgreen' if true_class else 'gray',
                     transform=fig.transFigure)
            plt.text(0.5, 0.51,
                     f"Features: {self._feat_str()}   "
                     f"Gewichte: K={CONTOUR_WEIGHT} KC={KONTUR_COV_WEIGHT} Cov={COVERAGE_WEIGHT}",
                     ha='center', fontsize=10, color='#444', transform=fig.transFigure)
            plt.text(0.5, 0.40, "Alle Scores (sortiert):", ha='center',
                     fontsize=10, fontweight='bold', transform=fig.transFigure)
            for i, (cls, sc, _, _, res) in enumerate(results[:8]):
                marker = " <- TRUE" if cls == true_class else ""
                plt.text(0.5, 0.32 - i*0.042,
                         f"{i+1}. {cls}:  {sc:.4f}"
                         f"  (K={res['contour_score']:.3f}"
                         f"  KC={res['kontur_cov_score']:.3f}"
                         f"  Cov={res['coverage_score']:.3f}){marker}",
                         ha='center', fontsize=9,
                         color='darkgreen' if cls == true_class else 'black',
                         transform=fig.transFigure)
            pdf.savefig(fig)
            plt.close()

            for rank, (cls, sc, aln, ref, res) in enumerate(results, 1):
                self._plot_step3_page(pdf, aln, ref, res,
                                      cls, true_class, rank)

        print(f"✓ {output_pdf}  ({len(results)} Seiten)")

    def _plot_step3_page(self, pdf, aligned, ref, res, ref_class,
                         true_class=None, rank=1):
        is_true = (true_class is not None and ref_class == true_class)
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.patch.set_facecolor('#f0fff0' if is_true else 'white')
        marker = "  ✓ TRUE CLASS" if is_true else ""
        fig.suptitle(
            f"#{rank}  {ref_class}{marker}\n"
            f"Score={res['score']:.4f}  K={res['contour_score']:.3f}  "
            f"KC={res['kontur_cov_score']:.3f}  Cov={res['coverage_score']:.3f}  "
            f"Scale={res['scale']:.2f}  X-Off={res['x_offset']:+.1f}px\n"
            f"[{self._feat_str()}]",
            fontsize=11, fontweight='bold',
            color='darkgreen' if is_true else 'black')

        axes[0].plot(ref[:, 0], ref[:, 1], 'r-', lw=1.5,
                     label='Referenz', alpha=0.7)
        axes[0].plot(aligned[:, 0], aligned[:, 1], 'b-', lw=1.5,
                     label='Fragment (1.0x)', alpha=0.7)
        axes[0].set_title('VOR Scale-Up')
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)
        axes[0].axis('equal')

        tf = res['transformed']
        axes[1].plot(ref[:, 0], ref[:, 1], 'r-', lw=1.5,
                     label='Referenz', alpha=0.7)
        axes[1].plot(tf[:, 0], tf[:, 1], 'b-', lw=1.5,
                     label=f'Fragment ({res["scale"]:.2f}x)', alpha=0.7)
        try:
            rp = Polygon(ref).buffer(0)
            fp = Polygon(tf).buffer(0)
            inter = rp.intersection(fp)
            if not inter.is_empty:
                geoms = [inter] if inter.geom_type == 'Polygon' else list(
                    inter.geoms)
                for g in geoms:
                    if g.geom_type == 'Polygon':
                        ix, iy = g.exterior.xy
                        axes[1].fill(ix, iy, alpha=0.35,
                                     color='green', label='Overlap')
        except Exception:
            pass

        y_lo, y_hi = tf[:, 1].min(), tf[:, 1].max()
        axes[1].axhline(y_lo, color='orange', ls='--', lw=1, alpha=0.7)
        axes[1].axhline(y_hi, color='orange', ls='--',
                        lw=1, alpha=0.7, label='Y-Streifen')

        if FEATURE_D_YWEIGHT:
            n_bands = 10
            y_bands = np.linspace(y_lo, y_hi, n_bands+1)
            for i in range(n_bands):
                w = ((y_bands[i]+y_bands[i+1])/2 - y_lo) / (y_hi - y_lo)
                w = w ** FEATURE_D_EXPONENT
                axes[1].axhspan(y_bands[i], y_bands[i+1],
                                xmin=0, xmax=0.02, alpha=w*0.6, color='blue')

        axes[1].set_title('NACH Scale-Up')
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)
        axes[1].axis('equal')
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

    def _plot_step1(self, pdf, orig, aligned, ref):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('SCHRITT 1: Alignment', fontsize=16, fontweight='bold')
        for ax, coords, title in zip(axes, [orig, aligned], ['VOR', 'NACH']):
            ax.plot(ref[:, 0], ref[:, 1], 'r-', lw=1.5,
                    label='Referenz', alpha=0.7)
            ax.plot(coords[:, 0], coords[:, 1], 'b-',
                    lw=1.5, label='Fragment', alpha=0.7)
            ax.scatter(*ref[np.argmax(ref[:, 1])], c='red', s=200, marker='*',
                       zorder=10, edgecolors='k', lw=2)
            ax.scatter(*coords[np.argmax(coords[:, 1])], c='blue', s=200, marker='*',
                       zorder=10, edgecolors='k', lw=2)
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.axis('equal')
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

    def _plot_step2(self, pdf, res):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f'SCHRITT 2: Kontur -> Scale={res["scale"]:.2f}  X-Off={res["x_offset"]:+.1f}px',
                     fontsize=14, fontweight='bold')
        axes[0].plot(res['scale_hist'], res['contour_hist'], 'b-o', lw=2, ms=4)
        axes[0].axvline(res['scale'], color='r', ls='--', lw=2,
                        label=f'Best={res["scale"]:.2f}')
        axes[0].set_xlabel('Scale')
        axes[0].set_ylabel('Kontur')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        if len(res['contour_hist']) > 1:
            imp = np.diff(res['contour_hist'])
            axes[1].plot(res['scale_hist'][1:], imp, 'b-o', lw=2, ms=4)
            axes[1].axhline(0, color='k', lw=0.5)
            axes[1].set_xlabel('Scale')
            axes[1].set_ylabel('Delta Kontur')
            axes[1].set_title('Early Stop')
            axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

    def load_references(self):
        print("\nLade Referenzen...")
        for p in self.reference_folder.glob('*.svg'):
            try:
                c = self._prep(p)
                self.reference_data[p.stem] = {'aligned': c}
            except Exception as e:
                print(f"  Fehler {p.name}: {e}")
        print(f"✓ {len(self.reference_data)} Referenzen\n")


# ===========================================================================
def main():
    print("="*70)
    print("TEMPLATE-MATCHING V17 + Kontur-Coverage Score")
    print(f"  A2 (sym. Y-Streifen): {'AN' if FEATURE_A2_SYMMETRIC else 'AUS'}")
    print(f"  Pre-X-Shift:          "
          f"'AN  Bereich=+/-'+str(PRE_XSHIFT_RANGE)+'px  Steps='+str(PRE_XSHIFT_STEPS) if PRE_XSHIFT_ENABLED else 'AUS'")
    print(
        f"  D  (Y-Gewichtung):    {'AN Exponent='+str(FEATURE_D_EXPONENT) if FEATURE_D_YWEIGHT else 'AUS'}")
    print(
        f"  Gewichte:  K={CONTOUR_WEIGHT}  KC={KONTUR_COV_WEIGHT}  Cov={COVERAGE_WEIGHT}")
    print(
        f"  KC Threshold: {KONTUR_THRESHOLD_REL*100:.0f}% der Fragment-Hoehe")
    print("="*70)

    ref_folder = input("\nReferenz-Ordner:  ").strip()
    test_folder = input("Test-Ordner:      ").strip()
    output_folder = input("Output-Ordner:    ").strip()

    clf = TemplateMatchingClassifier(ref_folder, test_folder, output_folder)
    clf.load_references()

    print("\nModus:")
    print("  1 = Klassifizierung (nur CSV + Konfusionsmatrix)")
    print("  2 = Klassifizierung + Top-K PDFs (1 PDF pro Fragment)")
    print("  3 = Batch-Visualisierung  (1 Fragment vs ALLE Referenzen)")
    mode = input("Modus [1/2/3]: ").strip()

    if mode == '2':
        gt = input("Ground Truth CSV: ").strip() or None
        top_k = int(input("Top-K [5]:        ").strip() or "5")
        vis_f = input("Ordner fuer PDFs:  ").strip()
        clf.classify_with_batch_vis(vis_f, gt, top_k)
    elif mode == '3':
        tf = input("Test-Datei:            ").strip()
        tc = input("True Class (optional): ").strip() or None
        if Path(tf).exists():
            out = Path(output_folder) / f"v17kc_batch_{Path(tf).stem}.pdf"
            clf.visualize_batch(tf, out, true_class=tc)
        else:
            print("Datei nicht gefunden.")
    else:
        gt = input("Ground Truth CSV: ").strip() or None
        top_k = int(input("Top-K [5]:        ").strip() or "5")
        clf.classify_with_ground_truth(gt, top_k)


if __name__ == "__main__":
    main()



#Beginn des Frontendteils

class ClassifierAPI:
    """
    Schnittstelle zwischen Backend-Logik und Frontend.

    Parameter
    ---------
    ref_folder    : Ordner mit Referenz-SVGs
    test_folder   : Ordner mit Test-Fragment-SVGs
    output_folder : Ordner für generierte Bilder / CSV
    top_k         : Wie viele Top-Treffer zurückgegeben werden (Standard: 5)
    """

    def __init__(self, ref_folder: str, test_folder: str,
                 output_folder: str, top_k: int = 5):
        self._clf = TemplateMatchingClassifier(
            ref_folder, test_folder, output_folder)
        self._clf.load_references()
        self.top_k = top_k

    # -----------------------------------------------------------------------
    # 1. LIST FRAGMENTS
    # -----------------------------------------------------------------------
    def list_fragments(self) -> dict:
        """
        Gibt alle verfügbaren Test-SVG-Dateinamen zurück.

        Rückgabe
        --------
        {
          "fragments": ["frag_001.svg", "frag_002.svg", ...]   # sortiert
        }

        Verwendung im Frontend
        ----------------------
        Dropdown-Menü befüllen: für jeden Eintrag in "fragments" einen
        Listeneintrag erstellen.
        """
        files = sorted(
            p.name for p in self._clf.test_folder.glob('*.svg'))
        return {"fragments": files}

    # -----------------------------------------------------------------------
    # 2. CLASSIFY FRAGMENT  (Haupt-Endpunkt)
    # -----------------------------------------------------------------------
    def classify_fragment(self, fragment_filename: str,
                          true_class: str = None) -> dict:
        """
        Klassifiziert ein einzelnes Fragment gegen alle Referenzen.

        Parameter
        ---------
        fragment_filename : Dateiname des Fragments (z. B. "frag_001.svg")
        true_class        : Optional — bekannte Klasse für Farbmarkierung

        Rückgabe
        --------
        {
          "fragment":   "frag_001.svg",
          "true_class": "Drag.33" | null,

          "top": [                          # top_k Einträge, nach Score sortiert
            {
              "rank":          1,
              "class":         "Drag.33",
              "score":         0.7821,      # Gesamtscore
              "contour_score": 1.234,       # K-Score
              "kc_score":      0.612,       # KC-Score
              "coverage_score":0.543,       # Coverage-Score
              "scale":         2.35,        # bester Scale-Faktor
              "x_offset":     +5.0,         # Pre-Shift in px
              "is_true_class": true | false,
              "overlap_image": "<base64-PNG>"  # Overlap-Plot als eingebettetes Bild
            },
            ...
          ],

          "fragment_image": "<base64-PNG>",  # Fragment-Kontur allein (links im UI)
          "predicted_class": "Drag.33",
          "correct": true | false | null     # null wenn true_class unbekannt
        }

        Verwendung im Frontend
        ----------------------
        - fragment_image  → linkes Bild (Fragment allein)
        - top[0].overlap_image → rechtes Bild (Overlap mit Referenz, Score etc.)
        - top[0..4]       → Score-Tabelle / Ranking-Liste
        - correct         → Hintergrundfarbe grün/rot setzen
        """
        svg_path = self._clf.test_folder / fragment_filename
        if not svg_path.exists():
            return {"error": f"Fragment nicht gefunden: {fragment_filename}"}

        try:
            tc = self._clf._prep(svg_path)
        except Exception as e:
            return {"error": f"SVG-Parsing fehlgeschlagen: {e}"}

        # Alle Referenzen matchen
        sims = []
        for cls, rd in self._clf.reference_data.items():
            ta = self._clf.align_fragment_to_reference(tc, rd['aligned'])
            res = self._clf.template_match_scaleup(ta, rd['aligned'])
            sims.append((cls, res['score'], ta, rd['aligned'], res))
        sims.sort(key=lambda x: x[1], reverse=True)

        top_entries = []
        for rank, (cls, sc, aln, ref, res) in enumerate(sims[:self.top_k], 1):
            overlap_b64 = self._render_overlap_png(
                aln, ref, res, cls, true_class, rank)
            top_entries.append({
                "rank":           rank,
                "class":          cls,
                "score":          round(float(sc), 4),
                "contour_score":  round(float(res['contour_score']), 4),
                "kc_score":       round(float(res['kontur_cov_score']), 4),
                "coverage_score": round(float(res['coverage_score']), 4),
                "scale":          round(float(res['scale']), 3),
                "x_offset":       round(float(res['x_offset']), 1),
                "is_true_class":  (true_class is not None and cls == true_class),
                "overlap_image":  overlap_b64,
            })

        pred = sims[0][0]
        correct = (pred == true_class) if true_class else None

        return {
            "fragment":        fragment_filename,
            "true_class":      true_class,
            "predicted_class": pred,
            "correct":         correct,
            "fragment_image":  self._render_fragment_png(tc),
            "top":             top_entries,
        }

    # -----------------------------------------------------------------------
    # 3. CLASSIFY ALL  (Batch-Endpunkt)
    # -----------------------------------------------------------------------
    def classify_all(self, ground_truth_csv: str = None) -> dict:
        """
        Klassifiziert alle Fragmente im Test-Ordner.
        Speichert Konfusionsmatrix-PNG und CSV im output_folder.

        Parameter
        ---------
        ground_truth_csv : Pfad zur CSV-Datei mit Spalten
                           'filename' und 'true_class' (optional)

        Rückgabe
        --------
        {
          "total":     41,
          "correct":   10,       # nur wenn ground_truth bekannt
          "accuracy":  0.2439,   # nur wenn ground_truth bekannt
          "results": [
            {
              "fragment":        "frag_001.svg",
              "true_class":      "Drag.33" | null,
              "predicted_class": "Drag.33",
              "score":           0.7821,
              "correct":         true | false | null,
              "top": [
                {"rank": 1, "class": "Drag.33",  "score": 0.7821},
                {"rank": 2, "class": "NB15",      "score": 0.6103},
                ...
              ]
            },
            ...
          ],
          "confusion_matrix_path": "output/confusion_matrix_....png",  # oder null
          "csv_path":              "output/results_....csv"
        }

        Verwendung im Frontend
        ----------------------
        - Fortschrittsbalken: results wird schrittweise befüllt
          (Implementierung mit Callback/Generator je nach Framework)
        - Tabelle mit allen Ergebnissen aufbauen
        - accuracy und confusion_matrix_path anzeigen
        """
        import pandas as pd
        from sklearn.metrics import accuracy_score

        gt = {}
        if ground_truth_csv and Path(ground_truth_csv).exists():
            df = pd.read_csv(ground_truth_csv)
            gt = dict(zip(df['filename'], df['true_class']))

        test_files = sorted(self._clf.test_folder.glob('*.svg'))
        all_results = []
        y_true, y_pred = [], []

        for tp in test_files:
            try:
                tc = self._clf._prep(tp)
                sims = []
                for cls, rd in self._clf.reference_data.items():
                    ta = self._clf.align_fragment_to_reference(
                        tc, rd['aligned'])
                    res = self._clf.template_match_scaleup(ta, rd['aligned'])
                    sims.append((cls, res['score'], res))
                sims.sort(key=lambda x: x[1], reverse=True)

                true_cls = gt.get(tp.name)
                pred_cls = sims[0][0]

                if true_cls:
                    y_true.append(true_cls)
                    y_pred.append(pred_cls)

                all_results.append({
                    "fragment":        tp.name,
                    "true_class":      true_cls,
                    "predicted_class": pred_cls,
                    "score":           round(float(sims[0][1]), 4),
                    "correct":         (pred_cls == true_cls) if true_cls else None,
                    "top": [
                        {"rank": i+1,
                         "class": cls,
                         "score": round(float(sc), 4)}
                        for i, (cls, sc, _) in enumerate(sims[:self.top_k])
                    ],
                })
                print(f"  ✓ {tp.name} -> {pred_cls}")
            except Exception as e:
                print(f"  ✗ {tp.name}: {e}")
                all_results.append({
                    "fragment": tp.name,
                    "error":    str(e),
                })

        # Konfusionsmatrix + CSV speichern
        cm_path = None
        csv_path = None
        if y_true:
            # Intern die bestehenden Spar-Methoden nutzen
            dummy_results = [
                {"filename": r["fragment"],
                 "true_class": r.get("true_class"),
                 "predicted_class": r.get("predicted_class", ""),
                 "confidence": r.get("score", 0),
                 "top_matches": [(e["class"], e["score"], {
                     "contour_score": 0, "kontur_cov_score": 0,
                     "coverage_score": 0})
                     for e in r.get("top", [])]}
                for r in all_results if "error" not in r
            ]
            self._clf._save_cm(y_true, y_pred)
            self._clf._save_csv(dummy_results, self.top_k)

            # Neueste Dateipfade ermitteln
            pngs = sorted(self._clf.output_folder.glob(
                'confusion_matrix_*.png'))
            csvs = sorted(self._clf.output_folder.glob('results_*.csv'))
            if pngs:
                cm_path = str(pngs[-1])
            if csvs:
                csv_path = str(csvs[-1])

        acc = accuracy_score(y_true, y_pred) if y_true else None
        return {
            "total":                  len(test_files),
            "correct":                sum(1 for r in all_results
                                          if r.get("correct") is True),
            "accuracy":               round(acc, 4) if acc is not None else None,
            "results":                all_results,
            "confusion_matrix_path":  cm_path,
            "csv_path":               csv_path,
        }

    # -----------------------------------------------------------------------
    # Interne Render-Hilfsmethoden (geben base64-PNG-Strings zurück)
    # -----------------------------------------------------------------------
    def _render_fragment_png(self, coords) -> str:
        """Fragment-Kontur allein als base64-PNG."""
        fig, ax = plt.subplots(figsize=(4, 5))
        ax.plot(coords[:, 0], coords[:, 1], 'b-', lw=1.5)
        ax.set_title('Fragment', fontsize=10)
        ax.axis('equal')
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#fafafa')
        plt.tight_layout()
        return self._fig_to_b64(fig)

    def _render_overlap_png(self, aligned, ref, res,
                            ref_class, true_class, rank) -> str:
        """
        Overlap-Plot (NACH Scale-Up) als base64-PNG.
        Enthält: Referenz (rot), Fragment (blau), Overlap (grün),
        Y-Streifen-Grenzen (orange), Y-Gewichtungs-Balken (blau links),
        Score-Annotation im Titel.
        """
        is_true = (true_class is not None and ref_class == true_class)
        fig, ax = plt.subplots(figsize=(5, 5))
        fig.patch.set_facecolor('#f0fff0' if is_true else 'white')

        tf = res['transformed']
        ax.plot(ref[:, 0], ref[:, 1],  'r-', lw=1.5,
                label='Referenz', alpha=0.8)
        ax.plot(tf[:,  0], tf[:,  1],  'b-', lw=1.5,
                label=f'Fragment ({res["scale"]:.2f}x)', alpha=0.8)

        # Overlap-Fläche grün einfärben
        try:
            rp = Polygon(ref).buffer(0)
            fp = Polygon(tf).buffer(0)
            inter = rp.intersection(fp)
            if not inter.is_empty:
                geoms = ([inter] if inter.geom_type == 'Polygon'
                         else list(inter.geoms))
                for g in geoms:
                    if g.geom_type == 'Polygon':
                        ix, iy = g.exterior.xy
                        ax.fill(ix, iy, alpha=0.35,
                                color='green', label='Overlap')
        except Exception:
            pass

        # Y-Streifen-Grenzen
        y_lo, y_hi = tf[:, 1].min(), tf[:, 1].max()
        ax.axhline(y_lo, color='orange', ls='--', lw=1, alpha=0.8,
                   label='Y-Streifen')
        ax.axhline(y_hi, color='orange', ls='--', lw=1, alpha=0.8)

        # Y-Gewichtungs-Balken (linker Rand)
        if FEATURE_D_YWEIGHT:
            n_bands = 10
            y_bands = np.linspace(y_lo, y_hi, n_bands + 1)
            for i in range(n_bands):
                w = (((y_bands[i] + y_bands[i+1]) / 2 - y_lo)
                     / (y_hi - y_lo)) ** FEATURE_D_EXPONENT
                ax.axhspan(y_bands[i], y_bands[i+1],
                           xmin=0, xmax=0.02, alpha=w * 0.6, color='blue')

        marker = '  ✓ TRUE' if is_true else ''
        ax.set_title(
            f"#{rank} {ref_class}{marker}\n"
            f"Score={res['score']:.4f}  K={res['contour_score']:.3f}  "
            f"KC={res['kontur_cov_score']:.3f}  Cov={res['coverage_score']:.3f}\n"
            f"Scale={res['scale']:.2f}x  X-Off={res['x_offset']:+.1f}px",
            fontsize=8,
            color='darkgreen' if is_true else 'black'
        )
        ax.axis('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc='lower right')
        plt.tight_layout()
        return self._fig_to_b64(fig)

    @staticmethod
    def _fig_to_b64(fig) -> str:
        """Matplotlib-Figure → base64-kodierter PNG-String."""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')



