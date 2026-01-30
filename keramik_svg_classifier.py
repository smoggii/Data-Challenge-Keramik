import numpy as np
from pathlib import Path
import xml.etree.ElementTree as ET
from scipy.spatial.distance import cosine, euclidean
from scipy.interpolate import interp1d
import csv
from datetime import datetime
import re


class SVGKeramikClassifier:
    """Klassifiziert Keramik-Profile basierend auf SVG-Pfad-Features"""

    def __init__(self, reference_folder, test_folder, output_folder):
        self.reference_folder = Path(reference_folder)
        self.test_folder = Path(test_folder)
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(exist_ok=True)

        self.reference_features = {}  # {filename: features}

    def parse_svg_path(self, svg_path):
        """
        Parst eine SVG-Datei und extrahiert die Pfad-Koordinaten

        Returns:
        - numpy array mit (x, y) Koordinaten
        """
        tree = ET.parse(svg_path)
        root = tree.getroot()

        # Verschiedene Namespaces die vorkommen können
        namespaces = {
            'svg': 'http://www.w3.org/2000/svg',
            'inkscape': 'http://www.inkscape.org/namespaces/inkscape',
            'sodipodi': 'http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd'
        }

        # Methode 1: Mit namespace
        paths = root.findall('.//svg:path', namespaces)

        # Methode 2: Ohne namespace
        if not paths:
            paths = root.findall('.//path')

        # Methode 3: Alle Elemente durchsuchen (für komplexe SVGs)
        if not paths:
            paths = []
            for elem in root.iter():
                # Entferne namespace aus tag
                tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
                if tag == 'path' and elem.get('d'):
                    paths.append(elem)

        # Methode 4: Auch polyline und polygon unterstützen
        if not paths:
            # Versuche polyline
            polylines = root.findall('.//polyline')
            if not polylines:
                polylines = []
                for elem in root.iter():
                    tag = elem.tag.split(
                        '}')[-1] if '}' in elem.tag else elem.tag
                    if tag == 'polyline' and elem.get('points'):
                        polylines.append(elem)

            if polylines:
                # Konvertiere polyline zu path
                polyline = max(
                    polylines, key=lambda p: len(p.get('points', '')))
                points_str = polyline.get('points', '')
                coords = self.parse_polyline_points(points_str)
                return coords

            # Versuche polygon
            polygons = root.findall('.//polygon')
            if not polygons:
                polygons = []
                for elem in root.iter():
                    tag = elem.tag.split(
                        '}')[-1] if '}' in elem.tag else elem.tag
                    if tag == 'polygon' and elem.get('points'):
                        polygons.append(elem)

            if polygons:
                polygon = max(polygons, key=lambda p: len(p.get('points', '')))
                points_str = polygon.get('points', '')
                coords = self.parse_polyline_points(points_str)
                return coords

        if not paths:
            # Debug: Zeige alle verfügbaren Elemente
            all_tags = set()
            for elem in root.iter():
                tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
                all_tags.add(tag)

            raise ValueError(
                f"Keine Pfade in {svg_path} gefunden\n"
                f"Verfügbare Elemente: {', '.join(sorted(all_tags))}\n"
                f"Tipp: Prüfe die SVG-Struktur"
            )

        # Nimm den längsten Pfad
        path = max(paths, key=lambda p: len(p.get('d', '')))
        path_data = path.get('d', '')

        if not path_data:
            raise ValueError(f"Pfad in {svg_path} hat keine 'd' Daten")

        # Parse den Pfad
        coords = self.parse_path_data(path_data)

        return coords

    def parse_polyline_points(self, points_str):
        """
        Parst polyline/polygon points Attribut
        Format: "x1,y1 x2,y2 x3,y3" oder "x1 y1 x2 y2"
        """
        # Entferne mehrfache Leerzeichen und Kommas
        points_str = points_str.replace(',', ' ')
        numbers = points_str.split()

        coords = []
        for i in range(0, len(numbers)-1, 2):
            try:
                x = float(numbers[i])
                y = float(numbers[i+1])
                coords.append([x, y])
            except (ValueError, IndexError):
                continue

        if not coords:
            raise ValueError(
                "Keine Koordinaten aus polyline/polygon extrahiert")

        return np.array(coords)

    def parse_path_data(self, path_data):
        """
        Parst SVG-Pfad-Daten und konvertiert zu Koordinaten

        Unterstützt: M (moveto), L (lineto), C (curveto), relative/absolute
        """
        coords = []

        # Entferne Kommas und ersetze durch Leerzeichen
        path_data = path_data.replace(',', ' ')

        # Split nach Kommandos
        commands = re.findall(r'[MLCZmlcz][^MLCZmlcz]*', path_data)

        current_pos = np.array([0.0, 0.0])

        for cmd in commands:
            cmd_type = cmd[0]
            # Extrahiere alle Zahlen (inklusive negative und Dezimalzahlen)
            numbers = re.findall(r'-?\d+\.?\d*', cmd[1:])
            numbers = [float(n) for n in numbers]

            if cmd_type == 'M':  # Absolute moveto
                if len(numbers) >= 2:
                    current_pos = np.array([numbers[0], numbers[1]])
                    coords.append(current_pos.copy())

            elif cmd_type == 'm':  # Relative moveto
                if len(numbers) >= 2:
                    current_pos += np.array([numbers[0], numbers[1]])
                    coords.append(current_pos.copy())

            elif cmd_type == 'L':  # Absolute lineto
                for i in range(0, len(numbers), 2):
                    if i + 1 < len(numbers):
                        current_pos = np.array([numbers[i], numbers[i+1]])
                        coords.append(current_pos.copy())

            elif cmd_type == 'l':  # Relative lineto
                for i in range(0, len(numbers), 2):
                    if i + 1 < len(numbers):
                        current_pos += np.array([numbers[i], numbers[i+1]])
                        coords.append(current_pos.copy())

            elif cmd_type == 'C':  # Absolute cubic Bezier
                # C hat 6 Parameter: x1,y1 x2,y2 x,y
                for i in range(0, len(numbers), 6):
                    if i + 5 < len(numbers):
                        # Vereinfachung: nehme nur Endpunkt
                        current_pos = np.array([numbers[i+4], numbers[i+5]])
                        coords.append(current_pos.copy())

            elif cmd_type == 'c':  # Relative cubic Bezier
                for i in range(0, len(numbers), 6):
                    if i + 5 < len(numbers):
                        current_pos += np.array([numbers[i+4], numbers[i+5]])
                        coords.append(current_pos.copy())

            # Z/z (closepath) ignorieren wir hier

        if not coords:
            raise ValueError("Keine Koordinaten aus Pfad extrahiert")

        return np.array(coords)

    def normalize_coordinates(self, coords):
        """
        Normalisiert Koordinaten:
        - Zentriert auf (0, 0)
        - Skaliert auf Einheitsgröße
        """
        # Zentrieren
        centroid = coords.mean(axis=0)
        coords_centered = coords - centroid

        # Skalieren (auf max. Ausdehnung = 1)
        max_extent = np.max(np.abs(coords_centered))
        if max_extent > 0:
            coords_normalized = coords_centered / max_extent
        else:
            coords_normalized = coords_centered

        return coords_normalized

    def resample_path(self, coords, n_points=200):
        """
        Resampled den Pfad auf eine feste Anzahl von Punkten
        für bessere Vergleichbarkeit
        """
        if len(coords) < 2:
            return coords

        # Berechne kumulative Distanzen entlang des Pfads
        distances = np.sqrt(np.sum(np.diff(coords, axis=0)**2, axis=1))
        cum_distances = np.concatenate([[0], np.cumsum(distances)])

        # Interpoliere
        if cum_distances[-1] == 0:
            # Alle Punkte identisch
            return np.tile(coords[0], (n_points, 1))

        # Erstelle Interpolationsfunktionen
        interp_x = interp1d(cum_distances, coords[:, 0], kind='linear',
                            fill_value='extrapolate')
        interp_y = interp1d(cum_distances, coords[:, 1], kind='linear',
                            fill_value='extrapolate')

        # Sample gleichmäßig entlang des Pfads
        sample_distances = np.linspace(0, cum_distances[-1], n_points)
        resampled = np.column_stack([interp_x(sample_distances),
                                     interp_y(sample_distances)])

        return resampled

    def extract_features(self, svg_path):
        """
        Extrahiert Features aus SVG-Pfad

        Returns:
        - Dictionary mit verschiedenen Feature-Typen
        """
        # Pfad parsen und Koordinaten extrahieren
        coords = self.parse_svg_path(svg_path)

        # Normalisieren
        coords_norm = self.normalize_coordinates(coords)

        # Resampling für konsistente Vergleiche
        coords_resampled = self.resample_path(coords_norm, n_points=200)

        features = {}

        # 1. Direkter Koordinaten-Descriptor
        # Flatten für einfacheren Vergleich
        features['coords_descriptor'] = coords_resampled.flatten()

        # 2. Geometrische Momente
        moments = self.compute_geometric_moments(coords_norm)
        features['moments'] = moments

        # 3. Fourier-Deskriptoren (Frequenzdomäne)
        fourier_desc = self.compute_fourier_descriptors(coords_resampled)
        features['fourier'] = fourier_desc

        # 4. Curvature-Features (Krümmung entlang des Pfads)
        curvature = self.compute_curvature(coords_resampled)
        features['curvature'] = curvature

        # 5. Statistische Form-Features
        shape_stats = self.compute_shape_statistics(coords_norm)
        features['shape_stats'] = shape_stats

        # 6. Aspect Ratio und Bounding Box
        bbox_features = self.compute_bbox_features(coords_norm)
        features['bbox'] = bbox_features

        return features

    def compute_geometric_moments(self, coords):
        """Berechnet geometrische Momente (ähnlich zu Hu-Momenten)"""
        x = coords[:, 0]
        y = coords[:, 1]

        # Zentrale Momente
        m00 = len(coords)
        m10 = np.sum(x)
        m01 = np.sum(y)
        m20 = np.sum(x**2)
        m02 = np.sum(y**2)
        m11 = np.sum(x * y)
        m30 = np.sum(x**3)
        m03 = np.sum(y**3)
        m21 = np.sum(x**2 * y)
        m12 = np.sum(x * y**2)

        # Normalisierte Momente
        moments = np.array([
            m20/m00 - (m10/m00)**2,  # Varianz X
            m02/m00 - (m01/m00)**2,  # Varianz Y
            m11/m00 - (m10/m00)*(m01/m00),  # Kovarianz
            m30/m00,
            m03/m00,
            m21/m00,
            m12/m00
        ])

        return moments

    def compute_fourier_descriptors(self, coords, n_descriptors=20):
        """
        Berechnet Fourier-Deskriptoren der Kontur
        (Rotations- und translationsinvariant)
        """
        # Komplexe Repräsentation
        complex_coords = coords[:, 0] + 1j * coords[:, 1]

        # FFT
        fft_result = np.fft.fft(complex_coords)

        # Nimm die ersten n_descriptors (niedrige Frequenzen)
        # Skip DC component (Index 0) für Translationsinvarianz
        descriptors = np.abs(fft_result[1:n_descriptors+1])

        # Normalisiere auf ersten Descriptor für Skalierungsinvarianz
        if descriptors[0] != 0:
            descriptors = descriptors / descriptors[0]

        return descriptors

    def compute_curvature(self, coords):
        """
        Berechnet Krümmung entlang des Pfads
        """
        # Erste und zweite Ableitungen (diskret)
        if len(coords) < 3:
            return np.array([0])

        # Zentrale Differenzen
        dx = np.gradient(coords[:, 0])
        dy = np.gradient(coords[:, 1])
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)

        # Krümmung: κ = (x'y'' - y'x'') / (x'^2 + y'^2)^(3/2)
        numerator = dx * ddy - dy * ddx
        denominator = (dx**2 + dy**2)**(3/2) + 1e-10
        curvature = numerator / denominator

        # Statistiken der Krümmung
        curvature_stats = np.array([
            np.mean(curvature),
            np.std(curvature),
            np.max(curvature),
            np.min(curvature),
            np.median(curvature)
        ])

        return curvature_stats

    def compute_shape_statistics(self, coords):
        """Berechnet statistische Form-Eigenschaften"""
        # Distanzen vom Zentrum
        distances = np.sqrt(np.sum(coords**2, axis=1))

        # Winkel
        angles = np.arctan2(coords[:, 1], coords[:, 0])

        stats = np.array([
            np.mean(distances),
            np.std(distances),
            np.max(distances),
            np.min(distances),
            np.mean(angles),
            np.std(angles)
        ])

        return stats

    def compute_bbox_features(self, coords):
        """Berechnet Bounding-Box-Features"""
        x_min, y_min = coords.min(axis=0)
        x_max, y_max = coords.max(axis=0)

        width = x_max - x_min
        height = y_max - y_min
        aspect_ratio = height / (width + 1e-10)
        area = width * height

        # Convex Hull Fläche (vereinfacht: Polygon-Fläche)
        polygon_area = self.compute_polygon_area(coords)

        features = np.array([
            width,
            height,
            aspect_ratio,
            area,
            polygon_area,
            polygon_area / (area + 1e-10)  # Konvexität
        ])

        return features

    def compute_polygon_area(self, coords):
        """Berechnet Fläche eines Polygons (Shoelace-Formel)"""
        x = coords[:, 0]
        y = coords[:, 1]
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    def compare_features(self, features1, features2):
        """
        Vergleicht zwei Feature-Sets

        Returns:
        - Similarity Score (0-1, höher = ähnlicher)
        - Details der einzelnen Vergleiche
        """
        similarities = []

        # 1. Koordinaten-Descriptor (Hauptgewicht)
        if 'coords_descriptor' in features1 and 'coords_descriptor' in features2:
            coords_sim = 1 - cosine(features1['coords_descriptor'],
                                    features2['coords_descriptor'])
            similarities.append(('coords', coords_sim, 0.30))

        # 2. Fourier-Deskriptoren (Formähnlichkeit)
        if 'fourier' in features1 and 'fourier' in features2:
            fourier_dist = euclidean(
                features1['fourier'], features2['fourier'])
            fourier_sim = 1 / (1 + fourier_dist)
            similarities.append(('fourier', fourier_sim, 0.25))

        # 3. Geometrische Momente
        if 'moments' in features1 and 'moments' in features2:
            moments_dist = euclidean(
                features1['moments'], features2['moments'])
            moments_sim = 1 / (1 + moments_dist)
            similarities.append(('moments', moments_sim, 0.20))

        # 4. Krümmung
        if 'curvature' in features1 and 'curvature' in features2:
            curv_dist = euclidean(
                features1['curvature'], features2['curvature'])
            curv_sim = 1 / (1 + curv_dist * 10)  # Skalierung
            similarities.append(('curvature', curv_sim, 0.10))

        # 5. Shape Statistics
        if 'shape_stats' in features1 and 'shape_stats' in features2:
            stats_dist = euclidean(
                features1['shape_stats'], features2['shape_stats'])
            stats_sim = 1 / (1 + stats_dist)
            similarities.append(('shape_stats', stats_sim, 0.10))

        # 6. Bounding Box Features
        if 'bbox' in features1 and 'bbox' in features2:
            bbox_dist = euclidean(features1['bbox'], features2['bbox'])
            bbox_sim = 1 / (1 + bbox_dist)
            similarities.append(('bbox', bbox_sim, 0.05))

        # Gewichteter Durchschnitt
        weighted_sum = sum(sim * weight for _, sim, weight in similarities)
        total_weight = sum(weight for _, _, weight in similarities)

        final_similarity = weighted_sum / total_weight if total_weight > 0 else 0

        # Details für Debug
        detail = {name: f"{sim:.3f}" for name, sim, _ in similarities}

        return final_similarity, detail

    def load_references(self):
        """Lädt alle Referenz-SVGs und extrahiert Features"""
        print("=" * 60)
        print("LADE REFERENZ-SVGs")
        print("=" * 60)

        reference_files = list(self.reference_folder.glob('*.svg'))

        print(f"Gefunden: {len(reference_files)} Referenz-SVGs\n")

        for idx, ref_path in enumerate(reference_files, 1):
            print(f"[{idx}/{len(reference_files)}] Verarbeite: {ref_path.name}")

            try:
                features = self.extract_features(ref_path)

                # Klassenname = Dateiname ohne Extension
                class_name = ref_path.stem

                self.reference_features[class_name] = features

            except Exception as e:
                print(f"  ⚠ Fehler: {e}")

        print(
            f"\n✓ {len(self.reference_features)} Referenzen erfolgreich geladen")

    def classify_test_images(self, top_k=5):
        """
        Klassifiziert alle Test-SVGs

        Parameters:
        - top_k: Wie viele Top-Matches zurückgeben
        """
        print("\n" + "=" * 60)
        print("KLASSIFIZIERE TEST-SVGs")
        print("=" * 60)

        if not self.reference_features:
            print("❌ Fehler: Keine Referenzen geladen!")
            return

        test_files = list(self.test_folder.glob('*.svg'))

        print(f"Gefunden: {len(test_files)} Test-SVGs\n")

        results = []

        for idx, test_path in enumerate(test_files, 1):
            print(f"\n[{idx}/{len(test_files)}] Klassifiziere: {test_path.name}")

            try:
                # Features extrahieren
                test_features = self.extract_features(test_path)

                # Mit allen Referenzen vergleichen
                similarities = []
                for class_name, ref_features in self.reference_features.items():
                    similarity, detail = self.compare_features(
                        test_features, ref_features)
                    similarities.append((class_name, similarity, detail))

                # Sortieren nach Similarity (höchste zuerst)
                similarities.sort(key=lambda x: x[1], reverse=True)

                # Top-K nehmen
                top_matches = similarities[:top_k]

                # Ausgeben
                print(f"  📊 Top {top_k} Matches:")
                for rank, (class_name, sim, detail) in enumerate(top_matches, 1):
                    print(
                        f"    {rank}. {class_name:30s} - Similarity: {sim:.3f}")
                    print(f"       Details: {detail}")

                # Ergebnis speichern
                result = {
                    'test_image': test_path.name,
                    'top_matches': top_matches
                }
                results.append(result)

            except Exception as e:
                print(f"  ❌ Fehler: {e}")
                import traceback
                traceback.print_exc()

        # Ergebnisse in CSV speichern
        self.save_results_csv(results, top_k)

        print(f"\n✅ Klassifizierung abgeschlossen!")
        return results

    def save_results_csv(self, results, top_k):
        """Speichert Ergebnisse in CSV-Datei"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = self.output_folder / \
            f"svg_classification_results_{timestamp}.csv"

        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # Header
            header = ['Test_SVG']
            for i in range(1, top_k + 1):
                header.extend([f'Match_{i}_Class', f'Match_{i}_Similarity'])
            writer.writerow(header)

            # Daten
            for result in results:
                row = [result['test_image']]
                for class_name, similarity, _ in result['top_matches']:
                    row.extend([class_name, f"{similarity:.4f}"])
                writer.writerow(row)

        print(f"\n📊 Ergebnisse gespeichert: {csv_path}")


def main():
    """Hauptfunktion"""

    print("=" * 60)
    print("SVG KERAMIK-KLASSIFIZIERER")
    print("Römische Keramik-Querschnitte")
    print("=" * 60)

    # Pfade eingeben
    print("\n📁 Ordner-Konfiguration:")
    reference_folder = input(
        "Ordner mit Referenz-SVGs (Literaturbilder): ").strip()
    test_folder = input("Ordner mit Test-SVGs (zu klassifizieren): ").strip()
    output_folder = input("Ausgabe-Ordner für Ergebnisse: ").strip()

    # Validierung
    if not Path(reference_folder).exists():
        print(f"❌ Fehler: Referenz-Ordner existiert nicht: {reference_folder}")
        return
    if not Path(test_folder).exists():
        print(f"❌ Fehler: Test-Ordner existiert nicht: {test_folder}")
        return

    # Classifier erstellen
    classifier = SVGKeramikClassifier(
        reference_folder, test_folder, output_folder)

    # Referenzen laden
    classifier.load_references()

    if not classifier.reference_features:
        print("\n❌ Keine Referenzen geladen. Abbruch.")
        return

    # Testbilder klassifizieren
    top_k = int(input("\nWie viele Top-Matches anzeigen? [5]: ") or "5")
    classifier.classify_test_images(top_k=top_k)


if __name__ == "__main__":
    main()
