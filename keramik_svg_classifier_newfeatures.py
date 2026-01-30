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

        # Nimm den längsten Pfad (sollte der Querschnitt sein)
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

        # 7. Symmetrie-Features (wichtig für Keramik!)
        symmetry_features = self.compute_symmetry_features(coords_resampled)
        features['symmetry'] = symmetry_features

        # 8. Radialer Profil-Descriptor (speziell für Rotationskörper)
        radial_profile = self.compute_radial_profile(coords_resampled)
        features['radial_profile'] = radial_profile

        # 9. Wandstärken-Analyse (Dicke des Gefäßes)
        wall_thickness = self.compute_wall_thickness(coords_resampled)
        features['wall_thickness'] = wall_thickness

        # 10. Kontur-Komplexität (Glätte vs. Verzierung)
        complexity = self.compute_contour_complexity(coords_resampled)
        features['complexity'] = complexity

        # 11. Lokale Features (kritische Punkte: Rand, Bauch, Fuß)
        local_features = self.compute_local_features(coords_resampled)
        features['local_features'] = local_features

        # 12. Winkel-Features (Neigungen, Schwünge)
        angle_features = self.compute_angle_features(coords_resampled)
        features['angle_features'] = angle_features

        # 13. Skelett-Features (Medial Axis)
        skeleton_features = self.compute_skeleton_features(coords_norm)
        features['skeleton'] = skeleton_features

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

    def compute_symmetry_features(self, coords):
        """
        Berechnet Symmetrie-Features (wichtig für Rotationskeramik)
        Keramik ist oft achsensymmetrisch!
        """
        # Annahme: Y-Achse ist Symmetrieachse (vertikale Achse des Gefäßes)
        x = coords[:, 0]
        y = coords[:, 1]

        # Spiegle Koordinaten an Y-Achse
        coords_mirrored = coords.copy()
        coords_mirrored[:, 0] = -coords_mirrored[:, 0]

        # Berechne Abstand zwischen Original und Spiegelung
        # (bei perfekter Symmetrie = 0)
        symmetry_error = np.mean(np.min(
            np.sqrt(np.sum(
                (coords[:, np.newaxis, :] - coords_mirrored[np.newaxis, :, :])**2, axis=2)),
            axis=1
        ))

        # Verteilung links vs. rechts von Achse
        left_points = np.sum(x < 0)
        right_points = np.sum(x > 0)
        balance = min(left_points, right_points) / \
            (max(left_points, right_points) + 1)

        # Momente-basierte Symmetrie
        m20 = np.sum(x**2)
        m02 = np.sum(y**2)
        m11 = np.sum(x * y)

        # Hauptachsen-Orientierung
        theta = 0.5 * np.arctan2(2*m11, m20 - m02)

        features = np.array([
            symmetry_error,
            balance,
            np.abs(theta),  # Abweichung von vertikaler Achse
            np.std(x[x < 0]) if np.any(x < 0) else 0,  # Variation links
            np.std(x[x > 0]) if np.any(x > 0) else 0   # Variation rechts
        ])

        return features

    def compute_radial_profile(self, coords, n_bins=50):
        """
        Radialer Profil-Descriptor
        Beschreibt die Form als Funktion des Radius vom Zentrum
        Sehr gut für Rotationskörper wie Keramik!
        """
        # Polarkoordinaten
        x = coords[:, 0]
        y = coords[:, 1]

        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)

        # Sortiere nach Winkel
        sorted_indices = np.argsort(theta)
        theta_sorted = theta[sorted_indices]
        r_sorted = r[sorted_indices]

        # Binning nach Winkel
        angle_bins = np.linspace(-np.pi, np.pi, n_bins)
        radial_profile = []

        for i in range(len(angle_bins) - 1):
            mask = (theta_sorted >= angle_bins[i]) & (
                theta_sorted < angle_bins[i+1])
            if np.any(mask):
                radial_profile.append(np.mean(r_sorted[mask]))
            else:
                radial_profile.append(0)

        return np.array(radial_profile)

    def compute_wall_thickness(self, coords):
        """
        Schätzt Wandstärke des Gefäßes
        Analysiert lokale "Dicke" der Kontur
        """
        x = coords[:, 0]
        y = coords[:, 1]

        # Teile in Höhen-Segmente
        y_min, y_max = y.min(), y.max()
        n_segments = 10
        thickness_profile = []

        for i in range(n_segments):
            y_low = y_min + (y_max - y_min) * i / n_segments
            y_high = y_min + (y_max - y_min) * (i + 1) / n_segments

            mask = (y >= y_low) & (y < y_high)
            if np.any(mask):
                x_segment = x[mask]
                thickness = x_segment.max() - x_segment.min()
                thickness_profile.append(thickness)
            else:
                thickness_profile.append(0)

        # Statistiken der Wandstärke
        thickness_array = np.array(thickness_profile)
        features = np.array([
            np.mean(thickness_array),
            np.std(thickness_array),
            np.max(thickness_array),
            np.min(thickness_array[thickness_array > 0]) if np.any(
                thickness_array > 0) else 0,
            np.median(thickness_array)
        ])

        return features

    def compute_contour_complexity(self, coords):
        """
        Misst Komplexität/Glätte der Kontur
        Hohe Komplexität = viele Details, Verzierungen
        Niedrige Komplexität = glatte Form
        """
        # 1. Fraktale Dimension (Box-Counting vereinfacht)
        path_length = np.sum(
            np.sqrt(np.sum(np.diff(coords, axis=0)**2, axis=1)))
        straight_distance = np.sqrt(np.sum((coords[-1] - coords[0])**2))
        tortuosity = path_length / (straight_distance + 1e-10)

        # 2. Anzahl der Richtungsänderungen
        angles = np.arctan2(np.diff(coords[:, 1]), np.diff(coords[:, 0]))
        angle_changes = np.abs(np.diff(angles))
        angle_changes = np.minimum(
            angle_changes, 2*np.pi - angle_changes)  # Kleinerer Winkel
        n_significant_changes = np.sum(angle_changes > 0.1)

        # 3. Hochfrequenz-Energie (aus Fourier)
        complex_coords = coords[:, 0] + 1j * coords[:, 1]
        fft_result = np.fft.fft(complex_coords)
        fft_power = np.abs(fft_result)**2

        # Verhältnis hohe zu niedrige Frequenzen
        low_freq_energy = np.sum(fft_power[:len(fft_power)//10])
        high_freq_energy = np.sum(fft_power[len(fft_power)//10:])
        freq_ratio = high_freq_energy / (low_freq_energy + 1e-10)

        # 4. Lokale Variabilität
        local_distances = np.sqrt(np.sum(np.diff(coords, axis=0)**2, axis=1))
        distance_variability = np.std(
            local_distances) / (np.mean(local_distances) + 1e-10)

        features = np.array([
            tortuosity,
            n_significant_changes,
            freq_ratio,
            distance_variability,
            path_length
        ])

        return features

    def compute_local_features(self, coords):
        """
        Extrahiert Features von kritischen Bereichen:
        - Rand/Öffnung (oben)
        - Bauch/maximale Breite (Mitte)
        - Fuß/Basis (unten)
        """
        y = coords[:, 1]
        x = coords[:, 0]

        y_min, y_max = y.min(), y.max()
        height = y_max - y_min

        # Definiere Bereiche
        top_mask = y > (y_max - height * 0.2)  # Oberste 20%
        middle_mask = (y > (y_min + height * 0.4)) & (y <
                                                      (y_max - height * 0.4))  # Mittlere 20%
        bottom_mask = y < (y_min + height * 0.2)  # Unterste 20%

        features = []

        # Rand-Features
        if np.any(top_mask):
            top_width = x[top_mask].max() - x[top_mask].min()
            top_curvature = np.std(x[top_mask])
        else:
            top_width = 0
            top_curvature = 0

        # Bauch-Features
        if np.any(middle_mask):
            middle_width = x[middle_mask].max() - x[middle_mask].min()
            middle_curvature = np.std(x[middle_mask])
        else:
            middle_width = 0
            middle_curvature = 0

        # Fuß-Features
        if np.any(bottom_mask):
            bottom_width = x[bottom_mask].max() - x[bottom_mask].min()
            bottom_curvature = np.std(x[bottom_mask])
        else:
            bottom_width = 0
            bottom_curvature = 0

        # Verhältnisse
        rim_to_belly = top_width / (middle_width + 1e-10)
        belly_to_base = middle_width / (bottom_width + 1e-10)

        features = np.array([
            top_width,
            middle_width,
            bottom_width,
            rim_to_belly,
            belly_to_base,
            top_curvature,
            middle_curvature,
            bottom_curvature
        ])

        return features

    def compute_angle_features(self, coords):
        """
        Analysiert Winkel und Neigungen entlang der Kontur
        Wichtig für Gefäßformen (ausladend, eingezogen, etc.)
        """
        # Tangenten-Winkel
        dx = np.diff(coords[:, 0])
        dy = np.diff(coords[:, 1])
        tangent_angles = np.arctan2(dy, dx)

        # Vertikale Komponenten (wichtig für Keramik)
        vertical_angles = np.abs(np.pi/2 - np.abs(tangent_angles))

        # Richtungsänderungen
        angle_changes = np.diff(tangent_angles)
        angle_changes = np.arctan2(
            np.sin(angle_changes), np.cos(angle_changes))  # Normalisieren

        # Konvexität/Konkavität
        convex_sections = np.sum(angle_changes > 0)
        concave_sections = np.sum(angle_changes < 0)

        features = np.array([
            np.mean(tangent_angles),
            np.std(tangent_angles),
            np.mean(vertical_angles),
            np.std(angle_changes),
            np.max(np.abs(angle_changes)),
            convex_sections / (len(angle_changes) + 1),
            concave_sections / (len(angle_changes) + 1)
        ])

        return features

    def compute_skeleton_features(self, coords):
        """
        Skelett-basierte Features (vereinfacht)
        Beschreibt die "Mittelachse" der Form
        """
        x = coords[:, 0]
        y = coords[:, 1]

        # Approximiere Skelett durch Mittelpunkte in Höhen-Schichten
        y_min, y_max = y.min(), y.max()
        n_layers = 20
        skeleton_points = []

        for i in range(n_layers):
            y_low = y_min + (y_max - y_min) * i / n_layers
            y_high = y_min + (y_max - y_min) * (i + 1) / n_layers

            mask = (y >= y_low) & (y < y_high)
            if np.any(mask):
                x_layer = x[mask]
                skeleton_x = (x_layer.max() + x_layer.min()) / 2
                skeleton_y = (y_low + y_high) / 2
                skeleton_points.append([skeleton_x, skeleton_y])

        if len(skeleton_points) < 2:
            return np.zeros(5)

        skeleton = np.array(skeleton_points)

        # Skelett-Features
        skeleton_length = np.sum(
            np.sqrt(np.sum(np.diff(skeleton, axis=0)**2, axis=1)))
        skeleton_straightness = np.sqrt(
            np.sum((skeleton[-1] - skeleton[0])**2)) / (skeleton_length + 1e-10)

        # Seitliche Abweichung vom Zentrum
        lateral_deviation = np.std(skeleton[:, 0])
        max_lateral = np.max(np.abs(skeleton[:, 0]))

        # Vertikale Gleichmäßigkeit
        vertical_spacing = np.diff(skeleton[:, 1])
        vertical_uniformity = np.std(
            vertical_spacing) / (np.mean(np.abs(vertical_spacing)) + 1e-10)

        features = np.array([
            skeleton_length,
            skeleton_straightness,
            lateral_deviation,
            max_lateral,
            vertical_uniformity
        ])

        return features

    def compare_features(self, features1, features2):
        """
        Vergleicht zwei Feature-Sets

        Returns:
        - Similarity Score (0-1, höher = ähnlicher)
        - Details der einzelnen Vergleiche
        """
        similarities = []

        # 1. Koordinaten-Descriptor (Hauptgewicht für Gesamtform)
        if 'coords_descriptor' in features1 and 'coords_descriptor' in features2:
            coords_sim = 1 - cosine(features1['coords_descriptor'],
                                    features2['coords_descriptor'])
            similarities.append(('coords', coords_sim, 0.15))

        # 2. Fourier-Deskriptoren (Formähnlichkeit in Frequenzdomäne)
        if 'fourier' in features1 and 'fourier' in features2:
            fourier_dist = euclidean(
                features1['fourier'], features2['fourier'])
            fourier_sim = 1 / (1 + fourier_dist)
            similarities.append(('fourier', fourier_sim, 0.15))

        # 3. Radialer Profil (SEHR wichtig für Rotationskeramik!)
        if 'radial_profile' in features1 and 'radial_profile' in features2:
            radial_sim = 1 - \
                cosine(features1['radial_profile'],
                       features2['radial_profile'])
            similarities.append(('radial_profile', radial_sim, 0.20))

        # 4. Symmetrie (Keramik ist meist symmetrisch)
        if 'symmetry' in features1 and 'symmetry' in features2:
            sym_dist = euclidean(features1['symmetry'], features2['symmetry'])
            sym_sim = 1 / (1 + sym_dist)
            similarities.append(('symmetry', sym_sim, 0.10))

        # 5. Lokale Features (Rand, Bauch, Fuß - typologisch wichtig!)
        if 'local_features' in features1 and 'local_features' in features2:
            local_dist = euclidean(
                features1['local_features'], features2['local_features'])
            local_sim = 1 / (1 + local_dist)
            similarities.append(('local_features', local_sim, 0.15))

        # 6. Geometrische Momente
        if 'moments' in features1 and 'moments' in features2:
            moments_dist = euclidean(
                features1['moments'], features2['moments'])
            moments_sim = 1 / (1 + moments_dist)
            similarities.append(('moments', moments_sim, 0.08))

        # 7. Wandstärke (charakteristisch für Herstellungstechnik)
        if 'wall_thickness' in features1 and 'wall_thickness' in features2:
            wall_dist = euclidean(
                features1['wall_thickness'], features2['wall_thickness'])
            wall_sim = 1 / (1 + wall_dist)
            similarities.append(('wall_thickness', wall_sim, 0.05))

        # 8. Krümmung
        if 'curvature' in features1 and 'curvature' in features2:
            curv_dist = euclidean(
                features1['curvature'], features2['curvature'])
            curv_sim = 1 / (1 + curv_dist * 10)
            similarities.append(('curvature', curv_sim, 0.03))

        # 9. Kontur-Komplexität
        if 'complexity' in features1 and 'complexity' in features2:
            comp_dist = euclidean(
                features1['complexity'], features2['complexity'])
            comp_sim = 1 / (1 + comp_dist)
            similarities.append(('complexity', comp_sim, 0.03))

        # 10. Winkel-Features
        if 'angle_features' in features1 and 'angle_features' in features2:
            angle_dist = euclidean(
                features1['angle_features'], features2['angle_features'])
            angle_sim = 1 / (1 + angle_dist)
            similarities.append(('angle_features', angle_sim, 0.03))

        # 11. Skelett-Features
        if 'skeleton' in features1 and 'skeleton' in features2:
            skel_dist = euclidean(features1['skeleton'], features2['skeleton'])
            skel_sim = 1 / (1 + skel_dist)
            similarities.append(('skeleton', skel_sim, 0.02))

        # 12. Shape Statistics
        if 'shape_stats' in features1 and 'shape_stats' in features2:
            stats_dist = euclidean(
                features1['shape_stats'], features2['shape_stats'])
            stats_sim = 1 / (1 + stats_dist)
            similarities.append(('shape_stats', stats_sim, 0.01))

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
