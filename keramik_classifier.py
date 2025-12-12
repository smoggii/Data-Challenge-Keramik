from PIL import Image
import numpy as np
from pathlib import Path
from skimage import measure
from skimage.feature import hog
from scipy.spatial.distance import cosine, euclidean
import csv
from datetime import datetime

class KeramikClassifier:
    """Klassifiziert Keramik-Profile basierend auf Shape-Features"""
    
    def __init__(self, reference_folder, test_folder, output_folder):
        self.reference_folder = Path(reference_folder)
        self.test_folder = Path(test_folder)
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(exist_ok=True)
        
        self.reference_features = {}  # {filename: features}
        self.reference_images = {}    # {filename: image}
        
    def extract_features(self, img_path):
        """
        Extrahiert mehrere Features aus einem Bild
        
        Returns:
        - Dictionary mit verschiedenen Feature-Typen
        """
        # Bild laden und vorbereiten
        img = Image.open(img_path).convert('L')
        img_array = np.array(img)
        
        # Bild normalisieren (Größe einheitlich machen)
        img_resized = img.resize((128, 128), Image.Resampling.LANCZOS)
        img_array_resized = np.array(img_resized)
        
        features = {}
        
        # 1. HOG Features (Histogram of Oriented Gradients)
        # Erfasst die Form/Kontur-Orientierung
        hog_features = hog(
            img_array_resized, 
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            visualize=False
        )
        features['hog'] = hog_features
        
        # 2. Hu Moments (invariant gegenüber Rotation, Skalierung)
        # Erfasst die grundlegende Form
        binary = img_array < 128  # Binarisieren
        moments = measure.moments(binary.astype(float))
        hu_moments = measure.moments_hu(moments)
        features['hu_moments'] = hu_moments
        
        # 3. Kontur-Eigenschaften
        contours = measure.find_contours(binary, 0.5)
        if contours:
            # Längste Kontur nehmen (sollte der Querschnitt sein)
            longest_contour = max(contours, key=len)
            
            # Kontur-Features
            contour_length = len(longest_contour)
            
            # Bounding Box
            rows = longest_contour[:, 0]
            cols = longest_contour[:, 1]
            height = rows.max() - rows.min()
            width = cols.max() - cols.min()
            aspect_ratio = height / (width + 1e-6)
            
            features['contour'] = np.array([
                contour_length,
                aspect_ratio,
                height,
                width
            ])
        else:
            features['contour'] = np.zeros(4)
        
        # 4. Pixel-Histogramm (Verteilung der Grauwerte)
        hist, _ = np.histogram(img_array_resized, bins=32, range=(0, 256))
        hist = hist / (hist.sum() + 1e-6)  # Normalisieren
        features['histogram'] = hist
        
        return features
    
    def compare_features(self, features1, features2):
        """
        Vergleicht zwei Feature-Sets und gibt Ähnlichkeit zurück
        
        Returns:
        - Similarity Score (0-1, höher = ähnlicher)
        """
        similarities = []
        
        # HOG Similarity (Cosine)
        if 'hog' in features1 and 'hog' in features2:
            hog_sim = 1 - cosine(features1['hog'], features2['hog'])
            similarities.append(('hog', hog_sim, 0.4))  # 40% Gewicht
        
        # Hu Moments Similarity (Euclidean, invertiert)
        if 'hu_moments' in features1 and 'hu_moments' in features2:
            # Log-transform für bessere Vergleichbarkeit
            hu1 = -np.sign(features1['hu_moments']) * np.log10(np.abs(features1['hu_moments']) + 1e-10)
            hu2 = -np.sign(features2['hu_moments']) * np.log10(np.abs(features2['hu_moments']) + 1e-10)
            hu_dist = euclidean(hu1, hu2)
            hu_sim = 1 / (1 + hu_dist)  # In Similarity umwandeln
            similarities.append(('hu_moments', hu_sim, 0.3))  # 30% Gewicht
        
        # Kontur Similarity
        if 'contour' in features1 and 'contour' in features2:
            contour_dist = euclidean(features1['contour'], features2['contour'])
            contour_sim = 1 / (1 + contour_dist / 100)  # Normalisieren
            similarities.append(('contour', contour_sim, 0.2))  # 20% Gewicht
        
        # Histogram Similarity
        if 'histogram' in features1 and 'histogram' in features2:
            hist_sim = 1 - cosine(features1['histogram'], features2['histogram'])
            similarities.append(('histogram', hist_sim, 0.1))  # 10% Gewicht
        
        # Gewichteter Durchschnitt
        weighted_sum = sum(sim * weight for _, sim, weight in similarities)
        total_weight = sum(weight for _, _, weight in similarities)
        
        final_similarity = weighted_sum / total_weight if total_weight > 0 else 0
        
        # Details für Debug
        detail = {name: f"{sim:.3f}" for name, sim, _ in similarities}
        
        return final_similarity, detail
    
    def load_references(self):
        """Lädt alle Referenzbilder und extrahiert Features"""
        print("=" * 60)
        print("LADE REFERENZBILDER")
        print("=" * 60)
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
        reference_files = [f for f in self.reference_folder.iterdir() 
                          if f.suffix.lower() in image_extensions]
        
        print(f"Gefunden: {len(reference_files)} Referenzbilder\n")
        
        for idx, ref_path in enumerate(reference_files, 1):
            print(f"[{idx}/{len(reference_files)}] Verarbeite: {ref_path.name}")
            
            try:
                features = self.extract_features(ref_path)
                img = Image.open(ref_path)
                
                # Klassenname = Dateiname ohne "processed_" und Extension
                class_name = ref_path.stem.replace('processed_', '')
                
                self.reference_features[class_name] = features
                self.reference_images[class_name] = img
                
            except Exception as e:
                print(f"  Fehler: {e}")
        
        print(f"\n✓ {len(self.reference_features)} Referenzen geladen")
    
    def classify_test_images(self, top_k=5):
        """
        Klassifiziert alle Testbilder
        
        Parameters:
        - top_k: Wie viele Top-Matches zurückgeben
        """
        print("\n" + "=" * 60)
        print("KLASSIFIZIERE TESTBILDER")
        print("=" * 60)
        
        if not self.reference_features:
            print("Fehler: Keine Referenzen geladen!")
            return
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
        test_files = [f for f in self.test_folder.iterdir() 
                     if f.suffix.lower() in image_extensions]
        
        print(f"Gefunden: {len(test_files)} Testbilder\n")
        
        results = []
        
        for idx, test_path in enumerate(test_files, 1):
            print(f"\n[{idx}/{len(test_files)}] Klassifiziere: {test_path.name}")
            
            try:
                # Features extrahieren
                test_features = self.extract_features(test_path)
                
                # Mit allen Referenzen vergleichen
                similarities = []
                for class_name, ref_features in self.reference_features.items():
                    similarity, detail = self.compare_features(test_features, ref_features)
                    similarities.append((class_name, similarity, detail))
                
                # Sortieren nach Similarity (höchste zuerst)
                similarities.sort(key=lambda x: x[1], reverse=True)
                
                # Top-K nehmen
                top_matches = similarities[:top_k]
                
                # Ausgeben
                print(f"  Top {top_k} Matches:")
                for rank, (class_name, sim, detail) in enumerate(top_matches, 1):
                    print(f"    {rank}. {class_name:30s} - Similarity: {sim:.3f}")
                    print(f"       Details: {detail}")
                
                # Ergebnis speichern
                result = {
                    'test_image': test_path.name,
                    'top_matches': top_matches
                }
                results.append(result)
                
            except Exception as e:
                print(f"  Fehler: {e}")
        
        # Ergebnisse in CSV speichern
        self.save_results_csv(results, top_k)
        
        print(f"\n✓ Klassifizierung abgeschlossen!")
        return results
    
    def save_results_csv(self, results, top_k):
        """Speichert Ergebnisse in CSV-Datei"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = self.output_folder / f"classification_results_{timestamp}.csv"
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header
            header = ['Test_Image']
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
    print("KERAMIK-KLASSIFIZIERER")
    print("=" * 60)
    
    # Pfade eingeben
    print("\n📁 Ordner-Konfiguration:")
    reference_folder = input("Ordner mit Referenzbildern (verarbeitete Literaturbilder): ").strip()
    test_folder = input("Ordner mit Testbildern (zu klassifizieren): ").strip()
    output_folder = input("Ausgabe-Ordner für Ergebnisse: ").strip()
    
    # Classifier erstellen
    classifier = KeramikClassifier(reference_folder, test_folder, output_folder)
    
    # Referenzen laden
    classifier.load_references()
    
    # Testbilder klassifizieren
    top_k = int(input("\nWie viele Top-Matches anzeigen? [5]: ") or "5")
    classifier.classify_test_images(top_k=top_k)


if __name__ == "__main__":
    main()