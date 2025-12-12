from PIL import Image, ImageDraw, ImageOps
import numpy as np
from scipy import ndimage
from skimage import measure
import os
from pathlib import Path

class KeramikPreprocessor:
    """Tool zum Verarbeiten von Keramik-Profilzeichnungen (Pillow Version)"""
    
    def __init__(self, input_folder, output_folder):
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(exist_ok=True)
        
    def crop_left_portion(self, img, percentage=50):
        """Schneidet den linken Teil des Bildes aus (wo der Querschnitt ist)"""
        w, h = img.size
        crop_width = int(w * percentage / 100)
        return img.crop((0, 0, crop_width, h))
    
    def crop_top_portion(self, img, percentage=25):
        """Schneidet den oberen Teil des Bildes aus"""
        w, h = img.size
        crop_height = int(h * percentage / 100)
        return img.crop((0, 0, w, crop_height))
    
    def fill_contours(self, img, min_area=100, threshold=127):
        """
        Findet Konturen in Linienzeichnung und füllt sie schwarz aus
        
        Parameters:
        - img: Input PIL Image
        - min_area: Minimale Konturfläche (filtert kleine Artefakte)
        - threshold: Schwellwert für Binarisierung
        """
        # In Graustufen konvertieren
        gray = img.convert('L')
        
        # Zu numpy array
        img_array = np.array(gray)
        
        # Invertieren falls nötig (Linien sollten dunkel sein)
        mean_val = np.mean(img_array)
        if mean_val > 127:
            img_array = 255 - img_array
        
        # Binarisierung
        binary = img_array < threshold
        
        # Konturen finden mit scikit-image
        contours = measure.find_contours(binary, 0.5)
        
        # Neues weißes Bild erstellen
        result = Image.new('L', img.size, 255)
        draw = ImageDraw.Draw(result)
        
        # Konturen füllen
        for contour in contours:
            # Kontur als Polygon
            # scikit-image gibt (row, col) zurück, PIL braucht (x, y)
            polygon = [(int(col), int(row)) for row, col in contour]
            
            # Fläche berechnen (approximiert)
            if len(polygon) > 2:
                area = self._polygon_area(polygon)
                if area > min_area:
                    draw.polygon(polygon, fill=0, outline=0)
        
        return result
    
    def _polygon_area(self, vertices):
        """Berechnet Fläche eines Polygons (Shoelace formula)"""
        n = len(vertices)
        if n < 3:
            return 0
        area = 0
        for i in range(n):
            j = (i + 1) % n
            area += vertices[i][0] * vertices[j][1]
            area -= vertices[j][0] * vertices[i][1]
        return abs(area) / 2
    
    def process_image(self, img_path, crop_percent=25, fill=True, min_area=100, 
                     threshold=127, left_portion=50):
        """
        Verarbeitet ein einzelnes Bild
        
        Parameters:
        - img_path: Pfad zum Bild
        - crop_percent: Wie viel Prozent von oben ausschneiden
        - fill: Ob Konturen gefüllt werden sollen
        - min_area: Minimale Konturfläche
        - threshold: Schwellwert für Binarisierung
        - left_portion: Wie viel % von links nehmen (Standard 50%)
        """
        try:
            # Bild laden
            img = Image.open(img_path)
            
            # In RGB konvertieren falls RGBA
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            
            # Erst linken Teil nehmen (nur Querschnitt)
            if left_portion < 100:
                img = self.crop_left_portion(img, left_portion)
            
            # Dann oberen Teil ausschneiden
            cropped = self.crop_top_portion(img, crop_percent)
            
            # Optional: Konturen füllen
            if fill:
                processed = self.fill_contours(cropped, min_area, threshold)
            else:
                processed = cropped.convert('L')
            
            return processed
            
        except Exception as e:
            print(f"Fehler beim Verarbeiten von {img_path}: {e}")
            return None
    
    def process_all(self, crop_percent=25, fill=True, min_area=100, threshold=127,
                   left_portion=50, preview=True):
        """
        Verarbeitet alle Bilder im Input-Ordner
        
        Parameters:
        - crop_percent: Prozent von oben
        - fill: Konturen füllen
        - min_area: Min. Konturfläche
        - threshold: Binarisierungs-Schwellwert
        - left_portion: Wie viel % von links (Standard 50%)
        - preview: Zeigt Vorschau vor dem Speichern
        """
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
        image_files = [f for f in self.input_folder.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        print(f"Gefunden: {len(image_files)} Bilder")
        
        for idx, img_path in enumerate(image_files, 1):
            print(f"\n[{idx}/{len(image_files)}] Verarbeite: {img_path.name}")
            
            # Verarbeiten
            result = self.process_image(img_path, crop_percent, fill, min_area, 
                                       threshold, left_portion)
            
            if result is None:
                continue
            
            # Preview
            if preview:
                # Original für Vergleich
                original = Image.open(img_path)
                if left_portion < 100:
                    original = self.crop_left_portion(original, left_portion)
                cropped_orig = self.crop_top_portion(original, crop_percent)
                
                # Side-by-side
                w1, h1 = cropped_orig.size
                w2, h2 = result.size
                
                # Höhe angleichen
                max_height = max(h1, h2)
                display = Image.new('RGB', (w1 + w2, max_height), 'white')
                
                # Original links
                display.paste(cropped_orig.convert('RGB'), (0, 0))
                
                # Verarbeitet rechts
                display.paste(result.convert('RGB'), (w1, 0))
                
                # Anzeigen
                display.show()
                
                choice = input("  Speichern? (j/n) [j]: ").strip().lower() or 'j'
                if choice != 'j':
                    print("  -> Übersprungen")
                    continue
            
            # Speichern
            output_path = self.output_folder / f"processed_{img_path.stem}.png"
            result.save(output_path)
            print(f"  -> Gespeichert: {output_path}")
        
        print(f"\n✓ Fertig! Ergebnisse in: {self.output_folder}")


def main():
    """Hauptfunktion mit Beispiel-Nutzung"""
    
    print("=" * 60)
    print("KERAMIK-PROFIL PREPROCESSING TOOL (Pillow Version)")
    print("=" * 60)
    
    # Pfade eingeben
    input_folder = input("\nEingabe-Ordner mit Literaturbildern: ").strip()
    output_folder = input("Ausgabe-Ordner für verarbeitete Bilder: ").strip()
    
    # Processor erstellen
    processor = KeramikPreprocessor(input_folder, output_folder)
    
    # Parameter eingeben
    print("\n--- Parameter ---")
    left_portion = int(input("Wie viel % von links nehmen (Querschnitt)? [40]: ") or "40")
    
    crop_percent = int(input("Wie viel % von oben ausschneiden? [25]: ") or "25")
    fill_choice = input("Konturen füllen? (j/n) [j]: ").strip().lower() or "j"
    fill = fill_choice == "j"
    
    if fill:
        min_area = int(input("Minimale Konturfläche (filtert Artefakte) [300]: ") or "300")
        threshold = int(input("Threshold für Binarisierung [150]: ") or "150")
    else:
        min_area = 300
        threshold = 150
    
    preview = input("Vorschau vor Speichern? (j/n) [j]: ").strip().lower() or "j"
    preview = preview == "j"
    
    print(f"\n--- Starte Verarbeitung ---")
    print(f"Von links: {left_portion}%")
    print(f"Von oben: {crop_percent}%")
    print(f"Füllen: {fill}")
    if fill:
        print(f"Min. Area: {min_area}")
        print(f"Threshold: {threshold}")
    print(f"Preview: {preview}")
    
    # Verarbeiten
    processor.process_all(crop_percent, fill, min_area, threshold, left_portion, preview)


if __name__ == "__main__":
    main()