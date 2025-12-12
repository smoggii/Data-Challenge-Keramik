from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from PIL import Image
from pathlib import Path
import time
import io

class SVGConverter:
    """Konvertiert SVG-Dateien zu PNG mit Chrome/Selenium"""
    
    def __init__(self, input_folder, output_folder):
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(exist_ok=True)
        self.driver = None
    
    def setup_chrome(self):
        """Startet Chrome im Headless-Modus"""
        print("🌐 Starte Chrome...")
        
        chrome_options = Options()
        chrome_options.add_argument('--headless=new')  # Kein Fenster
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--window-size=1920,1080')
        
        try:
            self.driver = webdriver.Chrome(options=chrome_options)
            print("✓ Chrome gestartet")
        except Exception as e:
            print(f"❌ Fehler beim Starten von Chrome: {e}")
            print("\nHinweis: Installiere chromedriver:")
            print("  pip install webdriver-manager")
            raise
    
    def close_chrome(self):
        """Schließt Chrome"""
        if self.driver:
            self.driver.quit()
            print("✓ Chrome geschlossen")
    
    def svg_to_screenshot(self, svg_path):
        """
        Öffnet SVG in Chrome und macht Screenshot
        
        Returns:
        - PIL Image
        """
        # SVG in Chrome öffnen
        svg_url = f"file:///{svg_path.absolute().as_posix()}"
        self.driver.get(svg_url)
        
        # Kurz warten bis SVG geladen ist
        time.sleep(0.5)
        
        # Screenshot machen (als PNG in Memory)
        png_data = self.driver.get_screenshot_as_png()
        
        # Zu PIL Image
        img = Image.open(io.BytesIO(png_data))
        
        return img
    
    def crop_to_content(self, img):
        """
        Schneidet weißen Rand weg (findet den SVG-Inhalt)
        
        Returns:
        - PIL Image (ohne Rand)
        """
        # Zu Graustufen
        gray = img.convert('L')
        
        # Finde Bounding Box (nicht-weiße Pixel)
        bbox = gray.point(lambda x: 0 if x > 250 else 255).getbbox()
        
        if bbox:
            return img.crop(bbox)
        else:
            return img
    
    def crop_left_and_vertical(self, img, left_percent=50, top_pixels=0, bottom_pixels=0):
        """
        Nimmt linke Hälfte und schneidet oben/unten ab
        
        Parameters:
        - img: PIL Image
        - left_percent: Wie viel % von links nehmen (50 = linke Hälfte)
        - top_pixels: Wie viele Pixel von oben abschneiden (z.B. 50)
        - bottom_pixels: Wie viele Pixel von unten abschneiden (z.B. 30)
        """
        w, h = img.size
        
        # Linke Hälfte
        crop_width = int(w * left_percent / 100)
        
        # Crop: (left, top, right, bottom)
        cropped = img.crop((
            0,                    # left
            top_pixels,           # top (direkt Pixel)
            crop_width,           # right
            h - bottom_pixels     # bottom (direkt Pixel)
        ))
        
        return cropped
    
    def process_svg(self, svg_path, left_percent=50, top_pixels=0, bottom_pixels=0):
        """
        Verarbeitet eine einzelne SVG-Datei
        
        Returns:
        - PIL Image (verarbeitet)
        """
        try:
            # 1. Screenshot in Chrome
            print(f"  Lade SVG in Chrome und erstelle Screenshot...")
            img = self.svg_to_screenshot(svg_path)
            
            print(f"  Screenshot-Größe: {img.size[0]}x{img.size[1]} px")
            
            # 2. Weißen Rand wegschneiden
            print(f"  Entferne weißen Rand...")
            img = self.crop_to_content(img)
            
            print(f"  Nach Rand-Entfernung: {img.size[0]}x{img.size[1]} px")
            
            # 3. Links/Oben/Unten croppen
            print(f"  Croppe: Links {left_percent}%, Oben -{top_pixels}px, Unten -{bottom_pixels}px")
            cropped = self.crop_left_and_vertical(img, left_percent, top_pixels, bottom_pixels)
            
            print(f"  Finale Größe: {cropped.size[0]}x{cropped.size[1]} px")
            
            return cropped
            
        except Exception as e:
            print(f"  ❌ Fehler: {e}")
            return None
    
    def process_all(self, left_percent=50, top_pixels=0, bottom_pixels=0, preview=False):
        """
        Verarbeitet alle SVG-Dateien im Input-Ordner
        
        Parameters:
        - left_percent: % von links
        - top_pixels: Pixel von oben abschneiden
        - bottom_pixels: Pixel von unten abschneiden
        - preview: Vorschau anzeigen
        """
        svg_files = list(self.input_folder.glob('*.svg'))
        
        if not svg_files:
            print("❌ Keine SVG-Dateien gefunden!")
            return
        
        print(f"✓ Gefunden: {len(svg_files)} SVG-Dateien\n")
        
        # Chrome starten
        try:
            self.setup_chrome()
        except:
            return
        
        successful = 0
        
        try:
            for idx, svg_path in enumerate(svg_files, 1):
                print(f"\n[{idx}/{len(svg_files)}] Verarbeite: {svg_path.name}")
                print("-" * 50)
                
                # Verarbeiten
                result = self.process_svg(svg_path, left_percent, top_pixels, bottom_pixels)
                
                if result is None:
                    continue
                
                # Preview
                if preview:
                    result.show()
                    choice = input("  Speichern? (j/n) [j]: ").strip().lower() or 'j'
                    if choice != 'j':
                        print("  → Übersprungen")
                        continue
                
                # Speichern
                output_name = svg_path.stem + '.png'
                output_path = self.output_folder / output_name
                result.save(output_path, 'PNG')
                print(f"  ✓ Gespeichert: {output_path}")
                successful += 1
        
        finally:
            # Chrome immer schließen
            self.close_chrome()
        
        print("\n" + "=" * 60)
        print(f"🎉 FERTIG! {successful}/{len(svg_files)} Bilder erfolgreich verarbeitet")
        print(f"📁 Ergebnisse in: {self.output_folder}")
        print("=" * 60)


def main():
    """Hauptfunktion"""
    
    print("=" * 60)
    print("SVG → PNG KONVERTER FÜR KERAMIK-TESTBILDER")
    print("(Chrome/Selenium Version)")
    print("=" * 60)
    
    # Pfade
    print("\n📁 Ordner-Konfiguration:")
    input_folder = input("Eingabe-Ordner mit SVG-Dateien: ").strip()
    output_folder = input("Ausgabe-Ordner für PNG-Dateien: ").strip()
    
    # Converter erstellen
    converter = SVGConverter(input_folder, output_folder)
    
    # Parameter
    print("\n⚙️ Parameter:")
    left_percent = int(input("Wie viel % von links nehmen? [50]: ") or "50")
    top_pixels = int(input("Wie viele Pixel von oben abschneiden? (Durchmesser-Nummer) [50]: ") or "50")
    bottom_pixels = int(input("Wie viele Pixel von unten abschneiden? (Legende) [80]: ") or "80")
    
    preview = input("Vorschau anzeigen? (j/n) [n]: ").strip().lower() or 'n'
    preview = preview == 'j'
    
    print(f"\n🚀 Starte Konvertierung...")
    print(f"   Links: {left_percent}%")
    print(f"   Oben abschneiden: {top_pixels} Pixel")
    print(f"   Unten abschneiden: {bottom_pixels} Pixel")
    print(f"   Preview: {preview}")
    
    # Verarbeiten
    converter.process_all(left_percent, top_pixels, bottom_pixels, preview)


if __name__ == "__main__":
    main()