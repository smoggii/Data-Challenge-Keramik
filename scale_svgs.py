#!/usr/bin/env python3
"""
SVG-Skalierung auf Zielhoehe (proportional, keine Verzerrung)

Liest die Koordinaten per demselben Parser wie der Klassifikator,
skaliert mit einheitlichem Faktor (Hoehe -> TARGET_HEIGHT px),
und schreibt die Punkte als <polyline> zurueck.
"""

import numpy as np
from pathlib import Path
import xml.etree.ElementTree as ET
import re

# ── CONFIG ────────────────────────────────────────────────────────
TARGET_HEIGHT = 150   # Zielhoehe in Pixeln
# ─────────────────────────────────────────────────────────────────


def parse_coords(svg_path):
    """Koordinaten aus SVG lesen (polyline ODER path), identisch zum Klassifikator."""
    tree = ET.parse(svg_path)
    root = tree.getroot()

    # 1) Polyline
    for elem in root.iter():
        tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
        if tag == 'polyline':
            pts = elem.get('points', '')
            if len(pts) > 20:
                nums = ' '.join(pts.replace(',', ' ').split()).split()
                coords = np.array([[float(nums[i]), float(nums[i+1])]
                                   for i in range(0, len(nums)-1, 2)])
                if len(coords) > 5:
                    return coords

    # 2) Path  (Kommando-basiertes Parsen wie im Klassifikator)
    for elem in root.iter():
        tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
        if tag == 'path':
            d = elem.get('d', '')
            if len(d) > 20:
                coords = _parse_path(d)
                if len(coords) > 5:
                    return coords

    raise ValueError("Keine verwertbaren Koordinaten gefunden")


def _parse_path(path_data):
    """SVG-Path d-Attribut parsen (wie im Klassifikator)."""
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
            cur = (cur + np.array([n[0], n[1]])
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


def scale_svg(svg_path, output_path, target_height=TARGET_HEIGHT):
    """
    Liest Koordinaten, skaliert proportional auf target_height,
    schreibt als <polyline> zurueck.
    Gibt (orig_h, new_w, factor) zurueck oder False bei Fehler.
    """
    try:
        coords = parse_coords(svg_path)
    except Exception as e:
        print(f"  ⚠  {svg_path.name}: {e}")
        return False

    orig_h = coords[:, 1].max() - coords[:, 1].min()
    if orig_h < 1:
        print(f"  ⚠  {svg_path.name}: Hoehe zu klein ({orig_h:.1f}px)")
        return False

    # Einheitlicher Skalierungsfaktor — X und Y identisch (keine Verzerrung)
    factor = target_height / orig_h

    x_min = coords[:, 0].min()
    y_min = coords[:, 1].min()
    scaled = coords.copy()
    scaled[:, 0] = (coords[:, 0] - x_min) * factor   # X proportional
    # Y proportional (gleicher factor)
    scaled[:, 1] = (coords[:, 1] - y_min) * factor

    new_w = scaled[:, 0].max() - scaled[:, 0].min()
    new_h = scaled[:, 1].max() - scaled[:, 1].min()  # == target_height

    # SVG neu aufbauen: minimales, sauberes SVG mit einer polyline
    margin = 5
    vb_w = new_w + 2 * margin
    vb_h = new_h + 2 * margin

    # Punkte mit kleinem Rand verschieben
    pts_shifted = scaled.copy()
    pts_shifted[:, 0] += margin
    pts_shifted[:, 1] += margin

    pts_str = ' '.join(f"{x:.3f},{y:.3f}" for x, y in pts_shifted)

    svg_content = (
        f'<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{vb_w:.1f}" height="{vb_h:.1f}" '
        f'viewBox="0 0 {vb_w:.1f} {vb_h:.1f}">\n'
        f'  <polyline points="{pts_str}" '
        f'fill="none" stroke="black" stroke-width="1"/>\n'
        f'</svg>\n'
    )

    Path(output_path).write_text(svg_content, encoding='utf-8')
    return orig_h, new_w, factor


def main():
    print("=" * 60)
    print("SVG-Skalierung auf Zielhoehe (proportional)")
    print(f"Zielhoehe: {TARGET_HEIGHT} px  (Breite skaliert mit)")
    print("=" * 60)

    input_folder = input("\nEingabe-Ordner (SVGs):  ").strip()
    output_folder = input("Ausgabe-Ordner:          ").strip()

    inp = Path(input_folder)
    out = Path(output_folder)

    if not inp.exists():
        print(f"Ordner nicht gefunden: {inp}")
        return

    out.mkdir(parents=True, exist_ok=True)

    svgs = sorted(inp.glob('*.svg'))
    if not svgs:
        print("Keine SVG-Dateien gefunden.")
        return

    print(f"\n{len(svgs)} SVGs gefunden\n")

    ok = 0
    for svg in svgs:
        result = scale_svg(svg, out / svg.name, TARGET_HEIGHT)
        if result:
            orig_h, new_w, factor = result
            print(f"  ✓ {svg.name:40s}  "
                  f"H={orig_h:6.1f}px -> {TARGET_HEIGHT}px   "
                  f"Breite x{factor:.3f}   "
                  f"(Faktor {factor:.3f}x)")
            ok += 1

    print(f"\n✓ {ok}/{len(svgs)} SVGs skaliert -> {out}")


if __name__ == "__main__":
    main()
