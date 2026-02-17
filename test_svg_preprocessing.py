import os
import xml.etree.ElementTree as ET

SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NS)


def is_interesting(elem):
    """
    Entscheidet, ob ein SVG-Element behalten werden soll
    """
    fill = elem.attrib.get("fill")

    if fill is None:
        return False

    fill = fill.lower()

    if fill == "none":
        return False

    if fill in ("#ffffff", "white"):
        return False

    return True


def clean_svg(input_path, output_path):
    tree = ET.parse(input_path)
    root = tree.getroot()

    # alle direkten Kinder prüfen
    for elem in list(root):
        tag = elem.tag.replace(f"{{{SVG_NS}}}", "")

        if tag in ("path", "polyline", "polygon"):
            if not is_interesting(elem):
                root.remove(elem)
            else:
                # optional: Stroke entfernen
                elem.attrib.pop("stroke", None)
                elem.attrib.pop("stroke-width", None)
        else:
            # alles andere (rect, text, etc.) löschen
            root.remove(elem)

    tree.write(output_path, encoding="utf-8", xml_declaration=True)


def batch_clean(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".svg"):
            in_path = os.path.join(input_dir, filename)
            out_path = os.path.join(output_dir, filename)

            clean_svg(in_path, out_path)
            print(f"✔ bereinigt: {filename}")


if __name__ == "__main__":
    INPUT_DIR = "clean_test_svgs\ground_truth"
    OUTPUT_DIR = "clean_test_svgs\ground_truth_clean"

    batch_clean(INPUT_DIR, OUTPUT_DIR)
