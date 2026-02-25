from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import os
import time
import keramik_svg_classifier_final as backend

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REF_FOLDER = os.path.join(BASE_DIR, 'svg_files')
TEST_FOLDER = os.path.join(BASE_DIR, 'preprocessing_File')
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'Output')

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

api = backend.ClassifierAPI(
    ref_folder=REF_FOLDER,
    test_folder=TEST_FOLDER,
    output_folder=OUTPUT_FOLDER
)

@app.route('/api/fragments', methods=['GET'])
def get_fragments():
    try:
        return jsonify(api.list_fragments())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/classify', methods=['POST'])
def classify():
    try:
        data = request.json
        filename = data.get('filename')
        result = api.classify_fragment(filename)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/download_report', methods=['POST'])
def download_report():
    data = request.json
    filename = data.get('filename')
    report_type = data.get('type', 'top5')

    clean_name = filename.replace('.svg', '')
    pdf_filename = f"{report_type}_{clean_name}.pdf"
    output_pdf = os.path.join(OUTPUT_FOLDER, pdf_filename)
    test_path = os.path.join(TEST_FOLDER, filename)

    if os.path.exists(output_pdf):
        os.remove(output_pdf)

    try:
        clf = api._clf
        ref_attr = 'reference_data' if hasattr(clf, 'reference_data') else 'references'

        if report_type == 'top5':
            results = api.classify_fragment(filename)
            top_5_classes = [item['class'] for item in results['top']]
            original_refs = getattr(clf, ref_attr).copy()

            if isinstance(original_refs, dict):
                filtered_refs = {k: v for k, v in original_refs.items() if k in top_5_classes}
            else:
                filtered_refs = [r for r in original_refs if r.name in top_5_classes]

            setattr(clf, ref_attr, filtered_refs)
            clf.visualize_batch(test_path, output_pdf)
            setattr(clf, ref_attr, original_refs)
        else:
            clf.visualize_batch(test_path, output_pdf)

        time.sleep(1)

        if os.path.exists(output_pdf):
            return send_file(output_pdf, as_attachment=True)
        else:
            return jsonify({"error": "PDF wurde nicht erstellt"}), 500

    except Exception as e:
        print(f"Fehler im PDF-Export: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001, host='0.0.0.0')