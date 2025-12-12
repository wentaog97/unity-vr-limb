import sys, json, joblib, numpy as np
from pathlib import Path

"""Usage: python export_rf_json.py <model.pkl> <out.json>
Extracts scaler, label encoder, and forest to portable JSON.
"""

if len(sys.argv) != 3:
    print("Usage: export_rf_json.py <model.pkl> <out.json>")
    sys.exit(1)

pkl_path, out_path = sys.argv[1:3]
obj = joblib.load(pkl_path)
clf = obj['classifier']
scaler = obj['scaler']
le = obj['label_encoder']

forest_dict = {
    'labels': le.classes_.tolist(),
    'scaler_mean': scaler.mean_.tolist(),
    'scaler_scale': scaler.scale_.tolist(),
    'n_features': clf.n_features_in_,
    'trees': []
}

for est in clf.estimators_:
    tree = est.tree_
    node_list = []
    for i in range(tree.node_count):
        # Determine majority class index at this node
        counts = tree.value[i][0]
        class_idx = int(np.argmax(counts))
        node_list.append({
            'feature': int(tree.feature[i]),
            'threshold': float(tree.threshold[i]),
            'left': int(tree.children_left[i]),
            'right': int(tree.children_right[i]),
            'class_idx': class_idx
        })
    forest_dict['trees'].append({'nodes': node_list})

Path(out_path).write_text(json.dumps(forest_dict))
print(f"Exported JSON model to {out_path}") 