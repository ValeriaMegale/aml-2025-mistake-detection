import csv
import json
import os

# Leggi activity names
with open('annotations/annotation_csv/activity_idx_step_idx.csv', 'r') as f:
    reader = csv.DictReader(f)
    activity_mapping = {}
    for row in reader:
        idx = row['activity_idx']
        name = row['activity_name']
        # Converti nome in filename (lowercase, niente spazi)
        filename = name.lower().replace(' ', '')
        activity_mapping[idx] = {
            'name': name,
            'task_graph_file': filename + '.json'
        }

# Verifica che i file esistano
task_graph_dir = 'annotations/task_graphs'
for idx, info in activity_mapping.items():
    filepath = os.path.join(task_graph_dir, info['task_graph_file'])
    if os.path.exists(filepath):
        print(f"Activity {idx}: {info['name']} -> {info['task_graph_file']} OK")
    else:
        print(f"Activity {idx}: {info['name']} -> {info['task_graph_file']} MISSING")

# Salva mapping
os.makedirs('substep3_step_detection', exist_ok=True)
with open('substep3_step_detection/activity_to_taskgraph.json', 'w') as f:
    json.dump(activity_mapping, f, indent=2)
print('\nSaved to substep3_step_detection/activity_to_taskgraph.json')
