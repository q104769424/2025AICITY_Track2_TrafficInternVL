import os
import json
import csv
import argparse
from tqdm import tqdm

def filter_data(data_path, save_path, split_name, reference_view_path, root_path='./data', area_threshold=1000):
    """
    Filter multimodal QA samples based on reference viewpoints and bounding box area.

    Args:
        data_path (str): Input JSON path.
        save_path (str): Output JSON path after filtering.
        split_name (str): Split name (e.g., 'train', 'val').
        reference_view_path (str): CSV path containing allowed overhead views.
        root_path (str): Root path of the dataset.
        area_threshold (int): Minimum area threshold for pedestrian/vehicle boxes.
    """
    stage_map = {
        'prerecognition': 0,
        'recognition': 1,
        'judgement': 2,
        'action': 3,
        'avoidance': 4
    }

    reference_views = {}
    with open(reference_view_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            reference_views[row[0]] = row[1:]

    with open(data_path, 'r') as f:
        data_json = json.load(f)

    filtered_data = []
    print(f"\nProcessing {split_name} split from: {data_path}")
    for data in tqdm(data_json):
        view = data['image'][0].split('/')[-2] if 'WTS' in data['image'][0] else data['image'][0].split('/')[-1].split('_')[0]
        event = data['image'][0].split('/')[-4] if 'WTS' in data['image'][0] else ''

        if 'WTS' in data['image'][0] and 'overhead_view' in data['image'][0]:
            if event in reference_views and view + '.mp4' not in reference_views[event]:
                continue  # View not in reference list

        stage = data['image'][0].split('/')[-1].split('.')[0].split('_')[-1]

        pedestrian_box_path = ''
        vehicle_box_path = ''
        if 'WTS' in data['image'][0]:
            if 'normal_trimmed' not in data['image'][0]:
                pedestrian_box_path = f"{root_path}/WTS/annotations/bbox_annotated/pedestrian/{split_name}/{event}/overhead_view/{view}_bbox.json"
                vehicle_box_path = f"{root_path}/WTS/annotations/bbox_annotated/vehicle/{split_name}/{event}/overhead_view/{view}_bbox.json"
            else:
                pedestrian_box_path = f"{root_path}/WTS/annotations/bbox_annotated/pedestrian/{split_name}/normal_trimmed/{event}/overhead_view/{view}_bbox.json"
                vehicle_box_path = f"{root_path}/WTS/annotations/bbox_annotated/vehicle/{split_name}/normal_trimmed/{event}/overhead_view/{view}_bbox.json"
        elif 'BDD_PC_5k' in data['image'][0]:
            pedestrian_box_path = f"{root_path}/BDD_PC_5k/annotations/bbox_annotated/{split_name}/{view}_bbox.json"

        pedestrian_data = {}
        vehicle_data = {}
        try:
            pedestrian_data = json.load(open(pedestrian_box_path))
        except FileNotFoundError:
            pass
        try:
            if vehicle_box_path:
                vehicle_data = json.load(open(vehicle_box_path))
        except FileNotFoundError:
            pass

        human_area = 0
        if pedestrian_data:
            for box in pedestrian_data.get('annotations', []):
                if str(stage_map[stage]) == str(box.get('phase_number')) or stage_map[stage] == box.get('phase_number'):
                    human_area = box['bbox'][2] * box['bbox'][3]
                    break

        vehicle_area = 0
        if vehicle_data:
            for box in vehicle_data.get('annotations', []):
                if str(stage_map[stage]) == str(box.get('phase_number')) or stage_map[stage] == box.get('phase_number'):
                    vehicle_area = box['bbox'][2] * box['bbox'][3]
                    break

        if human_area > area_threshold or vehicle_area > area_threshold:
            filtered_data.append(data)

    for i, item in enumerate(filtered_data):
        item['id'] = i

    print(f"{split_name} original: {len(data_json)} | filtered: {len(filtered_data)}")
    with open(save_path, 'w') as f:
        json.dump(filtered_data, f, indent=2, ensure_ascii=False)
    print(f"Filtered {split_name} saved to: {save_path}")

    return len(data_json), len(filtered_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference_view_path', type=str, required=True, help='Path to reference view CSV')
    args = parser.parse_args()

    # Train split
    train_data_path = './processed_anno/vqa_format/wts_bdd_multimodal_vqa_train.json'
    train_save_path = './processed_anno/vqa_format/wts_bdd_multimodal_vqa_train_filtered.json'
    original_train, filtered_train = filter_data(train_data_path, train_save_path, 'train', args.reference_view_path)

    # Validation split
    val_data_path = './processed_anno/vqa_format/wts_bdd_multimodal_vqa_val.json'
    val_save_path = './processed_anno/vqa_format/wts_bdd_multimodal_vqa_val_filtered.json'
    original_val, filtered_val = filter_data(val_data_path, val_save_path, 'val', args.reference_view_path)

    print("\n--- Final Stats ---")
    print(f"Train: original {original_train}, filtered {filtered_train}")
    print(f"Val: original {original_val}, filtered {filtered_val}")