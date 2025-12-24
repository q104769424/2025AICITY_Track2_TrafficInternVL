import os
import json
import csv
import argparse
from tqdm import tqdm

def filter_data(
    data_path,
    save_path,
    split_name,
    reference_view_path,
    root_path='./data',
    area_threshold=1000
):
    """
    Filter multimodal QA samples based on viewpoint reference and bounding box area.

    Args:
        data_path (str): Path to the input JSON file.
        save_path (str): Path to save the filtered JSON file.
        split_name (str): Dataset split name (e.g., 'train', 'val').
        reference_view_path (str): CSV path specifying valid overhead viewpoints.
        root_path (str): Root directory for annotation files.
        area_threshold (int): Minimum bbox area threshold for pedestrian/vehicle.
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
        next(reader)
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
                print(f"[View Filtered] {data['image'][0]}, view {view} not in reference list")
                continue

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

        pedestrian_bbox_data = ''
        vehicle_bbox_data = ''
        try:
            pedestrian_bbox_data = json.load(open(pedestrian_box_path))
        except FileNotFoundError:
            print(f"[Missing Pedestrian JSON] {pedestrian_box_path}")
        try:
            if vehicle_box_path:
                vehicle_bbox_data = json.load(open(vehicle_box_path))
        except FileNotFoundError:
            print(f"[Missing Vehicle JSON] {vehicle_box_path}")

        human_area = 0
        if pedestrian_bbox_data:
            for box in pedestrian_bbox_data.get('annotations', []):
                if str(stage_map[stage]) == str(box.get('phase_number')):
                    human_area = box['bbox'][2] * box['bbox'][3]
                    break

        vehicle_area = 0
        if vehicle_bbox_data:
            for box in vehicle_bbox_data.get('annotations', []):
                if str(stage_map[stage]) == str(box.get('phase_number')):
                    vehicle_area = box['bbox'][2] * box['bbox'][3]
                    break

        if human_area > area_threshold or vehicle_area > area_threshold:
            filtered_data.append(data)
        else:
            print(f"[Area Filtered] {data['image'][0]} | Human: {human_area:.1f}, Vehicle: {vehicle_area:.1f} < threshold {area_threshold}")

    for i, entry in enumerate(filtered_data):
        entry['id'] = i

    print(f"Original {split_name} size: {len(data_json)} | Filtered: {len(filtered_data)}")
    with open(save_path, 'w') as f:
        f.write(json.dumps(filtered_data, indent=2, ensure_ascii=False))
    print(f"Filtered {split_name} saved to: {save_path}")

    return len(data_json), len(filtered_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference_view_path', type=str, required=True, help='Path to reference view CSV')

    args = parser.parse_args()

    train_data_path = './processed_anno/qa_format/wts_bdd_multimodal_qa_train.json'
    train_save_path = './processed_anno/qa_format/wts_bdd_multimodal_qa_train_filtered.json'
    original_train, filtered_train = filter_data(train_data_path, train_save_path, 'train', args.reference_view_path)

    val_data_path = './processed_anno/qa_format/wts_bdd_multimodal_qa_val.json'
    val_save_path = './processed_anno/qa_format/wts_bdd_multimodal_qa_val_filtered.json'
    original_val, filtered_val = filter_data(val_data_path, val_save_path, 'val', args.reference_view_path)

    print("\n--- Final Statistics ---")
    print(f"Train: Original {original_train}, Filtered {filtered_train}")
    print(f"Val: Original {original_val}, Filtered {filtered_val}")