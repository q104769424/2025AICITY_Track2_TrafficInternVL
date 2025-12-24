import os
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, default='data', help='data root path')
parser.add_argument('--save-folder', type=str, default='processed_anno/qa_format', help='output directory for saving json file')
parser.add_argument('--wts-global-image-path', type=str, required=True, help='root path for wts global images')
args = parser.parse_args()

phrase_number_map = {
    '0': 'prerecognition',
    '1': 'recognition',
    '2': 'judgement',
    '3': 'action',
    '4': 'avoidance'
}
number_phrase_map = {v: k for k, v in phrase_number_map.items()}

camera_path_mapping = {}
train_samples = []

# ---------- Process WTS overhead_view only (train + val) ----------
for split in ['train', 'val']:
    wts_anno_path = os.path.join(args.root, 'WTS/annotations/caption', split)
    for item in os.listdir(wts_anno_path):
        sample_id = item
        overhead_path = os.path.join(wts_anno_path, item, 'overhead_view', f'{item}_caption.json')
        if not os.path.exists(overhead_path):
            continue

        overhead_view = json.load(open(overhead_path))
        for event in overhead_view['event_phase']:
            for image in overhead_view['overhead_videos']:
                train_samples.append({
                    'id': sample_id,
                    'segment': phrase_number_map[event['labels'][0]],
                    'view': 'overhead',
                    'start_time': event['start_time'],
                    'end_time': event['end_time'],
                    'conversations': [
                        {'from': 'human', 'value': '<image>\nPlease describe the interested pedestrian in the video.'},
                        {'from': 'gpt', 'value': event['caption_pedestrian']},
                        {'from': 'human', 'value': 'Please describe the interested vehicle in the video.'},
                        {'from': 'gpt', 'value': event['caption_vehicle']}
                    ],
                    'image': image
                })

# ---------- Build camera path mapping ----------
for split in ['train', 'val']:
    split_path = os.path.join(args.wts_global_image_path, split)
    for event in os.listdir(split_path):
        event_path = os.path.join(split_path, event)
        for view in os.listdir(event_path):
            view_path = os.path.join(event_path, view)
            for camera in os.listdir(view_path):
                camera_path_mapping[camera] = os.path.join(view_path, camera)

# ---------- Filter valid samples ----------
reserved_samples = []
for item in train_samples:
    image = item['image'].replace('.mp4', '')
    segment = item['segment']

    if 'video' in image:
        image_name = f'{image}_{segment}.jpg'
    else:
        image_name = f'{number_phrase_map[segment]}_{segment}.jpg'

    if image in camera_path_mapping:
        image_path = os.path.join(camera_path_mapping[image], image_name)
        if os.path.exists(image_path):
            item['image'] = image_path.replace('./data/', '')
            reserved_samples.append(item)

# ---------- Save ----------
os.makedirs(args.save_folder, exist_ok=True)
output_path = os.path.join(args.save_folder, 'wts_gaze_qa.json')
with open(output_path, 'w') as f:
    json.dump(reserved_samples, f, indent=4)

# ---------- Statistics ----------
print("\n--- WTS Gaze Export Summary ---")
print(f"Total valid samples: {len(reserved_samples)}")
print(f"Output saved to: {output_path}")
