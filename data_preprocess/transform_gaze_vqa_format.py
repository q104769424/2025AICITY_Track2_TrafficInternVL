import os
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, default='data', help='data root path')
parser.add_argument('--save-folder', type=str, default='processed_anno/vqa_gaze_format', help='output directory for saving json file')
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
raw_samples = []

# ---------- Step 1: Process WTS VQA overhead_view ----------
for split in ['train', 'val']:
    vqa_path = os.path.join(args.root, 'WTS/annotations/vqa', split)
    if not os.path.exists(vqa_path):
        continue

    for sample_id in os.listdir(vqa_path):
        overhead_json = os.path.join(vqa_path, sample_id, 'overhead_view', f'{sample_id}.json')
        if not os.path.exists(overhead_json):
            continue

        try:
            data_list = json.load(open(overhead_json))
            if not data_list or 'event_phase' not in data_list[0]:
                continue

            data = data_list[0]
            if 'overhead_videos' not in data or not data['overhead_videos']:
                continue

            for event in data['event_phase']:
                if 'conversations' not in event or not event['conversations']:
                    continue

                segment = phrase_number_map.get(event['labels'][0], event['labels'][0])
                for image in data['overhead_videos']:
                    raw_samples.append({
                        'id': sample_id,
                        'segment': segment,
                        'view': 'overhead',
                        'start_time': event['start_time'],
                        'end_time': event['end_time'],
                        'conversations': [],
                        'image': image,
                        'original_conversations': event['conversations']
                    })

        except Exception as e:
            print(f"Error processing {overhead_json}: {e}")

# ---------- Step 2: Build camera path mapping ----------
for split in ['train', 'val']:
    split_path = os.path.join(args.wts_global_image_path, split)
    for event in os.listdir(split_path):
        event_path = os.path.join(split_path, event)
        for view in os.listdir(event_path):
            view_path = os.path.join(event_path, view)
            for camera in os.listdir(view_path):
                camera_path_mapping[camera] = os.path.join(view_path, camera)

# ---------- Step 3: Filter samples and assign image path ----------
reserved_samples = []

def convert_vqa_to_conversations(vqa_qa_list):
    conversations = []
    for qa_pair in vqa_qa_list:
        question = qa_pair['question']
        correct_key = qa_pair['correct']

        choices = ""
        option_keys = sorted([k for k in qa_pair if len(k) == 1 and k.isalpha()])
        for k in option_keys:
            choices += f"{k}. {qa_pair[k]}\n"

        correct_answer = qa_pair.get(correct_key, '')
        conversations.append({
            'from': 'human',
            'value': f"<image>\n{question}\n{choices.strip()}\nAnswer with the option's letter and its corresponding text."
        })
        conversations.append({
            'from': 'gpt',
            'value': f"{correct_key}. {correct_answer}"
        })
    return conversations

for item in raw_samples:
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
            item['conversations'] = convert_vqa_to_conversations(item.pop('original_conversations'))
            reserved_samples.append(item)

# ---------- Step 4: Save ----------
os.makedirs(args.save_folder, exist_ok=True)
output_path = os.path.join(args.save_folder, 'wts_gaze_vqa.json')
with open(output_path, 'w') as f:
    json.dump(reserved_samples, f, indent=4)

# ---------- Summary ----------
print("\n--- WTS Gaze Export Summary ---")
print(f"Total valid VQA samples: {len(reserved_samples)}")
print(f"Output saved to: {output_path}")
