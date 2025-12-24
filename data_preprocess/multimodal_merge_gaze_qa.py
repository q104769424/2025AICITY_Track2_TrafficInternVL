import os
import json
import tqdm
import copy
from PIL import Image
import argparse

def process_conversations(file_path):
    processed_data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for entry in data:
        ped_entry = copy.deepcopy(entry)
        ped_entry['conversations'] = entry['conversations'][0:2]
        processed_data.append(ped_entry)
    return processed_data

def get_image_dimensions(image_path):
    try:
        with Image.open(image_path) as img:
            return img.height, img.width
    except Exception:
        return 0, 0

def process_multimodal_data(base_image_path, json_path):
    data_list = process_conversations(json_path)
    results = []
    filtered = 0
    current_id = 0

    for data in tqdm.tqdm(data_list, desc="Processing WTS pedestrian-only data"):
        rel_path = data['image']
        global_path = os.path.join(base_image_path, rel_path)
        h, w = get_image_dimensions(global_path)

        if h == 0 or w == 0:
            filtered += 1
            continue

        gpt_response = data['conversations'][1]['value']
        if 'line of sight' not in gpt_response.lower():
            filtered += 1
            continue

        ped_prompt = (
            f"This image show the interaction between the person in the green box "
            "and the car in the blue box, which should be the focus when answering the following question.\n"
            "The red fan-shaped area represents the line of sight of the person in the green box, so please also pay attention to that area.\n"
            "Image: <image>\n"
            "Based on the images, describe the pedestrian highlighted in green box or the one closest to the car in the blue box. Include their apparent age, height, and clothing, as well as their line of sight and position relative to the vehicle. Describe their movement behavior, and provide observations about the weather conditions and the road environment. "
            "Also indicate whether the pedestrian appears to be aware of the vehicle."
        )
        
        results.append({
            "video_id": data['id'],
            "id": current_id,
            "image": [global_path],
            "width_list": [w],
            "height_list": [h],
            "conversations": [
                {"from": "human", "value": ped_prompt},
                {"from": "gpt", "value": gpt_response}
            ]
        })
        current_id += 1

    print(f"Processed {len(results)} entries. Filtered {filtered} due to missing images or missing 'line of sight'.")
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-image-path', type=str, required=True, help='Base path to image root directory')
    args = parser.parse_args()

    output_dir = "./processed_anno/gaze_qa_format"
    os.makedirs(output_dir, exist_ok=True)

    input_json = os.path.join(output_dir, "wts_gaze_qa.json")
    output_json = os.path.join(output_dir, "wts_multimodal_gaze_qa_train.json")

    print("\n--- Processing WTS Pedestrian Gaze Data ---")
    processed = process_multimodal_data(args.base_image_path, input_json)

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(processed, f, indent=2, ensure_ascii=False)

    print(f"\n--- Done. Output written to: {output_json}")