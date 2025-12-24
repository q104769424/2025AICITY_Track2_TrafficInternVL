import os
import json
import tqdm
from PIL import Image
import argparse

def get_image_dimensions(image_path):
    try:
        with Image.open(image_path) as img:
            return img.height, img.width
    except Exception:
        return 0, 0

def process_vqa_gaze_to_multimodal(base_image_path, input_filepath):
    with open(input_filepath, 'r', encoding='utf-8') as f:
        data_list = json.load(f)

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

        prompt_prefix = (
            f"This image show the interaction between the person in the green box "
            "and the car in the blue box, which should be the focus when answering the following question.\n"
            "The red fan-shaped area represents the line of sight of the person in the green box, so please also pay attention to that area.\n"
            "Image: <image>"
        )

        conversations = data['conversations']
        for i in range(0, len(conversations), 2):
            q = conversations[i]
            a = conversations[i + 1] if i + 1 < len(conversations) else None

            if not a or q['from'] != 'human' or a['from'] != 'gpt':
                continue
            if 'line of sight' not in q['value'].lower():
                filtered += 1
                continue

            original_q = q['value'].replace('<image>\n', '', 1).strip()
            combined_question = f"{prompt_prefix}\n{original_q}"

            results.append({
                "video_id": data["id"],
                "id": current_id,
                "segment": data.get("segment", ""),
                "view": data.get("view", ""),
                "start_time": data.get("start_time", ""),
                "end_time": data.get("end_time", ""),
                "image": [global_path],
                "width_list": [w],
                "height_list": [h],
                "conversations": [
                    {"from": "human", "value": combined_question},
                    a
                ]
            })
            current_id += 1

    print(f"Processed {len(results)} entries. Filtered {filtered} due to missing images or missing 'line of sight'.")
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-image-path', type=str, required=True, help='Base path to image root directory')
    args = parser.parse_args()

    output_dir = "./processed_anno/gaze_vqa_format"
    os.makedirs(output_dir, exist_ok=True)

    input_json = os.path.join(output_dir, "wts_gaze_vqa.json")
    output_json = os.path.join(output_dir, "wts_multimodal_gaze_vqa_train.json")

    print("\n--- Processing WTS Pedestrian Gaze Data ---")
    processed = process_vqa_gaze_to_multimodal(args.base_image_path, input_json)

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(processed, f, ensure_ascii=False, indent=2)

    print(f"\n--- Done. Output written to: {output_json}")