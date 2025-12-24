import json
import tqdm
import copy
import os
from PIL import Image
import random
import numpy as np
import argparse

def get_image_dimensions(image_path):
    """
    Gets the height and width of an image.
    Returns (0, 0) if the file is not found or an error occurs.
    """
    try:
        with Image.open(image_path) as img:
            return img.height, img.width
    except FileNotFoundError:
        # Warning messages are suppressed for cleaner output during tqdm iteration
        return 0, 0
    except Exception as e:
        print(f"Error reading image {image_path}: {e}. Returning default dimensions (0, 0).")
        return 0, 0

def process_vqa_to_multimodal(base_image_path, input_filepath, output_filepath):
    print(f"\n--- Processing {os.path.basename(input_filepath)} for Multi-modal VQA conversion ---")
    
    with open(input_filepath, 'r', encoding='utf-8') as f:
        data_list = json.load(f)

    processed_data = []
    filtered_count = 0
    current_id = 0

    for data in tqdm.tqdm(data_list, desc="Converting to multi-modal VQA"):
        image_relative_path_original = data['image']

        global_image_path = os.path.join(base_image_path, image_relative_path_original)
        local_image_path = os.path.join(base_image_path, image_relative_path_original.replace("bbox_global", "bbox_local"))

        # Get image sizes
        h_global, w_global = get_image_dimensions(global_image_path)
        h_local, w_local = get_image_dimensions(local_image_path)

        if (h_global == 0 and w_global == 0) or (h_local == 0 and w_local == 0):
            filtered_count += 1
            continue

        width_list = [w_global, w_local]
        height_list = [h_global, h_local]

        filename = os.path.splitext(os.path.basename(global_image_path))[0]

        descriptive_prompt_prefix = (
            f"These two images show the interaction between the person in the green box "
            "and the car in the blue box, which should be the focus when answering the following question.\n"
            "The global image shows the full scene, while the local image highlights the relevant area.\n"
            "Global image: <image>\n"
            "Local image: <image>"
        )

        convs = data['conversations']
        assert len(convs) % 2 == 0, "Each conversation must be in Q&A pairs."

        for i in range(0, len(convs), 2):
            question = convs[i]
            answer = convs[i + 1]

            if question['from'] != 'human' or answer['from'] != 'gpt':
                continue

            original_q = question['value'].replace('<image>\n', '', 1).strip()
            combined_question = f"{descriptive_prompt_prefix}\n{original_q}"

            new_data = {
                "video_id": data['id'], 
                "id": current_id,
                "image": [global_image_path, local_image_path],
                "width_list": width_list,
                "height_list": height_list,
                "conversations": [
                    {"from": "human", "value": combined_question},
                    answer
                ]
            }

            processed_data.append(new_data)
            current_id += 1

    print(f"Finished processing. Total individual Q&A entries: {len(processed_data)}. Filtered entries due to image issues: {filtered_count}")

    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    with open(output_filepath, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=4)
    print(f"Multi-modal VQA data saved to {output_filepath}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-image-path', type=str, required=True, help='Base path to image root directory')
    args = parser.parse_args()

    output_dir = "./processed_anno/vqa_format"
    os.makedirs(output_dir, exist_ok=True)

    input_train_file = os.path.join(output_dir, "wts_bdd_train.json")
    output_train_file = os.path.join(output_dir, "wts_bdd_multimodal_vqa_train.json")

    input_val_file = os.path.join(output_dir, "wts_bdd_val.json")
    output_val_file = os.path.join(output_dir, "wts_bdd_multimodal_vqa_val.json")

    process_vqa_to_multimodal(args.base_image_path, input_train_file, output_train_file)
    process_vqa_to_multimodal(args.base_image_path, input_val_file, output_val_file)

    print("\n--- Processing Complete ---")
    print(f"Final training data is at: {output_train_file}")
    print(f"Final validation data is at: {output_val_file}")