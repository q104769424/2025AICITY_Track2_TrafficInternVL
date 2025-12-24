import json
import tqdm
import copy
import os
from PIL import Image
import random
import numpy as np
import argparse

def process_conversations(file_path):
    """
    Reads a JSON file with a specific format, splits conversation entries
    for pedestrians and vehicles into separate dictionary entries, and returns
    a new list of these split entries.
    """
    processed_data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for entry in data:
        # Create a deep copy to avoid modifying the original entry prematurely
        pedestrian_entry = copy.deepcopy(entry)
        vehicle_entry = copy.deepcopy(entry)

        # Extract the first conversation (pedestrian)
        pedestrian_entry['conversations'] = entry['conversations'][0:2]
        processed_data.append(pedestrian_entry)

        # Extract the second conversation (vehicle)
        vehicle_entry['conversations'] = entry['conversations'][2:4]
        processed_data.append(vehicle_entry)

    return processed_data

def get_image_dimensions(image_path):
    """
    Gets the height and width of an image.
    Returns (0, 0) if the file is not found or an error occurs.
    """
    try:
        with Image.open(image_path) as img:
            return img.height, img.width
    except FileNotFoundError:
        # print(f"Warning: Image file not found at {image_path}. Returning default dimensions (0, 0).")
        return 0, 0
    except Exception as e:
        print(f"Error reading image {image_path}: {e}. Returning default dimensions (0, 0).")
        return 0, 0

def process_multimodal_data(base_image_path, input_file):
    """
    Processes the raw data:
    1. Splits combined pedestrian/vehicle conversations into separate entries.
    2. Converts each entry to a multimodal (two-image) format.
    3. Filters out entries if either global or local image is not found.
    4. Assigns sequential IDs.
    """
    all_split_data = process_conversations(input_file)

    result = []
    filtered_count = 0
    current_id = 0

    for data in tqdm.tqdm(all_split_data, desc=f"Processing multimodal data from {os.path.basename(input_file)}"):
        image_relative_path_original = data['image']
        global_image_path = os.path.join(base_image_path, image_relative_path_original)
        local_image_path = os.path.join(base_image_path, image_relative_path_original.replace("bbox_global", "bbox_local"))

        h_global, w_global = get_image_dimensions(global_image_path)
        h_local, w_local = get_image_dimensions(local_image_path)

        if (h_global == 0 and w_global == 0) or (h_local == 0 and w_local == 0):
            filtered_count += 1
            continue

        width_list = [w_global, w_local]
        height_list = [h_global, h_local]

        filename = os.path.splitext(os.path.basename(global_image_path))[0]

        PEDESTRIAN_HUMAN_PROMPT = (
            "These two images show the interaction between the person in the green box "
            "and the car in the blue box, which should be the focus when answering the following question.\n"
            "The global image shows the full scene, while the local image highlights the relevant area.\n"
            "Global image: <image>\n"
            "Local image: <image>\n"
            "Based on the images, describe the pedestrian highlighted in green box or the one closest to the car in the blue box. Include their apparent age, height, and clothing, as well as their line of sight and position relative to the vehicle. Describe their movement behavior, and provide observations about the weather conditions and the road environment. "
            "Also indicate whether the pedestrian appears to be aware of the vehicle."
        )

        VEHICLE_HUMAN_PROMPT = (
            "These two images show the interaction between the person in the green box "
            "and the car in the blue box, which should be the focus when answering the following question.\n"
            "The global image shows the full scene, while the local image highlights the relevant area.\n"
            "Global image: <image>\n"
            "Local image: <image>\n"
            "Based on the images, describe the vehicle highlighted in blue box or the one closest to the pedestrian in the green box. Include its position relative to the pedestrian, driving behavior, weather conditions, and road environment. "
            "Also briefly describe the pedestrian in the green box, including their apparent age, height, and clothing."
        )

        human_question_original = data['conversations'][0]['value']
        gpt_response_original = data['conversations'][1]['value']

        if "pedestrian" in human_question_original.lower():
            human_value_to_use = PEDESTRIAN_HUMAN_PROMPT
        elif "vehicle" in human_question_original.lower():
            human_value_to_use = VEHICLE_HUMAN_PROMPT
        else:
            print(f"Warning: Unexpected question format: {human_question_original}")
            filtered_count += 1
            continue

        new_data = {
            "video_id": data['id'], 
            "id": current_id,
            "image": [global_image_path, local_image_path],
            "width_list": width_list,
            "height_list": height_list,
            "conversations": [
                {"from": "human", "value": human_value_to_use},
                {"from": "gpt", "value": gpt_response_original}
            ]
        }

        result.append(new_data)
        current_id += 1

    print(f"Len of processed multimodal data from {os.path.basename(input_file)} is {len(result)}. Filtered {filtered_count} entries due to missing images or unexpected questions.")
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-image-path', type=str, required=True, help='Base path to image root directory')
    args = parser.parse_args()

    output_dir = "./processed_anno/qa_format"
    os.makedirs(output_dir, exist_ok=True)

    train_input_file = os.path.join(output_dir, "wts_bdd_train.json")
    train_output_file = os.path.join(output_dir, "wts_bdd_multimodal_qa_train.json")

    val_input_file = os.path.join(output_dir, "wts_bdd_val.json")
    val_output_file = os.path.join(output_dir, "wts_bdd_multimodal_qa_val.json")

    print("\n--- Processing Training Data ---")
    processed_train_data = process_multimodal_data(args.base_image_path, train_input_file)
    with open(train_output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_train_data, f, indent=2, ensure_ascii=False)

    print("\n--- Processing Validation Data ---")
    processed_val_data = process_multimodal_data(args.base_image_path, val_input_file)
    with open(val_output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_val_data, f, indent=2, ensure_ascii=False)

    print(f"\n--- All Processing Complete ---")
    print(f"Final training data is at: {train_output_file}")
    print(f"Final validation data is at: {val_output_file}")