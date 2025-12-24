import os
import json
import copy
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, default='data', help='data root path')
parser.add_argument('--split', type=str, default='train')
parser.add_argument('--save-folder', type=str, default='processed_anno/vqa_format', help='dirname for saving json file')
parser.add_argument('--wts-global-image-path', type=str, required=True, help='root path for wts global images')
parser.add_argument('--bdd-global-image-path', type=str, required=True, help='root path for bdd global images')

args = parser.parse_args()

root = args.root

phrase_number_map = {
    '0': 'prerecognition',
    '1': 'recognition',
    '2': 'judgement',
    '3': 'action',
    '4': 'avoidance'
}
# Invert the map for easier lookup by event name
number_phrase_map = {v: k for k, v in phrase_number_map.items()}

train_samples = list()

overhead = 'overhead_view'
vehicle = 'vehicle_view'
environment = 'environment'

# Base path for WTS VQA annotations
wts_anno_root_path = os.path.join(root, 'WTS/annotations/vqa', args.split)
bdd_anno_path = os.path.join(root, 'BDD_PC_5k/annotations/vqa', args.split)

print(f"Checking WTS annotation root path: {wts_anno_root_path}, Exists: {os.path.exists(wts_anno_root_path)}")
print(f"Checking BDD annotation path: {bdd_anno_path}, Exists: {os.path.exists(bdd_anno_path)}")

# Initialize counters
processing_stats = {
    'WTS': {
        'overhead': 0,
        'vehicle': 0,
        'environment': 0,
        'normal_trimmed_overhead': 0,
        'normal_trimmed_vehicle': 0,
        'normal_trimmed_environment': 0
    },
    'BDD': {
        'prerecognition': 0,
        'recognition': 0,
        'judgement': 0,
        'action': 0,
        'avoidance': 0,
        'environment': 0 # Keep for specific environment-view processing if it exists
    },
    'total_samples': 0
}

def convert_vqa_to_conversations(vqa_qa_list):
    conversations = []
    for qa_pair in vqa_qa_list:
        question = qa_pair['question']
        correct_choice_key = qa_pair['correct']

        choices_str = ""
        # Dynamically get option keys and sort them to ensure consistent order (a, b, c, ...)
        option_keys = sorted([k for k in qa_pair.keys() if len(k) == 1 and k.isalpha() and k.islower() and k != 'correct'])

        for key in option_keys:
            choices_str += f"{key}. {qa_pair[key]}\n"  

        correct_answer_text = qa_pair.get(correct_choice_key, '')
        if not correct_answer_text:
            print(f"Warning: Correct answer for question '{question}' (key: {correct_choice_key}) is empty or missing from qa_pair's options.")

        conversations.append({
            'from': 'human',
            'value': f"<image>\n{question}\n{choices_str.strip()}\nAnswer with the option's letter and its corresponding text."
        })
        conversations.append({
            'from': 'gpt',
            'value': f"{correct_choice_key}. {correct_answer_text}"
        })
    return conversations

def process_wts_data(base_path):
    wts_items = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]

    for item in wts_items:
        sample_id = item 

        is_normal_trimmed_sample = '_normal_trimmed' in sample_id or '_normal' in sample_id

        overhead_view_json_path = os.path.join(base_path, item, overhead, f'{item}.json')
        vehicle_view_json_path = os.path.join(base_path, item, vehicle, f'{item}.json')
        environment_view_json_path = os.path.join(base_path, item, environment, f'{item}.json')

        # WTS Overhead View
        if os.path.exists(overhead_view_json_path):
            try:
                overhead_view_data_list = json.load(open(overhead_view_json_path))
                if overhead_view_data_list:
                    overhead_view_data = overhead_view_data_list[0]
                    if 'event_phase' in overhead_view_data and overhead_view_data['event_phase']:
                        for event in overhead_view_data['event_phase']:
                            if event.get('conversations'):
                                # Determine the segment name (phrase) for the JSON output
                                segment_phrase_for_json = phrase_number_map.get(event['labels'][0], event['labels'][0])

                                cur_data = {
                                    'id': sample_id,
                                    'segment': segment_phrase_for_json,
                                    'view': 'overhead',
                                    'start_time': event['start_time'],
                                    'end_time': event['end_time'],
                                    'conversations': convert_vqa_to_conversations(event['conversations'])
                                }
                                if 'overhead_videos' in overhead_view_data and overhead_view_data['overhead_videos']:
                                    for image_video_path in overhead_view_data['overhead_videos']:
                                        temp_data = copy.deepcopy(cur_data)

                                        camera_id_folder = os.path.splitext(os.path.basename(image_video_path))[0]

                                        raw_event_label = event['labels'][0]
                                        stage_id_num = number_phrase_map[raw_event_label] if raw_event_label in number_phrase_map else raw_event_label
                                        stage_name_phrase = phrase_number_map.get(stage_id_num, stage_id_num) 
                                        image_filename_jpg = f"{stage_id_num}_{stage_name_phrase}.jpg"

                                        image_mid_path_parts = ["bbox_global", args.split]
                                        if is_normal_trimmed_sample:
                                            image_mid_path_parts.append("normal_trimmed")
                                        image_mid_path = os.path.join(*image_mid_path_parts)

                                        temp_data['image'] = os.path.join("WTS", image_mid_path, sample_id, overhead, camera_id_folder, image_filename_jpg)
                                        
                                        train_samples.append(temp_data)
                                        if is_normal_trimmed_sample:
                                            processing_stats['WTS']['normal_trimmed_overhead'] += 1
                                        else:
                                            processing_stats['WTS']['overhead'] += 1
                                else:
                                    print(f"Warning: No 'overhead_videos' found in WTS Overhead VQA for {sample_id}. Skipping.")
            except json.JSONDecodeError:
                print(f"Error decoding JSON for {overhead_view_json_path}")
            except Exception as e:
                print(f"An unexpected error occurred loading {overhead_view_json_path}: {e}")

        # WTS Vehicle View
        if os.path.exists(vehicle_view_json_path):
            try:
                vehicle_view_data_list = json.load(open(vehicle_view_json_path))
                if vehicle_view_data_list:
                    vehicle_view_data = vehicle_view_data_list[0]
                    if 'event_phase' in vehicle_view_data and vehicle_view_data['event_phase']:
                        for event in vehicle_view_data['event_phase']:
                            if event.get('conversations'):
                                segment_phrase_for_json = phrase_number_map.get(event['labels'][0], event['labels'][0])

                                cur_data = {
                                    'id': sample_id,
                                    'segment': segment_phrase_for_json,
                                    'view': 'vehicle',
                                    'start_time': event['start_time'],
                                    'end_time': event['end_time'],
                                    'conversations': convert_vqa_to_conversations(event['conversations'])
                                }
                                if 'vehicle_view' in vehicle_view_data and vehicle_view_data['vehicle_view']:
                                    raw_event_label = event['labels'][0]
                                    stage_id_num = number_phrase_map[raw_event_label] if raw_event_label in number_phrase_map else raw_event_label
                                    stage_name_phrase = phrase_number_map.get(stage_id_num, stage_id_num)

                                    image_filename_jpg = f"{stage_id_num}_{stage_name_phrase}.jpg"

                                    image_mid_path_parts = ["bbox_global", args.split]
                                    if is_normal_trimmed_sample:
                                        image_mid_path_parts.append("normal_trimmed")
                                    image_mid_path = os.path.join(*image_mid_path_parts)
                                    cur_data['image'] = os.path.join("WTS", image_mid_path, sample_id, vehicle, sample_id+"_vehicle_view", image_filename_jpg)
                                    train_samples.append(cur_data)
                                    if is_normal_trimmed_sample:
                                        processing_stats['WTS']['normal_trimmed_vehicle'] += 1
                                    else:
                                        processing_stats['WTS']['vehicle'] += 1
                                else:
                                    print(f"Warning: No 'vehicle_view' found in WTS Vehicle VQA for {sample_id}. Skipping.")
            except json.JSONDecodeError:
                print(f"Error decoding JSON for {vehicle_view_json_path}")
            except Exception as e:
                print(f"An unexpected error occurred loading {vehicle_view_json_path}: {e}")

        # WTS Environment View (image reference logic)
        overhead_camera_id_folder = None
        # Only try to get camera_id if overhead_view_json_path exists
        if os.path.exists(overhead_view_json_path):
            try:
                temp_overhead_view_data_list = json.load(open(overhead_view_json_path))
                if temp_overhead_view_data_list and temp_overhead_view_data_list[0].get('overhead_videos'):
                    overhead_camera_id_folder = os.path.splitext(os.path.basename(temp_overhead_view_data_list[0]['overhead_videos'][0]))[0]
            except (json.JSONDecodeError, IndexError, KeyError):
                pass # Continue if we can't get this, it might be the reason for 0 samples
            except Exception as e:
                print(f"Error loading temp WTS overhead data for environment to get camera_id: {e}")

        # Process environment JSON for conversations, then find an image
        if os.path.exists(environment_view_json_path):
            try:
                environment_view_data_list = json.load(open(environment_view_json_path))
                if environment_view_data_list and 'environment' in environment_view_data_list[0] and environment_view_data_list[0]['environment']:
                    cur_data = {
                        'id': sample_id,
                        'segment': 'environment', # Environment view segment is fixed as 'environment'
                        'view': 'environment',
                        'start_time': "N/A", # Environment view typically doesn't have specific start/end times
                        'end_time': "N/A",
                        'conversations': convert_vqa_to_conversations(environment_view_data_list[0]['environment'])
                    }
                    image_found = False
                    
                    image_mid_path_parts = ["bbox_global", args.split]
                    if is_normal_trimmed_sample:
                        image_mid_path_parts.append("normal_trimmed")
                    image_mid_path = os.path.join(*image_mid_path_parts)

                    # Try to find the smallest phase image from overhead view for environment
                    sorted_phases = sorted(phrase_number_map.items(), key=lambda item: int(item[0]))
                    
                    if overhead_camera_id_folder:
                        for num_str, phrase in sorted_phases:
                            potential_image_filename = f"{num_str}_{phrase}.jpg"
                            full_image_path_candidate = os.path.join(args.wts_global_image_path, args.split, ("normal_trimmed" if is_normal_trimmed_sample else ""), sample_id, overhead, overhead_camera_id_folder, potential_image_filename)
       
                            if os.path.exists(full_image_path_candidate):
                                cur_data['image'] = os.path.join("WTS", image_mid_path, sample_id, overhead, overhead_camera_id_folder, potential_image_filename)
                                train_samples.append(cur_data)
                                image_found = True
                                if is_normal_trimmed_sample:
                                    processing_stats['WTS']['normal_trimmed_environment'] += 1
                                else:
                                    processing_stats['WTS']['environment'] += 1
                                break # Found the smallest, break and use this one
                    
                    # If no overhead image found or no overhead_camera_id_folder, try vehicle image
                    if not image_found:
                        for num_str, phrase in sorted_phases:
                            potential_image_filename = f"{num_str}_{phrase}.jpg"
                            full_image_path_candidate = os.path.join(args.wts_global_image_path, args.split, ("normal_trimmed" if is_normal_trimmed_sample else ""), sample_id, vehicle,  sample_id+"_vehicle_view", potential_image_filename)

                            if os.path.exists(full_image_path_candidate):
                                cur_data['image'] = os.path.join("WTS", image_mid_path, sample_id, vehicle,  sample_id+"_vehicle_view", potential_image_filename)
                                train_samples.append(cur_data)
                                image_found = True
                                if is_normal_trimmed_sample:
                                    processing_stats['WTS']['normal_trimmed_environment'] += 1
                                else:
                                    processing_stats['WTS']['environment'] += 1
                                break # Found the smallest, break and use this one

                    if not image_found:
                        print(f"Warning: No suitable smallest phase image found for WTS {'normal_trimmed ' if is_normal_trimmed_sample else ''}environment view {sample_id}. Skipping. path:{full_image_path_candidate}")
            except json.JSONDecodeError:
                print(f"Error decoding JSON for {environment_view_json_path}")
            except Exception as e:
                print(f"An unexpected error occurred loading {environment_view_json_path}: {e}")

# --- Call process_wts_data for both main and normal_trimmed paths ---
if os.path.exists(wts_anno_root_path):
    print(f"Processing main WTS data from: {wts_anno_root_path}")
    process_wts_data(wts_anno_root_path)

normal_trimmed_wts_anno_path = os.path.join(wts_anno_root_path, 'normal_trimmed')
if os.path.exists(normal_trimmed_wts_anno_path):
    print(f"Processing normal_trimmed WTS data from: {normal_trimmed_wts_anno_path}")
    process_wts_data(normal_trimmed_wts_anno_path)
else:
    print(f"Normal trimmed WTS annotation path not found: {normal_trimmed_wts_anno_path}")


# --- Process BDD VQA Data ---
if os.path.exists(bdd_anno_path):
    bdd_items = os.listdir(bdd_anno_path)
    for item in bdd_items:
        video_id = item

        # BDD Overhead View
        bdd_overhead_path = os.path.join(bdd_anno_path, video_id, 'overhead_view', f'{video_id}.json')
        if os.path.exists(bdd_overhead_path):
            try:
                bdd_overhead_data_list = json.load(open(bdd_overhead_path))
                if bdd_overhead_data_list and 'event_phase' in bdd_overhead_data_list[0] and bdd_overhead_data_list[0]['event_phase']:
                    for event in bdd_overhead_data_list[0]['event_phase']:
                        if event.get('conversations'):
                            segment_phrase_for_json = phrase_number_map.get(event['labels'][0], event['labels'][0])
                            cur_data = {
                                'id': video_id,
                                'segment': segment_phrase_for_json,
                                'view': 'overhead',
                                'start_time': event['start_time'],
                                'end_time': event['end_time'],
                                'conversations': convert_vqa_to_conversations(event['conversations'])
                            }
                            if 'videos' in bdd_overhead_data_list[0]:
                                segment_id_for_filename = event['labels'][0]
                                segment_name_for_filename = phrase_number_map.get(segment_id_for_filename, segment_id_for_filename)
                                image_filename_jpg = f"{video_id}_{segment_name_for_filename}.jpg"

                                cur_data['image'] = os.path.join("BDD_PC_5k", "bbox_global", args.split, image_filename_jpg)
                                train_samples.append(cur_data)
                                processing_stats['BDD'][segment_phrase_for_json] += 1
                            else:
                                print(f"Warning: No 'videos' key found in BDD Overhead VQA for {video_id}. Skipping.")
            except json.JSONDecodeError:
                print(f"Error decoding JSON for {bdd_overhead_path}")
            except Exception as e:
                print(f"An unexpected error occurred loading {bdd_overhead_path}: {e}")

        # BDD Vehicle View
        bdd_vehicle_path = os.path.join(bdd_anno_path, video_id, 'vehicle_view', f'{video_id}.json')
        if os.path.exists(bdd_vehicle_path):
            try:
                bdd_vehicle_data_list = json.load(open(bdd_vehicle_path))
                if bdd_vehicle_data_list and 'event_phase' in bdd_vehicle_data_list[0] and bdd_vehicle_data_list[0]['event_phase']:
                    for event in bdd_vehicle_data_list[0]['event_phase']:
                        if event.get('conversations'):
                            segment_phrase_for_json = phrase_number_map.get(event['labels'][0], event['labels'][0])
                            cur_data = {
                                'id': video_id,
                                'segment': segment_phrase_for_json,
                                'view': 'vehicle',
                                'start_time': event['start_time'],
                                'end_time': event['end_time'],
                                'conversations': convert_vqa_to_conversations(event['conversations'])
                            }
                            if 'videos' in bdd_vehicle_data_list[0]:
                                segment_id_for_filename = event['labels'][0]
                                segment_name_for_filename = phrase_number_map.get(segment_id_for_filename, segment_id_for_filename)
                                image_filename_jpg = f"{video_id}_{segment_name_for_filename}.jpg"

                                cur_data['image'] = os.path.join("BDD_PC_5k", "bbox_global", args.split, image_filename_jpg)
                                train_samples.append(cur_data)
                                processing_stats['BDD'][segment_phrase_for_json] += 1
                            else:
                                print(f"Warning: No 'videos' key found in BDD Vehicle VQA for {video_id}. Skipping.")
            except json.JSONDecodeError:
                print(f"Error decoding JSON for {bdd_vehicle_path}")
            except Exception as e:
                print(f"An unexpected error occurred loading {bdd_vehicle_path}: {e}")

        # BDD Environment View
        bdd_environment_path = os.path.join(bdd_anno_path, video_id, 'environment', f'{video_id}.json')
        if os.path.exists(bdd_environment_path):
            try:
                bdd_environment_data_list = json.load(open(bdd_environment_path))
                if bdd_environment_data_list and 'environment' in bdd_environment_data_list[0] and bdd_environment_data_list[0]['environment']:
                    cur_data = {
                        'id': video_id,
                        'segment': 'environment', # Environment view segment is fixed as 'environment'
                        'view': 'environment',
                        'start_time': "N/A",
                        'end_time': "N/A",
                        'conversations': convert_vqa_to_conversations(bdd_environment_data_list[0]['environment'])
                    }
                    image_found = False
                    
                    # Search for the smallest phase image for BDD environment view
                    sorted_phases = sorted(phrase_number_map.items(), key=lambda item: int(item[0]))
                    
                    for num_str, phrase in sorted_phases:
                        potential_image_filename = f"{video_id}_{phrase}.jpg" # BDD filenames directly use the phrase
                        full_image_path_candidate = os.path.join(args.bdd_global_image_path, args.split, potential_image_filename)

                        if os.path.exists(full_image_path_candidate):
                            cur_data['image'] = os.path.join("BDD_PC_5k", "bbox_global", args.split, potential_image_filename)
                            train_samples.append(cur_data)
                            image_found = True
                            processing_stats['BDD']['environment'] += 1
                            break # Found the smallest, break and use this one
                    
                    if not image_found:
                        print(f"Warning: No suitable smallest phase image found for BDD environment view {video_id}. Skipping. path:{full_image_path_candidate}")
            except json.JSONDecodeError:
                print(f"Error decoding JSON for {bdd_environment_path}")
            except Exception as e:
                print(f"An unexpected error occurred loading {bdd_environment_path}: {e}")

save_dir = args.save_folder
os.makedirs(save_dir, exist_ok=True)

output_path = os.path.join(save_dir, f'wts_bdd_{args.split}.json')
with open(output_path, 'w') as f:
    json.dump(train_samples, f, indent=4)

processing_stats['total_samples'] = len(train_samples)

print(f"\n--- Processing Statistics ---")
print(f"Processed {processing_stats['total_samples']} samples and saved to {output_path}")
print(f"\nBreakdown by Dataset and View:")

print(f"\n* WTS Dataset:")
print(f"    - Overhead View: {processing_stats['WTS']['overhead']} samples")
print(f"    - Vehicle View: {processing_stats['WTS']['vehicle']} samples")
print(f"    - Environment: {processing_stats['WTS']['environment']} samples")
print(f"    - Normal Trimmed Overhead View: {processing_stats['WTS']['normal_trimmed_overhead']} samples")
print(f"    - Normal Trimmed Vehicle View: {processing_stats['WTS']['normal_trimmed_vehicle']} samples")
print(f"    - Normal Trimmed Environment: {processing_stats['WTS']['normal_trimmed_environment']} samples")

print(f"\n* BDD Dataset:")
for phrase in phrase_number_map.values():
    print(f"    - {phrase.capitalize()} View: {processing_stats['BDD'].get(phrase, 0)} samples")
print(f"    - Environment: {processing_stats['BDD']['environment']} samples") # For the specific BDD environment view

print(f"\n--- End of Statistics ---")