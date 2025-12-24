import os
import re
import sys
import glob
import json
import time
import math
import torch
import random
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoModel, AutoTokenizer, AutoConfig
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )

        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()

    if 'InternVL3' in model_name:
        if path is None:
            raise ValueError("For InternVL3 models, 'path' must be provided to load config.")
        config = AutoConfig.from_pretrained(path, trust_remote_code=True)
        num_layers = config.llm_config.num_hidden_layers
    else:
        num_layers_dict = {
            'InternVL2_5-1B': 24,
            'InternVL2_5-2B': 24,
            'InternVL2_5-4B': 36,
            'InternVL2_5-8B': 32,
            'InternVL2_5-26B': 48,
            'InternVL2_5-38B': 64,
            'InternVL2_5-78B': 80
        }
        num_layers = num_layers_dict[model_name]

    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)

    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for _ in range(num_layer):
            if layer_cnt >= num_layers:
                break
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1

    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.model.rotary_emb'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

PHASE_NUMBER_MAP = {
    '0': 'prerecognition',
    '1': 'recognition',
    '2': 'judgement',
    '3': 'action',
    '4': 'avoidance'
}
LABEL_TO_INDEX = {v: k for k, v in PHASE_NUMBER_MAP.items()}

def extract_video_id(filename: str) -> str:
    name = os.path.splitext(filename)[0]
    name = re.sub(r'(_\d{1,3}(\.\d{1,3}){3}_\d+|_Camera\d+(_\d+)?|_vehicle_view)$', '', name)
    return name

def make_prompt(segment, question_type="pedestrian"):
    if question_type == "vehicle":
        return (
            # f"These two images from the '{segment}' stage show the interaction between the person in the green box "
            # "and the car in the blue box, which should be the focus when answering the following question.\n"
            # "The global image shows the full scene, while the local image highlights the relevant area.\n"
            # "Global image: <image>\n"
            # "Local image: <image>\n"
            # "Describe the vehicle in the blue box or the vehicle closest to the pedestrian based on the relative position to the pedestrian, "
            # "driving status, weather conditions and road environment. And describe the age, height, clothing of the pedestrian."
            "These two images show the interaction between the person in the green box "
            "and the car in the blue box, which should be the focus when answering the following question.\n"
            "The global image shows the full scene, while the local image highlights the relevant area.\n"
            "Global image: <image>\n"
            "Local image: <image>\n"
            "Based on the images, describe the vehicle highlighted in blue box or the one closest to the pedestrian in the green box. Include its position relative to the pedestrian, driving behavior, weather conditions, and road environment. "
            "Also briefly describe the pedestrian in the green box, including their apparent age, height, and clothing."
        )
    else:
        return (
            # f"These two images from the '{segment}' stage show the interaction between the person in the green box "
            # "and the car in the blue box, which should be the focus when answering the following question.\n"
            # "The global image shows the full scene, while the local image highlights the relevant area.\n"
            # "Global image: <image>\n"
            # "Local image: <image>\n"
            # "Describe the pedestrian in the green box or the pedestrian closest to the vehicle based on age, height, clothing, "
            # "line of sight, relative position to the vehicle, movement status, weather conditions and road environment."
            "These two images show the interaction between the person in the green box "
            "and the car in the blue box, which should be the focus when answering the following question.\n"
            "The global image shows the full scene, while the local image highlights the relevant area.\n"
            "Global image: <image>\n"
            "Local image: <image>\n"
            "Based on the images, describe the pedestrian highlighted in green box or the one closest to the car in the blue box. Include their apparent age, height, and clothing, as well as their line of sight and position relative to the vehicle. Describe their movement behavior, and provide observations about the weather conditions and the road environment. "
            "Also indicate whether the pedestrian appears to be aware of the vehicle."
        )

def inference(model, tokenizer, global_image_data_path, local_image_data_path, save_path):
    start_time = time.time()
    missing_images = 0
    results = {}
    valid_answers = {"pedestrian": [], "vehicle": []}

    phrase_number_map = {
        '0': 'prerecognition',
        '1': 'recognition',
        '2': 'judgement',
        '3': 'action',
        '4': 'avoidance'
    }

    all_scenarios = sorted(os.listdir(global_image_data_path))

    for scenario in tqdm(all_scenarios, desc="Running Inference"):
        scenario_res = []

        for i in range(5): 
            clip_name = f"{i}.jpg"
            global_image_path = os.path.join(global_image_data_path, scenario, clip_name)
            local_image_path = os.path.join(local_image_data_path, scenario, clip_name)
            segment = phrase_number_map.get(str(i), "avoidance")

            if not os.path.exists(global_image_path) or not os.path.exists(local_image_path):
                print(f"[Missing] {global_image_path} or {local_image_path}")
                response_ped = random.choice(valid_answers["pedestrian"]) if valid_answers["pedestrian"] else "unknown"
                response_veh = random.choice(valid_answers["vehicle"]) if valid_answers["vehicle"] else "unknown"
                scenario_res.append({
                    "labels": [str(i)],
                    "caption_pedestrian": response_ped,
                    "caption_vehicle": response_veh
                })
                missing_images += 1
                continue

            try:
                gv = load_image(global_image_path).to(torch.bfloat16).cuda()
                lv = load_image(local_image_path).to(torch.bfloat16).cuda()
                images = torch.cat([gv, lv], dim=0)
                npatches = [gv.size(0), lv.size(0)]

                # Ask about pedestrian
                prompt_ped = make_prompt(segment, "pedestrian")
                print(f"\n[Scenario / Segment / Target]\n{scenario} / {segment} / pedestrian")
                print(f"[Prompt]\n{prompt_ped}")
                t0 = time.time()
                response_ped = model.chat(
                    tokenizer, images, prompt_ped,
                    dict(max_new_tokens=8192, do_sample=False),
                    num_patches_list=npatches
                ).strip()
                print(f"[Response]\n{response_ped} (Inference Time: {time.time() - t0:.2f}s)")
                valid_answers["pedestrian"].append(response_ped)

                # Ask about vehicle
                prompt_veh = make_prompt(segment, "vehicle")
                print(f"\n[Scenario / Segment / Target]\n{scenario} / {segment} / vehicle")
                print(f"[Prompt]\n{prompt_veh}")
                t0 = time.time()
                response_veh = model.chat(
                    tokenizer, images, prompt_veh,
                    dict(max_new_tokens=8192, do_sample=False),
                    num_patches_list=npatches
                ).strip()
                print(f"[Response]\n{response_veh} (Inference Time: {time.time() - t0:.2f}s)")
                valid_answers["vehicle"].append(response_veh)

                scenario_res.append({
                    "labels": [str(i)],
                    "caption_pedestrian": response_ped,
                    "caption_vehicle": response_veh
                })

            except Exception as e:
                print(f"[Error] {scenario}/{clip_name}: {e}")
                response_ped = random.choice(valid_answers["pedestrian"]) if valid_answers["pedestrian"] else "unknown"
                response_veh = random.choice(valid_answers["vehicle"]) if valid_answers["vehicle"] else "unknown"
                scenario_res.append({
                    "labels": [str(i)],
                    "caption_pedestrian": response_ped,
                    "caption_vehicle": response_veh
                })

        results[scenario] = scenario_res

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nDONE in {time.time() - start_time:.1f}s")
    print(f"Total processed: {len(results)}")
    print(f"Missing count  : {missing_images}")
    print(f"Saved to       : {save_path}")

def inference_with_fewshot(model, tokenizer, global_image_data_path, local_image_data_path,
                           train_jsonl_path, save_path, target_video_id="20230922_37_CN9_T1"):
    start_time = time.time()
    missing_images = 0
    results = {}
    valid_answers = {"pedestrian": [], "vehicle": []}

    phrase_number_map = {
        '0': 'prerecognition',
        '1': 'recognition',
        '2': 'judgement',
        '3': 'action',
        '4': 'avoidance'
    }

    train_data = load_jsonl(train_jsonl_path)
    fewshot_image_path = None
    fewshot_text = ""

    for s in train_data:
        if s.get("video_id") != target_video_id:
            continue
        image_paths = s.get("image", [])
        if not image_paths or not os.path.basename(image_paths[0]).startswith("4_"):
            continue
        human_msg = next((c["value"] for c in s["conversations"] if c["from"] == "human"), None)
        gpt_msg = next((c["value"] for c in s["conversations"] if c["from"] == "gpt"), None)
        if not human_msg or not gpt_msg:
            continue
        fewshot_image_path = image_paths[0]
        fewshot_text = f"Q: {human_msg.strip()}\nA: {gpt_msg.strip()}"
        break

    fewshot_notice = (
        "Note: In the previous example, a red fan-shaped area showed the pedestrian's gaze direction. "
        "In the following questions, this visual gaze cue is not available. "
        "Please infer where the pedestrian is looking based on the scene and object positions. "
        "Do not guess randomly. Use context from the image to make a careful judgment."
    )

    if fewshot_image_path and os.path.exists(fewshot_image_path):
        fewshot_image_tensor = load_image(fewshot_image_path).to(torch.bfloat16).cuda()
        fewshot_patch_count = fewshot_image_tensor.size(0)
    else:
        print(f"[Warning] Few-shot image missing: {fewshot_image_path}")
        fewshot_image_tensor = None
        fewshot_patch_count = 0

    all_scenarios = sorted(os.listdir(global_image_data_path))

    for scenario in tqdm(all_scenarios, desc="Running Inference"):
        scenario_res = []

        for i in range(5):
            clip_name = f"{i}.jpg"
            global_image_path = os.path.join(global_image_data_path, scenario, clip_name)
            local_image_path = os.path.join(local_image_data_path, scenario, clip_name)
            segment = phrase_number_map.get(str(i), "avoidance")

            if not os.path.exists(global_image_path) or not os.path.exists(local_image_path):
                print(f"[Missing] {global_image_path} or {local_image_path}")

                response_ped = random.choice(valid_answers["pedestrian"]) if valid_answers["pedestrian"] else "unknown"
                response_veh = random.choice(valid_answers["vehicle"]) if valid_answers["vehicle"] else "unknown"
                scenario_res.append({
                    "labels": [str(i)],
                    "caption_pedestrian": response_ped,
                    "caption_vehicle": response_veh
                })
                missing_images += 1
                continue

            try:
                gv = load_image(global_image_path).to(torch.bfloat16).cuda()
                lv = load_image(local_image_path).to(torch.bfloat16).cuda()

                if fewshot_image_tensor is not None:
                    pv = torch.cat([fewshot_image_tensor, gv, lv], dim=0)
                    npatches = [fewshot_patch_count, gv.size(0), lv.size(0)]
                else:
                    pv = torch.cat([gv, lv], dim=0)
                    npatches = [gv.size(0), lv.size(0)]

                # --- Pedestrian ---
                prompt_ped = (
                    fewshot_text + "\n" +
                    fewshot_notice + "\n" +
                    make_prompt(segment, "pedestrian")
                )
                print(f"\n[Scenario / Segment / Target]\n{scenario} / {segment} / pedestrian")
                print(f"[Prompt]\n{prompt_ped}")
                t0 = time.time()
                response_ped = model.chat(
                    tokenizer, pv, prompt_ped,
                    dict(max_new_tokens=8192, do_sample=False),
                    num_patches_list=npatches
                ).strip()
                print(f"[Response]\n{response_ped} (Inference Time: {time.time() - t0:.2f}s)")
                valid_answers["pedestrian"].append(response_ped)

                # --- Vehicle ---
                prompt_veh = (
                    fewshot_text + "\n" +
                    fewshot_notice + "\n" +
                    make_prompt(segment, "vehicle")
                )
                print(f"\n[Scenario / Segment / Target]\n{scenario} / {segment} / vehicle")
                print(f"[Prompt]\n{prompt_veh}")
                t0 = time.time()
                response_veh = model.chat(
                    tokenizer, pv, prompt_veh,
                    dict(max_new_tokens=8192, do_sample=False),
                    num_patches_list=npatches
                ).strip()
                print(f"[Response]\n{response_veh} (Inference Time: {time.time() - t0:.2f}s)")
                valid_answers["vehicle"].append(response_veh)

                scenario_res.append({
                    "labels": [str(i)],
                    "caption_pedestrian": response_ped,
                    "caption_vehicle": response_veh
                })

            except Exception as e:
                print(f"[Error] {scenario}/{clip_name}: {e}")
                response_ped = random.choice(valid_answers["pedestrian"]) if valid_answers["pedestrian"] else "unknown"
                response_veh = random.choice(valid_answers["vehicle"]) if valid_answers["vehicle"] else "unknown"
                scenario_res.append({
                    "labels": [str(i)],
                    "caption_pedestrian": response_ped,
                    "caption_vehicle": response_veh
                })

        results[scenario] = scenario_res

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nDONE in {time.time() - start_time:.1f}s")
    print(f"Total processed: {len(results)}")
    print(f"Missing count  : {missing_images}")
    print(f"Saved to       : {save_path}")
    
if __name__ == "__main__":
    GLOBAL_IMAGE_DATA_PATH = "../data_preprocess/data/generate_test_frames/bbox_global"
    LOCAL_IMAGE_DATA_PATH = "../data_preprocess/data/generate_test_frames/bbox_local"
    SAVE_PATH = "./eval-metrics-AIC-Track2/inference_qa_result.json"
    # path = f'../model/InternVL2_5-38B-MPO'
    path = '../model/TrafficInternVL-38B-MPO'
    device_map = split_model("InternVL2_5-38B")

    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map=device_map
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(
        path,
        trust_remote_code=True,
        use_fast=False
    )
    
    inference(
        model,
        tokenizer,
        GLOBAL_IMAGE_DATA_PATH,
        LOCAL_IMAGE_DATA_PATH,
        SAVE_PATH
    )

    # Fewshot
    # TRAIN_JSON = "../data_preprocess/processed_anno/gaze_qa_format/wts_multimodal_gaze_qa_train.jsonl"
    # inference_with_fewshot(
    #     model,
    #     tokenizer,
    #     GLOBAL_IMAGE_DATA_PATH,
    #     LOCAL_IMAGE_DATA_PATH,
    #     TRAIN_JSON,
    #     SAVE_PATH
    # )