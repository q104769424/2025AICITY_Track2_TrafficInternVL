#!/bin/bash
# -*- coding: utf-8 -*-
num_worker=32
root="./data"
save_folder="./processed_anno" # Store json files
base_image_path="/mlsteam/data/A22/hsiufu/2025AICITY_TrafficInternVL/data_preprocess/data/" # Absolute path
reference_view_path="./data/test_part/view_used_as_main_reference_for_multiview_scenario.csv"
splits=("train" "val")
scale=1.5

for split in "${splits[@]}"; do
    python extract_wts_frame_bbox_anno.py --root $root --save-folder $save_folder/frame_bbox_anno --split $split
    python extract_bdd_frame_bbox_anno.py --root $root --save-folder $save_folder/frame_bbox_anno --split $split
done

for file in "$save_folder/frame_bbox_anno"/*train*; do
    python draw_bbox_on_frame.py --worker $num_worker --anno $file --scale $scale
done

for file in "$save_folder/frame_bbox_anno"/*val*; do
    python draw_bbox_on_frame.py --worker $num_worker --anno $file --scale $scale
done

# qa
for split in "${splits[@]}"; do
    python transform_qa_format.py \
        --root $root \
        --save-folder $save_folder/qa_format \
        --split $split \
        --wts-global-image-path $root/WTS/bbox_global \
        --bdd-global-image-path $root/BDD_PC_5k/bbox_global
done
python multimodal_merge_qa.py --base-image-path $base_image_path
python representative_sample_selection_qa.py --reference_view_path $reference_view_path
python json2jsonl.py --input  $save_folder/qa_format/wts_bdd_multimodal_qa_train_filtered.json --output  $save_folder/qa_format/wts_bdd_multimodal_qa_train_filtered.jsonl
python json2jsonl.py --input  $save_folder/qa_format/wts_bdd_multimodal_qa_val_filtered.json --output  $save_folder/qa_format/wts_bdd_multimodal_qa_val_filtered.jsonl

#vqa
for split in "${splits[@]}"; do
    python transform_vqa_format.py \
        --root $root \
        --save-folder $save_folder/vqa_format \
        --split $split \
        --wts-global-image-path $root/WTS/bbox_global \
        --bdd-global-image-path $root/BDD_PC_5k/bbox_global
done
python multimodal_merge_vqa.py --base-image-path $base_image_path
python representative_sample_selection_vqa.py --reference_view_path $reference_view_path
python json2jsonl.py --input  $save_folder/vqa_format/wts_bdd_multimodal_vqa_train_filtered.json --output  $save_folder/vqa_format/wts_bdd_multimodal_vqa_train_filtered.jsonl
python json2jsonl.py --input  $save_folder/vqa_format/wts_bdd_multimodal_vqa_val_filtered.json --output  $save_folder/vqa_format/wts_bdd_multimodal_vqa_val_filtered.jsonl

-------------------------------------------------------------------------------------------------------------
#For 3D gaze
for split in "${splits[@]}"; do
    python extract_wts_frame_bbox_gaze_anno.py --root $root --save-folder $save_folder/frame_bbox_gaze_anno --split $split
done

for file in "$save_folder/frame_bbox_gaze_anno"/*train*; do
    python draw_bbox_gaze_on_frame.py --worker $num_worker --anno $file --scale $scale
done

for file in "$save_folder/frame_bbox_gaze_anno"/*val*; do
    python draw_bbox_gaze_on_frame.py --worker $num_worker --anno $file --scale $scale
done

# qa
python transform_gaze_qa_format.py --root $root --save-folder $save_folder/gaze_qa_format --wts-global-image-path $root/WTS/bbox_gaze_global
python multimodal_merge_gaze_qa.py --base-image-path $base_image_path
python json2jsonl.py --input  $save_folder/gaze_qa_format/wts_multimodal_gaze_qa_train.json --output  $save_folder/gaze_qa_format/wts_multimodal_gaze_qa_train.jsonl

# vqa
python transform_gaze_vqa_format.py --root $root --save-folder $save_folder/gaze_vqa_format --wts-global-image-path $root/WTS/bbox_gaze_global
python multimodal_merge_gaze_vqa.py --base-image-path $base_image_path
python json2jsonl.py --input  $save_folder/gaze_vqa_format/wts_multimodal_gaze_vqa_train.json --output  $save_folder/gaze_vqa_format/wts_multimodal_gaze_vqa_train.jsonl

echo " Trainsets prepared."