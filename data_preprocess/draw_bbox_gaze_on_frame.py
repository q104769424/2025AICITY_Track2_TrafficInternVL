from decord import VideoReader
import cv2
import numpy as np
import json
import os
import re
from tqdm import tqdm
from multiprocessing import Pool
import copy
import argparse
import math

phase_number_map = {
    '0': 'prerecognition',
    '1': 'recognition',
    '2': 'judgement',
    '3': 'action',
    '4': 'avoidance'
}

def extract_frames(video_path, frame_indices, original_frame_indices):
    vr = VideoReader(video_path)
    if frame_indices[-1] >= len(vr):
        frame_indices[-1] = len(vr) - 1
    return {
        ori: vr[idx].asnumpy()
        for idx, ori in zip(frame_indices, original_frame_indices)
    }

def enlarge_bbox(bbox, scale):
    x,y,w,h = bbox
    cx, cy = x + w/2, y + h/2
    nw, nh = w*scale, h*scale
    return (cx - nw/2, cy - nh/2, nw, nh)

def calculate_combined_bbox(b1, b2):
    x1,y1,w1,h1 = b1; x2,y2,w2,h2 = b2
    xmin, ymin = min(x1,x2), min(y1,y2)
    xmax, ymax = max(x1+w1, x2+w2), max(y1+h1, y2+h2)
    return (xmin, ymin, xmax-xmin, ymax-ymin)

def constrain_bbox_within_frame(bbox, shape):
    x1,y1,x2,y2 = bbox; H,W = shape[0], shape[1]
    return (max(0,int(x1)), max(0,int(y1)),
            min(W,int(x2)), min(H,int(y2)))

def extract_info_from_video_path(video_path):
    bn = os.path.basename(video_path)
    m = re.match(r'(\d{8})_.*?_((?:\d{1,3}\.){3}\d{1,3})_(\d+)\.mp4', bn)
    if m:
        return m.group(1), f"{m.group(2)}-{m.group(3)}"
    m = re.match(r'(\d{8})_.*?_(Camera\d+_\d+)\.mp4', bn)
    if m:
        return m.group(1), m.group(2)
    raise ValueError(f"Cannot parse camera info from {bn}")

def draw_and_save_bboxes_scale_version(
    video_path, frames,
    ped_bboxes, veh_bboxes, phase_numbers, phase_number_map,
    scale, gaze_map, head_map,
    camera_params_base_path
):
    # Skip list: Manually identified samples with abnormal BBox or gaze
    skip_list = {
        "20230922_27_CN2_T1",
        "20230922_41_SY12_T1",
        "20230922_45_CN4_T1",
        "20230929_1_CN6_T1",
        "20230929_1_CN6_T3",
        "20230929_20_CY20_T1",
        "20230929_31_CY5_T1",
        "20231006_13_SY13_T1",
        "20231006_16_CN27_T1",  
        "20231006_17_CN26_T1",
        "20231006_17_CN26_T2"
    }
    parts = os.path.normpath(video_path).split(os.sep)
    skip_id = next((p for p in parts if p in skip_list), None)
    if skip_id:
        print(f"[SKIP] Skipping ID: {skip_id}")
        return 0  # Return 0 images saved

    # Load intrinsics
    _, cid = extract_info_from_video_path(video_path)
    int_path = os.path.join(camera_params_base_path, "camera_intrinsics", f"{cid}.json")
    with open(int_path, 'r', encoding='utf-8') as f:
        intr = json.load(f)[cid]['intrinsics']
    fx, fy, cx, cy = intr
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    saved_count = 0

    for fid, frame_np in frames.items():
        frame = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        combined = None

        # Draw pedestrian bbox
        if str(fid) in ped_bboxes and ped_bboxes[str(fid)] is not None:
            bx = enlarge_bbox(ped_bboxes[str(fid)], scale)
            if bx:
                x, y, w_, h_ = bx
                x2, y2 = x + w_, y + h_
                x1, y1, x2, y2 = constrain_bbox_within_frame((x, y, x2, y2), frame.shape)
                combined = (x1, y1, x2 - x1, y2 - y1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # Draw vehicle bbox
        if str(fid) in veh_bboxes and veh_bboxes[str(fid)] is not None:
            bx = enlarge_bbox(veh_bboxes[str(fid)], scale)
            if bx:
                x, y, w_, h_ = bx
                x2, y2 = x + w_, y + h_
                x1, y1, x2, y2 = constrain_bbox_within_frame((x, y, x2, y2), frame.shape)
                if combined:
                    combined = calculate_combined_bbox(combined, (x1, y1, x2 - x1, y2 - y1))
                else:
                    combined = (x1, y1, x2 - x1, y2 - y1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)

        # Draw gaze fan
        if combined and fid in gaze_map and fid in head_map:
            g = np.array(gaze_map[fid], dtype=float)
            n = np.linalg.norm(g)
            if n > 0:
                g /= n
                hx, hy = head_map[fid]
                px, py = int(hx), int(hy)
                cv2.circle(frame, (px, py), 5, (0, 0, 255), -1)
        
                z = 2.5
                p0 = np.array([(px - cx) * z / fx, (py - cy) * z / fy, z])
                angle_rad = np.radians(30)
                p1_gaze = p0 + g * 0.5
                ph_gaze = K.dot(p1_gaze)
        
                if ph_gaze[2] > 0:
                    ix_gaze, iy_gaze = int(ph_gaze[0] / ph_gaze[2]), int(ph_gaze[1] / ph_gaze[2])
                    dx, dy = ix_gaze - px, iy_gaze - py
                    base_angle = math.atan2(dy, dx)
        
                    radius = 150
                    start_angle = math.degrees(base_angle - angle_rad)
                    end_angle = math.degrees(base_angle + angle_rad)
        
                    overlay = np.zeros_like(frame, dtype=np.uint8)
                    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        
                    # Draw filled fan (wedge) using ellipse
                    cv2.ellipse(
                        overlay,
                        center=(px, py),
                        axes=(radius, radius),
                        angle=0,
                        startAngle=start_angle,
                        endAngle=end_angle,
                        color=(0, 0, 255),
                        thickness=-1
                    )
        
                    cv2.ellipse(
                        mask,
                        center=(px, py),
                        axes=(radius, radius),
                        angle=0,
                        startAngle=start_angle,
                        endAngle=end_angle,
                        color=255,
                        thickness=-1
                    )
        
                    α = 0.4
                    inv_m = cv2.bitwise_not(mask)
                    bg = cv2.bitwise_and(frame, frame, mask=inv_m)
                    blend = cv2.addWeighted(frame, 1 - α, overlay, α, 0)
                    fg = cv2.bitwise_and(blend, blend, mask=mask)
                    frame = cv2.add(bg, fg)

        # Save image
        ph = phase_numbers.get(str(fid), "")
        if ph:
            base = video_path.replace('.mp4','/').replace('/videos','/bbox_gaze_global')
            os.makedirs(base, exist_ok=True)
            out_fn = f"{base}{ph}_{phase_number_map.get(str(ph),'unknown')}.jpg"
            cv2.imwrite(out_fn, frame)
            saved_count += 1

    return saved_count

def process_video(args):
    video_path, data, phase_number_map, scale, camera_params_base_path = args
    idxs = list(map(int, data["phase_number"].keys()))
    if not idxs:
        print(f"[SKIP] {video_path} has no valid frames. Skipped.")
        return 0, 0

    proc = copy.deepcopy(idxs)
    if 'fps' in data and float(data['fps']) > 40:
        proc = [i // 2 for i in proc]

    frames = extract_frames(video_path, proc, idxs)
    gaze_map = {int(k): v for k, v in data["gaze"].items()}
    head_map = {int(k): v for k, v in data["head"].items()}

    saved = draw_and_save_bboxes_scale_version(
        video_path, frames,
        data["ped_bboxes"], data["veh_bboxes"],
        data["phase_number"], phase_number_map,
        scale, gaze_map, head_map,
        camera_params_base_path
    )

    if saved > 0:
        print(f"[DONE] {video_path} - {saved} images saved.")
        return 1, saved
    else:
        print(f"[SKIP] {video_path} - no image saved.")
        return 0, 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--anno',          type=str, required=True,
                        help='JSON with bbox+gaze+head annotations')
    parser.add_argument('--worker',        type=int, default=1,
                        help='number of parallel processes')
    parser.add_argument('--scale',         type=float, default=1.5,
                        help='scaling factor for bboxes')
    parser.add_argument('--camera-params', type=str, default='./data/camera_parameters',
                        help='base path for camera intrinsics/extrinsics')
    args = parser.parse_args()

    anno = json.load(open(args.anno, 'r'))
    jobs = [
        (vp, data, phase_number_map, args.scale, args.camera_params)
        for vp, data in anno.items()
    ]

    total_videos = 0
    total_images = 0

    with Pool(processes=args.worker) as pool:
        for processed, saved in tqdm(pool.imap(process_video, jobs), total=len(jobs)):
            total_videos += processed
            total_images += saved

    print(f"Total processed videos: {total_videos}")
    print(f"Total saved images: {total_images}")