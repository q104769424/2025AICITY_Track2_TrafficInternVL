import json
import os
import re
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--root',        type=str, default='data', help='data root path')
parser.add_argument('--split',       type=str, default='train')
parser.add_argument('--save-folder', type=str, default='processed_anno', help='dirname for saving json file')
args = parser.parse_args()

video_path      = os.path.join(args.root, 'WTS/videos',            args.split)
annotation_path = os.path.join(args.root, 'WTS/annotations/caption', args.split)
bbox_path       = os.path.join(args.root, 'WTS/annotations/bbox_annotated')
gaze_path       = os.path.join(args.root, 'WTS/annotations/3D_gaze',  args.split)
head_path       = os.path.join(args.root, 'WTS/annotations/head',     args.split)

results = {}
total_entries = 0
saved_entries = 0

def process_annotations(item, view, camera, camera_base, phases, view_path):
    key = os.path.join(view_path, camera)
    entry = {
        'start_time':   None,
        'end_time':     None,
        'ped_bboxes':   {},
        'veh_bboxes':   {},
        'phase_number': {},
        'gaze':         {},
        'head':         {}
    }

    # 1) load pedestrian & vehicle bboxes + phase_number
    for typ, short in (('pedestrian','ped'), ('vehicle','veh')):
        pth = os.path.join(bbox_path, typ, args.split, item,
                           f'{view}_view', f'{camera_base}_bbox.json')
        if not os.path.exists(pth):
            continue
        with open(pth, 'r', encoding='utf-8') as f:
            for ann in json.load(f)['annotations']:
                img_id = ann['image_id']
                entry[f'{short}_bboxes'][img_id]    = ann['bbox']
                entry['phase_number'][img_id] = int(ann['phase_number'])

    # 2) treat all phases as valid
    valid_phases = {int(lbl) for ph in phases for lbl in ph.get('labels', [])}

    # 3) map each valid phase to its first frame (min image_id)
    phase_first = {}
    for img_id, ph_no in entry['phase_number'].items():
        if ph_no in valid_phases:
            if ph_no not in phase_first or img_id < phase_first[ph_no]:
                phase_first[ph_no] = img_id

    # 4) extract IP & camera number from camera_base
    tokens = camera_base.split('_')
    ips = re.findall(r'\d+\.\d+\.\d+\.\d+', camera_base)
    if not ips:
        return key, entry
    ip = ips[0]
    try:
        idx = tokens.index(ip)
        cam_no = tokens[idx + 1]
    except (ValueError, IndexError):
        return key, entry
    if not cam_no.isdigit():
        return key, entry

    # 5) build gaze & head file paths
    scene     = item.split('/')[-1]
    gaze_file = os.path.join(gaze_path, scene, f"{scene}_{ip}-{cam_no}_gaze.json")
    head_file = os.path.join(head_path, scene, f"{scene}_{ip}_{cam_no}_head.json")
    if not (os.path.exists(gaze_file) and os.path.exists(head_file)):
        return key, entry

    # 6) load gaze & head annotations
    with open(gaze_file, 'r', encoding='utf-8') as f:
        gaze_map = {a['image_id']: a['gaze']
                    for a in json.load(f).get('annotations', [])}
    with open(head_file, 'r', encoding='utf-8') as f:
        head_map = {a['image_id']: a['head']
                    for a in json.load(f).get('annotations', [])}
    
    # compute the reference start frame so that
    # gaze indices become 0-based and head indices 1-based
    base_frame = min(phase_first.values())
    
    # 7) for each phase's first frame, extract gaze & head using the correct offsets
    for _, img_id in sorted(phase_first.items()):
        # for img_id=900→gaze_idx=0, img_id=902→gaze_idx=2, etc.
        gaze_idx = img_id - base_frame       # relative index for gaze
        g = gaze_map.get(gaze_idx)
        h = head_map.get(img_id)             # absolute image_id for head

        if g is not None and h is not None:
            entry['gaze'][str(img_id)] = g
            entry['head'][str(img_id)] = h

        
    # 8) filter bboxes & phase_number to only those phases that have gaze+head
    valid_imgs = {int(k) for k in entry['gaze'].keys()}
    entry['ped_bboxes']   = {str(i): box for i, box in entry['ped_bboxes'].items()   if i in valid_imgs}
    entry['veh_bboxes']   = {str(i): box for i, box in entry['veh_bboxes'].items()   if i in valid_imgs}
    entry['phase_number'] = {str(i): entry['phase_number'][i] for i in valid_imgs if i in entry['phase_number']}

    return key, entry

# ─── process main folders ─────────────────────────────────────────────────
for item in tqdm(os.listdir(video_path)):
    if 'normal' in item:
        continue
    for view in ('overhead','vehicle'):
        view_path = os.path.join(video_path, item, f'{view}_view')
        cap_file  = os.path.join(annotation_path, item,
                                 f'{view}_view', f'{item}_caption.json')
        if view == 'overhead':
            assert os.path.exists(cap_file), cap_file
        try:
            phases = json.load(open(cap_file, 'r', encoding='utf-8'))['event_phase']
        except:
            continue

        start_time = min(float(p['start_time']) for p in phases)
        end_time   = max(float(p['end_time'])   for p in phases)

        for camera in os.listdir(view_path):
            total_entries += 1
            camera_base = camera.replace('.mp4','')
            key, entry = process_annotations(
                item, view, camera, camera_base, phases, view_path)
            entry['start_time'], entry['end_time'] = start_time, end_time
            if entry['gaze'] and entry['head']:
                results[key] = entry
                saved_entries += 1

# ─── save & summary ────────────────────────────────────────────────────────
os.makedirs(args.save_folder, exist_ok=True)
out_file = os.path.join(
    args.save_folder,
    f'wts_{args.split}_all_video_with_bbox_gaze_anno_first_frame.json'
)
with open(out_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=4)

print(f"Originally processed {total_entries} entries, {saved_entries} remaining after filtering")
