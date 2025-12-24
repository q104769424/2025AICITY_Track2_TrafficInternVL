import json
import argparse

def convert_to_jsonl(input_file, output_file):
    with open(input_file, 'r') as f:
        data = json.load(f)

    with open(output_file, 'w') as out_f:
        for item in data:
            output_dict = {
                "video_id": item.get("video_id"),
                "id": item.get("id"),
                "image": item.get("image", []),
                "height_list": item.get("height_list", []),
                "width_list": item.get("width_list", []),
                "conversations": item.get("conversations", [])
            }
            out_f.write(json.dumps(output_dict, ensure_ascii=False) + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert JSON to JSONL format.')
    parser.add_argument('--input', '-i', required=True, help='Input JSON file path')
    parser.add_argument('--output', '-o', required=True, help='Output JSONL file path')

    args = parser.parse_args()
    convert_to_jsonl(args.input, args.output)
