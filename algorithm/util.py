import json
import argparse

def to_output(filepath, outpath):
    with open(filepath, encoding='utf-8') as f:
        data = json.load(f)
    output = []
    for d in data:
        temp = {}
        temp['id'] = d['id']
        temp['labels'] = d['pred_labels']
        output.append(temp)
    with open(outpath, 'w+') as f:
        json.dump(output, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert model prediction into acceptable format.')
    parser.add_argument('in_path', metavar='i', type=str)
    parser.add_argument('out_path', metavar='o', type=str)
    args = parser.parse_args()

    to_output(args.in_path, args.out_path)