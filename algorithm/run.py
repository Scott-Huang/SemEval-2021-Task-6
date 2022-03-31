import torch
from algorithm.algorithm import Baseline_v3, Entity_Algorithm
from algorithm.data import get_data, get_aug_data
import argparse
import logging
logging.basicConfig(level = logging.INFO)
from collections import defaultdict

MODELS = ['baseline', 'entity']

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Choose which model to run.')
    parser.add_argument('model', metavar='model', type=str, nargs='?', default='baseline')
    parser.add_argument('out_path', metavar='o', type=str, nargs='?', default=None,
                        help='The filepath of the prediction result on test dataset.')
    parser.add_argument('--augment', action='store_true', default=False,
                        help='Use augmented data.')
    parser.add_argument('--epoch', '-e', nargs='?', default=30, type=int,
                        help='Number of epoches to train')
    args = parser.parse_args()
    model_name = args.model
    if model_name not in MODELS:
        logging.info('invalid model name: ' + model_name)
        exit(1)

    tr_data, dev_data, test_data = get_data()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if model_name == 'entity':
        if args.augment:
            logging.info('Augmentation not implemented for entities yet')
            exit(1)
        algorithm = Entity_Algorithm(None, "sentence-transformers/all-mpnet-base-v2", device)
    else:
        algorithm = Baseline_v3(None, "sentence-transformers/all-mpnet-base-v2", device)
    if args.augment:
        aug_data = get_aug_data('data/aug_data3.json')
        tr_data = algorithm.preprocess_aug(tr_data, aug_data)
    else:
        tr_data = algorithm.preprocess(tr_data)
    dev_data = algorithm.preprocess(dev_data)

    algorithm.train(tr_data, dev_data, lr=8e-5, batch_size=32, epoch=args.epoch)

    test_data = algorithm.preprocess(test_data)
    d2,tl2 = algorithm.predict(test_data)
    print(f'test loss:{tl2}')

    _, _, test_data = get_data()
    for d in d2:
        d['labels'] = test_data[d['id']][0]

    pred_stats = defaultdict(lambda:[0,0,0]) # miss, false-positive, correct
    for i in d2:
        for l in i['labels']:
            if l in i['pred_labels']:
                pred_stats[l][2] += 1
            else:
                pred_stats[l][1] += 1
        for l in i['pred_labels']:
            if l not in i['labels']:
                pred_stats[l][0] += 1
    print('Stats for each categories: [miss, false-positive, correct]')
    print(dict(pred_stats))

    if args.out_path:
        import json
        with open(args.out_path, 'w+') as f:
            json.dump(d2, f, indent=4)
