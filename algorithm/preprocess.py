def sent_filter(sent):
    #sent = sent.lower()
    sent = sent.replace('\n\n', '\n')
    sent = sent.replace('\n', ' ')
    return sent.strip()

def augment_data(original_data, n1, n2=0):
    assert n1 > 0
    if n2 <= 0:
        n2 = n1
    
    import nlpaug.augmenter.word as naw
    import nlpaug.flow as naf

    aug_data = {}
    rared = {}
    d = {}
    for k,(s,rare) in original_data.items():
        if rare:
            rared[k] = sent_filter(s)
        else:
            d[k] = sent_filter(s)

    print('Start to build augmenter')
    aug = naf.Sometimes([
        naw.BackTranslationAug(
            from_model_name='facebook/wmt19-en-de', 
            to_model_name='facebook/wmt19-de-en',
            device='cuda'
        ),
        naw.RandomWordAug(action="swap", aug_p=0.2),
        naw.SynonymAug(aug_src='ppdb', model_path='ppdb-2.0-l-all', aug_p=0.2),
        naw.ContextualWordEmbsAug(model_path='roberta-large', action="substitute", device='cuda')
    ])
    
    print('Start to augment rare sents')
    rare_sent = list(rared.values())
    rare_aug_sent = aug.augment(rare_sent, n2)
    if n2 > 1:
        rare_aug_sent = list(zip(*rare_aug_sent))
    for (k,s),aug_sent in zip(rared.items(),rare_aug_sent):
        if n2 > 1:
            l = list(aug_sent)
            l.append(s)
        else:
            l = [aug_sent, s]
        aug_data[k] = l

    print('Start to augment the remaining')
    sent = list(d.values())
    aug_sent = aug.augment(sent, n1)
    if n1 > 1:
        aug_sent = list(zip(*aug_sent))
    for (k,s),aug_sent in zip(d.items(),aug_sent):
        if n1 > 1:
            l = list(aug_sent)
            l.append(s)
        else:
            l = [aug_sent, s]
        aug_data[k] = l
    print('Finished')
    
    return aug_data

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Augment the training data')
    parser.add_argument('n', type=int, nargs='?', default=1)
    parser.add_argument('n2', type=int, nargs='?', default=0)
    parser.add_argument('out_path', metavar='o', type=str,
                        help='The filepath of the augmented result.')
    args = parser.parse_args()

    import json
    #from algorithm.data import get_data
    with open('data/original_sentences.json') as f:
        data = json.load(f)
    aug_data = augment_data(data, args.n, args.n2)
    with open(args.out_path, 'w+') as f:
        json.dump(aug_data, f, indent=4, ensure_ascii=False)
