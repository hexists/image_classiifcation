#!/usr/bin/env python3


def read_tsv(path, ftype='test'):
    tsv = {}
    with open(path) as fp:
        for i, buf in enumerate(fp):
            line = buf.rstrip().split('\t')
            fname = line[0]
            if ftype == 'test':
                tsv[fname] = {'idx': i, 'val': ''}
            else:
                tsv[fname] = line[1:]

    return tsv


test_tsv = read_tsv('./nipa_dataset/test/test.tsv', 'test')
result_tsv = read_tsv('./result.resnet18_95.tsv', 'result')

for k, v in result_tsv.items():
    if k in test_tsv:
        test_tsv[k]['val'] = v
    else:
        print('ERROR: {}'.format(k))

from collections import OrderedDict
test_tsv = OrderedDict(sorted(test_tsv.items(), key=lambda t: t[1]['idx']))

for k, v in test_tsv.items():
    print('{}\t{}'.format(k, '\t'.join(v['val'])))
