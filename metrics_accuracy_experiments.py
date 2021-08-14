from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

import json
from json import encoder

encoder.FLOAT_REPR = lambda o: format(o, '.3f')
import random
from scipy.io import loadmat
import pandas as pd


def load_triplets(filepath='.experiment_data/consensus_abstract.mat'):
    consensus = loadmat(filepath)
    triplets = {}
    sent_to_index = {}
    index = 0
    for item in consensus['triplets'].tolist()[0]:
        A = str(item[0].tolist()[0][0][0])
        B = str(item[1].tolist()[0][0][0])
        C = str(item[2].tolist()[0][0][0])
        winner = item[3].tolist()[0][0]

        triplets[B + C] = triplets.get(B + C, [])
        triplets[B + C].append((A, B, C, winner))

        if not sent_to_index.get(B + C, False):
            sent_to_index[B + C] = index
            index += 1

    return triplets, sent_to_index


def compute_metrics(annFile, resFile):
    # create coco object and cocoRes object
    coco = COCO(annFile)
    cocoRes = coco.loadRes(resFile)

    # create cocoEval object by taking coco and cocoRes
    cocoEval = COCOEvalCap(coco, cocoRes)

    # evaluate on a subset of images by setting
    # cocoEval.params['image_id'] = cocoRes.getImgIds()
    # please remove this line when evaluating the full validation set
    cocoEval.params['image_id'] = cocoRes.getImgIds()

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    cocoEval.evaluate()

    results = {}
    for item in cocoEval.evalImgs:
        image_id = item['image_id']
        Bleu_4 = item['Bleu_4']
        METEOR = item['METEOR']
        ROUGE_L = item['ROUGE_L']
        CIDEr = item['CIDEr']
        CIDEr_R = item['CIDEr-R']
        SPICE = item['SPICE']['All']['f']

        results[image_id] = {'Bleu_4': Bleu_4,
                             'METEOR': METEOR,
                             'ROUGE_L': ROUGE_L,
                             'CIDEr': CIDEr,
                             'CIDEr-R': CIDEr_R,
                             'SPICE': SPICE}

    return results


def compute_accuracy(results_B, results_C, winners):
    counters = {'Bleu_4': 0,
                'METEOR': 0,
                'ROUGE_L': 0,
                'CIDEr': 0,
                'CIDEr-R': 0,
                'SPICE': 0}
    win_count = 0
    for img, winner in winners.items():
        if winner != 0:
            win_count += 1

        for metric in ['Bleu_4', 'METEOR', 'ROUGE_L', 'CIDEr', 'CIDEr-R', 'SPICE']:
            if (results_B[img][metric] > results_C[img][metric] and winner > 0) or \
                    (results_B[img][metric] < results_C[img][metric] and winner < 0):
                counters[metric] += 1

    for metric in ['Bleu_4', 'METEOR', 'ROUGE_L', 'CIDEr', 'CIDEr-R', 'SPICE']:
        counters[metric] = counters[metric] / win_count

    return counters


def exp_5_references_abstract_50s(triplets, sent_to_index):
    n_ref = 5
    ref_data = {'images': [],
                "licenses": [{"url": "http://creativecommons.org/licenses/by-nc-sa/2.0/", "id": 1,
                              "name": "Attribution-NonCommercial-ShareAlike License"},
                             {"url": "http://creativecommons.org/licenses/by-nc/2.0/", "id": 2,
                              "name": "Attribution-NonCommercial License"},
                             {"url": "http://creativecommons.org/licenses/by-nc-nd/2.0/", "id": 3,
                              "name": "Attribution-NonCommercial-NoDerivs License"},
                             {"url": "http://creativecommons.org/licenses/by/2.0/", "id": 4,
                              "name": "Attribution License"},
                             {"url": "http://creativecommons.org/licenses/by-sa/2.0/", "id": 5,
                              "name": "Attribution-ShareAlike License"},
                             {"url": "http://creativecommons.org/licenses/by-nd/2.0/", "id": 6,
                              "name": "Attribution-NoDerivs License"},
                             {"url": "http://flickr.com/commons/usage/", "id": 7,
                              "name": "No known copyright restrictions"},
                             {"url": "http://www.usa.gov/copyright.shtml", "id": 8,
                              "name": "United States Government Work"}],
                'type': 'captions',
                'info': {"description": "This is stable 1.0 version of the 2014 MS COCO dataset.",
                         "url": "http://mscoco.org",
                         "version": "1.0",
                         "year": 2014,
                         "contributor": "Microsoft COCO group", "date_created": "2015-01-27 09:11:52.357475"},
                'annotations': []}
    cand_b = []
    cand_c = []
    winners = {}
    img_to_index = {}
    for i, refs in enumerate(triplets.values()):
        ref_data['images'].append({'id': i})

        A, B, C, winner = refs[0]
        cand_b.append({"image_id": i, "caption": B})
        cand_c.append({"image_id": i, "caption": C})
        refs = random.sample(refs, n_ref)
        for ref in refs:
            A, B, C, winner = ref
            ref_data['annotations'].append({"image_id": i, "id": i, "caption": A})
            winners[i] = winners.get(i, 0) + winner
            img_to_index[i] = sent_to_index[B + C]

    with open('references.json', 'w') as file:
        json.dump(ref_data, file)

    with open('captions_B.json', 'w') as file:
        json.dump(cand_b, file)

    with open('captions_C.json', 'w') as file:
        json.dump(cand_c, file)

    annFile = 'references.json'
    resFile = 'captions_B.json'
    results_B = compute_metrics(annFile, resFile)

    resFile = 'captions_C.json'
    results_C = compute_metrics(annFile, resFile)

    HC = {'B': {img: value for img, value in results_B.items() if img_to_index[img] < 200},
          'C': {img: value for img, value in results_C.items() if img_to_index[img] < 200},
          'winners': {img: value for img, value in winners.items() if img_to_index[img] < 200}}

    HI = {'B': {img: value for img, value in results_B.items() if img_to_index[img] >= 200},
          'C': {img: value for img, value in results_C.items() if img_to_index[img] >= 200},
          'winners': {img: value for img, value in winners.items() if img_to_index[img] >= 200}}

    HC_accuracies = compute_accuracy(HC['B'], HC['C'], HC['winners'])
    HI_accuracies = compute_accuracy(HI['B'], HI['C'], HI['winners'])

    with open('results_abstract_50S.json', 'w') as file:
        json.dump({'HC': HC_accuracies, 'HI': HI_accuracies}, file)


def exp_5_references_pascal_50s(triplets, sent_to_index):
    n_ref = 5
    ref_data = {'images': [],
                "licenses": [{"url": "http://creativecommons.org/licenses/by-nc-sa/2.0/", "id": 1,
                              "name": "Attribution-NonCommercial-ShareAlike License"},
                             {"url": "http://creativecommons.org/licenses/by-nc/2.0/", "id": 2,
                              "name": "Attribution-NonCommercial License"},
                             {"url": "http://creativecommons.org/licenses/by-nc-nd/2.0/", "id": 3,
                              "name": "Attribution-NonCommercial-NoDerivs License"},
                             {"url": "http://creativecommons.org/licenses/by/2.0/", "id": 4,
                              "name": "Attribution License"},
                             {"url": "http://creativecommons.org/licenses/by-sa/2.0/", "id": 5,
                              "name": "Attribution-ShareAlike License"},
                             {"url": "http://creativecommons.org/licenses/by-nd/2.0/", "id": 6,
                              "name": "Attribution-NoDerivs License"},
                             {"url": "http://flickr.com/commons/usage/", "id": 7,
                              "name": "No known copyright restrictions"},
                             {"url": "http://www.usa.gov/copyright.shtml", "id": 8,
                              "name": "United States Government Work"}],
                'type': 'captions',
                'info': {"description": "This is stable 1.0 version of the 2014 MS COCO dataset.",
                         "url": "http://mscoco.org",
                         "version": "1.0",
                         "year": 2014,
                         "contributor": "Microsoft COCO group", "date_created": "2015-01-27 09:11:52.357475"},
                'annotations': []}
    cand_b = []
    cand_c = []
    winners = {}
    img_to_index = {}
    for i, refs in enumerate(triplets.values()):
        ref_data['images'].append({'id': i})

        A, B, C, winner = refs[0]
        cand_b.append({"image_id": i, "caption": B})
        cand_c.append({"image_id": i, "caption": C})
        refs = random.sample(refs, n_ref)
        for ref in refs:
            A, B, C, winner = ref
            ref_data['annotations'].append({"image_id": i, "id": i, "caption": A})
            winners[i] = winners.get(i, 0) + winner
            img_to_index[i] = sent_to_index[B + C]

    with open('references.json', 'w') as file:
        json.dump(ref_data, file)

    with open('captions_B.json', 'w') as file:
        json.dump(cand_b, file)

    with open('captions_C.json', 'w') as file:
        json.dump(cand_c, file)

    annFile = 'references.json'
    resFile = 'captions_B.json'
    results_B = compute_metrics(annFile, resFile)

    resFile = 'captions_C.json'
    results_C = compute_metrics(annFile, resFile)

    HC = {'B': {img: value for img, value in results_B.items() if img_to_index[img] < 1000},
          'C': {img: value for img, value in results_C.items() if img_to_index[img] < 1000},
          'winners': {img: value for img, value in winners.items() if img_to_index[img] < 1000}}

    HI = {
        'B': {img: value for img, value in results_B.items() if img_to_index[img] >= 1000 and img_to_index[img] < 2000},
        'C': {img: value for img, value in results_C.items() if img_to_index[img] >= 1000 and img_to_index[img] < 2000},
        'winners': {img: value for img, value in winners.items() if
                    img_to_index[img] >= 1000 and img_to_index[img] < 2000}}

    HM = {
        'B': {img: value for img, value in results_B.items() if img_to_index[img] >= 2000 and img_to_index[img] < 3000},
        'C': {img: value for img, value in results_C.items() if img_to_index[img] >= 2000 and img_to_index[img] < 3000},
        'winners': {img: value for img, value in winners.items() if
                    img_to_index[img] >= 2000 and img_to_index[img] < 3000}}

    MM = {'B': {img: value for img, value in results_B.items() if img_to_index[img] >= 3000},
          'C': {img: value for img, value in results_C.items() if img_to_index[img] >= 3000},
          'winners': {img: value for img, value in winners.items() if img_to_index[img] >= 3000}}

    HC_accuracies = compute_accuracy(HC['B'], HC['C'], HC['winners'])
    HI_accuracies = compute_accuracy(HI['B'], HI['C'], HI['winners'])

    HM_accuracies = compute_accuracy(HM['B'], HM['C'], HM['winners'])
    MM_accuracies = compute_accuracy(MM['B'], MM['C'], MM['winners'])

    with open('results_pascal_50S.json', 'w') as file:
        json.dump({'HC': HC_accuracies,
                   'HI': HI_accuracies,
                   'HM': HM_accuracies,
                   'MM': MM_accuracies, }, file)


def exp_varying_n_refs(triplets, imgfile, csvfile):
    results = {'n_ref': [],
               'Bleu_4': [],
               'METEOR': [],
               'ROUGE_L': [],
               'CIDEr': [],
               'CIDEr-R': [],
               'SPICE': []}

    for n_ref in range(1, 49):
        ref_data = {'images': [],
                    "licenses": [{"url": "http://creativecommons.org/licenses/by-nc-sa/2.0/", "id": 1,
                                  "name": "Attribution-NonCommercial-ShareAlike License"},
                                 {"url": "http://creativecommons.org/licenses/by-nc/2.0/", "id": 2,
                                  "name": "Attribution-NonCommercial License"},
                                 {"url": "http://creativecommons.org/licenses/by-nc-nd/2.0/", "id": 3,
                                  "name": "Attribution-NonCommercial-NoDerivs License"},
                                 {"url": "http://creativecommons.org/licenses/by/2.0/", "id": 4,
                                  "name": "Attribution License"},
                                 {"url": "http://creativecommons.org/licenses/by-sa/2.0/", "id": 5,
                                  "name": "Attribution-ShareAlike License"},
                                 {"url": "http://creativecommons.org/licenses/by-nd/2.0/", "id": 6,
                                  "name": "Attribution-NoDerivs License"},
                                 {"url": "http://flickr.com/commons/usage/", "id": 7,
                                  "name": "No known copyright restrictions"},
                                 {"url": "http://www.usa.gov/copyright.shtml", "id": 8,
                                  "name": "United States Government Work"}],
                    'type': 'captions',
                    'info': {"description": "This is stable 1.0 version of the 2014 MS COCO dataset.",
                             "url": "http://mscoco.org",
                             "version": "1.0",
                             "year": 2014,
                             "contributor": "Microsoft COCO group", "date_created": "2015-01-27 09:11:52.357475"},
                    'annotations': []}
        cand_b = []
        cand_c = []
        winners = {}
        for i, refs in enumerate(triplets.values()):
            ref_data['images'].append({'id': i})

            A, B, C, winner = refs[0]
            cand_b.append({"image_id": i, "caption": B})
            cand_c.append({"image_id": i, "caption": C})
            refs = random.sample(refs, n_ref)
            for ref in refs:
                A, B, C, winner = ref
                ref_data['annotations'].append({"image_id": i, "id": i, "caption": A})
                winners[i] = winners.get(i, 0) + winner

        with open('references.json', 'w') as file:
            json.dump(ref_data, file)

        with open('captions_B.json', 'w') as file:
            json.dump(cand_b, file)

        with open('captions_C.json', 'w') as file:
            json.dump(cand_c, file)

        annFile = 'references.json'
        resFile = 'captions_B.json'
        results_B = compute_metrics(annFile, resFile)

        resFile = 'captions_C.json'
        results_C = compute_metrics(annFile, resFile)

        accuracies = compute_accuracy(results_B, results_C, winners)

        results['n_ref'].append(n_ref)
        results['Bleu_4'].append(accuracies['Bleu_4'])
        results['METEOR'].append(accuracies['METEOR'])
        results['ROUGE_L'].append(accuracies['ROUGE_L'])
        results['CIDEr'].append(accuracies['CIDEr'])
        results['CIDEr-R'].append(accuracies['CIDEr-R'])
        results['SPICE'].append(accuracies['SPICE'])

    df_results = pd.DataFrame(results)
    plot = df_results.plot(x='n_ref')
    fig = plot.get_figure()
    fig.savefig(imgfile)
    df_results.to_csv(csvfile)


if __name__ == '__main__':
    triplets, sent_to_index = load_triplets(filepath='experiment_data/consensus_abstract.mat')
    exp_5_references_abstract_50s(triplets, sent_to_index)
    exp_varying_n_refs(triplets, imgfile='abstract_50S.png', csvfile='abstract_50S.csv')

    triplets, sent_to_index = load_triplets(filepath='experiment_data/consensus_pascal.mat')
    exp_5_references_pascal_50s(triplets, sent_to_index)
    exp_varying_n_refs(triplets, imgfile='pascal_50S.png', csvfile='pascal_50S.csv')
