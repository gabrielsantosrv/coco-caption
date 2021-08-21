from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

import json
from json import encoder

encoder.FLOAT_REPR = lambda o: format(o, '.3f')
import random
from scipy.io import loadmat
import pandas as pd


def load_pascal_triplets(filepath='experiment_data/consensus_pascal.mat'):
    consensus = loadmat(filepath)
    triplets = {}
    sent_to_index = {}
    index = 0
    for item in consensus['triplets'][0]:
        A = str(item[0][0][0][0]).encode('ascii', 'ignore').decode()
        B = str(item[1][0][0][0]).encode('ascii', 'ignore').decode()
        C = str(item[2][0][0][0]).encode('ascii', 'ignore').decode()
        winner = item[3][0][0]

        key = B + C
        triplets[key] = triplets.get(key, [])
        bucket = ''
        if len(triplets[key]) == 48:
            bucket = 1
            while len(triplets.get(key + str(bucket), [])) == 48:
                bucket += 1
            key = key + str(bucket)
            triplets[key] = triplets.get(key, [])

        triplets[key].append((A, B, C, winner, bucket))

        if sent_to_index.get(key, None) is None:
            sent_to_index[key] = index
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
        SPICE = 0#item['SPICE']['All']['f']
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
    print('computing accuracy...', len(winners), 'elements')
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


def get_class(pair):
    if pair[0] <= 5:
        if pair[1] <= 5:
            return 'MM'
        if pair[1] == 6:
            return 'HM'
    elif pair[0] == 6:
        if pair[1] <= 5:
            return 'HM'
        if pair[1] == 6:
            return 'HC'
        if pair[1] == 7:
            return 'HI'
    return None


def exp_5_references_pascal_50s(triplets, sent_to_index, pairs):
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
    results_B = {}
    results_C = {}
    for i, refs in enumerate(triplets.values()):
        ref_data['images'].append({'id': i})

        A, B, C, winner, bucket = refs[0]
        cand_b.append({"image_id": i, "caption": B})
        cand_c.append({"image_id": i, "caption": C})
        if n_ref <= len(refs):
            refs = random.sample(refs, n_ref)
        for ref in refs:
            A, B, C, winner, bucket = ref
            ref_data['annotations'].append({"image_id": i, "id": i, "caption": A})
            winners[i] = winners.get(i, 0) + winner
            img_to_index[i] = sent_to_index[B + C + str(bucket)]

        if i % 500 == 499:
            with open('references.json', 'w') as file:
                json.dump(ref_data, file)

            with open('captions_B.json', 'w') as file:
                json.dump(cand_b, file)

            with open('captions_C.json', 'w') as file:
                json.dump(cand_c, file)

            annFile = 'references.json'
            resFile = 'captions_B.json'
            results_B_aux = compute_metrics(annFile, resFile)
            results_B.update(results_B_aux)

            resFile = 'captions_C.json'
            results_C_aux = compute_metrics(annFile, resFile)
            results_C.update(results_C_aux)

            ref_data['images'] = []
            ref_data['annotations'] = []
            cand_b = []
            cand_c = []

    accuracies = compute_accuracy(results_B, results_C, winners)
    print(accuracies)
    HC = {'B': {img: value for img, value in results_B.items() if get_class(pairs['new_data'][img_to_index[img]]) == 'HC'},
          'C': {img: value for img, value in results_C.items() if get_class(pairs['new_data'][img_to_index[img]]) == 'HC'},
          'winners': {img: value for img, value in winners.items() if get_class(pairs['new_data'][img_to_index[img]]) == 'HC'}}

    HI = {
        'B': {img: value for img, value in results_B.items() if get_class(pairs['new_data'][img_to_index[img]]) == 'HI'},
        'C': {img: value for img, value in results_C.items() if get_class(pairs['new_data'][img_to_index[img]]) == 'HI'},
        'winners': {img: value for img, value in winners.items() if get_class(pairs['new_data'][img_to_index[img]]) == 'HI'}}

    HM = {
        'B': {img: value for img, value in results_B.items() if get_class(pairs['new_data'][img_to_index[img]]) == 'HM'},
        'C': {img: value for img, value in results_C.items() if get_class(pairs['new_data'][img_to_index[img]]) == 'HM'},
        'winners': {img: value for img, value in winners.items() if get_class(pairs['new_data'][img_to_index[img]]) == 'HM'}}

    MM = {'B': {img: value for img, value in results_B.items() if get_class(pairs['new_data'][img_to_index[img]]) == 'MM'},
          'C': {img: value for img, value in results_C.items() if get_class(pairs['new_data'][img_to_index[img]]) == 'MM'},
          'winners': {img: value for img, value in winners.items() if get_class(pairs['new_data'][img_to_index[img]]) == 'MM'}}

    HC_accuracies = compute_accuracy(HC['B'], HC['C'], HC['winners'])
    HI_accuracies = compute_accuracy(HI['B'], HI['C'], HI['winners'])

    HM_accuracies = compute_accuracy(HM['B'], HM['C'], HM['winners'])
    MM_accuracies = compute_accuracy(MM['B'], MM['C'], MM['winners'])

    with open('results_pascal_50S.json', 'w') as file:
        json.dump({'HC': HC_accuracies,
                   'HI': HI_accuracies,
                   'HM': HM_accuracies,
                   'MM': MM_accuracies, }, file)

def exp_varying_n_refs(triplets, imgfile, csvfile, sent_to_index, pairs=None, only_MM=False):
    results = {'n_ref': [],
               'Bleu_4': [],
               'METEOR': [],
               'ROUGE_L': [],
               'CIDEr': [],
               'CIDEr-R': [],
               'SPICE': []}
    all_refs = {}
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
        results_B = {}
        results_C = {}
        img_to_index = {}
        for i, refs in enumerate(triplets.values()):
            ref_data['images'].append({'id': i})

            A, B, C, winner, _ = refs[0]
            cand_b.append({"image_id": i, "caption": B})
            cand_c.append({"image_id": i, "caption": C})
            all_refs[i] = all_refs.get(i, [])

            if n_ref <= len(refs) and len(list(set(refs) - set(all_refs[i]))) > 0:
                ref = random.choice(list(set(refs) - set(all_refs[i])))
                all_refs[i].append(ref)

            for ref in all_refs[i]:
                A, B, C, winner, bucket = ref
                ref_data['annotations'].append({"image_id": i, "id": i, "caption": A})
                winners[i] = winners.get(i, 0) + winner
                img_to_index[i] = sent_to_index[B + C + str(bucket)]

            if i % 500 == 499:
                with open('references.json', 'w') as file:
                    json.dump(ref_data, file)

                with open('captions_B.json', 'w') as file:
                    json.dump(cand_b, file)

                with open('captions_C.json', 'w') as file:
                    json.dump(cand_c, file)

                annFile = 'references.json'
                resFile = 'captions_B.json'
                results_B_aux = compute_metrics(annFile, resFile)
                results_B.update(results_B_aux)

                resFile = 'captions_C.json'
                results_C_aux = compute_metrics(annFile, resFile)
                results_C.update(results_C_aux)

                ref_data['images'] = []
                ref_data['annotations'] = []
                cand_b = []
                cand_c = []

        if only_MM:
            MM = {'B': {img: value for img, value in results_B.items() if
                        get_class(pairs['new_data'][img_to_index[img]]) == 'MM'},

                  'C': {img: value for img, value in results_C.items() if
                        get_class(pairs['new_data'][img_to_index[img]]) == 'MM'},

                  'winners': {img: value for img, value in winners.items() if
                              get_class(pairs['new_data'][img_to_index[img]]) == 'MM'}}

            accuracies = compute_accuracy(MM['B'], MM['C'], MM['winners'])
        else:
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


def compute_accuracy_pracegover(data):
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

    results_B = {}
    results_C = {}
    for k, triplet in data.items():
        k = int(k)
        A, B, C, winner = triplet
        ref_data['images'].append({'id': k})

        cand_b.append({"image_id": k, "caption": B})
        cand_c.append({"image_id": k, "caption": C})
        ref_data['annotations'].append({"image_id": k, "id": k, "caption": A})
        winners[k] = winner

        if k % 500 == 499:
            with open('references.json', 'w') as file:
                json.dump(ref_data, file)

            with open('captions_B.json', 'w') as file:
                json.dump(cand_b, file)

            with open('captions_C.json', 'w') as file:
                json.dump(cand_c, file)

            annFile = 'references.json'
            resFile = 'captions_B.json'
            results_B_aux = compute_metrics(annFile, resFile)
            results_B.update(results_B_aux)

            resFile = 'captions_C.json'
            results_C_aux = compute_metrics(annFile, resFile)
            results_C.update(results_C_aux)

            ref_data['images'] = []
            ref_data['annotations'] = []
            cand_b = []
            cand_c = []

    if len(cand_b) > 0:
        with open('references.json', 'w') as file:
            json.dump(ref_data, file)

        with open('captions_B.json', 'w') as file:
            json.dump(cand_b, file)

        with open('captions_C.json', 'w') as file:
            json.dump(cand_c, file)

        annFile = 'references.json'
        resFile = 'captions_B.json'
        results_B_aux = compute_metrics(annFile, resFile)
        results_B.update(results_B_aux)

        resFile = 'captions_C.json'
        results_C_aux = compute_metrics(annFile, resFile)
        results_C.update(results_C_aux)

    return compute_accuracy(results_B, results_C, winners)

def exp_pracegover(output_file, filepath='experiment_data/pracegover_triplets.json'):
    with open(filepath) as file:
        data = json.load(file)

    accuracies = compute_accuracy_pracegover(data['HCI'])
    print('HCI', accuracies)

    accuracies = compute_accuracy_pracegover(data['HII'])
    print('HII', accuracies)


    # with open('results_pracegover.json', 'w') as file:
    #     json.dump(accuracies, file)


if __name__ == '__main__':
    # triplets, sent_to_index = load_pascal_triplets()
    # pairs = loadmat('experiment_data/pair_pascal.mat')
    # exp_5_references_pascal_50s(triplets, sent_to_index, pairs=pairs)
    #
    # exp_varying_n_refs(triplets, sent_to_index=sent_to_index, imgfile='pascal_50S.png', csvfile='pascal_50S.csv')
    # exp_varying_n_refs(triplets, sent_to_index=sent_to_index, imgfile='pascal_50S_only_MM.png', csvfile='pascal_50S_only_MM.csv',
    #                    only_MM=True, pairs=pairs)

    exp_pracegover(output_file='output.log')