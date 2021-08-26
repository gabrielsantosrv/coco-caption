import json

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

for i in range(5):
    ref_data['images'].append({'id': i})
    ref_data['annotations'].append({"image_id": i, "id": i, "caption": 'A young girl is preparing to blow out her candle'})
    ref_data['images'].append({'id': i + 5})
    ref_data['annotations'].append({"image_id": i + 5, "id": i + 5, "caption": 'a bird sitting on the back of a cow and a dog and bird standing on the ground next to the cow'})
    ref_data['images'].append({'id': i + 10})
    ref_data['annotations'].append({"image_id": i + 10, "id": i + 10, "caption": 'dog laying on couch looking into distance with remote control by paw'})

### correct candidate
correct_cand = [{"image_id": 0, "caption": 'A young girl is about to blow out her candle'},
                {"image_id": 1, "caption": 'A young girl is getting ready to blow out her candle'},
                {"image_id": 2, "caption": 'A young girl is getting ready to blow out a candle'},
                {"image_id": 3, "caption": 'A young girl is getting ready to blow out a candle on a small dessert'},
                {"image_id": 4, "caption": 'A kid is to blow out the single candle in a bowl of birthday goodness'},

                {"image_id": 5, "caption": 'a mottled brown dog and cow with two little birds outdoors'},
                {"image_id": 6, "caption": 'a cow standing next to a dog on dirt ground'},
                {"image_id": 7, "caption": 'a dog with a bird and a large cow on a street'},
                {"image_id": 8, "caption": 'a dog and a cow with a bird on its back'},
                {"image_id": 9, "caption": 'a cow and a dog on a street'}]

correct_cand.extend([{"image_id": i, "caption": 'a dog with a remote control'} for i in range(10,15)])

### incorrect candidate

incorrect_cand = [{"image_id": i, "caption": 'A young dog is preparing to blow out her candle'} for i in range(5)]
incorrect_cand.extend([{"image_id": i, "caption": 'a bird sitting on  the on of a cat and a dog and a dog and and standing on next to'} for i in range(5, 10)])

incorrect_cand.append({"image_id": 10, "caption": 'cat laying on couch looking into distance with remote control by paw'})
incorrect_cand.append({"image_id": 11, "caption": 'cat looking into distance with remote control by paw'})
incorrect_cand.append({"image_id": 12, "caption": 'cat looking into distance'})
incorrect_cand.append({"image_id": 13, "caption": 'cat with remote control by paw'})
incorrect_cand.append({"image_id": 14, "caption": 'a cat with a remote control'})

with open('toy_candidate_classification/toy_candidate_classification_references.json', 'w') as file:
    json.dump(ref_data, file)

with open('toy_candidate_classification/toy_candidate_classification_correct_candidates.json', 'w') as file:
    json.dump(correct_cand, file)

with open('toy_candidate_classification/toy_candidate_classification_incorrect_candidates.json', 'w') as file:
    json.dump(incorrect_cand, file)