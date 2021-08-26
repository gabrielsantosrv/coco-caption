import json
import random
from collections import defaultdict

with open('/home/gabriel/pracegover_projects/pracegover/dataset/ms_coco_annotations/captions_val2014.json') as file:
    data = json.load(file)

captions = defaultdict(lambda: [])

for input in data['annotations']:
    captions[input['image_id']].append(input['caption'])

captions = {k:v for i, (k, v) in enumerate(captions.items()) if i < 10000}
image_ids = list(captions.keys())

triplets = {'HCI':defaultdict(lambda: []), 'HII':defaultdict(lambda: [])}
n = len(captions)
index = 0
for img, refs in captions.items():
    ref_caption = random.choice(refs)
    refs = set(refs)
    refs.remove(ref_caption)
    refs = list(refs)
    random_id = img
    random_caption = None
    while random_id == img:
        random_id = random.choice(image_ids)
        random_caption = random.choice(captions[random_id])

    triplets['HCI'][index] = [refs, ref_caption, random_caption, 1]

    words = ref_caption.split()
    m = len(words)
    end = random.randint(m//4, m//2)
    triplets['HII'][index] = [refs, ' '.join(words[:end]), random_caption, 1]
    index += 1

print(len(triplets))
with open('mscoco_triplets_complete.json', 'w') as file:
    json.dump(triplets, file)
