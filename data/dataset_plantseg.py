import json
import os

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image

from bert.tokenization_bert import BertTokenizer


class PlantSegDataset(data.Dataset):
    def __init__(self,
                 args,
                 image_transforms=None,
                 target_transforms=None,
                 split='train'):

        self.image_transforms = image_transforms
        self.target_transform = target_transforms
        self.split = split
        self.data_root = args.plantseg_root
        self.caption_index = args.caption_index
        self.max_tokens = 20
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer)

        with open(os.path.join(self.data_root, 'main.json'), 'r', encoding='utf-8') as f:
            all_samples = json.load(f)

        self.samples = [sample for sample in all_samples if sample.get('split') == split]

        for sample in self.samples:
            captions = sample.get('caption', [])
            if len(captions) <= self.caption_index:
                raise ValueError(
                    'Sample {} does not contain caption index {}.'.format(
                        sample.get('id', 'unknown'), self.caption_index))

    def __len__(self):
        return len(self.samples)

    def _encode_caption(self, sentence_raw):
        attention_mask = [0] * self.max_tokens
        padded_input_ids = [0] * self.max_tokens

        input_ids = self.tokenizer.encode(text=sentence_raw, add_special_tokens=True)
        input_ids = input_ids[:self.max_tokens]

        padded_input_ids[:len(input_ids)] = input_ids
        attention_mask[:len(input_ids)] = [1] * len(input_ids)

        tensor_embeddings = torch.tensor(padded_input_ids, dtype=torch.long).unsqueeze(0)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long).unsqueeze(0)
        return tensor_embeddings, attention_mask

    def __getitem__(self, index):
        sample = self.samples[index]

        image_path = os.path.join(self.data_root, sample['image'])
        mask_path = os.path.join(self.data_root, sample['mask'])

        img = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        mask = np.array(mask)
        mask = (mask > 0).astype(np.uint8)
        annot = Image.fromarray(mask, mode="P")

        if self.image_transforms is not None:
            img, target = self.image_transforms(img, annot)
        else:
            target = torch.as_tensor(mask.copy(), dtype=torch.int64)

        tensor_embeddings, attention_mask = self._encode_caption(sample['caption'][self.caption_index])

        targets = {
            'mask': target,
            'cls': torch.tensor(0, dtype=torch.int64),
            'mask_path': sample['mask'],
            'image_path': sample['image'],
            'sample_id': sample['id'],
            'disease_label': sample.get('disease_label', ''),
        }
        return img, targets, tensor_embeddings, attention_mask
