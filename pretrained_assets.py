import os
import shutil

import requests


OFFICIAL_BERT_ID = 'bert-base-uncased'
OFFICIAL_BERT_FILES = {
    'config.json': 'https://huggingface.co/bert-base-uncased/resolve/main/config.json',
    'pytorch_model.bin': 'https://huggingface.co/bert-base-uncased/resolve/main/pytorch_model.bin',
    'vocab.txt': 'https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt',
    'tokenizer_config.json': 'https://huggingface.co/bert-base-uncased/resolve/main/tokenizer_config.json',
}
OFFICIAL_SWIN_URL = (
    'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/'
    'swin_base_patch4_window7_224_22k.pth'
)


def download_file(url, destination):
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    tmp_destination = destination + '.tmp'
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    with open(tmp_destination, 'wb') as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

    shutil.move(tmp_destination, destination)


def ensure_official_bert(asset_root):
    bert_root = os.path.join(asset_root, OFFICIAL_BERT_ID)
    os.makedirs(bert_root, exist_ok=True)

    for filename, url in OFFICIAL_BERT_FILES.items():
        destination = os.path.join(bert_root, filename)
        if not os.path.exists(destination):
            download_file(url, destination)

    return bert_root


def ensure_official_swin(asset_root):
    swin_root = os.path.join(asset_root, 'swin')
    os.makedirs(swin_root, exist_ok=True)
    destination = os.path.join(swin_root, 'swin_base_patch4_window7_224_22k.pth')
    if not os.path.exists(destination):
        download_file(OFFICIAL_SWIN_URL, destination)
    return destination


def prepare_official_pretrained_assets(args):
    if not getattr(args, 'auto_download_assets', False):
        return args

    asset_root = args.asset_cache_dir
    os.makedirs(asset_root, exist_ok=True)

    uses_official_bert = args.bert_tokenizer == OFFICIAL_BERT_ID or args.ck_bert == OFFICIAL_BERT_ID
    if uses_official_bert:
        bert_root = ensure_official_bert(asset_root)
        args.bert_tokenizer = bert_root
        args.ck_bert = bert_root

    if not args.pretrained_swin_weights:
        args.pretrained_swin_weights = ensure_official_swin(asset_root)

    return args
