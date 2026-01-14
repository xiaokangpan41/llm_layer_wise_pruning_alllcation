# Code adapted from https://github.com/IST-DASLab/sparsegpt/blob/master/datautils.py

import os
import numpy as np
import random
import torch
from datasets import load_dataset


def set_seed(seed: int):
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    random.seed(seed)


class TokenizerWrapper:
    def __init__(self, input_ids: torch.Tensor):
        self.input_ids = input_ids


def get_wikitext2(nsamples: int, seed: int, seqlen: int, tokenizer):
    """
    Wikitext-2 raw
    Returns:
      trainloader: list[(inp, tar)] with shapes [1, seqlen]
      testenc: tokenized full test set (pt tensors)
    """
    set_seed(seed)

    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    trainenc = tokenizer(" ".join(traindata["text"]), return_tensors="pt")
    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")

    trainloader = []
    max_start = trainenc.input_ids.shape[1] - seqlen - 1
    if max_start <= 0:
        raise ValueError(f"[wikitext2] seqlen={seqlen} too large for tokenized train set.")

    for _ in range(nsamples):
        i = random.randint(0, max_start)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    return trainloader, testenc


def _resolve_cache_dir(cache_dir: str = None) -> str:
    """
    HuggingFace datasets cache location:
      1) explicit cache_dir argument
      2) env HF_DATASETS_CACHE
      3) default (~/.cache/huggingface/datasets) handled by datasets itself
    """
    if cache_dir is not None and len(cache_dir) > 0:
        return cache_dir
    env_cache = os.environ.get("HF_DATASETS_CACHE", "").strip()
    return env_cache if env_cache else None


def get_c4_shard(
        nsamples: int,
        seed: int,
        seqlen: int,
        tokenizer,
        train_shard_id: int = 5,
        val_shard_id: int = 1,
        cache_dir: str = None,
        max_val_docs: int = 1100,
):
    """
    C4 English - fixed small subset by selecting specific shards.

    IMPORTANT:
      - We must pass BOTH 'train' and 'validation' in data_files,
        otherwise datasets will complain ExpectedMoreSplits {'validation'}.

    Defaults:
      - train_shard_id=0 -> en/c4-train.00000-of-01024.json.gz
      - val_shard_id=0   -> en/c4-validation.00000-of-00008.json.gz

    Returns:
      trainloader: list[(inp, tar)] each [1, seqlen]
      valenc: TokenizerWrapper of concatenated validation tokens
    """
    set_seed(seed)
    cache_dir = _resolve_cache_dir(cache_dir)

    train_path = f"en/c4-train.{train_shard_id:05d}-of-01024.json.gz"
    val_path = f"en/c4-validation.{val_shard_id:05d}-of-00008.json.gz"

    data_files = {
        "train": train_path,
        "validation": val_path,
    }
    # Load only the selected shards
    traindata = load_dataset(
        "allenai/c4",
        "en",
        data_files=data_files,
        split="train",
        cache_dir=cache_dir,
        verification_mode="no_checks",
    )
    valdata = load_dataset(
        "allenai/c4",
        "en",
        data_files=data_files,
        split="validation",
        cache_dir=cache_dir,
        verification_mode="no_checks",
    )

    # Build trainloader (random contiguous blocks)
    trainloader = []
    for _ in range(nsamples):
        while True:
            idx = random.randint(0, len(traindata) - 1)
            text = traindata[idx].get("text", "")
            if not isinstance(text, str) or len(text) == 0:
                continue
            enc = tokenizer(text, return_tensors="pt")
            if enc.input_ids.shape[1] > seqlen:
                break

        max_start = enc.input_ids.shape[1] - seqlen - 1
        i = random.randint(0, max_start)
        j = i + seqlen
        inp = enc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    # Validation tensor (like sparsegpt):
    # concatenate first max_val_docs documents of validation shard
    n_val = min(max_val_docs, len(valdata))
    val_text = " ".join(valdata[:n_val]["text"])
    valenc = tokenizer(val_text, return_tensors="pt").input_ids
    valenc = valenc[:, : (256 * seqlen)]
    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc


def get_loaders(name: str, nsamples: int = 128, seed: int = 0, seqlen: int = 2048, tokenizer=None):
    """
    Supported name:
      - 'wikitext2'
      - 'c4'  (shard-based small subset, fast + reproducible)
    """
    if tokenizer is None:
        raise ValueError("tokenizer must be provided")

    lname = name.lower()

    if "wikitext2" in lname or "wikitext" in lname:
        return get_wikitext2(nsamples, seed, seqlen, tokenizer)

    if "c4" in lname:
        # Scheme B: fixed shard(s) subset
        # You can change shard id here if you want another shard
        return get_c4_shard(
            nsamples=nsamples,
            seed=seed,
            seqlen=seqlen,
            tokenizer=tokenizer,
            train_shard_id=0,
            val_shard_id=0,
            cache_dir=None,
            max_val_docs=1100,
        )

    raise ValueError(f"Unknown dataset name: {name}")
