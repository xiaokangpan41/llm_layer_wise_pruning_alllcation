import argparse
import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM,LlamaTokenizer
# from importlib.metadata import version
from collections import defaultdict
from lib.prune_all import prune_wanda_outlier_structure_special,prune_wanda_outlier_structure,prune_sparsegpt_outlier,prune_wanda_outlier,prune_mag_outlier, prune_wanda,prune_magnitude,prune_sparsegpt, check_sparsity, find_layers
from lib.eval import eval_ppl
import sys
print('# of gpus: ', torch.cuda.device_count())
from lib.my_version1 import prune_wanda_outlier_plus
import os, csv, gc,  time

import json
import logging
import math

import random
from itertools import chain
from pathlib import Path

import datasets

from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from huggingface_hub import Repository, create_repo
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from transformers.utils import check_min_version, get_full_repo_name, send_example_telemetry
from transformers.utils.versions import require_version

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


logger = get_logger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

def get_llm(model, cache_dir="llm_weights"):
    model = model.strip()  # 关键：去掉末尾空格/换行
    assert os.path.isdir(model), f"Local model dir not found: {model}\nPWD={os.getcwd()}\nList parent={os.listdir(os.path.dirname(model))}"

    model = AutoModelForCausalLM.from_pretrained(
        model,
        torch_dtype=torch.float16,
        cache_dir=cache_dir,
        low_cpu_mem_usage=True,
        device_map="auto",
        local_files_only=True,   # 关键：强制只用本地文件
    )

    model.seqlen = 2048
    return model



def capture_base_state_dict_cpu(model):
    """Copy model weights to CPU RAM (fp16 stays fp16)."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

def restore_base_state_dict(model, base_sd):
    """Restore model weights from CPU copy."""
    model.load_state_dict(base_sd, strict=True)
    model.eval()

def run_once_prune_eval(args, model, tokenizer, device, prune_n=0, prune_m=0):
    """Your original single-run logic: prune -> check_sparsity -> eval_ppl."""
    model.eval()

    print("pruning starts")

    # -------- pruning --------
    if args.prune_method == "wanda":
        prune_wanda(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
    elif args.prune_method == "magnitude":
        prune_magnitude(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
    elif args.prune_method == "sparsegpt":
        prune_sparsegpt(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
    elif args.prune_method == "wanda_owl":
        prune_wanda_outlier(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
    elif args.prune_method == "prune_wanda_outlier_plus":
        prune_wanda_outlier_plus(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
    elif args.prune_method == "magnitude_owl":
        prune_mag_outlier(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
    elif args.prune_method == "sparsegpt_owl":
        prune_sparsegpt_outlier(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
    elif args.prune_method == "wanda_owl_structure":
        prune_wanda_outlier_structure(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
    elif args.prune_method == "wanda_owl_structure_special":
        prune_wanda_outlier_structure_special(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
    else:
        raise ValueError(f"Unknown prune_method: {args.prune_method}")

    # -------- check + eval --------
    print("*" * 30)
    sparsity_ratio = check_sparsity(model)
    print(f"sparsity sanity check {sparsity_ratio:.4f}")
    print("*" * 30)

    ppl = eval_ppl(model, tokenizer, device)
    print(f"ppl on wikitext {ppl}")

    torch.cuda.empty_cache()
    gc.collect()
    return float(ppl)

def load_done_keys_from_outcsv(out_csv):
    """Return set of keys 'k|probe|decay|decay_type|decay_beta|alpha|consider_current' already in out_csv."""
    done = set()
    if not os.path.exists(out_csv) or os.path.getsize(out_csv) == 0:
        return done
    with open(out_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                k = str(row["risk_k"]).strip()
                p = str(row["risk_probe"]).strip()
                d = str(row["risk_decay"]).strip()
                dt = str(row["risk_decay_type"]).strip()
                db = str(row.get("risk_decay_beta", "")).strip()
                a = str(row["risk_alpha"]).strip()
                c = str(row["consider_current_layer_in_risk"]).strip()
                if k and p and d and dt and db and a and c:
                    done.add(f"{k}|{p}|{d}|{dt}|{db}|{a}|{c}")
            except Exception:
                pass
    return done

def ensure_outcsv_header(out_csv):
    if (not os.path.exists(out_csv)) or os.path.getsize(out_csv) == 0:
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "risk_k", "risk_probe", "risk_decay", "risk_decay_type", "risk_decay_beta",
                "risk_alpha", "consider_current_layer_in_risk",
                "ppl", "exit_code", "seconds"
            ])




def main(argv=None):

    ########################## for prune ################################
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLaMA model')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=32, help='Number of calibration samples.')
    parser.add_argument('--sparsity_ratio', type=float, default=0.7, help='Sparsity level')
    parser.add_argument("--sparsity_type", type=str)
    parser.add_argument("--prune_method", type=str)
    parser.add_argument("--cache_dir", default="llm_weights", type=str )
    parser.add_argument('--use_variant', action="store_true", help="whether to use the wanda variant described in the appendix")
    parser.add_argument('--save', type=str, default=None, help='Path to save results.')
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')


    # ===== Propagation Risk (for OWL + propagation-aware sparsity) =====
    parser.add_argument(
        "--risk_k",
        type=int,
        default=4,
        help="Number of downstream layers used to measure propagation risk R_i"
    )

    parser.add_argument(
        "--risk_probe",
        type=float,
        default=0.02,
        help="Probe pruning ratio for estimating propagation risk (e.g., 0.01~0.05)"
    )

    parser.add_argument(
        "--risk_decay",
        type=float,
        default=0.9,
        help="Exponential decay factor for downstream layers when aggregating propagation risk, 0.9 for exponential decay"
    )

    parser.add_argument(
        "--risk_decay_beta",
        type=float,
        default=0.01,
        help="Linear decay beta for downstream layers when aggregating propagation risk (only used if risk_decay_type is linear), (0, 0.19] for linear decay"
    )

    parser.add_argument(
        "--risk_decay_type",
        type=str,
        default="exponential",  # or "linear"
        help="Type of decay to use for downstream layers when aggregating propagation risk"
    )

    parser.add_argument(
        "--consider_current_layer_in_risk",
        type=bool,
        default=False,
        help="Whether to include the current layer in the propagation risk calculation"
    )

    parser.add_argument(
        "--risk_alpha",
        type=float,
        default=0.3,
        help="Weight for propagation risk in D_hat = (1-alpha) * D + alpha * R"
    )

    ########################################### for train
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="wikitext",
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default="wikitext-2-raw-v1",
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=None,
        help=(
            "Optional input sequence length after tokenization. The training dataset will be truncated in block of"
            " this size for training. Default to the model max input length for single sentence inputs (take into"
            " account special tokens)."
        ),
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--no_keep_linebreaks", action="store_true", help="Do not keep line breaks when using TXT files."
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--low_cpu_mem_usage",
        action="store_true",
        help=(
            "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
            "If passed, LLM loading time and RAM consumption will be benefited."
        ),
    )





    #### saving parameters #####

    parser.add_argument(
        "--method",
        type=str,
        default=None,

    )



    #### data parameters #####

    parser.add_argument(
        "--Lamda",
        default=0.08,
        type=float,
        help="Lamda",
    )

    parser.add_argument("--grid_csv", type=str, default=None,
                        help="Path to grid csv with columns risk_k,risk_probe,risk_decay,risk_alpha. If set, run sweep in one process.")
    parser.add_argument("--out_csv", type=str, default="risk_sweep.csv",
                        help="Output CSV for sweep mode.")


    parser.add_argument(
            '--Hyper_m',
            type=float,
            default=3, )

    parser.add_argument(
        "--outlier_by_activation", action="store_true", help="outlier_by_activation")


    parser.add_argument(
        "--outlier_by_wmetric", action="store_true", help="outlier_by_wmetric")




    # ✅ 关键：外部传 argv 就用外部的；否则用默认的
    args = parser.parse_args(argv)


    print("[DEBUG] Parsed args:")
    print("  nsamples   =", args.nsamples)
    print("  risk_alpha     =", args.risk_alpha)
    print("  risk_k     =", args.risk_k)
    print("  risk_probe =", args.risk_probe)
    print("  risk_decay =", args.risk_decay)

    run_t0 = time.time()

    # print ("args.nsamples",args.nsamples)
    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    # Handling n:m sparsity
    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured":
        assert args.sparsity_ratio == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity"
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))


    model_name = args.model.split("/")[-1]
    # print(f"loading llm model {args.model}")
    model = get_llm(args.model, args.cache_dir)


    # print ("model is =================================================================================")
    # print (model.__class__.__name__)
    # print (model)


    model.eval()

    if "opt" in args.model:
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    elif "llama" in args.model:

        tokenizer = LlamaTokenizer.from_pretrained(args.model, use_fast=False)



    device = torch.device("cuda:0")
    if "30b" in args.model or "65b" in args.model: # for 30b and 65b we use device_map to load onto multiple A6000 GPUs, thus the processing here.
        device = model.hf_device_map["lm_head"]
    print("use device ", device)



    print ("target sparsity", args.sparsity_ratio)


    # =======================
    # Sweep mode (one process)
    # =======================
    if args.grid_csv is not None:
        ensure_outcsv_header(args.out_csv)
        done = load_done_keys_from_outcsv(args.out_csv)
        print(f"[SWEEP] grid_csv={args.grid_csv}")
        print(f"[SWEEP] out_csv={args.out_csv}")
        print(f"[SWEEP] resume: found {len(done)} done configs")

        print("[SWEEP] capturing base weights to CPU RAM ...")
        base_sd = capture_base_state_dict_cpu(model)
        print("[SWEEP] base weights captured.")

        with open(args.grid_csv, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                k = int(float(row["risk_k"]))
                probe = float(row["risk_probe"])
                decay = float(row["risk_decay"])
                decay_type = str(row["risk_decay_type"]).strip()
                decay_beta = float(row["risk_decay_beta"])
                alpha = float(row["risk_alpha"])
                consider_current = str(row["consider_current_layer_in_risk"]).strip()

                key = f"{k}|{probe}|{decay}|{decay_type}|{decay_beta}|{alpha}|{consider_current}"
                if key in done:
                    print(f"[SKIP] {key}")
                    continue

                # set args for this run
                args.risk_k = k
                args.risk_probe = probe
                args.risk_decay = decay
                args.risk_decay_type = decay_type
                args.risk_decay_beta = decay_beta
                args.risk_alpha = alpha
                args.consider_current_layer_in_risk = consider_current

                print(f"\n[RUN ] k={k}, probe={probe}, decay={decay}, decay_type={decay_type}, decay_beta={decay_beta}, alpha={alpha}, consider_current={consider_current}")
                t0 = time.time()

                # IMPORTANT: restore clean model (avoid interference)
                restore_base_state_dict(model, base_sd)

                exit_code = 0
                ppl = float("nan")
                try:
                    ppl = run_once_prune_eval(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
                except Exception as e:
                    exit_code = 1
                    print(f"[ERR ] {repr(e)}")

                sec = time.time() - t0

                # append result
                with open(args.out_csv, "a", newline="") as wf:
                    w = csv.writer(wf)
                    w.writerow([
                        k, probe, decay, args.risk_decay_type, args.risk_decay_beta,
                        alpha, args.consider_current_layer_in_risk,
                        ppl, exit_code, f"{sec:.2f}"
                    ])
                    wf.flush()

                done.add(key)

        sys.stdout.flush()
        return 0

    # =======================
    # Single-run mode (old behavior)
    # =======================
    ppl = run_once_prune_eval(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
    sys.stdout.flush()
    run_t1 = time.time()
    elapsed = run_t1 - run_t0
    print(f"[TIME] this run took {elapsed:.2f} seconds ({elapsed/60:.2f} min)")
    print(f"final ppl on wikitext {ppl}")
    return ppl




    if args.save_model:
        model.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)
        print(f"model saved to {args.save_model}")




import itertools
import csv
import math
import traceback



if __name__ == "__main__":
    main()   # 不要传 sys.argv[1:]

