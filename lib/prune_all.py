import time
import heapq
import torch
import torch.nn as nn
from .sparsegpt import SparseGPT
from .layerwrapper import WrappedGPT
from .data import get_loaders
import numpy as np
from pdb import set_trace as st
from collections import defaultdict
from contextlib import contextmanager

def _rel_l2(a, b, eps=1e-8):
    # a,b: [1, seq, hidden]
    num = (a - b).float().pow(2).mean().sqrt()
    den = a.float().pow(2).mean().sqrt().clamp_min(eps)
    return (num / den).item()

def _forward_one_layer(layer, x, attention_mask=None, position_ids=None, is_opt=False):
    if is_opt:
        return layer(x, attention_mask=attention_mask)[0]
    else:
        return layer(x, attention_mask=attention_mask, position_ids=position_ids)[0]

def _forward_range(layers, start, end, x, attention_mask=None, position_ids=None, is_opt=False):
    # run layers[start..end] sequentially, return list of outputs after each layer
    outs = []
    h = x
    for idx in range(start, end + 1):
        h = _forward_one_layer(layers[idx], h, attention_mask, position_ids, is_opt=is_opt)
        outs.append(h)
    return outs  # length = end-start+1

@contextmanager
def _perturb_layer_magnitude(layer: nn.Module, probe_ratio: float = 0.02):
    """
    Temporarily zero out the smallest |w| weights in every nn.Linear in this layer.
    This is ONLY for risk probing, not final pruning.
    """
    backups = []
    with torch.no_grad():
        for m in layer.modules():
            if isinstance(m, nn.Linear):
                W = m.weight
                backups.append((W, W.data.clone()))

                flat = W.data.abs().view(-1)
                k = int(probe_ratio * flat.numel())
                if k <= 0:
                    continue
                thresh = torch.kthvalue(flat, k).values
                mask = (W.data.abs() > thresh).to(W.data.dtype)
                W.data.mul_(mask)
    try:
        yield
    finally:
        with torch.no_grad():
            for W, orig in backups:
                W.data.copy_(orig)

def _same_device_end(model, i: int, j_end: int) -> int:
    """
    If model is sharded across devices (hf_device_map), restrict end index to the last layer
    that stays on the same device as layer i. This avoids device-mismatch in _forward_range.
    """
    if not hasattr(model, "hf_device_map"):
        return j_end
    dev_i = model.hf_device_map.get(f"model.layers.{i}", None)
    if dev_i is None:
        return j_end
    while j_end > i:
        dev_j = model.hf_device_map.get(f"model.layers.{j_end}", dev_i)
        if dev_j == dev_i:
            break
        j_end -= 1
    return j_end



def prepare_calibration_input_opt(model, dataloader, device):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    if "OPT" in model.__class__.__name__:
        layers=model.model.decoder.layers

    else:
        layers = model.model.layers

    # dev = model.hf_device_map["model.embed_tokens"]
    hf_device_map = getattr(model, "hf_device_map", None)
    if isinstance(hf_device_map, dict):
        # Transformers/Accelerate may use different module paths depending on architecture/version.
        for k in (
            "model.embed_tokens",
            "model.decoder.embed_tokens",
            "model.model.embed_tokens",
            "model.model.decoder.embed_tokens",
        ):
            if k in hf_device_map:
                device = hf_device_map[k]
                break

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((128, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None,}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    model.config.use_cache = use_cache

    position_ids=None

    return inps, outs, attention_mask, position_ids




def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def _get_model_layers(model):
    """
    Return the per-block transformer layers for common causal LM architectures.
    - LLaMA-style: model.model.layers
    - OPT-style:   model.model.decoder.layers
    - GPT2-style:  model.transformer.h
    """
    core = getattr(model, "model", None)
    if core is not None:
        decoder = getattr(core, "decoder", None)
        if decoder is not None and hasattr(decoder, "layers"):
            return decoder.layers
        if hasattr(core, "layers"):
            return core.layers
    transformer = getattr(model, "transformer", None)
    if transformer is not None and hasattr(transformer, "h"):
        return transformer.h
    raise AttributeError(
        f"Unsupported model architecture for layer access: {type(model).__name__}. "
        "Expected `model.model.layers` or `model.model.decoder.layers`."
    )

def check_sparsity(model):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    layers = _get_model_layers(model)
    count = 0
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W==0).sum().item()
            total_params += W.numel()

            sub_count += (W==0).sum().item()
            sub_params += W.numel()

        print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

    model.config.use_cache = use_cache
    return float(count)/total_params

def check_sparsity_mask(mask):


    W = mask
    count = 0
    total_params = 0
    count += (W!=0).sum().item()
    total_params += W.numel()



    print(f" density {float(count)/total_params:.6f}")



def check_outlier(mask,threshold):


    W = mask
    count = 0
    total_params = 0

    max_shred=torch.max(W)*threshold
    count += (W>max_shred).sum().item()
    total_params += W.numel()



    outlier_ratio=float(count)/total_params*100

    return outlier_ratio


def check_outlier_mean(mask,threshold):


    W = mask
    count = 0
    total_params = 0

    max_shred=torch.mean(W)*threshold
    count += (W>max_shred).sum().item()
    total_params += W.numel()



    outlier_ratio=float(count)/total_params*100

    return outlier_ratio


def prepare_calibration_input(model, dataloader, device, nsamples):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    if hasattr(model, "hf_device_map") and "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype

    inps = torch.zeros((nsamples, model.seqlen, model.config.hidden_size),
                       dtype=dtype, device=device)
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            if cache['i'] < nsamples:     # 防止越界
                inps[cache['i']] = inp
                cache['i'] += 1
                cache['attention_mask'] = kwargs.get('attention_mask', None)
                cache['position_ids'] = kwargs.get('position_ids', None)
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        if cache['i'] >= nsamples:
            break
        try:
            model(batch[0].to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module

    # outs 不一定必须放 GPU（见下面第2点）
    outs = torch.zeros_like(inps)

    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    model.config.use_cache = use_cache
    return inps, outs, attention_mask, position_ids

def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    thres_cumsum = sum_before * alpha
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1,1))
    thres = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True)-1)
    W_mask = (W_metric <= thres)
    cur_sparsity = (W_mask==True).sum() / W_mask.numel()
    return W_mask, cur_sparsity

def prune_magnitude(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):

    print ("model",model)

    layers = model.model.layers

    print (layers)
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        for name in subset:
            W = subset[name].weight.data
            W_metric = torch.abs(W)
            if prune_n != 0:
                W_mask = (torch.zeros_like(W)==1)
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.numel()*args.sparsity_ratio)].cpu()
                W_mask = (W_metric<=thresh)

            W[W_mask] = 0








def prune_wanda_outlier_structure_special(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    ##### calucalte outlier ratio
    all_layer_ratio=[]
    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("loading calibdation data")
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples, seed=args.seed, seqlen=2048, tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        if "OPT" in model.__class__.__name__:
            print('Experiments with OPT models')
            inps, outs, attention_mask, position_ids = prepare_calibration_input_opt(model, dataloader, device)
        else:
            inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device, args.nsamples)

    if "opt" in args.model:
        layers=model.model.decoder.layers
    else:
        layers = model.model.layers

    for i in range(len(layers)):
        layer = layers[i]

        subset = find_layers(layer)
        if hasattr(model, "hf_device_map") and f"model.layers.{i}" in model.hf_device_map:   ## handle multi-GPU device map
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()
        layer_wmetric=[]

        for name in subset:
            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
            activation_data=torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
            # layer_wmetric.append(activation_data)


            layer_wmetric.append(W_metric)

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps

        layer_wmetric = torch.cat([torch.flatten(x.cpu()) for x in layer_wmetric])



        for out_ratio in [args.Hyper_m]:
            out_ratio_layer = check_outlier_mean(layer_wmetric, out_ratio)
            print ("layer outlier ratio",out_ratio,out_ratio_layer)

        all_layer_ratio.append(out_ratio_layer)

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()

    print ("before adjustment", all_layer_ratio)

    all_layer_ratio=np.array(all_layer_ratio)
    all_layer_ratio = ((all_layer_ratio - all_layer_ratio.min()) * (1/(all_layer_ratio.max() - all_layer_ratio.min()) * args.Lamda))
    all_layer_ratio=all_layer_ratio-np.mean(all_layer_ratio)


    all_layer_ratio=np.round(all_layer_ratio)





    for i in range(len(all_layer_ratio)):
        if all_layer_ratio[i] ==1.0:
            all_layer_ratio[i]=2.0

    all_layer_ratio=prune_n-all_layer_ratio




    print ("after adjustment", all_layer_ratio)
    ############## prune
    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("loading calibdation data")
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=2048,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        if "OPT" in model.__class__.__name__:
            inps, outs, attention_mask, position_ids = prepare_calibration_input_opt(model, dataloader, device)
        else:
            inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device, args.nsamples)

    # print ("inps",inps)
    if "opt" in args.model:
        layers=model.model.decoder.layers
    else:
        layers = model.model.layers
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        if hasattr(model, "hf_device_map") and f"model.layers.{i}" in model.hf_device_map:   ## handle multi-GPU device map
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        prune_n = int(all_layer_ratio [i])
        print('Layer {} prune_n {} prune_m {}'.format(i, prune_n, prune_m))

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in subset:
            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
            activation_data=torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
            layer_sparsity_ratio= 1-all_layer_ratio[i]
            if layer_sparsity_ratio<=0:
                layer_sparsity_ratio=0.01

            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
            if prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)

            # print ("W_mask",W_mask)
            subset[name].weight.data[W_mask] = 0  ## set weights to zero 

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()



def prune_wanda(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("loading calibdation data")
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=2048,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device, args.nsamples)



    print ("inps",inps)
    layers = model.model.layers
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        if hasattr(model, "hf_device_map") and f"model.layers.{i}" in model.hf_device_map:   ## handle multi-GPU device map
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in subset:
            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
            if prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)

                if args.use_variant:
                    # wanda variant 
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0., 0.8]
                    W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    while (torch.abs(cur_sparsity - args.sparsity_ratio)>0.001) and (alpha_hist[1]-alpha_hist[0]>=0.001):
                        if cur_sparsity > args.sparsity_ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha

                        alpha = alpha_new
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                else:
                    # unstructured pruning
                    indices = sort_res[1][:,:int(W_metric.shape[1]*args.sparsity_ratio)]
                    W_mask.scatter_(1, indices, True)
            #             print ("W_mask",W_mask)
            subset[name].weight.data[W_mask] = 0  ## set weights to zero 

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()







def prune_mag_outlier(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    ##### calucalte outlier ratio



    all_layer_ratio=[]
    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("loading calibdation data")
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=2048,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():

        if "OPT" in model.__class__.__name__:

            inps, outs, attention_mask, position_ids = prepare_calibration_input_opt(model, dataloader, device)
        else:

            inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device, args.nsamples)




    print ("inps",inps)
    if "opt" in args.model:
        layers=model.model.decoder.layers

    else:
        layers = model.model.layers


    for i in range(len(layers)):
        layer = layers[i]

        subset = find_layers(layer)

        if hasattr(model, "hf_device_map") and f"model.layers.{i}" in model.hf_device_map:   ## handle multi-GPU device map
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()


        layer_wmetric=[]

        for name in subset:





            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))


            activation_data=torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))





            layer_wmetric.append(W_metric)


        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps



        layer_wmetric = torch.cat([torch.flatten(x.cpu()) for x in layer_wmetric])

        for out_ratio in [args.Hyper_m]:

            out_ratio_layer=check_outlier_mean(layer_wmetric,out_ratio)
            print ("layer outlier ratio",out_ratio,out_ratio_layer)


        all_layer_ratio.append(out_ratio_layer)




    print ("before adjustment",all_layer_ratio)





    all_layer_ratio=np.array(all_layer_ratio)

    all_layer_ratio = ((all_layer_ratio - all_layer_ratio.min()) * (1/(all_layer_ratio.max() - all_layer_ratio.min()) * args.Lamda*2))

    all_layer_ratio=all_layer_ratio-np.mean(all_layer_ratio)+(1-args.sparsity_ratio)

    print (all_layer_ratio,np.mean(all_layer_ratio),np.max(all_layer_ratio),np.min(all_layer_ratio))






    print ("after adjustment",all_layer_ratio  )





    ############## prune


    if "opt" in args.model:
        layers=model.model.decoder.layers

    else:
        layers = model.model.layers

    print (layers)
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        for name in subset:

            layer_sparsity_ratio= 1-all_layer_ratio[i]
            W = subset[name].weight.data
            W_metric = torch.abs(W)
            if prune_n != 0:
                W_mask = (torch.zeros_like(W)==1)
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.numel()*layer_sparsity_ratio)].cpu()
                W_mask = (W_metric<=thresh)

            W[W_mask] = 0




def prune_wanda_outlier_structure(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    ##### calucalte outlier ratio
    all_layer_ratio=[]
    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("loading calibdation data")
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples, seed=args.seed, seqlen=2048, tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        if "OPT" in model.__class__.__name__:
            print('Experiments with OPT models')
            inps, outs, attention_mask, position_ids = prepare_calibration_input_opt(model, dataloader, device)
        else:
            inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device, args.nsamples)


    if "opt" in args.model:
        layers=model.model.decoder.layers
    else:
        layers = model.model.layers

    for i in range(len(layers)):
        layer = layers[i]

        subset = find_layers(layer)
        if hasattr(model, "hf_device_map") and f"model.layers.{i}" in model.hf_device_map:   ## handle multi-GPU device map
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()
        layer_wmetric=[]

        for name in subset:
            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
            activation_data=torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
            # layer_wmetric.append(activation_data)


            layer_wmetric.append(W_metric)

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps

        layer_wmetric = torch.cat([torch.flatten(x.cpu()) for x in layer_wmetric])

        for out_ratio in [args.Hyper_m]:
            out_ratio_layer = check_outlier_mean(layer_wmetric, out_ratio)
            print ("layer outlier ratio",out_ratio,out_ratio_layer)

        all_layer_ratio.append(out_ratio_layer)

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()

    print ("before adjustment", all_layer_ratio)

    all_layer_ratio=np.array(all_layer_ratio)
    all_layer_ratio = ((all_layer_ratio - all_layer_ratio.min()) * (1/(all_layer_ratio.max() - all_layer_ratio.min()) * args.Lamda))
    all_layer_ratio=all_layer_ratio-np.mean(all_layer_ratio)

    all_layer_ratio=np.round(all_layer_ratio)


    all_layer_ratio=prune_n-all_layer_ratio


    print ("after adjustment", all_layer_ratio)
    ############## prune
    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("loading calibdation data")
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=2048,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        if "OPT" in model.__class__.__name__:
            inps, outs, attention_mask, position_ids = prepare_calibration_input_opt(model, dataloader, device)
        else:
            inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device, args.nsamples)


    # print ("inps",inps)
    if "opt" in args.model:
        layers=model.model.decoder.layers
    else:
        layers = model.model.layers
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        if hasattr(model, "hf_device_map") and f"model.layers.{i}" in model.hf_device_map:   ## handle multi-GPU device map
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        prune_n = int(all_layer_ratio [i])
        print('Layer {} prune_n {} prune_m {}'.format(i, prune_n, prune_m))

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in subset:
            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
            activation_data=torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
            layer_sparsity_ratio= 1-all_layer_ratio[i]
            if layer_sparsity_ratio<=0:
                layer_sparsity_ratio=0.01

            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
            if prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)

            # print ("W_mask",W_mask)
            subset[name].weight.data[W_mask] = 0  ## set weights to zero 

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()

def prune_wanda_outlier(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):


    ##### calucalte outlier ratio
    all_layer_ratio=[]
    use_cache = model.config.use_cache
    model.config.use_cache = False
    hf_device_map = getattr(model, "hf_device_map", None)

    print("loading calibdation data")
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=2048,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():

        if "OPT" in model.__class__.__name__:

            inps, outs, attention_mask, position_ids = prepare_calibration_input_opt(model, dataloader, device)
        else:

            inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device, args.nsamples)




    print ("inps",inps)
    if "opt" in args.model:
        layers=model.model.decoder.layers

    else:
        layers = model.model.layers


    for i in range(len(layers)):
        layer = layers[i]

        subset = find_layers(layer)

        if isinstance(hf_device_map, dict) and f"model.layers.{i}" in hf_device_map:   ## handle multi-GPU device map
            dev = hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()


        layer_wmetric=[]

        for name in subset:





            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))


            activation_data=torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
            layer_wmetric.append(W_metric)


        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps





        layer_wmetric = torch.cat([torch.flatten(x.cpu()) for x in layer_wmetric])

        for out_ratio in [args.Hyper_m]:

            out_ratio_layer=check_outlier_mean(layer_wmetric,out_ratio)
            print ("layer outlier ratio",out_ratio,out_ratio_layer)


        all_layer_ratio.append(out_ratio_layer)




    print ("before adjustment",all_layer_ratio)





    all_layer_ratio=np.array(all_layer_ratio)

    all_layer_ratio = ((all_layer_ratio - all_layer_ratio.min()) * (1/(all_layer_ratio.max() - all_layer_ratio.min()) * args.Lamda*2))

    all_layer_ratio=all_layer_ratio-np.mean(all_layer_ratio)+(1-args.sparsity_ratio)

    print (all_layer_ratio,np.mean(all_layer_ratio),np.max(all_layer_ratio),np.min(all_layer_ratio))






    print ("after adjustment",all_layer_ratio  )



    model.config.use_cache = use_cache
    torch.cuda.empty_cache()
    ############## prune





    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("loading calibdation data")
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=2048,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():

        if "OPT" in model.__class__.__name__:

            inps, outs, attention_mask, position_ids = prepare_calibration_input_opt(model, dataloader, device)
        else:

            inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device, args.nsamples)




    print ("inps",inps)
    if "opt" in args.model:
        layers=model.model.decoder.layers

    else:
        layers = model.model.layers


    for i in range(len(layers)):
        layer = layers[i]

        subset = find_layers(layer)

        if hasattr(model, "hf_device_map") and f"model.layers.{i}" in model.hf_device_map:   ## handle multi-GPU device map
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()




        for name in subset:


            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))


            activation_data=torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

            layer_sparsity_ratio= 1-all_layer_ratio[i]


            if layer_sparsity_ratio<=0:
                layer_sparsity_ratio=0.01

            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
            if prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)

                if args.use_variant:
                    # wanda variant 
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0., 0.8]
                    W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    while (torch.abs(cur_sparsity - layer_sparsity_ratio)>0.001) and (alpha_hist[1]-alpha_hist[0]>=0.001):
                        if cur_sparsity > layer_sparsity_ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha

                        alpha = alpha_new
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                else:
                    # unstructured pruning
                    indices = sort_res[1][:,:int(W_metric.shape[1]*layer_sparsity_ratio)]
                    W_mask.scatter_(1, indices, True)
            #             print ("W_mask",W_mask)
            subset[name].weight.data[W_mask] = 0  ## set weights to zero 

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps





    model.config.use_cache = use_cache
    torch.cuda.empty_cache()





@torch.no_grad()
def prune_sparsegpt(args, model, tokenizer, dev, prune_n=0, prune_m=0):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    print('Starting ...')
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=2048,tokenizer=tokenizer)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers
    hf_device_map = getattr(model, "hf_device_map", None)

    if isinstance(hf_device_map, dict):
        for k in (
            "model.embed_tokens",
            "model.decoder.embed_tokens",
            "model.model.embed_tokens",
            "model.model.decoder.embed_tokens",
        ):
            if k in hf_device_map:
                dev = hf_device_map[k]
                break

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')

    for i in range(len(layers)):
        layer = layers[i]
        if isinstance(hf_device_map, dict) and f"model.layers.{i}" in hf_device_map:
            dev = hf_device_map[f"model.layers.{i}"]
            print(f"layer {i} device {dev}")
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = SparseGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')

            gpts[name].fasterprune(args.sparsity_ratio, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
            gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()






@torch.no_grad()
def prune_sparsegpt_outlier(args, model, tokenizer, dev, prune_n=0, prune_m=0):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    ##### calucalte outlier ratio

    device=dev

    all_layer_ratio=[]
    use_cache = model.config.use_cache
    model.config.use_cache = False
    hf_device_map = getattr(model, "hf_device_map", None)

    print("loading calibdation data")
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=2048,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():

        if "OPT" in model.__class__.__name__:

            inps, outs, attention_mask, position_ids = prepare_calibration_input_opt(model, dataloader, device)
        else:

            inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device, args.nsamples)




    print ("inps",inps)
    if "opt" in args.model:
        layers=model.model.decoder.layers

    else:
        layers = model.model.layers


    for i in range(len(layers)):
        layer = layers[i]

        subset = find_layers(layer)

        if isinstance(hf_device_map, dict) and f"model.layers.{i}" in hf_device_map:   ## handle multi-GPU device map
            dev = hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()


        layer_wmetric=[]

        for name in subset:





            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))


            activation_data=torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

            layer_wmetric.append(W_metric)



        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps





        layer_wmetric = torch.cat([torch.flatten(x.cpu()) for x in layer_wmetric])

        for out_ratio in [args.Hyper_m]:

            out_ratio_layer=check_outlier_mean(layer_wmetric,out_ratio)
            print ("layer outlier ratio",out_ratio,out_ratio_layer)


        all_layer_ratio.append(out_ratio_layer)





    print ("before adjustment",all_layer_ratio)





    all_layer_ratio=np.array(all_layer_ratio)

    all_layer_ratio = ((all_layer_ratio - all_layer_ratio.min()) * (1/(all_layer_ratio.max() - all_layer_ratio.min()) * args.Lamda*2))

    all_layer_ratio=all_layer_ratio-np.mean(all_layer_ratio)+(1-args.sparsity_ratio)

    print (all_layer_ratio,np.mean(all_layer_ratio),np.max(all_layer_ratio),np.min(all_layer_ratio))








    print ("after adjustment",all_layer_ratio  )





    model.config.use_cache = use_cache
    torch.cuda.empty_cache()
    ############## prune
    print('Starting ...')


    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=2048,tokenizer=tokenizer)





    use_cache = model.config.use_cache
    model.config.use_cache = False
    if "opt" in args.model:
        layers=model.model.decoder.layers

    else:
        layers = model.model.layers

    if isinstance(hf_device_map, dict):
        for k in (
            "model.embed_tokens",
            "model.decoder.embed_tokens",
            "model.model.embed_tokens",
            "model.model.decoder.embed_tokens",
        ):
            if k in hf_device_map:
                dev = hf_device_map[k]
                break

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')

    for i in range(len(layers)):



        layer_sparsity_ratio= 1-all_layer_ratio[i]


        if layer_sparsity_ratio<=0:
            layer_sparsity_ratio=0.01


        layer = layers[i]
        if isinstance(hf_device_map, dict) and f"model.layers.{i}" in hf_device_map:
            dev = hf_device_map[f"model.layers.{i}"]
            print(f"layer {i} device {dev}")
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)
            # inps, outs, attention_mask = inps.to(dev), outs.to(dev), attention_mask.to(dev)

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = SparseGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            # outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')

            gpts[name].fasterprune(layer_sparsity_ratio, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
            gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            # outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        layers[i] = layer
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()




@torch.no_grad()
def prune_wanda_outlier_plus(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    """
    OWL-style layerwise sparsity allocation + Wanda pruning,
    with your modification:
        D_hat[i] = D[i] + R_norm[i]
    where R[i] measures propagation risk: perturb/prune layer i slightly and measure drift in next k layers.
    """

    # -------------------------
    # Pass 1: compute D (outlier ratio) and R (propagation risk)
    # -------------------------
    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("loading calibration data")
    dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed, seqlen=2048, tokenizer=tokenizer)
    print("dataset loading complete")

    with torch.no_grad():
        if "OPT" in model.__class__.__name__:
            inps, outs, attention_mask, position_ids = prepare_calibration_input_opt(model, dataloader, device)
        else:
            inps, outs, attention_mask, position_ids = prepare_calibration_input(
                model, dataloader, device, args.nsamples)

    layers = model.model.layers

    D_list = []
    R_raw = []

    # Hyperparameters for risk (you said only k is a "real" hyperparam; others can be fixed/default)
    risk_k = getattr(args, "risk_k", 4)
    risk_probe = getattr(args, "risk_probe", 0.1)   # can be fixed if you want
    risk_decay = getattr(args, "risk_decay", 0.9)    # can be fixed if you want
    risk_nsamples = getattr(args, "risk_nsamples", min(args.nsamples, 16))  # to keep runtime manageable

    is_opt = ("OPT" in model.__class__.__name__)

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        # Handle multi-GPU sharding
        if hasattr(model, "hf_device_map") and f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps = inps.to(dev)
            outs = outs.to(dev)
            attention_mask = attention_mask.to(dev)
            position_ids = position_ids.to(dev)

        # 1) Collect scaler_row stats via hooks (WrappedGPT) and also compute outs for this layer once.
        wrapped_layers = {name: WrappedGPT(subset[name]) for name in subset}

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            if is_opt:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
            else:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        for h in handles:
            h.remove()

        # 2) Compute D_i (outlier ratio) from Wanda metric A = |W| * sqrt(scaler_row)
        layer_wmetric = []
        for name in subset:
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))
            layer_wmetric.append(W_metric)

        layer_wmetric = torch.cat([torch.flatten(x.cpu()) for x in layer_wmetric])
        out_ratio_layer = check_outlier_mean(layer_wmetric, args.Hyper_m)   # this is D_i
        D_list.append(float(out_ratio_layer))

        # 3) Compute R_i (propagation risk) BEFORE swapping inps/outs
        #    Use the current inps[j] which is the true input to layer i.
        j_start = i + 1
        j_end = min(len(layers) - 1, i + risk_k)
        j_end = _same_device_end(model, i, j_end)  # avoid cross-device forward

        if j_start > j_end:
            R_i = 0.0
        else:
            risk_sum, w_sum = 0.0, 0.0
            # To reduce cost, only use first `risk_nsamples` calibration samples
            for j in range(min(risk_nsamples, args.nsamples)):
                x0 = inps[j].unsqueeze(0)

                base_outs = _forward_range(
                    layers, i, j_end, x0,
                    attention_mask=attention_mask, position_ids=position_ids, is_opt=is_opt
                )
                with _perturb_layer_magnitude(layers[i], probe_ratio=risk_probe):
                    pert_outs = _forward_range(
                        layers, i, j_end, x0,
                        attention_mask=attention_mask, position_ids=position_ids, is_opt=is_opt
                    )

                # base_outs[t] corresponds to layer (i+t)
                # Compare downstream layers: i+1..j_end  => t=1..(j_end-i)
                for t in range(1, len(base_outs)):
                    w = (risk_decay ** (t - 1))
                    d = _rel_l2(base_outs[t], pert_outs[t])
                    risk_sum += w * d
                    w_sum += w

            R_i = risk_sum / max(w_sum, 1e-12)

        R_raw.append(float(R_i))

        # 4) Now swap to feed next layer (outs computed above is correct for unpruned model)
        inps, outs = outs, inps

    # Normalize R to [0,1] and form D_hat
    D = np.array(D_list, dtype=np.float32)
    R = np.array(R_raw, dtype=np.float32)
    if R.max() - R.min() > 1e-12:
        Rn = (R - R.min()) / (R.max() - R.min())
    else:
        Rn = np.zeros_like(R)
    D_hat = D + Rn

    # Map (D_hat) -> per-layer density (keep ratio) using your existing OWL mapping
    # (This keeps the rest of OWL unchanged.)
    all_layer_ratio = D_hat.copy()
    all_layer_ratio = ((all_layer_ratio - all_layer_ratio.min()) *
                       (1.0 / (all_layer_ratio.max() - all_layer_ratio.min() + 1e-12) * args.Lamda * 2))
    all_layer_ratio = all_layer_ratio - np.mean(all_layer_ratio) + (1 - args.sparsity_ratio)

    print("after adjustment", all_layer_ratio, "mean", np.mean(all_layer_ratio),
          "max", np.max(all_layer_ratio), "min", np.min(all_layer_ratio))

    # -------------------------
    # Pass 2: actual pruning using per-layer sparsity = 1 - density
    # -------------------------
    # Rebuild calibration inputs again (common practice, and avoids device/mutation surprises)
    model.config.use_cache = False
    with torch.no_grad():
        if "OPT" in model.__class__.__name__:
            inps, outs, attention_mask, position_ids = prepare_calibration_input_opt(model, dataloader, device)
        else:
            inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device, args.nsamples)

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        # multi-GPU
        if hasattr(model, "hf_device_map") and f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps = inps.to(dev)
            outs = outs.to(dev)
            attention_mask = attention_mask.to(dev)
            position_ids = position_ids.to(dev)

        # Collect scaler_row for pruning metric (before pruning)
        wrapped_layers = {name: WrappedGPT(subset[name]) for name in subset}

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            if is_opt:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
            else:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        for h in handles:
            h.remove()

        # Now do pruning for each linear in this layer
        for name in subset:
            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))

            # per-layer sparsity
            layer_sparsity_ratio = 1.0 - float(all_layer_ratio[i])
            if layer_sparsity_ratio <= 0:
                layer_sparsity_ratio = 0.01

            W_mask = (torch.zeros_like(W_metric) == 1)  # all False

            # structured N:M (optional)
            if prune_n != 0:
                # Example N:M mask per row, group size prune_m, prune_n zeros
                W_metric_reshape = W_metric.view(W_metric.shape[0], -1, prune_m)
                _, idx = torch.topk(W_metric_reshape, prune_n, dim=2, largest=False)
                W_mask_reshape = torch.zeros_like(W_metric_reshape, dtype=torch.bool)
                W_mask_reshape.scatter_(2, idx, True)
                W_mask = W_mask_reshape.view_as(W_metric)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)

                if getattr(args, "use_variant", False):
                    # wanda variant (binary search alpha)
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0., 0.8]
                    W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)

                    while (torch.abs(torch.tensor(cur_sparsity) - layer_sparsity_ratio) > 0.001) and \
                            (alpha_hist[1] - alpha_hist[0] >= 0.001):
                        if cur_sparsity > layer_sparsity_ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha
                        alpha = alpha_new
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                else:
                    # standard wanda: prune smallest fraction per row
                    thresh_idx = int(W_metric.shape[1] * layer_sparsity_ratio)
                    if thresh_idx <= 0:
                        thresh_idx = 1
                    W_mask.scatter_(1, sort_res[1][:, :thresh_idx], True)

            # apply mask
            subset[name].weight.data[W_mask] = 0

        # After pruning this layer, recompute outs with pruned weights to feed next layer
        for j in range(args.nsamples):
            if is_opt:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
            else:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()
