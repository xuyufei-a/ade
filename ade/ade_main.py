import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rome.layer_stats import layer_stats
from util import nethook
from util.generate import generate_fast
from util.globals import *

from memit.memit_main import get_context_templates
from ade_hparams import ADEHyperParams

# Cache variable(s)
CONTEXT_TEMPLATES_CACHE = None
COV_CACHE = {}

def apply_ade_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: ADEHyperParams,
    copy=False,
    return_orig_weights=False,
    **kwargs: Any,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    """
    Returns a model with the desired changes.
    :param copy: If true, will preserve the original model while creating a new one to edit.
        Note that you are responsible for deallocating the new model's memory to avoid leaks.
    :return: (1) the updated model, (2) the weights that changed
    """

    weights_copy = {}
    if copy:
        model = deepcopy(model)

    deltas = execute_ade(model, tok, requests, hparams)

    with torch.no_grad():
        for w_name, upd_matrix in deltas.items():
            w = nethook.get_parameter(model, w_name)
            if return_orig_weights and w_name not in weights_copy:
                weights_copy[w_name] = w.detach().clone()

            w[...] += upd_matrix

    print(f"New weights successfully inserted into {list(deltas.keys())}")

    return model, weights_copy


def execute_ade(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: ADEHyperParams,
    **kwargs: Any,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the FT update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """

    # Update target and print info
    requests = deepcopy(requests)
    for request in requests:
        if request["target_new"]["str"][0] != " ":
            # Space required for correct tokenization
            request["target_new"]["str"] = " " + request["target_new"]["str"]
        print(
            f"Executing ADE algo for: "
            f"[{request['prompt'].format(request['subject'])}] -> [{request['target_new']['str']}]"
        )

    # Retrieve weights that user desires to change
    weights = {
        n: p
        for n, p in model.named_parameters()
        for layer in hparams.layers
        if hparams.rewrite_module_tmp.format(layer) in n
    }
    # Save old weights for future restoration
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}
    print(f"Weights to be updated: {list(weights.keys())}")

    # Define inputs
    if hparams.use_context:
        context_templates = get_context_templates(model, tok)
        texts = [
            context.format(r["prompt"]).format(r["subject"]) for r in requests
            for context_types in context_templates
            for context in context_types    
        ]
        targets = [
            r["target_new"]["str"] for r in requests
            for context_types in context_templates
            for context in context_types    
        ]
    else:
        texts = [r["prompt"].format(r["subject"]) for r in requests]
        targets = [r["target_new"]["str"] for r in requests]

    # Configure optimizer / gradients
    wd = (
        hparams.weight_decay
        if not isinstance(hparams.wd_power_law, tuple)
        else (len(requests) ** hparams.wd_power_law[0])
        * np.exp(hparams.wd_power_law[1])
    )
    print(f"Using weight decay of {wd} for {len(requests)} edits")
    opt = torch.optim.SGD(
        [v for _, v in weights.items()],
        lr=hparams.lr,
        weight_decay=wd,
    )
    for name, w in model.named_parameters():
        w.requires_grad = name in weights
    
    # set weight projections to U[:, :r]
    # use P ^ T W to project W to subspace
    weight_projs = {
        torch.linalg.svd(weight)[0] if hparams.ade_rank is None
        else torch.linalg.svd(weight)[0][:, :hparams.ade_rank]
        for name, weight in weights.item()
    }

    # Update loop: intervene at layers simultaneously
    loss_meter = AverageMeter()
    for it in range(hparams.num_steps):
        print(20 * "=")
        print(f"Epoch: {it}")
        print(20 * "=")
        loss_meter.reset()

        for txt, tgt in zip(
            chunks(texts, hparams.batch_size), chunks(targets, hparams.batch_size)
        ):
            inputs = tok(txt, return_tensors="pt", padding=True).to("cuda")
            target_ids = tok(tgt, return_tensors="pt", padding=True)["input_ids"].to(
                "cuda"
            )
            last_token_inds = inputs["attention_mask"].sum(dim=1) - 1
            loss_mask = target_ids != tok.unk_token_id

            opt.zero_grad()
            bs = inputs["input_ids"].shape[0]
            probs = torch.nn.functional.log_softmax(
                model(**inputs).logits[torch.arange(bs), last_token_inds], dim=-1
            )
            loss = -(torch.gather(probs, 1, target_ids) * loss_mask).sum(
                1
            ) / loss_mask.sum(1)
            loss = loss.mean()
            print(f"Batch loss {loss.item()}")
            loss_meter.update(loss.item(), n=bs)

            # TODO: why neglect small loss value
            if loss.item() >= 1e-2:
                loss.backward()

                # update weights in ade way
                update_gradient_by_ade(loss, weights, weight_projs, hparams)

                opt.step()

            if type(hparams.norm_constraint) is float:
                eps = hparams.norm_constraint
                with torch.no_grad():
                    for k, v in weights.items():
                        v[...] = torch.clamp(
                            v, min=weights_copy[k] - eps, max=weights_copy[k] + eps
                        )

        print(f"Total loss {loss_meter.avg}")

        if loss_meter.avg < 1e-2:
            break

    deltas = {k: (weights[k] - weights_copy[k]).detach() for k in weights}

    # Restore state of original model
    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]

    print(f"Deltas successfully computed for {list(weights.keys())}")

    return deltas

def update_gradient_by_ade(loss, weights, weight_projs, hparams: ADEHyperParams):
    if hparams.ade_method != "galore":
        raise Exception(f"unemplemented ade_method {hparams.ade_method}")

    restrain_method = RESTRAIN_METHOD_DICT[hparams.param_select]
    for name, weight in weights.items():
        weight.grad = weight_projs[name] @ restrain_method(weight_projs[name].T @ weight.grad, hparams, weight=weight)

def valilla_param_select(weight_grad, hparams: ADEHyperParams, **kwargs):
    return weight_grad

def topk_param_select(weight_grad, hparams: ADEHyperParams, **kwargs):
    weight = kwargs.pop("weight")
    if hparams.param_select_metric == "change":
        eps = hparams.div_eps
        score = weight_grad.abs() / (weight.abs() + eps)
    elif hparams.param_select_metric == "salience":
        score = weight_grad.abs() * weight.abs()
    elif hparams.param_select_metric == "valilla":
        score = weight_grad.clone().abs()
    else:
        # TODO: maybe try pca
        raise Exception(f"unimplemented param select metric {hparams.param_select_metric}")

    shape = weight_grad.shape
    flatten_weight_grad = weight_grad.flatten()
    flatten_score = score.flatten()

    num_zeros = int(flatten_score.numel() * hparams.zero_ratio)
    zero_pos = np.argpartition(flatten_score.cpu().numpy(), num_zeros)[:num_zeros]

    flatten_weight_grad[zero_pos] = 0
    return flatten_weight_grad.reshape(shape)

def randomk_param_select(weight_grad, hparams: ADEHyperParams, **kwargs):
    shape = weight_grad.shape
    flatten_weight_grad = weight_grad.flatten()

    num_zeros = int(flatten_weight_grad.numel() * hparams.zero_ratio)
    zero_pos = torch.randperm(int(flatten_weight_grad.numel()))[:num_zeros]

    flatten_weight_grad[zero_pos] = 0
    return flatten_weight_grad.reshape(shape)

def direction_param_select(weight_grad, hparams: ADEHyperParams, **kwargs):
    weight = kwargs.pop("weight")
    if hparams.param_select_metric == "change":
        eps = hparams.div_eps
        score = weight_grad.abs() / (weight.abs() + eps)
    elif hparams.param_select_metric == "salience":
        score = weight_grad.abs() * weight.abs()
    elif hparams.param_select_metric == "valilla":
        score = weight_grad.clone().abs()
    else:
        # TODO: maybe try pca
        raise Exception(f"unimplemented param select metric {hparams.param_select_metric}")

    shape = weight_grad.shape
    num_zeros = int(shape[0] * hparams.zero_ratio)

    score = torch.sum(score, dim=1, keepdim=False)
    zero_pos = np.argpartition(score.cpu().numpy(), num_zeros)[:num_zeros]

    weight_grad[zero_pos, :] = 0
    return weight_grad


RESTRAIN_METHOD_DICT = {
    "valilla": valilla_param_select, 
    "topk": topk_param_select,
    "randomk": randomk_param_select,
    "direction": direction_param_select,
}

def chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    chunk = []
    for a in arr:
        chunk.append(a)
        if len(chunk) == n:
            yield chunk
            chunk = []
    if len(chunk) > 0:
        yield chunk


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    ## check param select method

    # hparams = ADEHyperParams(param_select='valilla', param_select_metric='salience')

    # a = torch.tensor([[0, 2, 3], [4, 5, 6]])
    # a.grad = a.clone() - 1
    
    # print(valilla_param_select(a.grad, hparams, weight=a))
    # print(topk_param_select(a.grad.clone(), hparams, weight=a.clone()))
    # print(randomk_param_select(a.grad.clone(), hparams, weight=a.clone()))
    # print(direction_param_select(a.grad.clone(), hparams, weight=a.clone()))