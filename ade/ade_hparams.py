from dataclasses import dataclass
from typing import List, Literal

from util.hparams import HyperParams


@dataclass
class ADEHyperParams(HyperParams):
# class ADEHyperParams():
    # # Method
    # layers: List[int]
    # layer_selection: Literal["all", "random"]
    # fact_token: Literal[
    #     "last", "subject_first", "subject_last", "subject_first_after_last"
    # ]
    # v_num_grad_steps: int
    # v_lr: float
    # v_loss_layer: int
    # v_weight_decay: float
    # clamp_norm_factor: float
    # kl_factor: float
    # mom2_adjustment: bool
    # mom2_update_weight: float

    # # Module templates
    # rewrite_module_tmp: str
    # layer_module_tmp: str
    # mlp_module_tmp: str
    # attn_module_tmp: str
    # ln_f_module: str
    # lm_head_module: str

    # # Statistics
    # mom2_dataset: str
    # mom2_n_samples: int
    # mom2_dtype: str

    use_context: bool

    # Ade param setting
    ade_method: Literal["galore", "lora"]
    ade_rand: int | None

    # Ade param select setting
    param_select: Literal["valilla", "topk", "randomk", "direction"]
    param_select_metric: Literal["change", "salience", "valilla"]
    div_eps: float = 1e-8
    zero_ratio: float = 0.8