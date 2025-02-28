import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

# Paths to canonical (closest to L0 100) for each layer
GEMMA_SCOPE_16K_CANONICAL_PATHS = {
    "layer_0/width_16k/canonical": "layer_0/width_16k/average_l0_105",
    "layer_1/width_16k/canonical": "layer_1/width_16k/average_l0_102",
    "layer_2/width_16k/canonical": "layer_2/width_16k/average_l0_141",
    "layer_3/width_16k/canonical": "layer_3/width_16k/average_l0_59",
    "layer_4/width_16k/canonical": "layer_4/width_16k/average_l0_124",
    "layer_5/width_16k/canonical": "layer_5/width_16k/average_l0_68",
    "layer_6/width_16k/canonical": "layer_6/width_16k/average_l0_70",
    "layer_7/width_16k/canonical": "layer_7/width_16k/average_l0_69",
    "layer_8/width_16k/canonical": "layer_8/width_16k/average_l0_71",
    "layer_9/width_16k/canonical": "layer_9/width_16k/average_l0_73",
    "layer_10/width_16k/canonical": "layer_10/width_16k/average_l0_77",
    "layer_11/width_16k/canonical": "layer_11/width_16k/average_l0_80",
    "layer_12/width_16k/canonical": "layer_12/width_16k/average_l0_82",
    "layer_13/width_16k/canonical": "layer_13/width_16k/average_l0_84",
    "layer_14/width_16k/canonical": "layer_14/width_16k/average_l0_84",
    "layer_15/width_16k/canonical": "layer_15/width_16k/average_l0_78",
    "layer_16/width_16k/canonical": "layer_16/width_16k/average_l0_78",
    "layer_17/width_16k/canonical": "layer_17/width_16k/average_l0_77",
    "layer_18/width_16k/canonical": "layer_18/width_16k/average_l0_74",
    "layer_19/width_16k/canonical": "layer_19/width_16k/average_l0_73",
    "layer_20/width_16k/canonical": "layer_20/width_16k/average_l0_71",
    "layer_21/width_16k/canonical": "layer_21/width_16k/average_l0_70",
    "layer_22/width_16k/canonical": "layer_22/width_16k/average_l0_72",
    "layer_23/width_16k/canonical": "layer_23/width_16k/average_l0_75",
    "layer_24/width_16k/canonical": "layer_24/width_16k/average_l0_73",
    "layer_25/width_16k/canonical": "layer_25/width_16k/average_l0_116",
}


class JumpReLUSAE(nn.Module):
    def __init__(self, d_model, d_sae):
        super().__init__()
        self.W_enc = nn.Parameter(torch.zeros(d_model, d_sae))
        self.W_dec = nn.Parameter(torch.zeros(d_sae, d_model))
        self.threshold = nn.Parameter(torch.zeros(d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_model))

        self.ablation_indices = []

    def encode(self, input_acts):
        pre_acts = input_acts @ self.W_enc + self.b_enc
        mask = pre_acts > self.threshold

        acts = mask * torch.nn.functional.relu(pre_acts)
        return acts

    def decode(self, acts):
        return acts @ self.W_dec + self.b_dec

    def forward(self, acts):
        acts = self.encode(acts)
        recon = self.decode(acts)
        return recon

    @classmethod
    def from_pretrained(cls, layer: int):
        file_name = GEMMA_SCOPE_16K_CANONICAL_PATHS[
            f"layer_{layer}/width_16k/canonical"
        ]
        repo_id = "google/gemma-scope-2b-pt-res"

        path_to_params = hf_hub_download(
            repo_id=repo_id,
            filename=file_name + "/params.npz",
        )

        params = np.load(path_to_params)
        pt_params = {k: torch.from_numpy(v) for k, v in params.items()}
        model = cls(params["W_enc"].shape[0], params["W_enc"].shape[1])
        model.load_state_dict(pt_params)
        return model
