"""Most of this is just copied over from Arthur's code and slightly simplified:
https://github.com/ArthurConmy/sae/blob/main/sae/model.py
"""

import gzip
import os
import pickle
from typing import Any

import einops
import torch
from torch import nn
from transformer_lens.hook_points import HookedRootModule, HookPoint
from dataclasses import dataclass
from typing import Optional, cast
from abc import ABC

@dataclass
class RunnerConfig(ABC):
    """
    The config that's shared across all runners.
    """

    # Data Generating Function (Model + Training Distibuion)
    model_name: str = "gelu-2l"
    hook_point: str = "blocks.{layer}.hook_mlp_out"
    hook_point_layer: int = 0
    hook_point_head_index: Optional[int] = None
    dataset_path: str = "NeelNanda/c4-tokenized-2b"
    is_dataset_tokenized: bool = True
    context_size: int = 128
    use_cached_activations: bool = False
    cached_activations_path: Optional[str] = (
        None  # Defaults to "activations/{dataset}/{model}/{full_hook_name}_{hook_point_head_index}"
    )

    # SAE Parameters
    d_in: int = 512

    # Activation Store Parameters
    n_batches_in_buffer: int = 20
    total_training_tokens: int = 2_000_000
    store_batch_size: int = 32

    # Misc
    device: str | torch.device = "cpu"
    seed: int = 42
    dtype: torch.dtype = torch.float32

    def __post_init__(self):
        # Autofill cached_activations_path unless the user overrode it
        if self.cached_activations_path is None:
            self.cached_activations_path = f"activations/{self.dataset_path.replace('/', '_')}/{self.model_name.replace('/', '_')}/{self.hook_point}"
            if self.hook_point_head_index is not None:
                self.cached_activations_path += f"_{self.hook_point_head_index}"

@dataclass
class LanguageModelSAERunnerConfig(RunnerConfig):
    """
    Configuration for training a sparse autoencoder on a language model.
    """

    # SAE Parameters
    b_dec_init_method: str = "geometric_median"
    expansion_factor: int = 4
    from_pretrained_path: Optional[str] = None
    d_sae: Optional[int] = None

    # Training Parameters
    l1_coefficient: float = 1e-3
    lp_norm: float = 1
    lr: float = 3e-4
    lr_scheduler_name: str = (
        "constantwithwarmup"  # constant, constantwithwarmup, linearwarmupdecay, cosineannealing, cosineannealingwarmup
    )
    lr_warm_up_steps: int = 500
    train_batch_size: int = 4096

    # Resampling protocol args
    use_ghost_grads: bool = False  # want to change this to true on some timeline.
    feature_sampling_window: int = 2000
    dead_feature_window: int = 1000  # unless this window is larger feature sampling,

    dead_feature_threshold: float = 1e-8

    # WANDB
    log_to_wandb: bool = True
    wandb_project: str = "mats_sae_training_language_model"
    run_name: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_log_frequency: int = 10

    # Misc
    n_checkpoints: int = 0
    checkpoint_path: str = "checkpoints"

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.expansion_factor, list):
            self.d_sae = self.d_in * self.expansion_factor
        self.tokens_per_buffer = (
            self.train_batch_size * self.context_size * self.n_batches_in_buffer
        )

        if self.run_name is None:
            self.run_name = f"{self.d_sae}-L1-{self.l1_coefficient}-LR-{self.lr}-Tokens-{self.total_training_tokens:3.3e}"

        if self.b_dec_init_method not in ["geometric_median", "mean", "zeros"]:
            raise ValueError(
                f"b_dec_init_method must be geometric_median, mean, or zeros. Got {self.b_dec_init_method}"
            )
        if self.b_dec_init_method == "zeros":
            print(
                "Warning: We are initializing b_dec to zeros. This is probably not what you want."
            )

        self.device = torch.device(self.device)

        unique_id = cast(
            Any, wandb
        ).util.generate_id()  # not sure why this type is erroring
        self.checkpoint_path = f"{self.checkpoint_path}/{unique_id}"

        print(
            f"Run name: {self.d_sae}-L1-{self.l1_coefficient}-LR-{self.lr}-Tokens-{self.total_training_tokens:3.3e}"
        )
        # Print out some useful info:
        n_tokens_per_buffer = (
            self.store_batch_size * self.context_size * self.n_batches_in_buffer
        )
        print(f"n_tokens_per_buffer (millions): {n_tokens_per_buffer / 10 **6}")
        n_contexts_per_buffer = self.store_batch_size * self.n_batches_in_buffer
        print(
            f"Lower bound: n_contexts_per_buffer (millions): {n_contexts_per_buffer / 10 **6}"
        )

        total_training_steps = self.total_training_tokens // self.train_batch_size
        print(f"Total training steps: {total_training_steps}")

        total_wandb_updates = total_training_steps // self.wandb_log_frequency
        print(f"Total wandb updates: {total_wandb_updates}")

        # how many times will we sample dead neurons?
        # assert self.dead_feature_window <= self.feature_sampling_window, "dead_feature_window must be smaller than feature_sampling_window"
        n_feature_window_samples = total_training_steps // self.feature_sampling_window
        print(
            f"n_tokens_per_feature_sampling_window (millions): {(self.feature_sampling_window * self.context_size * self.train_batch_size) / 10 **6}"
        )
        print(
            f"n_tokens_per_dead_feature_window (millions): {(self.dead_feature_window * self.context_size * self.train_batch_size) / 10 **6}"
        )

        if self.use_ghost_grads:
            print("Using Ghost Grads.")

        print(
            f"We will reset the sparsity calculation {n_feature_window_samples} times."
        )
        # print("Number tokens in dead feature calculation window: ", self.dead_feature_window * self.train_batch_size)
        print(
            f"Number tokens in sparsity calculation window: {self.feature_sampling_window * self.train_batch_size:.2e}"
        )
        
class SparseAutoencoder(HookedRootModule):
    """ """

    def __init__(
        self,
        cfg: LanguageModelSAERunnerConfig,
    ):
        super().__init__()
        self.cfg = cfg
        self.d_in = cfg.d_in
        if not isinstance(self.d_in, int):
            raise ValueError(
                f"d_in must be an int but was {self.d_in=}; {type(self.d_in)=}"
            )
        assert cfg.d_sae is not None  # keep pyright happy
        self.d_sae = cfg.d_sae
        self.l1_coefficient = cfg.l1_coefficient
        self.lp_norm = cfg.lp_norm
        self.dtype = cfg.dtype
        self.device = cfg.device

        # NOTE: if using resampling neurons method, you must ensure that we initialise the weights in the order W_enc, b_enc, W_dec, b_dec
        self.W_enc = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(self.d_in, self.d_sae, dtype=self.dtype, device=self.device)
            )
        )
        self.b_enc = nn.Parameter(
            torch.zeros(self.d_sae, dtype=self.dtype, device=self.device)
        )

        self.W_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(self.d_sae, self.d_in, dtype=self.dtype, device=self.device)
            )
        )

        with torch.no_grad():
            # Anthropic normalize this to have unit columns
            self.W_dec.data /= torch.norm(self.W_dec.data, dim=1, keepdim=True)

        self.b_dec = nn.Parameter(
            torch.zeros(self.d_in, dtype=self.dtype, device=self.device)
        )

        self.hook_sae_in = HookPoint()
        self.hook_hidden_pre = HookPoint()
        self.hook_hidden_post = HookPoint()
        self.hook_sae_out = HookPoint()

        self.setup()  # Required for `HookedRootModule`s

    def forward(self, x: torch.Tensor, dead_neuron_mask: torch.Tensor | None = None):
        # move x to correct dtype
        x = x.to(self.dtype)
        sae_in = self.hook_sae_in(
            x - self.b_dec
        )  # Remove decoder bias as per Anthropic

        hidden_pre = self.hook_hidden_pre(
            einops.einsum(
                sae_in,
                self.W_enc,
                "... d_in, d_in d_sae -> ... d_sae",
            )
            + self.b_enc
        )
        feature_acts = self.hook_hidden_post(torch.nn.functional.relu(hidden_pre))

        sae_out = self.hook_sae_out(
            einops.einsum(
                feature_acts,
                self.W_dec,
                "... d_sae, d_sae d_in -> ... d_in",
            )
            + self.b_dec
        )

        # add config for whether l2 is normalized:
        x_centred = x - x.mean(dim=0, keepdim=True)
        mse_loss = (
            torch.pow((sae_out - x.float()), 2)
            / (x_centred**2).sum(dim=-1, keepdim=True).sqrt()
        )

        mse_loss_ghost_resid = torch.tensor(0.0, dtype=self.dtype, device=self.device)
        # gate on config and training so evals is not slowed down.
        if (
            self.cfg.use_ghost_grads
            and self.training
            and dead_neuron_mask is not None
            and dead_neuron_mask.sum() > 0
        ):
            # ghost protocol

            # 1.
            residual = x - sae_out
            residual_centred = residual - residual.mean(dim=0, keepdim=True)
            l2_norm_residual = torch.norm(residual, dim=-1)

            # 2.
            feature_acts_dead_neurons_only = torch.exp(hidden_pre[:, dead_neuron_mask])
            ghost_out = feature_acts_dead_neurons_only @ self.W_dec[dead_neuron_mask, :]
            l2_norm_ghost_out = torch.norm(ghost_out, dim=-1)
            norm_scaling_factor = l2_norm_residual / (1e-6 + l2_norm_ghost_out * 2)
            ghost_out = ghost_out * norm_scaling_factor[:, None].detach()

            # 3.
            mse_loss_ghost_resid = (
                torch.pow((ghost_out - residual.detach().float()), 2)
                / (residual_centred.detach() ** 2).sum(dim=-1, keepdim=True).sqrt()
            )
            mse_rescaling_factor = (mse_loss / (mse_loss_ghost_resid + 1e-6)).detach()
            mse_loss_ghost_resid = mse_rescaling_factor * mse_loss_ghost_resid

            mse_loss_ghost_resid = mse_loss_ghost_resid.mean()

        mse_loss = mse_loss.mean()
        sparsity = feature_acts.norm(p=self.lp_norm, dim=1).mean(dim=(0,))
        l1_loss = self.l1_coefficient * sparsity
        loss = mse_loss + l1_loss + mse_loss_ghost_resid

        return sae_out, feature_acts, loss, mse_loss, l1_loss, mse_loss_ghost_resid

    @torch.no_grad()
    def initialize_b_dec_with_precalculated(self, origin: torch.Tensor):
        out = torch.tensor(origin, dtype=self.dtype, device=self.device)
        self.b_dec.data = out

    @torch.no_grad()
    def initialize_b_dec(self, all_activations: torch.Tensor):
        if self.cfg.b_dec_init_method == "geometric_median":
            self.initialize_b_dec_with_geometric_median(all_activations)
        elif self.cfg.b_dec_init_method == "mean":
            self.initialize_b_dec_with_mean(all_activations)
        elif self.cfg.b_dec_init_method == "zeros":
            pass
        else:
            raise ValueError(
                f"Unexpected b_dec_init_method: {self.cfg.b_dec_init_method}"
            )

    @torch.no_grad()
    def initialize_b_dec_with_mean(self, all_activations: torch.Tensor):
        previous_b_dec = self.b_dec.clone().cpu()
        out = all_activations.mean(dim=0)

        previous_distances = torch.norm(all_activations - previous_b_dec, dim=-1)
        distances = torch.norm(all_activations - out, dim=-1)

        print("Reinitializing b_dec with mean of activations")
        print(
            f"Previous distances: {previous_distances.median(0).values.mean().item()}"
        )
        print(f"New distances: {distances.median(0).values.mean().item()}")

        self.b_dec.data = out.to(self.dtype).to(self.device)

    @torch.no_grad()
    def get_test_loss(self, batch_tokens: torch.Tensor, model: HookedRootModule):
        """
        A method for running the model with the SAE activations in order to return the loss.
        returns per token loss when activations are substituted in.
        """
        head_index = self.cfg.hook_point_head_index

        def standard_replacement_hook(activations: torch.Tensor, hook: Any):
            activations = self.forward(activations)[0].to(activations.dtype)
            return activations

        def head_replacement_hook(activations: torch.Tensor, hook: Any):
            new_actions = self.forward(activations[:, :, head_index])[0].to(
                activations.dtype
            )
            activations[:, :, head_index] = new_actions
            return activations

        replacement_hook = (
            standard_replacement_hook if head_index is None else head_replacement_hook
        )

        ce_loss_with_recons = model.run_with_hooks(
            batch_tokens,
            return_type="loss",
            fwd_hooks=[(self.cfg.hook_point, replacement_hook)],
        )

        return ce_loss_with_recons

    @torch.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        self.W_dec.data /= torch.norm(self.W_dec.data, dim=1, keepdim=True)

    @torch.no_grad()
    def remove_gradient_parallel_to_decoder_directions(self):
        """
        Update grads so that they remove the parallel component
            (d_sae, d_in) shape
        """

        parallel_component = einops.einsum(
            self.W_dec.grad,
            self.W_dec.data,
            "d_sae d_in, d_sae d_in -> d_sae",
        )
        assert parallel_component is not None  # keep pyright happy

        self.W_dec.grad -= einops.einsum(
            parallel_component,
            self.W_dec.data,
            "d_sae, d_sae d_in -> d_sae d_in",
        )

    def save_model(self, path: str):
        """
        Basic save function for the model. Saves the model's state_dict and the config used to train it.
        """

        # check if path exists
        folder = os.path.dirname(path)
        os.makedirs(folder, exist_ok=True)

        state_dict = {"cfg": self.cfg, "state_dict": self.state_dict()}

        if path.endswith(".pt"):
            torch.save(state_dict, path)
        elif path.endswith("pkl.gz"):
            with gzip.open(path, "wb") as f:
                pickle.dump(state_dict, f)
        else:
            raise ValueError(
                f"Unexpected file extension: {path}, supported extensions are .pt and .pkl.gz"
            )

        print(f"Saved model to {path}")

    @classmethod
    def load_from_pretrained(cls, path: str):
        """
        Load function for the model. Loads the model's state_dict and the config used to train it.
        This method can be called directly on the class, without needing an instance.
        """

        # Ensure the file exists
        if not os.path.isfile(path):
            raise FileNotFoundError(f"No file found at specified path: {path}")

        # Load the state dictionary
        if path.endswith(".pt"):
            try:
                if torch.backends.mps.is_available():
                    state_dict = torch.load(path, map_location="mps")
                    state_dict["cfg"].device = "mps"
                else:
                    state_dict = torch.load(path)
            except Exception as e:
                raise IOError(f"Error loading the state dictionary from .pt file: {e}")

        elif path.endswith(".pkl.gz"):
            try:
                with gzip.open(path, "rb") as f:
                    state_dict = pickle.load(f)
            except Exception as e:
                raise IOError(
                    f"Error loading the state dictionary from .pkl.gz file: {e}"
                )
        elif path.endswith(".pkl"):
            try:
                with open(path, "rb") as f:
                    state_dict = pickle.load(f)
            except Exception as e:
                raise IOError(f"Error loading the state dictionary from .pkl file: {e}")
        else:
            raise ValueError(
                f"Unexpected file extension: {path}, supported extensions are .pt, .pkl, and .pkl.gz"
            )

        # Ensure the loaded state contains both 'cfg' and 'state_dict'
        if "cfg" not in state_dict or "state_dict" not in state_dict:
            raise ValueError(
                "The loaded state dictionary must contain 'cfg' and 'state_dict' keys"
            )

        # Create an instance of the class using the loaded configuration
        instance = cls(cfg=state_dict["cfg"])
        instance.load_state_dict(state_dict["state_dict"])

        return instance

    def get_name(self):
        sae_name = f"sparse_autoencoder_{self.cfg.model_name}_{self.cfg.hook_point}_{self.cfg.d_sae}"
        return sae_name
