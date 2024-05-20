from dataclasses import dataclass

@dataclass
class CacheConfig:
    num_batches: int = 2_250
    minibatch_size: int = 150
    seed: int = 22

    batch_len: int = 128
    n_examples = 20

    l_ctx: int = 15
    r_ctx: int = 4

    l: str = "<<"
    r: str = ">>"


    activation_threshold: float = 0.4