from dataclasses import dataclass, field

PROVIDER: str = "openai"


@dataclass
class EnvConfig:
    num_batches: int = 2_000
    minibatch_size: int = 150
    seed: int = 22

    batch_len: int = 128
    n_examples = 40

    l_ctx: int = 15
    r_ctx: int = 4


@dataclass
class ExplainerConfig:
    max_tokens : int = 2000
    temperature : float = 0.8

    batch_size: int = 8
    n_batches : int = 2
    runs_per_batch: int = 2

    activation_threshold: float = 0.2

    l = "<<"
    r = ">>"


@dataclass
class CondenserConfig:
    max_tokens : int = 1000
    temperature : float = 0.0


@dataclass
class DetectionScorerConfig:
    max_tokens: int = 1000
    temperature: float = 0.0

    n_batches: int = 1

    real_ids : list = field(default_factory=lambda: [0, 2, 5, 6, 9, 10, 11, 12, 18, 19])
    @property
    def n_real(self):
        return len(self.real_ids)

    @property
    def batch_size(self):
        return 2 * self.n_real
    

@dataclass
class GenerationScorerConfig:
    n_examples : int = 10
    temperature : float = 0.7