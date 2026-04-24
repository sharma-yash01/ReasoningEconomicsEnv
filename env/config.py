"""Environment configuration (EnvConfig) for v2 post-training RL environment."""

import os
import warnings
from dataclasses import dataclass, replace
from typing import Optional


# Keep the server settings adjustable from shell exports on Lambda.
# Blank values mean "use the normal default".
def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean for {name}: {raw!r}")


def _env_int(name: str, default: int | None) -> int | None:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def _env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip()
    return value or default


@dataclass
class EnvConfig:
    """Fully configurable episode and environment parameters."""

    # If True, run full experiment settings. If False, run baby test settings.
    prod: bool = False
    # Baby-test subset window on MetaMathQA: [subset_start_idx, subset_start_idx + subset_size)
    subset_start_idx: int = 0
    subset_size: int = 500
    # Same indexing for NuminaMath-TIR baby runs
    numina_subset_start_idx: int = 0
    numina_subset_size: int = 500

    # Budget policy: hard-cap preserves existing clipping/early-stop behavior.
    hard_cap_mode: bool = True
    # Soft-budget controls (used when hard_cap_mode=False).
    soft_allow_negative_budget: bool = True
    soft_overspend_penalty: float = 0.25

    num_questions: int = 10
    total_budget: Optional[int] = None
    budget_ratio: float = 2.0
    min_tokens: int = 10
    max_tokens: int = 800
    max_tokens_per_step: int = 2048
    tokenizer_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    beta: float = 0.05
    gamma: float = 0.1
    lambda_ep: float = 0.5
    target_utilization: float = 0.9
    seed: Optional[int] = None

    @classmethod
    def from_env(cls):
        """Build config from REASON_BUDGET_* env vars.

        This let us switch between 4 question smokes, 10 question runs, hard
        budgets, and soft budgets without editing this file on the VM.
        """
        return cls(
            prod=_env_bool("REASON_BUDGET_PROD", cls.prod),
            subset_start_idx=_env_int("REASON_BUDGET_SUBSET_START_IDX", cls.subset_start_idx),
            subset_size=_env_int("REASON_BUDGET_SUBSET_SIZE", cls.subset_size),
            numina_subset_start_idx=_env_int(
                "REASON_BUDGET_NUMINA_SUBSET_START_IDX",
                cls.numina_subset_start_idx,
            ),
            numina_subset_size=_env_int(
                "REASON_BUDGET_NUMINA_SUBSET_SIZE",
                cls.numina_subset_size,
            ),
            hard_cap_mode=_env_bool("REASON_BUDGET_HARD_CAP_MODE", cls.hard_cap_mode),
            soft_allow_negative_budget=_env_bool(
                "REASON_BUDGET_SOFT_ALLOW_NEGATIVE_BUDGET",
                cls.soft_allow_negative_budget,
            ),
            soft_overspend_penalty=_env_float(
                "REASON_BUDGET_SOFT_OVERSPEND_PENALTY",
                cls.soft_overspend_penalty,
            ),
            num_questions=_env_int("REASON_BUDGET_NUM_QUESTIONS", cls.num_questions),
            total_budget=_env_int("REASON_BUDGET_TOTAL_BUDGET", cls.total_budget),
            budget_ratio=_env_float("REASON_BUDGET_BUDGET_RATIO", cls.budget_ratio),
            min_tokens=_env_int("REASON_BUDGET_MIN_TOKENS", cls.min_tokens),
            max_tokens=_env_int("REASON_BUDGET_MAX_TOKENS", cls.max_tokens),
            max_tokens_per_step=_env_int(
                "REASON_BUDGET_MAX_TOKENS_PER_STEP",
                cls.max_tokens_per_step,
            ),
            tokenizer_name=_env_str("REASON_BUDGET_TOKENIZER_NAME", cls.tokenizer_name),
            beta=_env_float("REASON_BUDGET_BETA", cls.beta),
            gamma=_env_float("REASON_BUDGET_GAMMA", cls.gamma),
            lambda_ep=_env_float("REASON_BUDGET_LAMBDA_EP", cls.lambda_ep),
            target_utilization=_env_float(
                "REASON_BUDGET_TARGET_UTILIZATION",
                cls.target_utilization,
            ),
            seed=_env_int("REASON_BUDGET_SEED", cls.seed),
        )

    def get_total_budget(self) -> int:
        """Compute total_budget from budget_ratio if not set.

        NOTE: This formula uses min_tokens/max_tokens as abstract units. It does
        NOT tokenize any content with ``tokenizer_name``. When a policy tokenizer
        is active, prefer ``ReasonBudgetEnvironment._compute_tokenizer_native_budget``
        (called automatically on ``reset`` when ``tokenizer_name`` is provided).
        """
        if self.total_budget is not None:
            return self.total_budget
        avg_tokens = (self.min_tokens + self.max_tokens) / 2.0
        derived = int(self.budget_ratio * self.num_questions * avg_tokens)
        if self.tokenizer_name and self.tokenizer_name != "Qwen/Qwen2.5-0.5B-Instruct":
            warnings.warn(
                f"EnvConfig.tokenizer_name is set to {self.tokenizer_name!r} but "
                f"total_budget is derived from min_tokens/max_tokens ({derived} abstract "
                "units), not from the tokenizer. The budget cap and per-step token "
                "counts may be in different unit systems. Pass total_budget explicitly "
                "or use tokenizer_name on reset() for tokenizer-native budgets.",
                UserWarning,
                stacklevel=2,
            )
        return derived


def env_config_for_server() -> EnvConfig:
    """Defaults for new OpenEnv sessions, with optional Docker/deploy overrides.

    ``REE_DEFAULT_TOKENIZER_NAME``: Hugging Face model id used when the client
    does not send ``tokenizer_name`` on reset (post-training clients should send it).
    """
    cfg = EnvConfig.from_env()
    tok = os.environ.get("REE_DEFAULT_TOKENIZER_NAME", "").strip()
    if tok:
        cfg = replace(cfg, tokenizer_name=tok)
    prod = os.environ.get("REE_PROD", "").strip().lower()
    if prod in ("1", "true", "yes"):
        cfg = replace(cfg, prod=True)
    return cfg
