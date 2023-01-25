import numpy as np
from numpy.random import default_rng
from typing import Any, Dict

from src.data.ate.data_class import PVTestDataSet, PVTrainDataSet

def psi(t: np.ndarray) -> np.ndarray:
    return 2 * ((t - 5) ** 4 / 600 + np.exp(-4 * (t - 5) ** 2) + t / 10 - 2)


def generate_demand_core(n_sample: int, rng, W_noise: float = 1, Z_noise: float = 1):
    demand = rng.uniform(0, 10, n_sample)
    cost1 = 2 * np.sin(demand * np.pi * 2 / 10) + rng.normal(0, Z_noise, n_sample)
    cost2 = 2 * np.cos(demand * np.pi * 2 / 10) + rng.normal(0, Z_noise, n_sample)
    price = 35 + (cost1 + 3) * psi(demand) + cost2 + rng.normal(0, 1.0, n_sample)
    views = 7 * psi(demand) + 45 + rng.normal(0, W_noise, n_sample)
    outcome = cal_outcome(price, views, demand)

    return demand, cost1, cost2, price, views, outcome


def generate_train_demand_pv(n_sample: int, Z_noise: float = 1, W_noise: float = 1,
                                n_extra_W: int = 0, n_extra_Z: int = 0,
                                seed=42, **kwargs):
    rng = default_rng(seed=seed)
    demand, cost1, cost2, price, views, outcome = generate_demand_core(n_sample, rng, W_noise, Z_noise)
    outcome = (outcome + rng.normal(0, 1.0, n_sample)).astype(float)
    views = np.concatenate( ( views[:, np.newaxis],
                np.random.normal( 0, W_noise, (n_sample, n_extra_W) ) ),
                axis = 1 )
    cost = np.concatenate( (np.c_[cost1, cost2],
                np.random.normal( 0, Z_noise, (n_sample, n_extra_Z) ) ),
                axis = 1 )
    return PVTrainDataSet(treatment=price[:, np.newaxis],
                          treatment_proxy=cost,
                          outcome_proxy=views,
                          outcome=outcome[:, np.newaxis],
                          backdoor=None)


def cal_outcome(price, views, demand):
    return np.clip(np.exp((views - price) / 10.0), None, 5.0) * price - 5 * psi(demand)

def cal_structural(p: float, n_sample: int = 10**4, W_noise: float = 1):
    rng = default_rng(seed=42)
    demand = rng.uniform(0, 10.0, n_sample)
    views = 7 * psi(demand) + 45 + rng.normal(0, W_noise, n_sample)
    outcome = cal_outcome(p, views, demand)
    return np.mean(outcome)

def generate_test_demand_pv(data_config: Dict[str, Any],
        analysis_config: Dict[str, Any], n_sample: int = 10**4,
        W_noise: float = 1):
    W_noise = data_config.get("W_noise", 1)
    price = analysis_config["intervention_array"]
    structural = np.array([cal_structural(p, n_sample, W_noise) for p in price])
    return PVTestDataSet(structural=structural[:, np.newaxis],
                         treatment=price[:, np.newaxis])
