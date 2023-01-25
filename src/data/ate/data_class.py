from typing import NamedTuple, Optional

import numpy as np
import torch
from sklearn.model_selection import train_test_split


class PVTrainDataSet(NamedTuple):
    treatment: np.ndarray
    treatment_proxy: np.ndarray
    outcome_proxy: np.ndarray
    outcome: np.ndarray
    backdoor: Optional[np.ndarray]

    def fill_backdoor(self):
        if self.backdoor is None:
            backdoor = np.zeros((len(self.treatment), 0))
            return PVTrainDataSet(treatment=self.treatment,
                                    treatment_proxy=self.treatment_proxy,
                                    outcome_proxy=self.outcome_proxy,
                                    backdoor=backdoor,
                                    outcome=self.outcome)
        else:
            return(self)


class PVTestDataSet(NamedTuple):
    treatment: np.ndarray
    structural: Optional[np.ndarray]
    
    def fill_structural(self):
        if self.structural is None:
            structural = np.zeros((len(self.treatment), 0))
            return PVTestDataSet(treatment=self.treatment,
                                    structral=structural)
        else:
            return(self)

class RHCTestDataSet(NamedTuple):
    treatment: np.ndarray
    outcome_proxy: np.ndarray
    backdoor: np.ndarray

    def fill_backdoor(self):
        if self.backdoor is None:
            backdoor = np.zeros((len(self.treatment), 0))
            return RHCTestDataSet(treatment=self.treatment,
                                    outcome_proxy=self.outcome_proxy,
                                    backdoor=backdoor)
        else:
            return(self)

class PVTrainDataSetTorch(NamedTuple):
    treatment: torch.Tensor
    treatment_proxy: torch.Tensor
    outcome_proxy: torch.Tensor
    outcome: torch.Tensor
    backdoor: Optional[torch.Tensor]

    @classmethod
    def from_numpy(cls, train_data: PVTrainDataSet):
        if train_data.backdoor is None:
            backdoor = torch.zeros((len(train_data.treatment), 0), dtype=torch.float32)
        else:
            backdoor = torch.tensor(train_data.backdoor, dtype=torch.float32)
        return PVTrainDataSetTorch(treatment=torch.tensor(train_data.treatment, dtype=torch.float32),
                                    treatment_proxy=torch.tensor(train_data.treatment_proxy, dtype=torch.float32),
                                    outcome_proxy=torch.tensor(train_data.outcome_proxy, dtype=torch.float32),
                                    backdoor=backdoor,
                                    outcome=torch.tensor(train_data.outcome, dtype=torch.float32))

    def to_gpu(self, device_name = "cpu"):
        if self.backdoor is None:
            backdoor = torch.zeros((len(self.treatment), 0), dtype=torch.float32)
        else: backdoor = self.backdoor
        if device_name == "cpu":
            return PVTrainDataSetTorch(treatment=self.treatment,
                                        treatment_proxy=self.treatment_proxy,
                                        outcome_proxy=self.outcome_proxy,
                                        backdoor=backdoor,
                                        outcome=self.outcome)
        else:
            return PVTrainDataSetTorch(treatment=self.treatment.to(device_name),
                                        treatment_proxy=self.treatment_proxy.to(device_name),
                                        outcome_proxy=self.outcome_proxy.to(device_name),
                                        backdoor=backdoor.to(device_name),
                                        outcome=self.outcome.to(device_name))


class PVTestDataSetTorch(NamedTuple):
    treatment: torch.Tensor
    structural: Optional[torch.Tensor]

    @classmethod
    def from_numpy(cls, test_data: PVTestDataSet):
        if test_data.structural is None:
            structural = torch.zeros((len(test_data.treatment), 0), dtype=torch.float32)
        else:
            structural = torch.tensor(test_data.structural, dtype=torch.float32)
        return PVTestDataSetTorch(treatment=torch.tensor(test_data.treatment, dtype=torch.float32),
                                    structural=structural)

    def to_gpu(self, device_name = "cpu"):
        if self.structural is None:
                structural = torch.zeros((len(self.treatment), 0), dtype=torch.float32)
        else:
            structural = self.structural
        
        if device_name == "cpu":
            return PVTestDataSetTorch(treatment=self.treatment, structural=structural)
        else:
            return PVTestDataSetTorch(treatment=self.treatment.to(device_name),
                                        structural=structural.to(device_name))
            


class RHCTestDataSetTorch(NamedTuple):
    treatment: torch.Tensor
    outcome_proxy: torch.Tensor
    backdoor: torch.Tensor

    @classmethod
    def from_numpy(cls, test_data: RHCTestDataSet):
        if test_data.backdoor is None:
            backdoor = torch.zeros((len(test_data.treatment), 0), dtype=torch.float32)
        else: backdoor = torch.tensor(test_data.backdoor, dtype=torch.float32)

        return RHCTestDataSetTorch(treatment=torch.tensor(test_data.treatment, dtype=torch.float32),
                                   outcome_proxy=torch.tensor(test_data.outcome_proxy, dtype=torch.float32),
                                   backdoor=backdoor)

    def to_gpu(self, device_name = "cpu"):
        if self.backdoor is None:
            backdoor = torch.zeros((len(self.treatment), 0), dtype=torch.float32)
        else: backdoor = self.backdoor
        if device_name == "cpu":
            return RHCTestDataSetTorch(treatment=self.treatment,
                                        outcome_proxy=self.outcome_proxy,
                                        backdoor=backdoor)
        else:
            return RHCTestDataSetTorch(treatment=self.treatment.to(device_name),
                                        outcome_proxy=self.outcome_proxy.to(device_name),
                                        backdoor=backdoor.to(device_name))


def split_train_data(train_data: PVTrainDataSet, split_ratio=0.5):
    if split_ratio < 0.0:
        return train_data, train_data

    n_data = train_data[0].shape[0]
    idx_train_1st, idx_train_2nd = train_test_split(np.arange(n_data), train_size=split_ratio)

    def get_data(data, idx):
        return data[idx] if data is not None else None

    train_1st_data = PVTrainDataSet(*[get_data(data, idx_train_1st) for data in train_data])
    train_2nd_data = PVTrainDataSet(*[get_data(data, idx_train_2nd) for data in train_data])
    return train_1st_data, train_2nd_data
