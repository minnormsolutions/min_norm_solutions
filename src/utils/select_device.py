# Function to determine optimal available device (accelerator).

import torch

def select_device(device_name:str = None):
    """Determines optimal available device (accelerator)."""

    # Builds prioritized list of available device types (cuda > mps > cpu).

    device_list = []

    if torch.cuda.is_available():
        device_list.append("cuda")
    # if torch.backends.mps.is_available():
    #     device_list.append("mps")
    device_list.append("cpu")

    # Uses requested device if available, otherwise uses highest priority
    # available device (cuda > mps > cpu).

    if device_name not in device_list:
        device_name = device_list[0]

    # As of 1.13 MPS performs poorly due to an incomplete implementation,
    # so even if MPS is detected, another device is used instead.

    if device_name == "mps":
        mps_index = device_list.index("mps")
        if mps_index > 0:
            device_name = device_list[ mps_index - 1 ]
        else:
            device_name = device_list[ mps_index + 1 ]
    
    return device_name