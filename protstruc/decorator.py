import torch
import numpy as np


def with_tensor(func):
    def wrapper(*args, **kwargs):
        found_tensor = False

        new_args = []
        for arg in args:
            if isinstance(arg, np.ndarray):
                new_args.append(torch.tensor(arg))
            elif isinstance(arg, torch.Tensor):
                found_tensor = True
                new_args.append(arg)
            else:
                new_args.append(arg)

        new_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, np.ndarray):
                new_kwargs[key] = torch.tensor(value)
            elif isinstance(value, torch.Tensor):
                found_tensor = True
                new_kwargs[key] = value
            else:
                new_kwargs[key] = value

        out = func(*new_args, **new_kwargs)

        if found_tensor:
            return out

        # convert all tensors to numpy arrays
        if isinstance(out, tuple):
            return tuple([x.numpy() if isinstance(x, torch.Tensor) else x for x in out])
        elif isinstance(out, list):
            return list([x.numpy() if isinstance(x, torch.Tensor) else x for x in out])
        elif isinstance(out, dict):
            return {k: v.numpy() if isinstance(v, torch.Tensor) else v for k, v in out.items()}
        elif isinstance(out, torch.Tensor):
            return out.numpy()

    return wrapper
