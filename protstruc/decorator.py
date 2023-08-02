import torch
import numpy as np


def with_tensor(func):
    def wrapper(*args, **kwargs):
        found_tensor = False

        new_args = []
        for arg in args:
            if isinstance(arg, np.ndarray):
                t = torch.tensor(arg)
                if arg.dtype in [np.float32, np.float64]:
                    t = t.float()
                new_args.append(t)
            elif isinstance(arg, torch.Tensor):
                found_tensor = True
                new_args.append(arg)
            else:
                new_args.append(arg)

        new_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, np.ndarray):
                t = torch.tensor(value)
                if arg.dtype in [np.float32, np.float64]:
                    t = t.float()
                new_kwargs[key] = t
            elif isinstance(value, torch.Tensor):
                found_tensor = True
                new_kwargs[key] = value
            else:
                new_kwargs[key] = value

        out = func(*new_args, **new_kwargs)

        # if at least one torch.tensor is in the input,
        # return the output in torch.tensor type
        if found_tensor:
            return out

        # if there are no tensors in the input but are only numpy arrays,
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
