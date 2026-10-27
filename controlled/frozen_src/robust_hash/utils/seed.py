import os
import random

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """
    Set random seed for python / numpy / torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    # 为了结果更稳一点
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False