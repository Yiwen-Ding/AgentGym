import random

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


chat_template_mapping = {
    "llama2": {
        "bos": "",
        "human": "<s>[INST] {value} [/INST]",
        "gpt": " {value}</s>"
    },
    "llama3": {
        "bos": "<|begin_of_text|>",
        "human": "<|start_header_id|>user<|end_header_id|>\n\n{value}<|eot_id|>",
        "gpt": "<|start_header_id|>assistant<|end_header_id|>\n\n{value}<|eot_id|>"
    },
    "deepseek": {
        "bos": "<｜begin▁of▁sentence｜>",
        "human": "User: {value}\n\n",
        "gpt": "Assistant: {value}<｜end▁of▁sentence｜>"
    }
}