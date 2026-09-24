import os

import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer

from transformer import Transformer

if __name__ == "__main__":
    load_dotenv()
    HF_ACCESS_TOKEN = os.getenv("HF_ACCESS_TOKEN")

    tokenizer = AutoTokenizer.from_pretrained("google/mt5-small", token=HF_ACCESS_TOKEN)

    vocab_size = len(tokenizer)
    pad_token_id = tokenizer.pad_token_id
    d_model = 128
    head_num = 8
    max_len = 64

    special_tokens = {}
    if tokenizer.pad_token_id is None:
        special_tokens["pad_token"] = "<|pad|>"
    if tokenizer.bos_token_id is None:
        special_tokens["bos_token"] = "<|bos|>"
    if tokenizer.eos_token_id is None:
        special_tokens["eos_token"] = "<|eos|>"

    tokenizer.add_special_tokens(special_tokens)
    tokenizer.padding_side = "right"

    model = Transformer(vocab_size, d_model, head_num, max_len)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    criterion = torch.nn.CrossEntropyLoss()

    # train

    # used for inference
    batch_input_texts = [
        "Sphinx of black quartz, judge my vow.",
        "Pack my box with five dozen liquor jugs.",
        "How vexingly quick daft zebras jump!",
    ]
