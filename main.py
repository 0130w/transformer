import os

from dotenv import load_dotenv
from transformers import AutoTokenizer

if __name__ == "__main__":
    load_dotenv()
    HF_ACCESS_TOKEN = os.getenv("HF_ACCESS_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained("google/mt5-small", token=HF_ACCESS_TOKEN)
    batch_input_texts = [
        "Sphinx of black quartz, judge my vow.",
        "Pack my box with five dozen liquor jugs.",
        "How vexingly quick daft zebras jump!",
    ]
    pad_token_id = tokenizer.pad_token_id
    batch = tokenizer(
        batch_input_texts,
        padding=True,
        return_tensors="pt",
    )  # host on cpu
    print(batch)
