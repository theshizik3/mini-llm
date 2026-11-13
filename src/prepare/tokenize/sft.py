from datasets import Dataset, Features, Sequence, Value
from transformers import AutoTokenizer

import duckdb


FEATURES = Features({
    "input_ids":      Sequence(Value("int32")),
    "assistant_mask": Sequence(Value("bool")),
})

def run_tokenize_sft(cfg):    
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer)
    it = duckdb.sql("SELECT conversation FROM read_parquet(?)", params=[cfg.input_dir+"/**/*.parquet"]).fetchall()
    
    convs = []

    for row in it:
        messages = []
        for m in row[0]:
            role = m.get("role")
            content = m.get("content")
            messages.append({"role": role, "content": content})
        enc = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_assistant_tokens_mask=True,
            add_generation_prompt=False,
        )
        convs.append({
            "input_ids": enc["input_ids"],
            "assistant_mask": enc["assistant_masks"],
        })
    ds = Dataset.from_list(convs, features=FEATURES)
    ds.save_to_disk(cfg.output_dir, max_shard_size=cfg.shard_size)
