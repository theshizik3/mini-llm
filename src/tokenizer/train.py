from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

from src.tokenizer.tokens import EOS_TOKEN, UNK_TOKEN, CHAT_TEMPLATE, PAD_TOKEN, ADDITIONAL_SPECIAL_TOKENS
from src.tokenizer.tokenizer import build_tokenizer
from src.tokenizer.trainer import build_trainer

import duckdb


def make_iterator(cfg):
    cur = duckdb.sql("SELECT text FROM read_parquet(?)", params=[cfg.dataset_path+"/**/*.parquet"])
    length = 0
    offset = 0
    while True:
        batch = cur.offset(offset).limit(cfg.batch_size).fetchall()
        if not batch:
            break
        offset += len(batch)
        for row in batch:
            yield row[0]
            length += len(row[0])
            if length > cfg.train_seq_len:
                return

def run_train_tokenizer(cfg):
    tokenizer = build_tokenizer()
    trainer = build_trainer(cfg.vocab)

    tokenizer.train_from_iterator(
        iterator=make_iterator(cfg),
        trainer=trainer,
    )

    tokenizer.post_processor = TemplateProcessing(
        single=f"$A {EOS_TOKEN}",
        pair=f"$A {EOS_TOKEN} $B:1 {EOS_TOKEN}:1",
        special_tokens=[
            (EOS_TOKEN, tokenizer.token_to_id(EOS_TOKEN)),
        ],
    )

    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token=UNK_TOKEN,
        pad_token=PAD_TOKEN,
        eos_token=EOS_TOKEN,
        bos_token=EOS_TOKEN,
        additional_special_tokens=ADDITIONAL_SPECIAL_TOKENS,
        chat_template=CHAT_TEMPLATE,
    )
    fast.save_pretrained(cfg.path)
