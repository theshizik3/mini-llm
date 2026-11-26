from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

from src.tokenizer.tokens import *
from src.tokenizer.tokenizer import build_tokenizer
from src.tokenizer.trainer import build_trainer

import duckdb


def make_iterator(cfg):
    rel = duckdb.sql("SELECT text FROM read_parquet(?)", params=[cfg.dataset_path+"/**/*.parquet"])
    length = 0

    for batch in rel.fetch_record_batch(cfg.batch_size):
        texts = batch.column("text").to_pylist()
        for text in texts:
            yield text
            length += len(text)
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
        single=f"{BOS_TOKEN} $A {EOS_TOKEN}",
        pair=f"{BOS_TOKEN} $A {EOS_TOKEN} $B:1 {EOS_TOKEN}:1",
        special_tokens=[
            (BOS_TOKEN, tokenizer.token_to_id(BOS_TOKEN)),
            (EOS_TOKEN, tokenizer.token_to_id(EOS_TOKEN)),
        ],
    )

    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token=UNK_TOKEN,
        pad_token=PAD_TOKEN,
        eos_token=EOS_TOKEN,
        bos_token=BOS_TOKEN,
        additional_special_tokens=ADDITIONAL_SPECIAL_TOKENS,
        chat_template=CHAT_TEMPLATE,
    )
    fast.save_pretrained(cfg.path)
