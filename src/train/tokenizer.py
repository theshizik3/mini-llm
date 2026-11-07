from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

from src.configs import TokenizerConfig
from src.readers import ColumnParquetReader, DirReader
from src.tokenizer import *


def make_iterator(cfg):
    ds = DirReader(
        ColumnParquetReader,
        dir=cfg.dataset_path,
        pat="*.parquet",
        column="text",
        batch_size=cfg.batch_size,
    )
    length = 0
    for s in ds:
        length += len(s)
        yield s
        if length > cfg.train_seq_len:
            return

def train():
    cfg = TokenizerConfig()

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
