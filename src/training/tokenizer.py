from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

from src.config import TrainConfig, ModelConfig
from src.datasets.pretraining import *
from src.tokenizer import *


def make_iterator(cfg: TrainConfig):
    ds = MultiParquetDataset(
        CulturaX(cfg.hf_token, 2),
        TaigaProza(cfg.hf_token, 4),
        Wikipedia(cfg.hf_token, 10),
        TaigaStihi(cfg.hf_token, 2),
        Kolmogorov(cfg.hf_token),
        MathD0rj(cfg.hf_token),
        ZeroAgency(cfg.hf_token),
        shuffle=False
    )
    length = 0
    for shard in ds:
        length += len(shard)
        yield shard
        print(f"{(length/cfg.tokenizer_len):02}")
        if length > cfg.tokenizer_len:
            return

def train():
    train_cfg = TrainConfig()
    model_cfg = ModelConfig()

    tokenizer = build_tokenizer()
    trainer = build_trainer(model_cfg.vocab_size)

    tokenizer.train_from_iterator(
        iterator=make_iterator(train_cfg),
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
    fast.save_pretrained(train_cfg.tokenizer)
