from tokenizers.trainers import BpeTrainer
from src.tokenizer.tokens import SPECIAL_TOKENS, ALPHABET


def build_trainer(vocab_size):
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=ALPHABET,
        show_progress=True,
    )
    return trainer
