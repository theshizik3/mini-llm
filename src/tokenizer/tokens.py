CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "<|im_start|>system\n{{ message['content'] }}<|im_end|>\n"
    "{% elif message['role'] == 'user' %}"
    "<|im_start|>user\n{{ message['content'] }}<|im_end|>\n"
    "{% elif message['role'] == 'assistant' %}"
    "<|im_start|>assistant\n"
    "{{ message['content'] or '' }}<|im_end|>\n"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)

UNK_TOKEN = "<|unk|>"
PAD_TOKEN = "<|padding|>"
EOS_TOKEN = "<|endoftext|>"

ADDITIONAL_SPECIAL_TOKENS = [
    "<|im_start|>",
    "<|im_end|>",
]

SPECIAL_TOKENS = [UNK_TOKEN, PAD_TOKEN, EOS_TOKEN] + ADDITIONAL_SPECIAL_TOKENS

CYRILLIC = [chr(c) for c in range(0x0410, 0x044F + 1)]
CYRILLIC_EXTRA = list("ЁёІіЇїЄєҐґЪъЫыЬьЭэЮюЯя")
LATIN = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
DIGITS = list("0123456789")
PUNCT = list(" .,:;!?()[]{}<>\"'`~@#$%^&*+-=/_\\|")
WHITESPACE = [" ", "\n", "\t"]

ALPHABET = list(dict.fromkeys(
    CYRILLIC + CYRILLIC_EXTRA + LATIN + DIGITS + PUNCT + WHITESPACE
))
