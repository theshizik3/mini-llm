from src.datasets.sft.sft_dataset import SFTDataset


class Ru0x7o(SFTDataset):
    repo = "0x7o/ru-instruction"
    columns = ("conversation",)

    def handler(self, row):
        conv = row[0]
        messages = []
        for m in conv:
            role = m.get("role")
            content = m.get("content")
            if not isinstance(content, str) or not content.strip():
                continue
            if role not in ("system", "user", "assistant"):
                continue
            messages.append({"role": role, "content": content})
        return messages
