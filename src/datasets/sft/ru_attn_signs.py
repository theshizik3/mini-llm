from src.datasets.sft.sft_dataset import SFTDataset


class RuAttnSigns(SFTDataset):
    repo = "attn-signs/ru-instructions"
    columns = ("prompt", "answer")

    def handler(self, row):
        prompt, answer = row
        if not (isinstance(prompt, str) and prompt.strip() and
                isinstance(answer, str) and answer.strip()):
            return None
        return [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ]