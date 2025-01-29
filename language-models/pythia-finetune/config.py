from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import get_device, set_all_seeds


class Config:
    def __init__(self, model_name):
        self.model_name = model_name

    def config(self) -> tuple[AutoModelForCausalLM, AutoTokenizer, list[int]]:
        device = get_device()
        model = AutoModelForCausalLM.from_pretrained(self.model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        tokenizer.pad_token = (
            tokenizer.eos_token
        )  # Set padding token to eos token for CLM
        return model, tokenizer

    def forward(self):
        set_all_seeds(42, warn_only=True)
        model, tokenizer = self.config()
        return model, tokenizer
