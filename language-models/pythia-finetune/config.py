from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import get_device, set_all_seeds


class Config:
    def __init__(self):
        self.model_name = "EleutherAI/pythia-70m"

    def config(self) -> tuple[AutoModelForCausalLM, AutoTokenizer, list[int]]:
        device = get_device()
        model = AutoModelForCausalLM.from_pretrained(self.model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return model, tokenizer

    def forward(self):
        set_all_seeds(42, warn_only=True)
        model, tokenizer = self.config()
        return model, tokenizer
