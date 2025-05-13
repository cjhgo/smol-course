from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from trl import setup_chat_format
import torch

model_name = "HuggingFaceTB/SmolLM2-135M"

# Dynamically set the device
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

model_name = "HuggingFaceTB/SmolLM2-135M"
model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_name
).to(device)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)
model, tokenizer = setup_chat_format(model=model, tokenizer=tokenizer)

ds = load_dataset("HuggingFaceTB/smoltalk", "everyday-conversations")


def process_dataset(sample):
    # TODO: 🐢 Convert the sample into a chat format
    # use the tokenizer's method to apply the chat template
    sample['text'] = tokenizer.apply_chat_template(sample['messages'],
                                                   tokenize=False,
                                                   add_generation_prompt=False)
    return sample


import ipdb; ipdb.set_trace()
first = ds['train'][0]
print(first)
ds = ds.map(process_dataset)
print(first, ds['train'][0])



ds = load_dataset("openai/gsm8k", "main")


def process_dataset(sample):
    # TODO: 🐕 Convert the sample into a chat format
    # 1. create a message format with the role and content
    messages = [{"role": "user", "content": sample['question']}, {"role": "assistant", "content": sample['answer']}]
    # 2. apply the chat template to the samples using the tokenizer's method
    sample['text'] = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    return sample

first = ds['train'][0]

ds = ds.map(process_dataset)
print(first, ds['train'][0]['text'])