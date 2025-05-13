
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import setup_chat_format

prompt = "Write a haiku about programming"

# Format with template
messages = [{"role": "user", "content": prompt}]


def hub_generate():
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
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)

    outputs = model.generate(**inputs, max_new_tokens=100)
    print("Before training:")
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


def local_generate():
    # Dynamically set the device
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    model_name = "./sft_output/checkpoint-1000"
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_name,
        local_files_only=True,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)

    # model, tokenizer = setup_chat_format(model=model, tokenizer=tokenizer)
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)

    outputs = model.generate(**inputs, max_new_tokens=100)
    print("After training:")
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

hub_generate()
local_generate()