import argparse

from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from trl import (
    SFTConfig,
    SFTTrainer,
    setup_chat_format,
)


def main():
    device = ( "cuda")
    model_name = "HuggingFaceTB/SmolLM2-135M"
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_name
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)

    if tokenizer.chat_template is None:
        model, tokenizer = setup_chat_format(model, tokenizer, format="chatml")

    dataset = load_dataset(path="HuggingFaceTB/smoltalk", name="everyday-conversations")

    sft_config = SFTConfig(
        output_dir="./sft_output",
        max_steps=1000,  # Adjust based on dataset size and desired training duration
        per_device_train_batch_size=4,  # Set according to your GPU memory capacity
        learning_rate=5e-5,  # Common starting point for fine-tuning
        logging_steps=10,  # Frequency of logging training metrics
        save_steps=100,  # Frequency of saving model checkpoints
        evaluation_strategy="steps",  # Evaluate the model at regular intervals
        eval_steps=50,  # Frequency of evaluation
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer,
    )

    trainer.train()





if __name__ == "__main__":
    main()
