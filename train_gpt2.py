from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling

# Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# ✅ Set pad token
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

# Load dataset
dataset = load_dataset("json", data_files="dataset.jsonl", split="train")

# Tokenize
def tokenize(example):
    input_text = f"###Input: {example['input']}\n###Output: {example['output']}"
    return tokenizer(input_text, truncation=True, max_length=512)


tokenized_dataset = dataset.map(tokenize)

# Training setup
training_args = TrainingArguments(
    output_dir="./my_model",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=50,
    save_total_limit=1
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
)

trainer.train()

# Save model
model.save_pretrained("./my_model")
tokenizer.save_pretrained("./my_model")
