from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset

dataset = load_dataset("json", data_files={"train": "../data/dpo_emoji_english_withgpt.jsonl"})["train"]
model_name = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

# 加载本地数据


# DPO配置
dpo_config = DPOConfig(
    beta=0.1,
    learning_rate=1e-5,
    per_device_train_batch_size=4,
    num_train_epochs=1,
    logging_steps=5,
    gradient_accumulation_steps=2,
    max_length=512,
)

training_args = TrainingArguments(
    per_device_train_batch_size=4,
    num_train_epochs=1,
    learning_rate=1e-5,
    logging_steps=5,
    output_dir="./dpo-emoji-checkpoint",
    save_strategy="epoch",
    fp16=True,
    report_to=[]
)

trainer = DPOTrainer(
    model=model,
    ref_model=model_name,   # 用基线模型当参考模型
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    max_length=512,
    dpo_config=dpo_config,
)

trainer.train()
# 保存模型
trainer.save_model("./dpo-emoji-final")
# 保存tokenizer
tokenizer.save_pretrained("./dpo-emoji-final")