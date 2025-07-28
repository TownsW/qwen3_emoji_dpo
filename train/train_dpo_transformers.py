import os
import sys
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import DPOTrainer, DPOConfig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--train_file", type=str, required=True, help="Path to the jsonl training data file.")
    parser.add_argument("--output_dir", type=str, default="./dpo-emoji-checkpoint")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--fp16", action='store_true')
    parser.add_argument("--bf16", action='store_true')
    parser.add_argument("--save_strategy", type=str, default="epoch")
    parser.add_argument("--report_to", type=str, default="", help="Reporting type for Huggingface (wandb, tensorboard, etc.), empty for none")
    parser.add_argument("--ref_model_name", type=str, default=None, help="Reference model, default is base model_name")
    args = parser.parse_args()

    print("==== Arguments ====")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print("===================")


    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True)

    # 数据集加载
    dataset = load_dataset("json", data_files={"train": args.train_file})["train"]

    dpo_config = DPOConfig(
        beta=args.beta,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        logging_steps=args.logging_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_length=args.max_length,
        fp16=args.fp16,
        bf16=args.bf16,
        report_to=args.report_to.split(',') if args.report_to else [],
        ddp_find_unused_parameters=False,  # DDP常用设置
        save_strategy = args.save_strategy,
        output_dir=args.output_dir,
    )




    # 参考模型可指定，默认为base
    ref_model = args.ref_model_name if args.ref_model_name else args.model_name

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer
    )

    trainer.train()
    trainer.save_model(os.path.join(args.output_dir, "final"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "final"))

if __name__ == "__main__":
    main()
