from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from collections import OrderedDict
# 加载原始和微调后模型
orig_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
org_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)

ft_model = AutoModelForCausalLM.from_pretrained("./dpo-emoji-final", trust_remote_code=True)
ft_tokenizer = AutoTokenizer.from_pretrained("./dpo-emoji-final", trust_remote_code=True)

def qa(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors='pt')
    outputs = model.generate(**inputs, max_new_tokens=50, eos_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(outputs[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)

test_questions = [
    "帮我写一条新年祝福。",
    "你好，可以帮我推荐一本书吗？",
    "今天心情不好，能安慰一下我吗？"
]

for q in test_questions:
    print(f"用户提问: {q}")
    print("原始模型回复:", qa(orig_model, org_tokenizer, q))
    print("DPO增强模型回复:", qa(ft_model, ft_tokenizer, q))
    print("="*50)