from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import OrderedDict
import torch

"""
load original and dpo models
"""

# original model
org = AutoModelForCausalLM.from_pretrained("/nfs-151/disk6/townswu/pretrained_models/Qwen/Qwen3-0.6B", trust_remote_code=True, torch_dtype=torch.bfloat16).cuda(0)
org_tk = AutoTokenizer.from_pretrained("/nfs-151/disk6/townswu/pretrained_models/Qwen/Qwen3-0.6B", trust_remote_code=True)

# sharegpt_vicuna, simple insert emoji
ft_vc_simple = AutoModelForCausalLM.from_pretrained("/nfs-151/disk6/townswu/exp/output/save/DPO/Qwen/Qwen3_1_6B/emoji-checkpoint-via-transformers_sharegpt_simple/final", trust_remote_code=True, torch_dtype=torch.bfloat16).cuda(1)
ft_tk_vc_simple = AutoTokenizer.from_pretrained("/nfs-151/disk6/townswu/exp/output/save/DPO/Qwen/Qwen3_1_6B/emoji-checkpoint-via-transformers_sharegpt_simple/final", trust_remote_code=True)

# sharegpt_vicuna, strategically inserted emoji, correct healthy cheak and gpt refine
ft_vc_rf_gpt = AutoModelForCausalLM.from_pretrained("/nfs-151/disk6/townswu/exp/output/save/DPO/Qwen/Qwen3_1_6B/emoji-checkpoint-via-transformers_sharegpt_diverse_correct_gpt_refine/final", trust_remote_code=True, torch_dtype=torch.bfloat16).cuda(2)
ft_tk_vc_rf_gpt = AutoTokenizer.from_pretrained("/nfs-151/disk6/townswu/exp/output/save/DPO/Qwen/Qwen3_1_6B/emoji-checkpoint-via-transformers_sharegpt_diverse_correct_gpt_refine/final", trust_remote_code=True)

# alpaca-gpt4, simple insert emoji
ft_alpaca_simple = AutoModelForCausalLM.from_pretrained("/nfs-151/disk6/townswu/exp/output/save/DPO/Qwen/Qwen3_1_6B/emoji-checkpoint-via-transformers_alpaca_simple/final", trust_remote_code=True, torch_dtype=torch.bfloat16).cuda(3)
ft_tk_alpaca_simple = AutoTokenizer.from_pretrained("/nfs-151/disk6/townswu/exp/output/save/DPO/Qwen/Qwen3_1_6B/emoji-checkpoint-via-transformers_alpaca_simple/final", trust_remote_code=True)

# alpaca-gpt4, strategically inserted emoji, correct healthy cheak and gpt refine
ft_alpaca_rf_gpt = AutoModelForCausalLM.from_pretrained("/nfs-151/disk6/townswu/exp/output/save/DPO/Qwen/Qwen3_1_6B/emoji-checkpoint-via-transformers_alpaca_diverse_correct_gpt_refine/final", trust_remote_code=True, torch_dtype=torch.bfloat16).cuda(4)
ft_tk_alpaca_rf_gpt = AutoTokenizer.from_pretrained("/nfs-151/disk6/townswu/exp/output/save/DPO/Qwen/Qwen3_1_6B/emoji-checkpoint-via-transformers_alpaca_diverse_correct_gpt_refine/final", trust_remote_code=True)

# gpt generation data，directly generate conversation emoji data
ft_gpt = AutoModelForCausalLM.from_pretrained("/nfs-151/disk6/townswu/exp/output/save/DPO/Qwen/Qwen3_1_6B/emoji-checkpoint-via-transformers_gpt_v1/final", trust_remote_code=True, torch_dtype=torch.bfloat16).cuda(5)
ft_tk_gpt = AutoTokenizer.from_pretrained("/nfs-151/disk6/townswu/exp/output/save/DPO/Qwen/Qwen3_1_6B/emoji-checkpoint-via-transformers_gpt_v1/final", trust_remote_code=True)

# gpt generation data，for more task format
ft_mtask_gpt = AutoModelForCausalLM.from_pretrained("/nfs-151/disk6/townswu/exp/output/save/DPO/Qwen/Qwen3_1_6B/emoji-checkpoint-via-transformers_gpt_multi_task/final", trust_remote_code=True, torch_dtype=torch.bfloat16).cuda(6)
ft_tk_mtask_gpt = AutoTokenizer.from_pretrained("/nfs-151/disk6/townswu/exp/output/save/DPO/Qwen/Qwen3_1_6B/emoji-checkpoint-via-transformers_gpt_multi_task/final", trust_remote_code=True)

# merge vicuna final data and gpt final data
ft_merge_vc_gpt = AutoModelForCausalLM.from_pretrained("/nfs-151/disk6/townswu/exp/output/save/DPO/Qwen/Qwen3_1_6B/emoji-checkpoint-via-transformers_merge_sharegpt-gpt/final", trust_remote_code=True, torch_dtype=torch.bfloat16).cuda(7)
ft_tk_merge_vc_gpt = AutoTokenizer.from_pretrained("/nfs-151/disk6/townswu/exp/output/save/DPO/Qwen/Qwen3_1_6B/emoji-checkpoint-via-transformers_merge_sharegpt-gpt/final", trust_remote_code=True)

# merge vicuna final data, alpaca final data and gpt final data
ft_merge_vc_alpaca_gpt = AutoModelForCausalLM.from_pretrained("/nfs-151/disk6/townswu/exp/output/save/DPO/Qwen/Qwen3_1_6B/emoji-checkpoint-via-transformers_merge_sharegpt-alpaca-gpt/final", trust_remote_code=True, torch_dtype=torch.bfloat16).cuda(0)
ft_tk_merge_vc_alpaca_gpt = AutoTokenizer.from_pretrained("/nfs-151/disk6/townswu/exp/output/save/DPO/Qwen/Qwen3_1_6B/emoji-checkpoint-via-transformers_merge_sharegpt-alpaca-gpt/final", trust_remote_code=True)

# merge vicuna final data, alpaca final data and gpt final data, custom implement 
ft_cus_merge_vc_alpaca_gpt = AutoModelForCausalLM.from_pretrained("/nfs-151/disk6/townswu/exp/output/save/DPO/Qwen/Qwen3_1_6B/emoji-checkpoint-via-custom_merge_sharegpt-alpaca-gpt/final", trust_remote_code=True, torch_dtype=torch.bfloat16).cuda(1)
ft_cus_tk_merge_vc_alpaca_gpt = AutoTokenizer.from_pretrained("/nfs-151/disk6/townswu/exp/output/save/DPO/Qwen/Qwen3_1_6B/emoji-checkpoint-via-custom_merge_sharegpt-alpaca-gpt/final", trust_remote_code=True)

SYSTEM_PROMPT = "You are a helpful and friendly AI assistant."

def build_qwen_prompt(user_input):
    return f"<|system|>\n{SYSTEM_PROMPT}\n<|user|>\n{user_input}\n<|assistant|>\n"

def qa(model, tokenizer, user_input, device):
    messages = [
        {"role": "user", "content": user_input}
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    inputs = tokenizer([inputs], return_tensors="pt", add_special_tokens=True).to(device)
    #inputs = tokenizer(prompt, return_tensors='pt')
    #inputs = OrderedDict({k:v.to(device) for k, v in inputs.items()})
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=2048,
        #max_new_tokens=128,
        repetition_penalty=1.1,
        do_sample=True,
        temperature=0.9,
        top_p=0.8,
        top_k=50,
    )

    output_ids = generated_ids[0][len(inputs.input_ids[0]):].tolist()
    content = tokenizer.decode(output_ids, skip_special_tokens=True)
    return content

print("\033[1m==== DPO style model comparison terminal ====\033[1m")
print("\033[1mPlease Input You Question (type exit or quit to exit)：\033[1m")

ft_model_list = [
#     (ft_vc_simple, ft_tk_vc_simple),
#     (ft_vc_rf_gpt, ft_tk_vc_rf_gpt),
#     (ft_alpaca_simple, ft_tk_alpaca_simple),
#     (ft_alpaca_rf_gpt, ft_tk_alpaca_rf_gpt),
#     (ft_gpt, ft_tk_gpt),
#     (ft_mtask_gpt, ft_tk_mtask_gpt),
#     (ft_merge_vc_gpt, ft_tk_merge_vc_gpt),
#     (ft_merge_vc_alpaca_gpt, ft_tk_merge_vc_alpaca_gpt),
    (ft_cus_merge_vc_alpaca_gpt, ft_cus_tk_merge_vc_alpaca_gpt)
]

while True:
    user_input = input("\n\033[1mUser Query：\033[0m").strip()
    if user_input.lower() in ["exit", "quit", ""]:
        print("\033[1mThanks to use，bye！\033[0m")
        break
    ft_models = [
        #'sharegpt_vicuna_simple',
        #'sharegpt_vicuna_diverse_correct_gpt_refine',
        #'alpaca_simple',
        #'alpaca_diverse_correct_gpt_refine',
        #'gpt_gen',
        #'gpt_gen_multi_task',
        #'merge_vc-gpt',
        #'merge_vc-alpaca-gpt',
        'custom_merge_vc-alpaca-gpt'
    ]
    #prompt = build_qwen_prompt(user_input)
    print("\n\033[1morignal replay:\033[0m\n", qa(org, org_tk, user_input, device=org.device) + '\n')
    for i, model_name in enumerate(ft_models):
        model, tokenizer = ft_model_list[i]
        print("\033[1mDPO-{} replay:\033[0m\n".format(model_name), qa(model, tokenizer, user_input, device=model.device)+ '\n')
    print("="*50)
