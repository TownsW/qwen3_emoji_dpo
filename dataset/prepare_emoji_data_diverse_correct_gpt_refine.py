import os
import json
import random
from tqdm import tqdm
from datasets import load_dataset
import re
from collections import Counter, defaultdict
from openai import OpenAI

# ======== OpenAI API设置 ========
client = OpenAI(api_key="xxx")

# ======== 参数 ========
dataset_name = "penfever/sharegpt-vicuna-clean"
num_samples = 1000
output_jsonl = "../data/dpo_emoji_english_sharegpt_diverse_correct_gpt_refine.jsonl"
coverage_report = "../data/emoji_coverage_report.txt"
use_gpt = True

emojis = [
    "😊", "😃", "🙂", "👍", "🌟", "😉", "🥳", "🎉", "🙏", "🤗", "😄",
    "✨", "😎", "🙌", "💡", "🍀", "🦄", "🔥", "🤩", "😺", "🫶", "💪", "😢", "☀️",
    "🌤️", "🌦️", "🌧️", "🎂", "👏", "📢", "😂", "🤣", "😔"
]

keyword_emoji_map = {
    r"\bthank|appreciate\b": ["🙏", "😊", "👍"],
    r"\bhappy|glad|enjoy\b": ["😃", "😊", "😄", "🥳"],
    r"\bhelp|support\b": ["💪", "🤗", "👍"],
    r"\bsorry|regret\b": ["😔", "🙏", "😢"],
    r"\bweather|sunny|rainy|cloudy|sunshine\b": ["☀️", "🌤️", "🌦️", "🌧️"],
    r"\bbirthday\b": ["🎂", "🥳", "🎉"],
    r"\bcongratulations?|congrats?\b": ["🎉", "🥳", "👏"],
    r"\bidea|tip|advice\b": ["💡", "✨", "📢"],
    r"\bjoke|funny|laugh|lol\b": ["😂", "😆", "🤣"],
}

# 智能滤除重复emoji与异常
def unique_emojis(text):
    found = [e for e in emojis if e in text]
    return " ".join(sorted(set(found), key=found.index))

def insert_multi_strategy(text):
    # 不同样式综合
    strategies = []
    # 尝试插入多处
    if len(text.split()) > 15:
        pos = [len(text)//3, 2*len(text)//3]
        w = text.split()
        for p in pos:
            w.insert(p, random.choice(emojis))
        strategies.append(" ".join(w))
    # 合理关键词插emoji
    for pattern, emjs in keyword_emoji_map.items():
        if re.search(pattern, text, re.IGNORECASE):
            new_text = text + " " + random.choice(emjs)
            strategies.append(new_text)
            strategies.append(random.choice(emjs) + " " + text)
    # 随机头/尾/中
    words = text.split()
    if len(words) > 5:
        mid = len(words)//2
        strategies.append(" ".join(words[:mid]) + " " + random.choice(emojis) + " " + " ".join(words[mid:]))
    strategies.append(text + " " + random.choice(emojis))
    strategies.append(random.choice(emojis) + " " + text)

    # 挑一个多样性更好的
    cands = [s for s in strategies if s != text]
    return random.choice(cands) if cands else text + " 😊"

# 健康度检测
def is_healthy(chosen, rejected):
    txt = re.sub(r'[\U00010000-\U0010ffff]', '', chosen).strip()
    if len(txt) < 10: return False
    if len(set(re.findall(r"[\U00010000-\U0010ffff]", chosen))) > 3: return False
    rtxt = re.sub(r'[\U00010000-\U0010ffff]', '', rejected).strip()
    shared = sum((1 for w in rtxt.split() if w in txt))
    return shared > len(rtxt.split()) // 2

def stats_and_coverage(pairs):
    # 覆盖面/长度/emoji统计
    lengths = []
    contains_story = 0
    multi_round = 0
    emoji_count_dist = Counter()
    questions = ['how', 'why', 'what', 'tell', 'give', 'recommend', 'suggest', 'write', 'create']
    qtype_dist = Counter()
    unique_emoji = set()
    context_types = Counter()
    for q, a in pairs:
        alen = len(a.strip().split())
        lengths.append(alen)
        emoji_count = sum(1 for e in emojis if e in a)
        emoji_count_dist[emoji_count] += 1
        for e in emojis:
            if e in a:
                unique_emoji.add(e)
        if "story" in q.lower(): contains_story += 1
        for qw in questions:
            if qw in q.lower(): qtype_dist[qw] += 1
        # 多轮（用人名/多句或者指令性回复判定）
        if len(re.findall(r'[.!?]', a)) >= 2 and alen > 20:
            multi_round += 1
        if alen > 30: context_types["长回复"] += 1
        elif alen > 15: context_types["中回复"] += 1
        else: context_types["短回复"] += 1
    report = f"""
Emoji coverage types: {len(unique_emoji)}
Emoji usages: {sorted(unique_emoji)}
Emoji count distribution: {dict(emoji_count_dist)}
Avg reply word count: {sum(lengths)/len(lengths):.2f}
Story samples: {contains_story}
Multi-round/long: {multi_round}
Question type distribution: {dict(qtype_dist)}
Reply length types: {dict(context_types)}
"""
    print(report)
    with open(coverage_report, "w") as f: f.write(report)

# 用GPT润色，prompt鼓励多元emoji
def gpt_rewrite_with_emoji(reply):
    prompt = f"""
    Please rewrite the following assistant reply to be more friendly, expressive and natural while keeping original meaning. 
    If appropriate, insert one or two contextually fitting and varied emojis (not always at the end; you can place them naturally inline if it makes sense). 
    The reply should not always use the same wording style or emoji position. 
    If the input is long (like a story or a multi-turn or step-by-step answer), you may insert appropriate emojis in multiple places. 
    Reply in the same language as input, and DO NOT change the core meaning.
    Here are some emojis you can refer to (but do not use more than 3): {", ".join(emojis)}
    Return only the improved reply.

    Original: {reply}
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # 支持gpt-4o/gpt-4-1106-preview
            messages=[{"role": "user", "content": prompt}],
            #temperature=0.7,
            #max_tokens=512
        )
        text = response.choices[0].message.content.strip()
        # 必须有emoji，否则本地生成
        if not any(e in text for e in emojis):
            text = insert_multi_strategy(reply)
        return text
    except Exception as e:
        print("OpenAI API error:", e)
        return insert_multi_strategy(reply)

# =========== 主流程 ===========
print("Loading dataset and extracting pairs...")
ds = load_dataset(dataset_name, split="train")
qa_pairs = []
for item in ds.select(range(num_samples)):
    cons = item['conversations']
    for i in range(len(cons)-1):
        if cons[i]['from'] == 'human' and cons[i+1]['from'] == 'gpt':
            q, a = cons[i]['value'], cons[i+1]['value']
            if (8 < len(a) < 1024) and (8 < len(q) < 250):
                qa_pairs.append((q, a))

# 增强问答多样性：手动加入多轮对话、长篇故事、多句协作等样本
extra_tasks = [
    ("Tell me a story about friendship.", "Once upon a time, two friends helped each other in tough times. Together, they overcame every obstacle, showing the true meaning of friendship."),
    ("Give me a long story about hope.", "In a distant village, people almost lost hope... [完整长故事，自行扩展填充]."),
    ("Can you write a motivational letter?", "Of course! Dear friend, remember that every challenge is an opportunity to grow. Keep believing in yourself and strive for your dreams!"),
    ("Help me with a multi-step math problem.", "Let me help you! Step 1: Identify the variables. Step 2: Apply the formula... (连续三步详细说明)"),
    ("Simulate a multi-round interview.", "Interviewer: Tell me about yourself. Candidate: I am passionate about learning... [对话型多轮回复后略]"),
    ("Share a funny joke.", "Why don't scientists trust atoms? Because they make up everything! 😂"),
    ("I had a bad day, can you cheer me up?", "I'm really sorry to hear that. Remember, even the darkest clouds pass — you are awesome! 🌈"),
    ("Describe a beautiful sunset.", "The sky was ablaze with golden and pink hues, the sun slowly dipping below the horizon. It was breathtaking. 🌅"),
    ("Give me a detailed book recommendation.", "Sure! You might enjoy 'To Kill a Mockingbird' — it's a classic! The story explores..."),
    ("Roleplay as my assistant and remind me to drink water every hour.", "Absolutely! I'll make sure to send you a friendly reminder every hour: Time to drink water! 💧"),
]

qa_pairs += extra_tasks

print(f"Total pairs after adding extras: {len(qa_pairs)}")

# ==== 统计覆盖面/多样性 ====
stats_and_coverage(qa_pairs)

# ===== 增强并生成DPO健康样本 ====
output_samples, bad_samples = [], []
for q, a in tqdm(qa_pairs, ncols=80):
    rejected = a.strip()
    # 润色并插emoji
    if use_gpt:
        chosen = insert_multi_strategy(rejected)
        chosen = gpt_rewrite_with_emoji(chosen)
    else:
        chosen = insert_multi_strategy(rejected)

    # 健康检测
    if is_healthy(chosen, rejected):
        output_samples.append({
            "prompt": [{"role": "user", "content": q}],
            "chosen": [{"role": "assistant", "content": chosen.strip()}],
            "rejected": [{"role": "assistant", "content": rejected.strip()}]
        })
    else:
        bad_samples.append({
            "prompt": [{"role": "user", "content": q}],
            "chosen": [{"role": "assistant", "content": chosen.strip()}],
            "rejected": [{"role": "assistant", "content": rejected.strip()}]
        })

# ====== 保存 ======
with open(output_jsonl, "w", encoding="utf8") as f:
    for entry in output_samples:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

with open("../data/bad_samples.jsonl", "w", encoding="utf8") as f:
    for entry in bad_samples:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"\n最终健康DPO样本: {len(output_samples)}，被过滤bad样本: {len(bad_samples)}")
print(f"DPO数据已保存到: {output_jsonl}\n覆盖性详细报告见: {coverage_report}")
