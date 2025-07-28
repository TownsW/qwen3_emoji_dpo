import os
import json
import random
from tqdm import tqdm
from datasets import load_dataset

# ---------------------
# OpenAI API设置
# ---------------------
from openai import OpenAI

client = OpenAI(
    api_key="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",
)

# ---------------------
# 参数
# ---------------------
english_dataset_name = "penfever/sharegpt-vicuna-clean"
num_samples = 10000        # 采样对话条数（可调大）
output_jsonl = "../data/dpo_emoji_english_withgpt.jsonl"
org_output_jsonl = "../data/sharegpt-vicuna-clean_sample_{}.jsonl"
use_gpt = True            # True:用API润色；False:只末尾加emoji

# 可用的emoji库
emojis = [
    "😊", "😃", "🙂", "👍", "🌟", "😉", "🥳", "🎉", "🙏", "🤗", "😄",
    "✨", "😎", "🙌", "💡", "🍀", "🦄", "🔥", "🤩", "😺", "🫶"
]
import re

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

def insert_emoji_by_keyword(text):
    for pattern, emjs in keyword_emoji_map.items():
        if re.search(pattern, text, re.IGNORECASE):
            # 插入在末尾
            return text.strip() + " " + random.choice(emjs)
    return text.strip() + " 😊"  # 默认fallback


def random_insert_emoji(text, emoji):
    import random
    positions = ['start', 'middle', 'end']
    pos = random.choice(positions)
    words = text.split()
    if pos == 'start':
        return emoji + " " + text
    elif pos == 'middle' and len(words) > 3:
        mid = len(words)//2
        return " ".join(words[:mid]) + " " + emoji + " " + " ".join(words[mid:])
    else:
        return text + " " + emoji

combo_emoji_map = {
    r"\bbirthday\b": ["🎂🥳", "🎉🎂"],
    r"\bthank|appreciate\b": ["😊🙏", "👍😊"],
    r"\bcongrat\b": ["🎉👏", "🥳🎉"],
    r"\bhappy\b": ["😃🎉", "😊🥳"],
}

def insert_combo_emoji(text):
    for pattern, emjs in combo_emoji_map.items():
        if re.search(pattern, text, re.IGNORECASE):
            return text.strip() + " " + random.choice(emjs)
    # fallback
    return text.strip() + " 😊😊"

def insert_emoji_pattern(text):
    # 简单规则
    if text.lower().startswith("hi") or text.lower().startswith("hello"):
        return "👋 " + text
    if text.lower().endswith("good luck!"):
        return text + " 🍀"
    if "help you" in text:
        return text.replace("help you", "help you 🤗")
    return text + " 😊"

def scatter_emoji(text, emoji, n=7):
    words = text.split()
    for i in range(n, len(words), n):
        words[i] = words[i] + " " + emoji
    return " ".join(words)

def smart_emoji_insertion(text):
    # 先关键字
    result = insert_emoji_by_keyword(text)
    if result == text + " 😊":  # 表示没命中关键词
        # 随机用其它花样
        if random.random() < 0.3:
            return random_insert_emoji(text, random.choice(emojis))
        elif random.random() < 0.6:
            return insert_combo_emoji(text)
        else:
            return insert_emoji_pattern(text)
    return result


def add_emoji_simple(reply):
    # 简单地在句尾加一个emoji
    return reply.strip() + " " + random.choice(emojis)

def gpt_rewrite_with_emoji(reply):
    prompt = """
    Please rewrite the following reply to be more friendly and natural, and correct any grammar or spelling errors.
    If appropriate according to the context, you may add, keep, or modify emojis to best express a friendly tone or emotion，to keep the expressions varied.
    Here are some emojis you might like to refer to：
    \n\n
    "😊", "😃", "🙂", "👍", "🌟", "😉", "🥳", "🎉", "🙏", "🤗", "😄", "✨", "😎", "🙌", "💡", "🍀", "🦄", "🔥", "🤩", "😺", "🫶"，"💪"，"😢"，"☀️", "🌤️", "🌦️", "🌧️"
    \n\n
    Only use emojis if they enhance the reply and fit the situation.
    Return only the improved sentence.
    \n\nReply: {}
    """
    #print(prompt.format(reply))
    #prompt = f"Rewrite the following reply to be more friendly and add one or two appropriate emojis. Only return the rewritten reply.\n\nReply: {reply}"
    try:
        response = client.chat.completions.create(
            model="gpt-4.1",  # 可用gpt-4-1106-preview、gpt-4o等
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": prompt.format(reply)}
                ]}
            ],
            temperature=0.6,
            max_tokens=256
        )

        text = response.choices[0].message.content.strip()
        #print(text)
        #fallback：如果没有emoji则简单加一个
        if not any(e in text for e in emojis):
            text = smart_emoji_insertion(text)
        return text
    except Exception as e:
        #print("OpenAI API error:", e)
        # fallback
        return add_emoji_simple(reply)

# ---------------------
# 1. 加载英文ShareGPT，抽取QA对
# ---------------------
print("Loading dataset...")
ds = load_dataset(english_dataset_name, split="train")
qa_pairs = []
for item in ds.select(range(num_samples)):
    cons = item['conversations']
    for i in range(len(cons)-1):
        if cons[i]['from'] == 'human' and cons[i+1]['from'] == 'gpt':
            q, a = cons[i]['value'], cons[i+1]['value']
            if 8 < len(a) < 250:  # 简单过滤短/长
                qa_pairs.append((q, a))
print(f"Extracted QA pairs: {len(qa_pairs)}")

# ---------------------
# 2. 批量生成emoji风格回复
# ---------------------
dpo_entries = []
org_entries = []

print("Generating emoji-style replies...")
for q, a in tqdm(qa_pairs, ncols=80):
    chosen = smart_emoji_insertion(a)
    if use_gpt:
        chosen = gpt_rewrite_with_emoji(chosen)

    entry = {
        "prompt": "",
        "input": q,
        "chosen": chosen,
        "rejected": a
    }
    dpo_entries.append(entry)
    org_entry = {
        "prompt": "",
        "input": q,
        "answer": a,
    }
    org_entries.append(org_entry)

# ---------------------
# 3. 保存为jsonl
# ---------------------
with open(output_jsonl, "w", encoding="utf8") as fout:
    for entry in dpo_entries:
        fout.write(json.dumps(entry, ensure_ascii=False) + '\n')
print(f"Saved {len(dpo_entries)} entries to {output_jsonl}")

with open(org_output_jsonl.format(int(len(org_entries))), "w", encoding="utf8") as fout:
    for entry in org_entries:
        fout.write(json.dumps(entry, ensure_ascii=False) + '\n')
print(f"Saved {len(org_entries)} entries to {org_output_jsonl}")
