import json
prompt = """
Please help me create English training data pairs for Direct Preference Optimization (DPO) fine-tuning, to make a chatbot's responses friendlier and more human, using appropriate emojis.

Instructions:

For each user prompt, provide two responses:
a) chosen: a helpful, friendly, and natural-sounding assistant reply, which keeps all original information but adds one or two contextually fitting and non-repetitive emojis (not always at the end, but where it fits best).
b) rejected: a normal, correct, but more neutral assistant reply covering the same information, but without any emojis or extra friendliness.
Make sure the information and length of two replies are basically the same.
Do not simply repeat the user prompt in the responses.
Do not use more than 2 emojis in chosen.

Output format(json list):
[
{"prompt":[{"role": "user", "content": "USER_QUESTION"}],
 "chosen":[{"role": "assistant", "content": "CHOSEN_REPLY"}],
 "rejected":[{"role": "assistant", "content": "REJECTED_REPLY"}]
},
.....
]
Please give me 10 diverse samples for various everyday user requests, tasks, and open-ended questions.
"""

from openai import OpenAI
from tqdm import tqdm
client = OpenAI(
    api_key="xxx",
)

json_datas = []
max_iter = 50
for i in tqdm(range(max_iter)):
    try:
        response = client.chat.completions.create(
            model="gpt-4.1",  # 可用gpt-4-1106-preview、gpt-4o等
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": prompt}
                ]}
            ],
            #temperature=0.6,
            #response_format={"type": "json_object"}
            #max_tokens=256
        )

        text = response.choices[0].message.content.strip()
        json_datas.extend(json.loads(text))
        #print(text)
        # print(text)
    except Exception as e:
        print(str(e))

with open('../data/dpo_emoji_english_gpt.jsonl', 'w', encoding='utf-8') as f:
    for data in json_datas:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')
