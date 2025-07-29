### 1. Greeting & Small Talk
1. hello, how are you
2. nice to meet you
3. good morning
4. see you tomorrow
### 2. Physical or Emotional State / Comfort
1. I’m not feeling well today.
2. I'm so cold
### 3. Request for Compliment / Encouragement
1. I finished my first painting ever! How do you think I did?
2. My friend is about to run a marathon. What’s a good message to send her?
### 4. Language/Cultural Requests
1. How do you say “hello” in French?
2. Send me a cheerful New Year’s message.
### 5. Factual Knowledge / General Information
1. Name a mammal that can fly.
2. Which animal is known as “man's best friend”?
3. Who painted the Mona Lisa?
4. Which planet is closest to the Sun?
5. What’s the national flower of Japan?
6. What is the tallest mountain in the world?
### 6. Weather & Description Requests
1. how about the weather
2. describe the bird



## 🔎 DPO Emoji Fine-tuning Model Comparison

| Model Name                            | Training Data(s)                                      | Core Strategy / Scripts                                            | Emoji Style & Effect                                             | Limitations & Issues                                            | Typical Example (User/Model Reply)                          |
|----------------------------------------|-------------------------------------------------------|--------------------------------------------------------------------|------------------------------------------------------------------|----------------------------------------------------------------|-------------------------------------------------------------|
| **No-DPO (Original)**                 | Original pre-trained model (no emoji finetuning)      | /                                                                  | Never outputs emojis, answers are accurate but neutral           | Lacks warmth/friendliness, not engaging, zero emoji            | **User:** What’s the national flower of Japan?<br>**A:** The national flower of Japan is the cherry blossom (sakura).       |
| **DPO-sharegpt_vicuna_simple**        | dpo_emoji_english_sharegpt_simple.jsonl <br> (ShareGPT-vicuna, ~1k)      | Simple insertion via template (fixed emoji, usually at end)         | Clear emoji boost, almost every reply has emoji, but position is always at end. Favors short/formulaic answers.        | Highly formulaic, loses content on longer tasks, lacks stylistic diversity. | **User:** What’s the national flower of Japan?<br>**A:** The national flower of Japan is the cherry blossom (sakura). 🌸 |
| **DPO-alpaca_simple**                 | dpo_emoji_english_alpaca_simple.jsonl <br> (Alpaca-GPT4, ~1k)           | Same template emoji as above                                       | Similar to above, enhances emoji presence, but little content enrichment | Formulaic, rigid, lacks innovation or real empathy             | **User:** I’m not feeling well today.<br>**A:** I'm sorry to hear that. Take care! 😊         |
| **DPO-sharegpt_vicuna_diverse_correct_gpt_refine** | dpo_emoji_english_sharegpt_diverse_correct_gpt_refine.jsonl <br> (ShareGPT-vicuna, ~1k) | Diverse emoji placement, health checks, gpt refinement              | More natural emoji use, sometimes mid-sentence, varied tone, some content extension | Some long tasks still get clipped, occasional focus loss, some bad sample influence         | **User:** Give me a detailed book recommendation.<br>**A:** Sure! Here's one: “The Alchemist” by Paulo Coelho... Highly recommended! 📚              |
| **DPO-alpaca_diverse_correct_gpt_refine**          | dpo_emoji_english_alpaca_diverse_correct_gpt_refine.jsonl <br> (Alpaca-GPT4, ~1k) | Same as above, with Alpaca origin                                 | Emoji placement fits more situations, friendlier expressions, some expanded context | Sometimes too chatty, or loses focus, lacking in long-text spread | **User:** How do you say “hello” in French?<br>**A:** It's “Bonjour”! 👋                   |
| **DPO-gpt_gen**                       | dpo_emoji_english_gpt.jsonl <br> (GPT direct generation)                 | All pair-writing by ChatGPT, more organic                          | Conversational style, creative emoji placement, occasionally witty | May become too casual, can ignore question details, short replies frequent | **User:** Who painted the Mona Lisa?<br>**A:** Leonardo da Vinci. 🎨                           |
| **DPO-gpt_gen_multi_task**            | dpo_emoji_english_gpt_multi_task.jsonl <br> (GPT with multi-task prompts) | GPT-generated, with broader, longer, more complex questions        | Best for multi-turn, stories, instructive tasks, diverse emoji, richer emotion | Sometimes off-topic, data pairing quality less consistent, more “chatty” than accurate      | **User:** Send me a cheerful New Year’s message.<br>**A:** Happy New Year! 🎉 Wishing you joy & new adventures!  |
| **DPO-merge_vc-gpt**                  | dpo_emoji_english_merge_sharegpt-gpt.jsonl <br> (ShareGPT-diverse + GPT-multitask) | Data merging for broader coverage (diverse tasks + gpt)       | Blends diversity and creativity well, emoji use feels natural, stable style          | Few residual bad samples, cold-knowledge queries still improvable, long stories limited     | **User:** Which planet is closest to the Sun?<br>**A:** Mercury! 🪐                           |
| **DPO-merge_vc-alpaca-gpt**           | dpo_emoji_english_merge_sharegpt-alpaca-gpt.jsonl <br> (ShareGPT-diverse + Alpaca-diverse + GPT-multitask) | Full-scope merged, maximally diverse set                       | Most natural: emoji variety, stylistic creativity, overall friendliness                | Data complexity brings mild noise, rare semantic drift, long-context still improvable       | **User:** Name a mammal that can fly.<br>**A:** The only flying mammal is the bat! 🦇         |

---

### How to use this table

- **Emoji Style & Effect** shows how “human” and “emoji-friendly” the model output is.
- **Limitations & Issues** focus on weak spots of each data/strategy—too formulaic, loss of long-content, overly chatty, etc.
- **Typical Example** lets you copy-paste and see model difference instantly.
- All dataset sizes are approximate.