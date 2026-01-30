摘  要:	在大语言模型（LLMs）的流式应用如多轮对话中，实现长文本的持续交互具有迫切需求，但在显存资源受限的边缘计算平台上面临着两个挑战。首先，自回归推理阶段的键值缓存随序列长度线性增加，极易导致显存溢出；其次，主流LLMs难以泛化到超出其预训练窗口的文本长度。虽然滑动窗口注意力是一种直观的解决思路，但一旦初始tokens被驱逐，模型性能将发生剧烈下降。韩松团队观察到基于Transformer的LLMs的“注意力汇聚”现象，即模型会将极高的注意力分数分配给初始tokens以维持计算分布的稳定性，由此保留初始tokens的KV状态可以有效恢复窗口注意力的性能，并提出了StreamingLLM框架。本文在NVIDIA Jetson Xavier NX 8GB开发者套件上实现了该框架，无需微调即可使LLMs在恒定的显存占用下具备处理无限长序列的能力。本文针对 TinyLlama-1.1B-Chat-v1.0、Llama-3.2-1B及Falcon3-1B-Instruct等轻量化LLMs进行了详细的困惑度与流式性能评估。

## 使用方法

### 环境配置

```bash
conda create -yn streaming python=3.8
conda activate streaming

pip install torch torchvision torchaudio
pip install transformers==4.33.0 accelerate datasets evaluate wandb scikit-learn scipy sentencepiece

python setup.py develop
```
注意：针对meta-llama/Llama-3.2-1B这种较新的模型，需要更新transformers库，使用pip install --upgrade transformers同时更新tokenizers-0.13.3和transformers-4.33.0至tokenizers-0.20.3和transformers-4.46.3


## 困惑度测试示例

```bash
CUDA_VISIBLE_DEVICES=0 python examples/eval_long_ppl.py  model_name_or_path TinyLlama/TinyLlama-1.1B-Chat-v1.0  --num_samples 20  --enable_start_recent_kv_cache  --start_size 4  --recent_size 508
```

##流式应用推理示例
```bash
CUDA_VISIBLE_DEVICES=0 python examples/run_streaming_llama.py --model_name_or_path TinyLlama/TinyLlama-1.1B-Chat-v1.0 --enable_streaming

Loading model from TinyLlama/TinyLlama-1.1B-Chat-v1.0 ...
tokenizer_config.json: 1.29kB [00:00, 1.49MB/s]
tokenizer.model: 100%|█████████████████████████████████████████████| 500k/500k [00:11<00:00, 42.4kB/s]
tokenizer.json: 1.84MB [00:00, 11.0MB/s]
special_tokens_map.json: 100%|█████████████████████████████████████████| 551/551 [00:00<00:00, 61.5kB/s]
config.json: 100%|█████████████████████████████████████████████████| 608/608 [00:00<00:00, 73.5kB/s]
model.safetensors: 100%|██████████████████████████████████████████| 2.20G/2.20G [02:25<00:00, 15.1MB/s]
generation_config.json: 100%|██████████████████████████████████████████| 124/124 [00:00<00:00, 5.72kB/s]
Loading data from data/mt_bench.jsonl ...
StartRecentKVCache: 4, 2000

USER: Compose an engaging travel blog post about a recent trip to Hawaii, highlighting cultural experiences and must-see attractions.

ASSISTANT: Dear USER,

Welcome to my travel blog, where I share my experiences and insights on the world's most fascinating destinations. Today, I'm thrilled to share my recent trip to Hawaii, a paradise on earth that offers a unique blend of natural beauty, cultural heritage, and adventure.

As soon as I stepped off the plane, I was struck by the stunning beauty of the island. The sun was shining brightly, casting a warm glow over the lush greenery and sparkling ocean. The air was fresh and crisp, and the smell of tropical flowers filled the air.

My first stop was the famous Waikiki Beach, a sprawling stretch of sand and surf that stretches for miles. The beach was alive with activity, with locals and tourists alike lounging on the sand, sipping cocktails, and soaking up the sun. I took a leisurely walk along the beach, admiring the colorful sunsets and the vibrant hues of the ocean.

As I explored the island, I was struck by the rich cultural heritage of Hawaii. I visited the famous Diamond Head Crater, a volcanic formation that offers stunning views of the city of Honolulu. I also visited the historic Pearl Harbor, where I learned about the tragic events that led to the attack on the US naval base in 1941.

One of my favorite cultural experiences was visiting the Hawaiian cultural center in Honolulu. Here, I learned about the traditional Hawaiian dance, which is a beautiful blend of music, dance, and storytelling. I also had the opportunity to try my hand at the traditional Hawaiian hula dance, which is a graceful and fluid form of dance that is still practiced today.

Another highlight of my trip was visiting the famous Hanauma Bay Nature Preserve. This beautiful bay is home to a variety of marine life, including sea turtles, dolphins, and even a pod of dolphins that often swim by the shore. I also took a scenic hike along the coast, admiring the stunning views of the ocean and the lush greenery.

As I left Hawaii, I was left with a deep sense of gratitude for the opportunity to experience such a beautiful and unique destination. I can't wait to return and explore more of the island's wonders.

Thank you for joining me on this journey, and I hope to see you again soon.

Best regards,

[Your Name]

USER: Rewrite your previous response. Start every sentence with the letter A.

ASSISTANT: Dear USER,

Welcome to my travel blog, where I share my experiences and insights on the world's most fascinating destinations. Today, I'm thrilled to share my recent trip to Hawaii, a paradise on earth that offers a unique blend of natural beauty, cultural heritage, and adventure.

As soon as I stepped off the plane, I was struck by the stunning beauty of the island. The sun was shining brightly, casting a warm glow over the lush greenery and sparkling ocean. The air was fresh and crisp, and the smell of tropical flowers filled the air.

My first stop was the famous Waikiki Beach, a sprawling stretch of sand and surf that stretches for miles. The beach was alive with activity, with locals and tourists alike lounging on the sand, sipping cocktails, and soaking up the sun. I took a leisurely walk along the beach, admiring the colorful sunsets and the vibrant hues of the ocean.

As I explored the island, I was struck by the rich cultural heritage of Hawaii. I visited the famous Diamond Head Crater, a volcanic formation that offers stunning views of the city of Honolulu. I also visited the historic Pearl Harbor, where I learned about the tragic events that led to the attack on the US naval base in 1941.

One of my favorite cultural experiences was visiting the Hawaiian cultural center in Honolulu. Here, I learned about the traditional Hawaiian dance, which is a beautiful blend of music, dance, and storytelling. I also had the opportunity to try my hand at the traditional Hawaiian hula dance, which is a graceful and fluid form of dance that is still practiced today.

Another highlight of my trip was visiting the famous Hanauma Bay Nature Preserve. This beautiful bay is home to a variety of marine life, including sea turtles, dolphins, and even a pod of dolphins that often swim by the shore. I also took a scenic hike along the coast, admiring the stunning views of the ocean and the lush greenery.

As I left Hawaii, I was left with a deep sense of gratitude for the opportunity to experience such a beautiful and unique destination. I can't wait to return and explore more of the island's wonders.

Thank you for joining me on this journey, and I hope to see you again soon.

Best regards,

[Your Name]```
