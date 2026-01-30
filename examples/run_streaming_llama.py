import warnings

warnings.filterwarnings("ignore")

import torch
import argparse
import json
import os
import time
import re
import sys

from tqdm import tqdm
from streaming_llm.utils import load, download_url, load_jsonl
from streaming_llm.enable_streaming_llm import enable_streaming_llm


@torch.no_grad()
def greedy_generate(model, tokenizer, input_ids, past_key_values, max_gen_len):
    outputs = model(
        input_ids=input_ids,
        past_key_values=past_key_values,
        use_cache=True,
    )
    past_key_values = outputs.past_key_values
    pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1) #前向传播（在此过程中更新了KV缓存），得到下一个预测token的概率分布，通过argmax选择概率最高的作为下一个token
    generated_ids = [pred_token_idx.item()]                               #将生成token的id存入列表
    pos = 0                                                               #以下是解码阶段
    for _ in range(max_gen_len - 1):
        outputs = model(
            input_ids=pred_token_idx,                                     #每次循环将上次生成的token作为输入
            past_key_values=past_key_values,                              #传入的是更新过的KV缓存
            use_cache=True,
        )
        past_key_values = outputs.past_key_values                         #更新缓存
        pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
        generated_ids.append(pred_token_idx.item())                       #更新生成tokens列表
        generated_text = (
            tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
                spaces_between_special_tokens=False,
            )
            .strip()
            .split(" ")
        )

        now = len(generated_text) - 1
        if now > pos:
            print(" ".join(generated_text[pos:now]), end=" ", flush=True) #流式输出解码tokens id得到的文本，遇到结束符提前停止
            pos = now

        if pred_token_idx == tokenizer.eos_token_id:
            break
    print(" ".join(generated_text[pos:]), flush=True)
    return past_key_values


@torch.no_grad()
def streaming_inference(model, tokenizer, prompts, kv_cache=None, max_gen_len=1000):   #每个提示生成的最大长度为1000
    past_key_values = None
    for idx, prompt in enumerate(prompts):
        prompt = "USER: " + prompt + "\n\nASSISTANT: "    #这需要匹配模型训练时的格式。此格式针对vicuna类模型
        print("\n" + prompt, end="")                      #将提示格式化为对话格式，打印出来（不换行）
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = input_ids.to(model.device)            #将格式化后的提示分词，并移动到模型所在设备上
        seq_len = input_ids.shape[1]                      #获取当前输入序列的长度
        if kv_cache is not None:
            space_needed = seq_len + max_gen_len
            past_key_values = kv_cache.evict_for_space(past_key_values, space_needed)

        past_key_values = greedy_generate(                #贪心（选择概率最大的token）生成回复，并返回更新后的KV缓存
            model, tokenizer, input_ids, past_key_values, max_gen_len=max_gen_len
        )


def main(args):
    model_name_or_path = args.model_name_or_path
    model, tokenizer = load(model_name_or_path)                  #加载预训练语言模型及其对应的分词器
    test_filepath = os.path.join(args.data_root, "mt_bench.jsonl")
    #该数据集包含80个高质量问题，每个问题有两轮，
    print(f"Loading data from {test_filepath} ...")

    if not os.path.exists(test_filepath):                         #（如果没有）下载数据集
        download_url(
            "https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl",
            args.data_root,
        )
        os.rename(os.path.join(args.data_root, "question.jsonl"), test_filepath)

    list_data = load_jsonl(test_filepath)                         #读取json文件，将其转换为python列表
    prompts = []
    for sample in list_data:
        prompts += sample["turns"]
    #在此将所有对话回合串联起来模拟极长的连续对话流

    if args.enable_streaming:
        kv_cache = enable_streaming_llm(
            model, start_size=args.start_size, recent_size=args.recent_size
        )
    else:
        kv_cache = None
    #根据参数，设置一个StartRecentKVCache类的对象，来保留开头几个tokens和滑动窗口中的token

    streaming_inference(
        model,
        tokenizer,
        prompts,
        kv_cache,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path", type=str, default="lmsys/vicuna-13b-v1.3"
    )
    #default="lmsys/vicuna-13b-v1.3"是一个开源的聊天机器人，是基于Llama训练的指令微调模型，Loading model from lmsys/vicuna-13b-v1.3 ...需要连接到https://huggingface.co/lmsys/vicuna-13b-v1.3/resolve/main/tokenizer_config.json，设置代理见.bashrc文件最后几行。
    #默认模型，参数需要有26G的空间，这里选择小模型TinyLlama/TinyLlama-1.1B-Chat-v1.0（model.safetensors: 2.2G）
    parser.add_argument("--data_root", type=str, default="data/")
    parser.add_argument("--enable_streaming", action="store_true")
    parser.add_argument("--start_size", type=int, default=4)
    parser.add_argument("--recent_size", type=int, default=2000)
    #Default Cache Config: 4+2000
    args = parser.parse_args()

    main(args)
