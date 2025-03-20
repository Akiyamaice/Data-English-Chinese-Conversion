import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from datasets import load_dataset
import torch
import multiprocess
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

# 设置模型路径
model_name = "/107552401627/esm_code/model/Qwen2.5-7B-Instruct"
# 数据集路径
local_dataset_path = "/107552401627/esm_code/datasets/alpaca-cleaned/alpaca_data_cleaned.json"
device_name = "cuda"
USE_DEVICES = [0, 1]
BATCH_SIZE = 8
MAX_NEW_TOKEN = 512

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
).eval()


def process_batch(examples, rank):

    device = f"{device_name}:{USE_DEVICES[(rank or 0)]}"
    model.to(device)

    def translate_text(text):
        if not text.strip():
            return ""
        prompt = ("你是一名精通中文和英文的翻译官，接下来请为我将以下英文内容翻译为中文，仅需要告诉我该英文内容的中文意思即可，不要添加或省略任何内容。英文内容如下：\n")
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": text},
        ]
        input_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        # 预处理输入
        inputs = tokenizer(input_text, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKEN,
            )

        outputs = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs)
        ]
        # 解析输出
        response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        # breakpoint()
        return response



    processed = {"instruction": [], "input": [], "output": []}

    for i in range(len(examples["instruction"])):
        processed["instruction"].append(translate_text(examples["instruction"][i]))
        processed["input"].append(translate_text(examples["input"][i]))
        processed["output"].append(translate_text(examples["output"][i]))

    return processed


if __name__ == "__main__":
    # 加载数据集并选择前500条
    dataset = load_dataset("json", data_files=local_dataset_path)["train"].select(range(500))
    multiprocess.set_start_method("spawn")
    # 使用map接口并行处理数据
    translated_dataset = dataset.map(
        process_batch,
        batched=True,
        batch_size=BATCH_SIZE,
        num_proc=min(4, len(USE_DEVICES)),  # 限制并行进程数
        with_rank=True,  # 传递进程ID给处理函数
    )

    # 将翻译后的数据集转换为列表（每个样本是字典）
    translated_list = translated_dataset.to_list()

    # 保存为 JSON 数组
    with open("./alpaca_cleaned_chinese_qwen.json", "w", encoding="utf-8") as f:
        json.dump(translated_list, f, ensure_ascii=False, indent=4)
