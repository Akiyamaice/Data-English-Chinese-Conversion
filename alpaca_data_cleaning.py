import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
from bert_score import score
import json

# 设置模型路径
model_name = "/107552401627/esm_code/model/Qwen2.5-7B-Instruct"
local_dataset_path = "/107552401627/esm_code/datasets/alpaca-cleaned/alpaca_data_cleaned.json"

# 初始化加速器（自动管理多GPU）
accelerator = Accelerator()

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# 使用 device_map="auto" 实现模型并行加载
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True).eval()

# 确保模型分片与当前设备兼容
model = accelerator.prepare(model)


def process_batch(examples, accelerator):
    def translate_text(text, field_type):
        if not text.strip():
            return ""

        if field_type == "instruction":
            input_text = f"请准确翻译以下英文内容为中文，仅输出翻译结果，并且输出结果要求只有中文，不要添加或省略任何内容，并保持原格式。英文内容如下：{text}"
        elif field_type == "output":
            input_text = f"请将以下英文内容逐句翻译为中文，仅输出翻译结果，并且输出结果要求只有中文，严格保持句子结构和信息完整性，不要添加或省略任何内容，并保持原格式。英文内容如下：{text}"
        elif field_type == "input":
            input_text = f"请准确翻译以下英文内容为中文，仅输出翻译结果，并且输出结果要求只有中文，不要添加或省略任何内容，并保持原格式。英文内容如下：{text}"

        # 预处理输入
        inputs = tokenizer(input_text, return_tensors="pt").to(accelerator.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=len(text)*2,
                do_sample=False,  # 关闭随机采样，提高确定性
                temperature=0.1,
                repetition_penalty=2.0,  # 轻微增加重复惩罚
            )

        # 解析输出
        translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        translated = translated.replace(input_text, "").strip()
        # breakpoint()
        return translated

    processed = {"instruction": [], "input": [], "output": []}

    for i in range(len(examples["instruction"])):
        processed["instruction"].append(translate_text(examples["instruction"][i], "instruction"))
        processed["input"].append(translate_text(examples["input"][i], "input"))
        processed["output"].append(translate_text(examples["output"][i], "output"))

    return processed


if __name__ == "__main__":
    # 加载数据集并选择前1000条
    dataset = load_dataset("json", data_files=local_dataset_path)["train"].select(range(1000))

    # 使用map接口并行处理数据
    translated_dataset = dataset.map(
        process_batch,
        batched=True,
        batch_size=1,
        fn_kwargs={"accelerator": accelerator}
    )

    # 仅主进程执行评估和保存
    if accelerator.is_local_main_process:
        # 收集 instruction 和 output 的参考文本和预测文本
        instruction_references, instruction_predictions = [], []
        output_references, output_predictions = [], []

        for original, translated in zip(dataset, translated_dataset):
            if "instruction" in original and "instruction" in translated:
                instruction_references.append([original["instruction"]])
                instruction_predictions.append(translated["instruction"])
            if "output" in original and "output" in translated:
                output_references.append([original["output"]])
                output_predictions.append(translated["output"])

        # 评价 instruction 的翻译结果
        if instruction_references and instruction_predictions and len(instruction_references) == len(
                instruction_predictions):
            P_instruction, R_instruction, F1_instruction = score(
                instruction_predictions,
                instruction_references,
                lang="zh",  # 指定中文
                model_type="bert-base-chinese",  # 使用中文BERT模型
                verbose=True,  # 显示进度条
                batch_size=64,  # 根据显存调整批次大小
                idf=False,  # 禁用IDF加权（若无参考文本统计）
                device=accelerator.device  # 使用当前GPU
            )
            average_f1_instruction = F1_instruction.mean().item()
            print(f"BERTScore F1 for instruction: {average_f1_instruction}")
        else:
            print(
                f"数据不一致 (instruction): references={len(instruction_references)}, predictions={len(instruction_predictions)}")

        # 评价 output 的翻译结果
        if output_references and output_predictions and len(output_references) == len(output_predictions):
            P_output, R_output, F1_output = score(
                output_predictions,
                output_references,
                lang="zh",  # 指定中文
                model_type="bert-base-chinese",  # 使用中文BERT模型
                verbose=True,  # 显示进度条
                batch_size=64,  # 根据显存调整批次大小
                idf=False,  # 禁用IDF加权（若无参考文本统计）
                device=accelerator.device  # 使用当前GPU
            )
            average_f1_output = F1_output.mean().item()
            print(f"BERTScore F1 for output: {average_f1_output}")
        else:
            print(f"数据不一致 (output): references={len(output_references)}, predictions={len(output_predictions)}")

        # 如果你想计算总的 F1 分数，可以将两部分合并
        all_references = instruction_references + output_references
        all_predictions = instruction_predictions + output_predictions
        if all_references and all_predictions and len(all_references) == len(all_predictions):
            P_all, R_all, F1_all = score(
                all_predictions,
                all_references,
                lang="zh",  # 指定中文
                model_type="bert-base-chinese",  # 使用中文BERT模型
                verbose=True,  # 显示进度条
                batch_size=64,  # 根据显存调整批次大小
                idf=False,  # 禁用IDF加权（若无参考文本统计）
                device=accelerator.device  # 使用当前GPU
            )
            average_f1_all = F1_all.mean().item()
            print(f"BERTScore F1 for all: {average_f1_all}")
        else:
            print(f"数据不一致 (all): references={len(all_references)}, predictions={len(all_predictions)}")

        # 将翻译后的数据集转换为列表（每个样本是字典）
        translated_list = translated_dataset.to_list()
        print(translated_list)
        # 保存为 JSON 数组
        with open("./alpaca_cleaned_chinese_qwen.json", "w", encoding="utf-8") as f:
            json.dump(translated_list, f, ensure_ascii=False, indent=4)

    accelerator.wait_for_everyone()
