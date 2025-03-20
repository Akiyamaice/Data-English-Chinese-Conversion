import json
import re

def calculate_english_ratio(text):
    # 统计英文字母数量（A-Za-z）
    english_chars = len(re.findall(r'[A-Za-z]', text))
    total_chars = len(text)
    if total_chars == 0:
        return 0.0
    return (english_chars / total_chars) * 100

def evaluate_translation(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    passed_count = 0
    total_count = len(data)

    for item in data:
        # 拼接instruction、input、output（空字段自动忽略）
        combined_text = item['instruction'] + item['input'] + item['output']
        ratio = calculate_english_ratio(combined_text)
        if ratio <= 8.0:
            passed_count += 1
    # 计算达标率（保留两位小数）
    compliance_rate = round((passed_count / total_count) * 100, 2) if total_count > 0 else 0.0
    return compliance_rate


if __name__ == "__main__":
    score = evaluate_translation("/107552401627/esm_code/code/practice_code/alpaca_cleaned_chinese_qwen.json")
    print(f"score：{score}%")