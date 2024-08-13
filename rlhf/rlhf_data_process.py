from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from tqdm import tqdm
import torch
import random
import os
def split_jsonl():
    input_file = './rlhf.jsonl'
    output_files = ['./rlhf_part1.jsonl', './rlhf_part2.jsonl', './rlhf_part3.jsonl', './rlhf_part4.jsonl']

    # 读取输入文件的内容
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 打乱顺序
    random.shuffle(lines)

    # 计算每个文件的行数
    num_lines = len(lines)
    chunk_size = num_lines // 4

    # 将行分成 4 组
    chunks = [lines[i * chunk_size: (i + 1) * chunk_size] for i in range(4)]
    
    # 如果有多余的行，均匀分配到各个文件
    for i in range(num_lines % 4):
        chunks[i].append(lines[4 * chunk_size + i])

    # 将每组写入不同的输出文件
    for i, output_file in enumerate(output_files):
        with open(output_file, 'w', encoding='utf-8') as out_f:
            for line in chunks[i]:
                out_f.write(line)


def generate_rlhf_data():
    input_file = './rlhf_part4.jsonl'
    model_path = './sft_model'
    output_file = './rlhf_generate_part4.jsonl'
    device = "cuda"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # 读取输入文件的内容
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 检查 output_file 是否存在以及已经写了多少条数据
    existing_data = []
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as out_f:
            existing_data = out_f.readlines()

    processed_prompts = set()
    for line in existing_data:
        data = json.loads(line)
        processed_prompts.add(data['prompt'])

    # 处理每一行 JSON 对象
    with open(output_file, 'a', encoding='utf-8') as out_f:
        for line in tqdm(lines, desc="Processing"):
            data = json.loads(line)
            prompt = data['prompt']

            answer = data['answer']
            messages = [
                {"role": "user", "content": prompt}
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            if text in processed_prompts:
                continue  # Skip already processed prompts
            model_inputs = tokenizer([text], return_tensors="pt").to(device)

            with torch.no_grad():
                generated_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=512,
                    do_sample=False
                )
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]

            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            result = {
                'prompt': text,
                'response': answer + tokenizer.eos_token,
                'chosen': answer + tokenizer.eos_token,
                'rejected': response + tokenizer.eos_token
            }
            # 将结果写入输出文件
            out_f.write(json.dumps(result, ensure_ascii=False) + '\n')


def process_step_2_3_data():
    input_files = [
        './rlhf_generate_part1.jsonl',
        './rlhf_generate_part2.jsonl',
        './rlhf_generate_part3.jsonl',
        './rlhf_generate_part4.jsonl'
    ]
    output_step2_train_file = './step2_data/train.jsonl'
    output_step2_eval_file = './step2_data/eval.jsonl'
    output_step3_train_file = './step3_data/train.jsonl'
    output_step3_eval_file = './step3_data/eval.jsonl'

    data = []

    # 读取所有输入文件
    for file in input_files:
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))

    # 随机打乱数据
    random.shuffle(data)

    # 分割数据
    total_size = len(data)
    step3_train_size = int(total_size * 0.95)
    step3_eval_size = total_size - step3_train_size
    step2_train_size = int(total_size * 0.475)
    step2_eval_size = int(total_size * 0.025)

    step3_train_data = data[:step3_train_size]
    step3_eval_data = data[step3_train_size:]

    step2_train_data = data[:step2_train_size]
    step2_eval_data = data[step2_train_size:step2_train_size + step2_eval_size]

    # 写入输出文件
    with open(output_step3_train_file, 'w', encoding='utf-8') as f:
        for item in step3_train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    with open(output_step3_eval_file, 'w', encoding='utf-8') as f:
        for item in step3_eval_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    with open(output_step2_train_file, 'w', encoding='utf-8') as f:
        for item in step2_train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    with open(output_step2_eval_file, 'w', encoding='utf-8') as f:
        for item in step2_eval_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')





def main():
    #split_jsonl()

    #generate_rlhf_data()
    process_step_2_3_data()




if __name__ == "__main__":
    main()
