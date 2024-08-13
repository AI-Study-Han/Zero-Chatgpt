
import json
import time
from tqdm import tqdm
import re
import os
import pandas as pd
from datasketch import MinHash, MinHashLSH
import random


def process_baike():
    input_file = './563w_baidubaike.json'
    output_file = './baidubaike_no_depulication.json'
    batch_size = 100000

    processed_lines = 0
    start_time = time.time()
    # 正则表达式模式匹配 [1]、[2]、[3]、[1-2] 等内容
    bracket_pattern = re.compile(r'\[\d+(-\d+)?\]')
    punctuation_pattern = re.compile(r'[。！？：]$')
    chinese_char_pattern = re.compile(r'[\u4e00-\u9fa5]')
    repeated_punctuation_pattern = re.compile(r'([。！？])\1+')
    whitespace_pattern = re.compile(r'\s+|　+')


    def process_lines(lines, outfile):
        nonlocal processed_lines, start_time
        for line in lines:
            try:
                data = json.loads(line)
                text = ""
                title = data.get("title", "")
                summary = data.get("summary", "")
                if summary is None or summary.strip() == "":
                    text = f"{title}。"
                elif summary.startswith(title):
                    text = f"{summary}"
                    if not punctuation_pattern.search(text):
                        text += "。"
                else:
                    text = f"{title}，{summary}"
                    if not punctuation_pattern.search(text):
                        text += "。"
                skip_line = False
                sections = data.get("sections", [])
                for section in sections:
                    section_title = section.get("title", "")
                    if "重要参数" in section_title or "项目简介" in section_title or "产品简介" in section_title or "个人资料" in section_title or "个人简介" in section_title:
                        skip_line = True
                        break
                    section_content = section.get("content", "")
                    text += f"{section_title}，{section_content}"
                    if not punctuation_pattern.search(text):
                        text += "。"
                
                chinese_chars = chinese_char_pattern.findall(text)
                if skip_line or len(chinese_chars) < 30 or text.count(' ') > 10:
                    continue
                
                 # 移除所有空白字符（包括全角空格）
                text = re.sub(whitespace_pattern, '', text)
                # 移除文本中的 [1]、[2]、[3] 等内容
                text = re.sub(bracket_pattern, '', text)
                # 合并重复的标点符号
                text = re.sub(repeated_punctuation_pattern, r'\1', text)
                new_data = {
                    "text": text,
                    "source": "baidubaike"
                }
                
                outfile.write(json.dumps(new_data, ensure_ascii=False) + '\n')
                processed_lines += 1

            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
            except Exception as e:
                print(f"Error processing line: {e}")

        # Print total processed lines and processing speed
        elapsed_time = time.time() - start_time
        speed = processed_lines / elapsed_time
        tqdm.write(f"Processed {processed_lines} lines at {speed:.2f} lines/second")

    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        batch_lines = []
        for line in tqdm(infile, desc="Reading lines"):
            batch_lines.append(line)
            if len(batch_lines) == batch_size:
                process_lines(batch_lines, outfile)
                batch_lines = []

        # Process remaining lines
        if batch_lines:
            process_lines(batch_lines, outfile)

def process_cn_wiki():
    input_file = "./wikipedia-cn-20230720-filtered.json"
    output_file = "./wiki_cn.json"
    
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        data = json.load(infile)
        for entry in data:
            text = entry.get("completion", "")
            new_entry = {
                "text": text,
                "source": "wiki_cn"
            }
            json.dump(new_entry, outfile, ensure_ascii=False)
            outfile.write('\n')

    print("Processing complete. Output saved to", output_file)

def process_skypile():
    input_dir = "./SkyPile-50/"
    output_file = "./skypile.json"

    # 获取所有 .jsonl 文件列表
    jsonl_files = [f for f in os.listdir(input_dir) if f.endswith(".jsonl")]

    with open(output_file, 'w', encoding='utf-8') as outfile:
        # 初始化文件级别的进度条
        for filename in tqdm(jsonl_files, desc="Processing files"):
            input_file = os.path.join(input_dir, filename)
            with open(input_file, 'r', encoding='utf-8') as infile:
                for line in infile:
                    try:
                        data = json.loads(line)
                        text = data.get("text", "")
                        new_entry = {
                            "text": text,
                            "source": "skypile"
                        }
                        json.dump(new_entry, outfile, ensure_ascii=False)
                        outfile.write('\n')
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON in file {filename}: {e}")
                    except Exception as e:
                        print(f"Error processing line in file {filename}: {e}")

    print("Processing complete. Output saved to", output_file)

def ngrams(text, n=2):
    return [text[i:i+n] for i in range(len(text)-n+1)]

def process_line(line, num_perm):
    data = json.loads(line)
    text = data["text"]
    minhash = MinHash(num_perm=num_perm)
    for d in ngrams(text, 2):
        minhash.update(d.encode('utf-8'))
    return data, minhash

def depulication_cn_file(input_file, output_file, threshold):
    # MinHash-LSH 参数
    num_perm = 128
    lsh = MinHashLSH(threshold, num_perm=num_perm)
    key_counter = 0

    retained_lines = 0
    processed_lines = 0

    # 创建进度条
    pbar = tqdm(desc="Processing lines", unit="line", mininterval=0.1)

    with open(output_file, 'w', encoding='utf-8') as out_file:
        start_time = time.time()
        with open(input_file, 'r', encoding='utf-8') as file:
            for line in file:
                data, minhash = process_line(line, num_perm)
                unique_key = f"{data['source']}_{key_counter}"
                key_counter += 1
                if not lsh.query(minhash):
                    lsh.insert(unique_key, minhash)
                    json.dump(data, out_file, ensure_ascii=False)
                    out_file.write('\n')
                    retained_lines += 1
                processed_lines += 1
                pbar.update(1)
                elapsed_time = time.time() - start_time
                lines_per_second = processed_lines / elapsed_time if elapsed_time > 0 else 0
                pbar.set_postfix({"Retained": retained_lines, "Processed": processed_lines, "Speed": f"{lines_per_second:.2f} lines/sec"})

    # 关闭进度条
    pbar.close()

def depulication_cn_files():
    # 定义路径
    input_dir = "/home/"
    output_file = "/home/deduplicated_cn_data.json"
    # MinHash-LSH 参数
    num_perm = 128
    lsh = MinHashLSH(threshold=0.6, num_perm=num_perm)
    key_counter = 0

    retained_lines = 0
    processed_lines = 0

    # 创建进度条
    pbar = tqdm(desc="Processing lines", unit="line", mininterval=0.1)

    with open(output_file, 'w', encoding='utf-8') as out_file:
        start_time = time.time()
        for filename in os.listdir(input_dir):
            if filename.endswith(".json"):
                file_path = os.path.join(input_dir, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    for line in file:
                        data, minhash = process_line(line, num_perm)
                        unique_key = f"{data['source']}_{key_counter}"
                        key_counter += 1
                        if not lsh.query(minhash):
                            lsh.insert(unique_key, minhash)
                            json.dump(data, out_file, ensure_ascii=False)
                            out_file.write('\n')
                            retained_lines += 1
                        processed_lines += 1
                        pbar.update(1)
                        elapsed_time = time.time() - start_time
                        lines_per_second = processed_lines / elapsed_time if elapsed_time > 0 else 0
                        pbar.set_postfix({"Retained": retained_lines, "Processed": processed_lines, "Speed": f"{lines_per_second:.2f} lines/sec"})

    # 关闭进度条
    pbar.close()


def merge_data():
    input_files = [
        './baidubaike.json',
        './wiki_cn.json',
        './skypile.json'
    ]
    output_file = './pretrain.json'
    sampling_ratios = [1, 1, 0.57]  # 分别从每个文件中抽取100%、100%和57%的数据

    assert len(input_files) == len(sampling_ratios), "输入文件数和抽样比例数不匹配"
    
    line_counts = {}
    
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for file, ratio in zip(input_files, sampling_ratios):
            line_counts[file] = 0
            with open(file, 'r', encoding='utf-8') as in_f:
                for line in in_f:
                    if random.random() <= ratio:
                        data = json.loads(line.strip())
                        out_f.write(json.dumps(data, ensure_ascii=False) + '\n')
                        line_counts[file] += 1
    
    for file, count in line_counts.items():
        print(f"{file} 写入了 {count} 行")


def generate_train_tokenizer_data():
    input_files = [
        './baidubaike.json',
        './wiki_cn.json',
        './skypile.json'
    ]
    output_file = './train_tokenizer.json'
    sampling_ratios = [1, 0.5, 0.02]  # 分别从每个文件中抽取100%、50%和2%的数据

    assert len(input_files) == len(sampling_ratios), "输入文件数和抽样比例数不匹配"
    
    line_counts = {}
    
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for file, ratio in zip(input_files, sampling_ratios):
            line_counts[file] = 0
            with open(file, 'r', encoding='utf-8') as in_f:
                for line in in_f:
                    if random.random() < ratio:
                        data = json.loads(line.strip())
                        out_f.write(json.dumps(data, ensure_ascii=False) + '\n')
                        line_counts[file] += 1
    
    for file, count in line_counts.items():
        print(f"{file} 写入了 {count} 行")

def sft_process_firefly():
    input_data_path = './firefly-cn-train-1.1M.jsonl'
    output_data_path = './processed_firefly.jsonl'

    line_count = 0

    with open(input_data_path, 'r', encoding='utf-8') as infile, open(output_data_path, 'w', encoding='utf-8') as outfile:
        for line in infile:
            data = json.loads(line)
            conversations = data.get("conversations", [])
            if len(conversations) == 2 and conversations[0].get("from") == "human" and conversations[1].get("from") == "gpt":
                human_value = conversations[0].get("value", "")
                if len(human_value) > 5:
                    new_data = {
                        "messages": [
                            {"from": "user", "value": human_value},
                            {"from": "assistant", "value": conversations[1].get("value", "")}
                        ]
                    }
                    outfile.write(json.dumps(new_data, ensure_ascii=False) + '\n')
                    line_count += 1

    print(f"Total lines written: {line_count}")

def process_sft_line(line, num_perm):
    data = json.loads(line)
    messages = data.get("messages", [])
    combined_text = ''.join([msg['value'] for msg in messages if msg['from'] in ['user', 'assistant']])
    minhash = MinHash(num_perm=num_perm)
    for d in ngrams(combined_text, 2):
        minhash.update(d.encode('utf-8'))
    return data, minhash

def depulication_cn_firefly():
    # 定义路径
    input_file = "./processed_firefly.jsonl"
    output_file = "./depulication_firefly.jsonl"
    # MinHash-LSH 参数
    num_perm = 128
    lsh = MinHashLSH(threshold=0.4, num_perm=num_perm)
    key_counter = 0

    retained_lines = 0
    processed_lines = 0

    # 创建进度条
    pbar = tqdm(desc="Processing lines", unit="line", mininterval=0.1)

    with open(output_file, 'w', encoding='utf-8') as out_file:
        start_time = time.time()
        with open(input_file, 'r', encoding='utf-8') as file:
            for line in file:
                data, minhash = process_sft_line(line, num_perm)
                unique_key = f"{key_counter}"
                key_counter += 1
                if not lsh.query(minhash):
                    lsh.insert(unique_key, minhash)
                    json.dump(data, out_file, ensure_ascii=False)
                    out_file.write('\n')
                    retained_lines += 1
                processed_lines += 1
                pbar.update(1)
                elapsed_time = time.time() - start_time
                lines_per_second = processed_lines / elapsed_time if elapsed_time > 0 else 0
                pbar.set_postfix({"Retained": retained_lines, "Processed": processed_lines, "Speed": f"{lines_per_second:.2f} lines/sec"})

    # 关闭进度条
    pbar.close()

def generate_sft_rlfh_data():
    cn_firefly_file = './depulication_firefly.jsonl'
    ruozhiba_file = './ruozhiout_qa_cn.jsonl'
    total_count = 400000
    sft_count = 300000
    #rlhf_count = total_count - sft_count
    output_file_sft = './sft.jsonl'
    output_file_rlfh = './rlhf.jsonl'
    # Load data from files
    with open(ruozhiba_file, 'r', encoding='utf-8') as f:
        ruozhiba_data = [json.loads(line) for line in f]

    with open(cn_firefly_file, 'r', encoding='utf-8') as f:
        cn_firefly_data = [json.loads(line) for line in f]

    # Extract conversations from ruozhiba_data
    sft_data = []
    for item in ruozhiba_data:
        for conversation in item['conversations']:
            if conversation['from'] == 'human':
                prompt = conversation['value']
            elif conversation['from'] == 'gpt':
                answer = conversation['value']
        sft_data.append({'prompt': prompt, 'answer': answer})

    # Calculate the number of entries to pick from cn_firefly_data
    ruozhiba_count = len(sft_data)
    cn_firefly_count = total_count - ruozhiba_count

    # Randomly select entries from cn_firefly_data
    random_cn_firefly_data = random.sample(cn_firefly_data, cn_firefly_count)

    # Extract messages from cn_firefly_data
    for item in random_cn_firefly_data:
        for message in item['messages']:
            if message['from'] == 'user':
                prompt = message['value']
            elif message['from'] == 'assistant':
                answer = message['value']
        sft_data.append({'prompt': prompt, 'answer': answer})

    # Randomly shuffle the data
    random.shuffle(sft_data)

    # Split the data into two parts
    #split_index = len(sft_data) // 2
    sft_part = sft_data[:sft_count]
    rlhf_part = sft_data[sft_count:]

    # Write data to output files
    with open(output_file_sft, 'w', encoding='utf-8') as f:
        for item in sft_part:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    with open(output_file_rlfh, 'w', encoding='utf-8') as f:
        for item in rlhf_part:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def main():
    # process_baike()#保留2,763,469行
    #process_cn_wiki()
    #process_skypile()
    #百度百科去重
    # depulication_cn_file('./baidubaike_no_depulication.json', './baidubaike.json', 0.4)
    # merge_data()
    # generate_train_tokenizer_data()
    #sft_process_firefly()
    #depulication_cn_firefly()
    generate_sft_rlfh_data()

    
if __name__ == '__main__':
    main()