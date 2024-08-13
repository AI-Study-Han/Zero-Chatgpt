import os
import json
import time
import random
import numpy as np
from multiprocessing import Process, Manager
from transformers import AutoTokenizer

def split_file(data_path, num_splits=20):
    file_handles = [open(f"{data_path}.part{i}", 'w', encoding='utf-8') for i in range(num_splits)]
    
    try:
        total_lines = 0
        with open(data_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                part_idx = i % num_splits
                file_handles[part_idx].write(line)
                total_lines += 1
                if total_lines % 1000 == 0:  # 每处理1000行打印一次进度
                    print(f"Processed lines: {total_lines}")
    finally:
        for handle in file_handles:
            handle.close()
        print(f"Total lines processed: {total_lines}")

def process_file(part_path, bin_path, tokenizer, ratio, result_dict):
    source_token_counts = {}
    total_token_count = 0
    line_count = 0
    start_time = time.time()
    
    with open(part_path, 'r', encoding='utf-8') as f, open(bin_path, 'wb') as f2:
        for line in f:
            if random.random() > ratio:
                continue
            data = json.loads(line)
            text = data['text']
            source = data['source']
            text_id = tokenizer.encode(text, add_special_tokens=False)
            text_id.append(tokenizer.eos_token_id)
            
            token_count = len(text_id)
            if source not in source_token_counts:
                source_token_counts[source] = 0
            source_token_counts[source] += token_count
            
            total_token_count += token_count
            
            arr = np.array(text_id, dtype=np.uint16)
            f2.write(arr.tobytes())
            
            line_count += 1
            elapsed_time = time.time() - start_time
            print(f"Processed lines: {line_count}, Time elapsed: {elapsed_time:.2f} seconds")

    result_dict[part_path] = (source_token_counts, total_token_count)

def merge_bins(bin_paths, final_bin_path, chunk_size=10*1024*1024):
    with open(final_bin_path, 'wb') as f_out:
        for bin_path in bin_paths:
            with open(bin_path, 'rb') as f_in:
                while True:
                    chunk = f_in.read(chunk_size)
                    if not chunk:
                        break
                    f_out.write(chunk)

def main(data_path, bin_path, ratio=1):
    tokenizer_path = './miaomiao_tokenizer'
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path,trust_remote_code=True)  # 主进程加载tokenizer
    num_splits = 20
    
    # Split the file into parts
    split_file(data_path, num_splits)
    
    manager = Manager()
    result_dict = manager.dict()
    
    processes = []
    bin_paths = [f"{bin_path}.part{i}.bin" for i in range(num_splits)]
    
    for i in range(num_splits):
        part_path = f"{data_path}.part{i}"
        bin_part_path = bin_paths[i]
        p = Process(target=process_file, args=(part_path, bin_part_path, tokenizer, ratio, result_dict))
        processes.append(p)
        p.start()
    
    for p in processes:
        p.join()
    
    # Merge binary files
    merge_bins(bin_paths, bin_path)
    
    # Output combined statistics
    combined_source_token_counts = {}
    combined_total_token_count = 0
    
    for source_token_counts, total_token_count in result_dict.values():
        for source, count in source_token_counts.items():
            if source not in combined_source_token_counts:
                combined_source_token_counts[source] = 0
            combined_source_token_counts[source] += count
        combined_total_token_count += total_token_count
    
    print("Token counts by source:", combined_source_token_counts)
    print("Total token count:", combined_total_token_count)

if __name__ == "__main__":
    #一共15M行
    data_path = "./pretrain_data_train.json"
    bin_path = "./pretrain_data_train.bin"
    main(data_path, bin_path, ratio=1)
