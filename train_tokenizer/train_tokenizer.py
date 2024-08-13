import random
from tqdm import tqdm
from transformers import AutoTokenizer
import json 
from datasets import load_dataset
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)
import os

random.seed(42)

def train_tokenizer():
    # 读取JSON文件并提取文本数据
    def read_texts_from_json(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                yield data['text']
    
    data_path = './tokenizer_data/tokenizer_data.json'
    
    # 初始化tokenizer
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    
    # 设置训练器
    trainer = trainers.BpeTrainer(
        vocab_size=32000,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )
    
    # 读取文本数据
    texts = read_texts_from_json(data_path)
    
    # 训练tokenizer
    tokenizer.train_from_iterator(texts, trainer=trainer)
    
    # 设置解码器
    tokenizer.decoder = decoders.ByteLevel()
    
    # 保存tokenizer
    tokenizer_dir = "./miaomiao_tokenizer"
    os.makedirs(tokenizer_dir, exist_ok=True)
    tokenizer.save(os.path.join(tokenizer_dir, "tokenizer.json"))
    tokenizer.model.save("./miaomiao_tokenizer")
    
    # 手动创建配置文件
    config = {
        "auto_map": {
            "AutoTokenizer": [
                "tokenization_miaomiao.MiaomiaoTokenizer",
                None
            ]
        },
        "add_prefix_space": False,
        "added_tokens_decoder": {
            "32000": {
                "content": "system",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            },
            "32001": {
                "content": "user",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            },
            "32002": {
                "content": "assistant",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            },
            "32003": {
                "content": "<|endoftext|>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            },
            "32004": {
                "content": "<|im_start|>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            },
            "32005": {
                "content": "<|im_end|>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            }
        },
        "additional_special_tokens": ["<|im_start|>", "<|im_end|>"],
        "bos_token": None,
        "chat_template": "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\n你是一个由喵阿姨开发的喵喵小助手<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}",
        "clean_up_tokenization_spaces": False,
        "eos_token": "<|im_end|>",
        "errors": "replace",
        "model_max_length": 32768,
        "pad_token": "<|endoftext|>",
        "split_special_tokens": False,
        "tokenizer_class": "MiaomiaoTokenizer",
        "unk_token": None
    }
    
    # 保存配置文件
    with open(os.path.join(tokenizer_dir, "tokenizer_config.json"), "w", encoding="utf-8") as config_file:
        json.dump(config, config_file, ensure_ascii=False, indent=4)
    
    print("Tokenizer training completed and saved.")

def test_tokenizer():
    # 加载保存的分词器
    tokenizer = Tokenizer.from_file("./tokenizer/custom/tokenizer.json")
    
    # 测试分词器
    text = "hello word.You are a helpful assistant.今天，我们来训练一个大模型<|im_end|><|endoftext|>"
    encoding = tokenizer.encode(text)
    
    print("Original text:", text)
    print("Tokens:", encoding.tokens)
    print("Token IDs:", encoding.ids)
    # 获取词汇表
    vocab = tokenizer.get_vocab()
    
    # 获取特殊token的ID
    special_tokens=["<unk>", "<|endoftext|>", "<|im_start|>", "<|im_end|>", "system", "user", "assistant"]
    token_ids = {token: vocab[token] for token in special_tokens if token in vocab}
    
    print("Special tokens IDs:", token_ids)
    eos_token_id = token_ids.get("<|im_end|>", None)
    print("EOS token ID:", eos_token_id)
    print(vocab['<|im_end|>'])
    print(tokenizer.eos_token_id)



def main():
    
    train_tokenizer()
    #test_tokenizer()
    
if __name__ == '__main__':
    main()
