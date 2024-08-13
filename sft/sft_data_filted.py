from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
import json
from tqdm import tqdm

def calculate_perplexity(model, tokenizer, messages, device):
    
    formatted_messages = [
        {"role": "user", "content": messages[0]['value']},
        {"role": "assistant", "content": messages[1]['value']}
    ]
    user_input = [
        {"role": "user", "content": messages[0]['value']}
    ]

    # 编码输入
    inputs_text = tokenizer.apply_chat_template(
        user_input,
        tokenize=False,
        add_generation_prompt=True
    )
    #print(inputs_text)
    inputs = tokenizer(inputs_text, return_tensors="pt").to(device)

     # 编码输入
    full_text = tokenizer.apply_chat_template(
        formatted_messages,
        tokenize=False,
        add_generation_prompt=False
    )
    #print(full_text)
    
    full_inputs = tokenizer(full_text, return_tensors="pt").to(device)

    # 计算给定用户输入情况下生成助理响应的困惑度
    with torch.no_grad():
        outputs = model(**full_inputs)
        logits = outputs.logits

        # 只关注助理响应部分的logits
        start_pos = inputs.input_ids.size(1)
        shift_logits = logits[:, start_pos:-1, :].contiguous()
        shift_labels = full_inputs['input_ids'][:, start_pos+1:].contiguous()

        # 计算交叉熵损失
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss = loss.view(shift_labels.size())
        perplexity_given_user = torch.exp(loss.mean())
        #print(f"给定输入的困惑度：{perplexity_given_user}")
    # 计算直接生成助理响应的困惑度
    with torch.no_grad():
        assistant_input = messages[1]["value"]
        assistant_inputs = tokenizer(assistant_input, return_tensors="pt").to(device)
        
        outputs = model(**assistant_inputs)
        logits = outputs.logits

        # Shift the logits and labels to ignore the first token
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = assistant_inputs['input_ids'][:, 1:].contiguous()

        # Flatten the logits and labels
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss = loss.view(shift_labels.size())

        # 计算每个token的困惑度
        perplexity_direct = torch.exp(loss.mean())
        #print(f"直接生成的困惑度：{perplexity_direct}")

    return perplexity_given_user.item(), perplexity_direct.item()

def main():
    model_name = "./Qwen2-0.5B-Instruct"
    device = "cuda"  # 设备
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    input_data = './depulication_firefly.jsonl'
    output_data = './depulication_firefly_ppl.jsonl'
    # 打开输入文件
    with open(input_data, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    with open(output_data, 'w', encoding='utf-8') as out_f:
        # 逐行处理输入数据并计算困惑度
        for line in tqdm(lines, desc="Processing"):
            data = json.loads(line)
            messages = data["messages"]
            perplexity_given_user, perplexity_direct = calculate_perplexity(model, tokenizer, messages, device)
            result = {
                "messages": messages,
                "ppl_a_q": perplexity_given_user,
                "ppl_a": perplexity_direct,
                "ifd": perplexity_given_user / perplexity_direct,
            }
            out_f.write(json.dumps(result, ensure_ascii=False) + '\n')
            # print(f"Perplexity given user input: {perplexity_given_user}")
            # print(f"Perplexity of direct assistant response: {perplexity_direct}")

if __name__ == "__main__":
    main()
