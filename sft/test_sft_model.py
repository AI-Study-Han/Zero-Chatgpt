from transformers import AutoModelForCausalLM, AutoTokenizer


device = "cuda" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained(
    './model',
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained('./miaomiao_tokenizer', trust_remote_code=True)

prompt_list = ["你知道北京吗？  ",
               "你知道杭州有哪些美食吗？",
               "你知道中国的四大名著吗？",
               "你了解美国的历史吗？",
               "左手一只鸭，右手一只鸡。交换两次后左右手里各是什么？",
               "鸡兔同笼，共35只头，94只脚，问鸡兔各多少？",
               "世界上最大的动物是什么？",
               "介绍一下刘德华。",
               "介绍一下中国。"
               ]
for prompt in prompt_list:
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature = 0.9,
        top_k = 30
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(f"question:{prompt}")
    print(f"response:{response}")