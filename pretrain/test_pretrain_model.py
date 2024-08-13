from transformers import AutoModelForCausalLM, AutoTokenizer


device = "cuda" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained(
    './pretrain/model',
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained('./miaomiao_tokenizer', trust_remote_code=True)
text = "床前明月光，"
model_inputs = tokenizer([text], return_tensors="pt").to(device)
print(model_inputs)
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=1024
)
print(generated_ids)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
print(generated_ids)
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
