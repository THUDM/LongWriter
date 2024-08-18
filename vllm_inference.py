from vllm import LLM, SamplingParams

model = LLM(
    model= "THUDM/LongWriter-glm4-9b",
    dtype="auto",
    trust_remote_code=True,
    tensor_parallel_size=1,
    max_model_len=32768,
    gpu_memory_utilization=1,
)
tokenizer = model.get_tokenizer()

stop_token_ids = [tokenizer.eos_token_id, tokenizer.get_command("<|user|>"), tokenizer.get_command("<|observation|>")]
generation_params = SamplingParams(
    temperature=0.5,
    top_p=0.8,
    top_k=50,
    max_tokens=32768,
    repetition_penalty=1,
    stop_token_ids=stop_token_ids
)

query = "Write a 10000-word China travel guide"
input_ids = tokenizer.build_chat_input(query, history=[], role='user').input_ids[0].tolist()
outputs = model.generate(
    sampling_params=generation_params,
    prompt_token_ids=[input_ids],
)
output = outputs[0]
print(output.outputs[0].text)