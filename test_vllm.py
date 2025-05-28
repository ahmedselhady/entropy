from vllm import LLM, SamplingParams

# Initialize the model
model = LLM(model="Qwen/Qwen2.5-3B")

# Define sampling parameters
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.95,
    max_tokens=256,
    stop=["</s>", "\n\n"]  # Common stop tokens
)

# Example prompt
prompt = "Write a short story about a robot learning to paint:"

# Generate response
outputs = model.generate([prompt], sampling_params)

# Get the generated text
generated_text = outputs[0].outputs[0].text
print(f"Generated text:\n{generated_text}")
