from transformers import AutoTokenizer

model_id = "hub/models--Qwen--Qwen3-VL-32B-Thinking"
try:
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    print(f"Tokenizer type: {type(tokenizer)}")
    print(f"Has all_special_tokens_extended: {hasattr(tokenizer, 'all_special_tokens_extended')}")
    print(f"Has num_special_tokens_to_add: {hasattr(tokenizer, 'num_special_tokens_to_add')}")
    
    # List all attributes containing 'special_tokens'
    special_attrs = [attr for attr in dir(tokenizer) if 'special_tokens' in attr]
    print(f"Special token attributes: {special_attrs}")
except Exception as e:
    print(f"Error loading tokenizer: {e}")
