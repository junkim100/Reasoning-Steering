from typing import List
from transformers import PreTrainedTokenizer

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
BASE_INPUT = "Input:"
BASE_RESPONSE = "\nResponse:"

ADD_FROM_POS_CHAT = E_INST
ADD_FROM_POS_BASE = BASE_RESPONSE


def tokenize_llama_chat(
    tokenizer: PreTrainedTokenizer,
    user_input: str,
    model_output: str = None,
    system_prompt: str = None,
) -> List[int]:
    input_content = ""
    if system_prompt is not None:
        input_content += B_SYS + system_prompt + E_SYS
    input_content += f"{B_INST} {user_input.strip()} {E_INST}"
    if model_output is not None:
        input_content += f" {model_output.strip()}"
    return tokenizer.encode(input_content)


def tokenize_llama_base(
    tokenizer, user_input: str, model_output: str = None
) -> List[int]:
    input_content = ""
    input_content += f"{BASE_INPUT} {user_input.strip()}"
    if model_output is not None:
        input_content += f"{BASE_RESPONSE} {model_output.strip()}"
    return tokenizer.encode(input_content)


def get_instruction_start_index(tokenizer, sys_prompt: str, inst_prompt: str) -> int:
    if sys_prompt is None or inst_prompt is None:
        return 0  # No system prompt provided; start at the beginning.
    prefix = B_SYS + sys_prompt + E_SYS + B_INST + inst_prompt + E_INST
    prefix_tokens = tokenizer.encode(prefix)
    return len(prefix_tokens)
