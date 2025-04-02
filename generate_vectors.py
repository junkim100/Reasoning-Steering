"""
Generates steering vectors for each layer of the model by averaging the activations of all the positive and negative examples.

Example usage:
python generate_vectors.py --layers $(seq 0 31) --save_activations --use_base_model --model_size 7b --behaviors sycophancy
"""

import json
import torch as t
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import os
from dotenv import load_dotenv
from llama_wrapper import LlamaWrapper
import argparse
from typing import List
from utils.tokenize import (
    tokenize_llama_base,
    tokenize_llama_chat,
    get_instruction_start_index,
)
from behaviors import (
    get_vector_dir,
    get_activations_dir,
    get_ab_data_path,
    get_vector_path,
    get_activations_path,
    ALL_BEHAVIORS,
)

load_dotenv()

HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN")
SYSTEM_PROMPT = "You are an helpful assistant."


class ComparisonDataset(Dataset):
    def __init__(self, data_path, token, model_name_path, use_chat, system_prompt):
        with open(data_path, "r") as f:
            self.data = json.load(f)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_path, token=token)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.use_chat = use_chat
        self.system_prompt = system_prompt

    def prompt_to_tokens(self, instruction, model_output):
        if self.use_chat:
            start_index = get_instruction_start_index(
                self.tokenizer, self.system_prompt
            )
            tokens = tokenize_llama_chat(
                self.tokenizer,
                user_input=instruction,
                model_output=model_output,
                system_prompt=self.system_prompt,
            )[start_index:]
        else:
            tokens = tokenize_llama_base(
                self.tokenizer,
                user_input=instruction,
                model_output=model_output,
            )
        return t.tensor(tokens).unsqueeze(0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        p_text = item["answer_matching_behavior"]
        n_text = item["answer_not_matching_behavior"]
        q_text = item["question"]
        p_tokens = self.prompt_to_tokens(q_text, p_text)
        n_tokens = self.prompt_to_tokens(q_text, n_text)
        return p_tokens, n_tokens


def generate_save_vectors_for_behavior(
    layers: List[int],
    save_activations: bool,
    behavior: List[str],
    model: LlamaWrapper,
    debug: bool = False,  # set to True to enable debug prints
):
    data_path = get_ab_data_path(behavior)
    if not os.path.exists(get_vector_dir(behavior)):
        os.makedirs(get_vector_dir(behavior))
    if save_activations and not os.path.exists(get_activations_dir(behavior)):
        os.makedirs(get_activations_dir(behavior))

    model.set_save_internal_decodings(False)
    model.reset_all()

    # These dictionaries will now hold, for each sample, the
    # activation tensor for every token in the sliced input.
    pos_activations = {layer: [] for layer in layers}
    neg_activations = {layer: [] for layer in layers}

    dataset = ComparisonDataset(
        data_path,
        HUGGINGFACE_TOKEN,
        model.model_name_path,
        model.use_chat,
        SYSTEM_PROMPT,
    )

    # --- Debug block: Check that the tokens returned from prompt_to_tokens
    # contain only the user instruction (and model output) without the system prompt.
    if debug:
        # Get the first positive example.
        p_tokens_example, _ = dataset[0]  # shape: [1, seq_length]
        # Decode and print the text.
        sliced_token_ids = p_tokens_example.squeeze().tolist()  # removes batch dim
        decoded_text = dataset.tokenizer.decode(sliced_token_ids)
        expected_token_len = len(sliced_token_ids)
        print("DEBUG: Sliced tokenized text (after system prompt):")
        print(decoded_text)
        print(
            "DEBUG: Expected number of tokens after system prompt:", expected_token_len
        )
    # -------------------------------------------------------

    for p_tokens, n_tokens in tqdm(dataset, desc="Processing prompts"):
        # p_tokens and n_tokens are the sliced tokens (without system prompt)
        p_tokens = p_tokens.to(model.device)  # shape: [1, seq_length]
        n_tokens = n_tokens.to(model.device)

        # Process positive example
        model.reset_all()
        model.get_logits(p_tokens)
        for layer in layers:
            # Get the activations for the entire sequence (all tokens after system prompt)
            # get_last_activations returns a tensor of shape [batch_size, seq_length, hidden_dim]
            act = model.get_last_activations(layer)[0, :, :].detach().cpu()
            pos_activations[layer].append(act)
            if debug:
                print(
                    f"DEBUG: Layer {layer} positive activation shape: {act.shape} "
                    f"(should be ({p_tokens.size(1)}, hidden_dim))"
                )

        # Process negative example
        model.reset_all()
        model.get_logits(n_tokens)
        for layer in layers:
            act = model.get_last_activations(layer)[0, :, :].detach().cpu()
            neg_activations[layer].append(act)
            if debug:
                print(
                    f"DEBUG: Layer {layer} negative activation shape: {act.shape} "
                    f"(should be ({n_tokens.size(1)}, hidden_dim))"
                )

    # For each layer, we now aggregate the activations across all examples.
    # Here, for example, we average over all token positions for each example
    # and then average over all examples to get a single vector.
    for layer in layers:
        pos_sample_avg = [
            act.mean(dim=0) for act in pos_activations[layer]
        ]  # each: [hidden_dim]
        neg_sample_avg = [act.mean(dim=0) for act in neg_activations[layer]]
        pos_avg = t.stack(pos_sample_avg).mean(dim=0)
        neg_avg = t.stack(neg_sample_avg).mean(dim=0)
        vec = pos_avg - neg_avg
        t.save(vec, get_vector_path(behavior, layer, model.model_name_path))
        if save_activations:
            t.save(
                pos_activations[layer],
                get_activations_path(behavior, layer, model.model_name_path, "pos"),
            )
            t.save(
                neg_activations[layer],
                get_activations_path(behavior, layer, model.model_name_path, "neg"),
            )


def generate_save_vectors(
    layers: List[int],
    save_activations: bool,
    use_base_model: bool,
    model_size: str,
    behaviors: List[str],
    debug: bool = False,
):
    """
    layers: list of layers to generate vectors for
    save_activations: if True, save the activations for each layer
    use_base_model: Whether to use the base model instead of the chat model
    model_size: size of the model to use"
    behaviors: behaviors to generate vectors for
    """
    model = LlamaWrapper(
        HUGGINGFACE_TOKEN, size=model_size, use_chat=not use_base_model
    )
    for behavior in behaviors:
        generate_save_vectors_for_behavior(
            layers, save_activations, behavior, model, debug
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", nargs="+", type=int, default=list(range(32)))
    parser.add_argument("--save_activations", action="store_true", default=False)
    parser.add_argument("--use_base_model", action="store_true", default=False)
    parser.add_argument("--model_size", type=str, choices=["8B", "70B"], default="8B")
    parser.add_argument("--behaviors", nargs="+", type=str, default=ALL_BEHAVIORS)
    parser.add_argument("--debug", action="store_true", default=False)

    args = parser.parse_args()
    generate_save_vectors(
        args.layers,
        args.save_activations,
        args.use_base_model,
        args.model_size,
        args.behaviors,
        args.debug,
    )
