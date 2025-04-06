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
    get_reasoning_data_path,
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
            self.data = json.load(f)#, encoding="utf-8")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_path, token=token)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.use_chat = use_chat
        self.system_prompt = system_prompt

    def prompt_to_tokens(self, instruction, model_output):
        if self.use_chat:
            start_index = get_instruction_start_index(
                self.tokenizer, self.system_prompt, instruction
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
    use_svd: bool = False,
):
    data_path = get_reasoning_data_path(behavior="reasoning")
    if not os.path.exists(get_vector_dir(behavior)):
        os.makedirs(get_vector_dir(behavior))
    if save_activations and not os.path.exists(get_activations_dir(behavior)):
        os.makedirs(get_activations_dir(behavior))

    print(f"the data path is : {data_path}\n")
    model.set_save_internal_decodings(False)
    model.reset_all()

    pos_activations = dict([(layer, []) for layer in layers])
    neg_activations = dict([(layer, []) for layer in layers])

    dataset = ComparisonDataset(
        data_path,
        HUGGINGFACE_TOKEN,
        model.model_name_path,
        model.use_chat,
        SYSTEM_PROMPT,
    )

    # Process in smaller batches
    batch_size = 1  # Adjust based on your GPU memory
    all_data = [(dataset[i][0], dataset[i][1]) for i in range(len(dataset))]

    for i in range(0, len(all_data), batch_size):
        batch_data = all_data[i : i + batch_size]
        for p_tokens, n_tokens in tqdm(
            batch_data,
            desc=f"Processing batch {i//batch_size + 1}/{len(all_data)//batch_size + 1}",
        ):
            try:
                p_tokens = p_tokens.to(model.device)
                model.reset_all()
                model.get_logits(p_tokens)
                for layer in layers:
                    p_activations_all = model.get_last_activations(layer)
                    p_activations_mean = (
                        p_activations_all[0, :, :].mean(dim=0).detach().cpu()
                    )
                    pos_activations[layer].append(p_activations_mean)

                # Clear GPU memory
                del p_tokens, p_activations_all
                t.cuda.empty_cache()

                n_tokens = n_tokens.to(model.device)
                model.reset_all()
                model.get_logits(n_tokens)
                for layer in layers:
                    n_activations_all = model.get_last_activations(layer)
                    n_activations_mean = (
                        n_activations_all[0, :, :].mean(dim=0).detach().cpu()
                    )
                    neg_activations[layer].append(n_activations_mean)

                # Clear GPU memory
                del n_tokens, n_activations_all
                t.cuda.empty_cache()

            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    print(f"CUDA OOM detected, skipping this sample and continuing")
                    t.cuda.empty_cache()
                else:
                    raise e

    for layer in layers:
        all_pos_layer = t.stack(pos_activations[layer])
        all_neg_layer = t.stack(neg_activations[layer])

        if use_svd:
            # Apply SVD to positive activations
            U_pos, S_pos, Vh_pos = t.linalg.svd(all_pos_layer, full_matrices=False)
            # Extract the principal direction vector from positive activations
            pos_direction = Vh_pos[0]

            # Apply SVD to negative activations
            U_neg, S_neg, Vh_neg = t.linalg.svd(all_neg_layer, full_matrices=False)
            # Extract the principal direction vector from negative activations
            neg_direction = Vh_neg[0]

            pos_direction = pos_direction / pos_direction.norm()
            neg_direction = neg_direction / neg_direction.norm()
            steering_vector = pos_direction - neg_direction
        else:
            # Calculate the difference between means
            steering_vector = all_pos_layer.mean(dim=0) - all_neg_layer.mean(dim=0)
            # Normalize the steering vector to unit length
            steering_vector = steering_vector / steering_vector.norm()

        # Verify that the vector is properly normalized (within a small tolerance)
        expected_norm = 1.0  # Unit length
        actual_norm = steering_vector.norm().item()
        assert abs(actual_norm - expected_norm) < 1e-5, f"Vector not properly normalized: expected {expected_norm}, got {actual_norm}"

        # Save both the raw difference vector and the direction vector
        t.save(
            steering_vector,
            get_vector_path(behavior, layer, model.model_name_path),
        )

        if save_activations:
            t.save(
                all_pos_layer,
                get_activations_path(behavior, layer, model.model_name_path, "pos"),
            )
            t.save(
                all_neg_layer,
                get_activations_path(behavior, layer, model.model_name_path, "neg"),
            )


def generate_save_vectors(
    layers: List[int],
    save_activations: bool,
    use_base_model: bool,
    model_size: str,
    behaviors: List[str],
    use_svd: bool = False,
    normalize_across_behaviors: bool = True,
):
    """
    layers: list of layers to generate vectors for
    save_activations: if True, save the activations for each layer
    use_base_model: Whether to use the base model instead of the chat model
    model_size: size of the model to use, either "7b" or "13b"
    behaviors: behaviors to generate vectors for
    use_svd: Whether to use SVD to extract principal directions
    normalize_across_behaviors: Whether to normalize vectors across behaviors to have the same norm per layer
    """
    model = LlamaWrapper(
        HUGGINGFACE_TOKEN, size=model_size, use_chat=not use_base_model
    )
    print(model)

    # First, generate all vectors for all behaviors
    all_vectors = {layer: {} for layer in layers}

    for behavior in behaviors:
        generate_save_vectors_for_behavior(layers, save_activations, behavior, model, use_svd)

        # Load the generated vectors for normalization across behaviors
        if normalize_across_behaviors:
            for layer in layers:
                vector_path = get_vector_path(behavior, layer, model.model_name_path)
                if os.path.exists(vector_path):
                    all_vectors[layer][behavior] = t.load(vector_path)

    # Normalize vectors across behaviors if requested
    if normalize_across_behaviors:
        # Create normalized_vectors directory
        normalized_vectors_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "normalized_vectors")
        if not os.path.exists(normalized_vectors_dir):
            os.makedirs(normalized_vectors_dir)

        for layer in layers:
            if not all_vectors[layer]:  # Skip if no vectors for this layer
                continue

            # Calculate norms for each behavior's vector
            norms = {behavior: vec.norm().item() for behavior, vec in all_vectors[layer].items()}

            # Calculate mean norm across behaviors for this layer
            mean_norm = t.tensor(list(norms.values())).mean().item()

            # Normalize all vectors to have the same norm
            for behavior, vec in all_vectors[layer].items():
                # Normalize the vector to have the mean norm
                normalized_vec = vec * mean_norm / norms[behavior]

                # Verify normalization (within a small tolerance)
                assert abs(normalized_vec.norm().item() - mean_norm) < 1e-5, f"Vector for {behavior} at layer {layer} not properly normalized"

                # Save the normalized vector
                normalized_path = get_vector_path(behavior, layer, model.model_name_path).replace("vectors", "normalized_vectors")
                if not os.path.exists(os.path.dirname(normalized_path)):
                    os.makedirs(os.path.dirname(normalized_path))
                t.save(normalized_vec, normalized_path)

            print(f"Layer {layer}: Normalized vectors across {len(all_vectors[layer])} behaviors to norm {mean_norm:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", nargs="+", type=int, default=list(range(32)))
    parser.add_argument("--save_activations", action="store_true", default=False)
    parser.add_argument("--use_base_model", action="store_true", default=False)
    parser.add_argument("--model_size", type=str, choices=["7b", "13b"], default="7b")
    parser.add_argument("--behaviors", nargs="+", type=str, default=ALL_BEHAVIORS)
    parser.add_argument("--use_svd", action="store_true", default=False)
    parser.add_argument("--normalize", action="store_true", default=True,
                      help="Normalize vectors across behaviors to have the same norm per layer")

    args = parser.parse_args()
    generate_save_vectors(
        args.layers,
        args.save_activations,
        args.use_base_model,
        args.model_size,
        args.behaviors,
        args.use_svd,
        args.normalize,
    )
