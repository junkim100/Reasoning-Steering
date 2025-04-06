"""
Evaluate models using saved steering vectors with lm-eval

Usage:
python evaluate_with_steering.py --behavior reasoning --layers 10 --multipliers 0.1 0.5 1 2 5 10 --task gsm8k --model_size 7b

This script evaluates language models with steering vectors applied using the lm-evaluation-harness.
By default, it uses the GSM8K dataset for mathematical reasoning evaluation.
"""

import os
import json
import torch as t
import argparse
from typing import List, Dict, Optional, Tuple, Union
from tqdm import tqdm
from dotenv import load_dotenv
import lm_eval
from lm_eval.utils import setup_logging
from lm_eval.api.model import LM

from llama_wrapper import LlamaWrapper
from behaviors import (
    get_steering_vector,
    get_system_prompt,
    get_results_dir,
    ALL_BEHAVIORS,
)
from steering_settings import SteeringSettings

load_dotenv()

HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN")


class LlamaWrapperForLMEval(LM):
    """
    Adapter class to make LlamaWrapper compatible with lm-eval's LM interface
    """
    def __init__(
        self,
        llama_wrapper: LlamaWrapper,
        layer: int = None,
        vector = None,
        multiplier: float = None,
        system_prompt: Optional[str] = None,
        batch_size: int = 1
    ):
        super().__init__()
        self.model = llama_wrapper
        self.layer = layer
        self.vector = vector
        self.multiplier = multiplier
        self.system_prompt = system_prompt
        self.batch_size = batch_size

        # Apply steering vector if provided
        if layer is not None and vector is not None and multiplier is not None:
            self.model.reset_all()
            self.model.set_add_activations(layer, multiplier * vector)

    def loglikelihood(self, requests):
        """
        Calculate log-likelihood for each request
        """
        results = []
        for request in tqdm(requests, desc="Calculating log-likelihoods"):
            context, continuation = request.args

            # Get logits for the context
            logits = self.model.get_logits_from_text(
                user_input=context,
                system_prompt=self.system_prompt
            )

            # This is a simplified implementation - in a real scenario,
            # you would need to calculate the actual log-likelihood of the continuation
            # based on the logits from the model
            # For now, we'll return a placeholder
            results.append((0.0, False))  # (log_likelihood, is_greedy)

        return results

    def loglikelihood_rolling(self, requests):
        """
        Calculate rolling log-likelihood for each request
        """
        # Similar to loglikelihood but with rolling window
        # This is a simplified implementation
        return [0.0 for _ in requests]

    def generate_until(self, requests):
        """
        Generate text until a stopping condition is met
        """
        results = []
        for request in tqdm(requests, desc="Generating responses"):
            context, gen_kwargs = request.args

            # Generate text with the model
            generated = self.model.generate_text(
                user_input=context,
                system_prompt=self.system_prompt,
                max_new_tokens=gen_kwargs.get('max_new_tokens', 1024)  # Adjust as needed
            )

            # Extract the generated part (after the context)
            # This is a simplified approach
            results.append(generated.split(context)[-1].strip())

        return results

    @property
    def eot_token_id(self):
        """Return the end-of-text token ID"""
        return self.model.tokenizer.eos_token_id

    @property
    def max_length(self):
        """Return the maximum sequence length"""
        return 2048  # Adjust based on model capabilities

    @property
    def tokenizer_name(self):
        """Return the tokenizer name"""
        return self.model.model_name_path


def evaluate_with_steering(
    layers: List[int],
    multipliers: List[float],
    settings: SteeringSettings,
    tasks: List[str],
    num_fewshot: int = 8,
    limit: Optional[int] = None,
    overwrite: bool = False,
):
    """
    Evaluate model performance with steering vectors applied

    Args:
        layers: List of layers to apply steering vectors
        multipliers: List of multipliers to scale steering vectors
        settings: SteeringSettings object with configuration
        tasks: List of tasks to evaluate (e.g., "gsm8k")
        num_fewshot: Number of few-shot examples to use
        limit: Limit the number of examples to evaluate
        overwrite: Whether to overwrite existing results
    """
    # Create results directory
    save_results_dir = get_results_dir(settings.behavior)
    if not os.path.exists(save_results_dir):
        os.makedirs(save_results_dir)

    # Initialize model
    model = LlamaWrapper(
        HUGGINGFACE_TOKEN,
        size=settings.model_size,
        use_chat=not settings.use_base_model,
        override_model_weights_path=settings.override_model_weights_path,
    )

    # Set up logging for lm-eval
    setup_logging("INFO")

    # Evaluate for each layer and multiplier combination
    for layer in layers:
        # Get the steering vector
        name_path = model.model_name_path
        if settings.override_vector_model is not None:
            name_path = settings.override_vector_model

        if settings.override_vector is not None:
            vector = get_steering_vector(
                settings.behavior, settings.override_vector, name_path, normalized=True
            )
        else:
            vector = get_steering_vector(
                settings.behavior, layer, name_path, normalized=True
            )

        if settings.model_size != "7b":
            vector = vector.half()

        vector = vector.to(model.device)

        for multiplier in multipliers:
            # Create a unique suffix for this configuration
            result_save_suffix = settings.make_result_save_suffix(
                layer=layer, multiplier=multiplier
            )

            save_filename = os.path.join(
                save_results_dir,
                f"lmeval_results_{result_save_suffix}.json",
            )

            if os.path.exists(save_filename) and not overwrite:
                print("Found existing", save_filename, "- skipping")
                continue

            print(f"Evaluating with layer {layer}, multiplier {multiplier}")

            # Reset model state
            model.reset_all()

            # Create lm-eval compatible model
            lm_obj = LlamaWrapperForLMEval(
                llama_wrapper=model,
                layer=layer,
                vector=vector,
                multiplier=multiplier,
                system_prompt=get_system_prompt(settings.behavior, settings.system_prompt),
                batch_size=1  # Adjust as needed
            )

            # Run evaluation
            results = lm_eval.simple_evaluate(
                model=lm_obj,
                tasks=tasks,
                num_fewshot=num_fewshot,
                limit=limit,
                log_samples=True,
            )

            # Save results
            with open(save_filename, "w") as f:
                json.dump(results, f, indent=4)

            print(f"Results saved to {save_filename}")

            # Print summary
            print(lm_eval.utils.make_table(results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", nargs="+", type=int, required=True)
    parser.add_argument("--multipliers", nargs="+", type=float, required=True)
    parser.add_argument("--behaviors", type=str, nargs="+", default=["reasoning"])
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=["gsm8k"],
        help="Tasks to evaluate (e.g., gsm8k, mmlu)"
    )
    parser.add_argument(
        "--num_fewshot",
        type=int,
        default=8,
        help="Number of few-shot examples to use"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of examples to evaluate"
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default="pos",
        choices=["pos", "neg"],
        required=False,
    )
    parser.add_argument("--override_vector", type=int, default=None)
    parser.add_argument("--override_vector_model", type=str, default=None)
    parser.add_argument("--use_base_model", action="store_true", default=False)
    parser.add_argument("--model_size", type=str, choices=["7b", "13b"], default="7b")
    parser.add_argument("--override_model_weights_path", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true", default=False)

    args = parser.parse_args()

    steering_settings = SteeringSettings()
    steering_settings.system_prompt = args.system_prompt
    steering_settings.override_vector = args.override_vector
    steering_settings.override_vector_model = args.override_vector_model
    steering_settings.use_base_model = args.use_base_model
    steering_settings.model_size = args.model_size
    steering_settings.override_model_weights_path = args.override_model_weights_path

    for behavior in args.behaviors:
        steering_settings.behavior = behavior
        print(f"Evaluating steering for behavior: {behavior}")

        evaluate_with_steering(
            layers=args.layers,
            multipliers=args.multipliers,
            settings=steering_settings,
            tasks=args.tasks,
            num_fewshot=args.num_fewshot,
            limit=args.limit,
            overwrite=args.overwrite,
        )
