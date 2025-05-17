import logging
import os
import time
from typing import Tuple
from dotenv import load_dotenv
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import notebook_login
from huggingface_hub import login

from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="google/pegasus-xsum", filename="config.json")

logger = logging.getLogger(__name__)

# load_dotenv()
token = "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"
login(token=token)

# if not token:
#     raise ValueError("Hugging Face token is missing! Check your .env file or environment variables.")

# print(f"Loaded token: {token[:5]}********")  # Print first few characters for verification

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name, token=token)

# def chat(prompt):
#     inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
#     outputs = model.generate(**inputs, max_length=512)
#     return tokenizer.decode(outputs[0], skip_special_tokens=True)

# response = chat("Explain quantum physics in simple terms.")
# print(response)

def load_model() -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    """
    Load the model and tokenizer with 8-bit quantization configuration to optimize memory usage
    and inference performance.
    """
    load_dotenv()
    #model_name = os.getenv("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

    token = "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"
    logger.info(f"Loading model: {model_name}")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Define the quantization configuration for 8-bit
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True  # Remove llm_int8_threshold unless needed
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=token,
        device_map="auto",  #! Dynamically balancing between CPU and GPU
        quantization_config=quantization_config,  #! Quantization
    )

    logger.info(f"Model ({model_name}) loaded.")
    return tokenizer, model

#tokenizer, model = load_model()

def generate_chat_response(
    prompt: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    max_length: int = 3500,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
) -> str:
    """
        Generate a response from the model based on the input prompt.

        Args:
            - prompt (str): The input prompt.
            - tokenizer (AutoTokenizer): The tokenizer to preprocess the input.
            - model (AutoModelForCausalLM): The model used for generating the response.
            - max_length (int): The maximum length of the generated output.
            - temperature (float): The randomness of the output.
            - top_k (int): The number of top token choices.
            - top_p (float): The cumulative probability threshold for nucleus sampling.

        Returns:
            Tuple[str, str]: The thinking steps and the final answer from the model.

    .   #* About temp, top_k, top_p
        Temperature controls the randomness of the generated text, with higher values
        leading to more creative but less coherent output, and lower values resulting
        in more predictable, deterministic responses.

        Top-k limits token choices to the top k most likely options, reducing irrelevant
        text but potentially limiting creativity.

        Top-p (nucleus sampling) selects tokens dynamically until a cumulative probability
        threshold is met, balancing diversity and coherence, often used in combination
        with top-k.
    """
    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors="pt")

    # Move input tensors to the same device as the model
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Start timing the response generation process
    start_time = time.time()

    # Generate logits and outputs
    with torch.no_grad():
        logits = model(**inputs).logits
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        logger.debug(
            f"Intermediate logits shape: {logits.shape}"
        )  # Debugging: inspect logits

    # Calculate the time elapsed for thinking
    elapsed_time = time.time() - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    time_str = f"{int(minutes):02}:{int(seconds):02}"

    # Decode the full response
    final_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Log the thinking time and final response
    logger.info(f"Thinking time: {time_str}")
    logger.info(f"Response from generate_chat_response function:\n{final_answer}")

    return final_answer