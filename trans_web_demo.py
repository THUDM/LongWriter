"""
This script creates an interactive web demo for the GLM-4-9B model using Gradio,
a Python library for building quick and easy UI components for machine learning models.
It's designed to showcase the capabilities of the GLM-4-9B model in a user-friendly interface,
allowing users to interact with the model through a chat-like interface.
"""

import os
from pathlib import Path
from threading import Thread
from typing import Union

import gradio as gr
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer
)

ModelType = Union[PreTrainedModel]
TokenizerType = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]

MODEL_PATH = os.environ.get('MODEL_PATH', 'THUDM/LongWriter-glm4-9b')
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", MODEL_PATH)


def _resolve_path(path: Union[str, Path]) -> Path:
    return Path(path).expanduser().resolve()


def load_model_and_tokenizer(
        model_dir: Union[str, Path], trust_remote_code: bool = True
) -> tuple[ModelType, TokenizerType]:
    model_dir = _resolve_path(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, trust_remote_code=trust_remote_code, device_map='auto'
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, trust_remote_code=trust_remote_code, use_fast=False
    )
    return model, tokenizer


model, tokenizer = load_model_and_tokenizer(MODEL_PATH, trust_remote_code=True)


class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # stop_ids = model.config.eos_token_id
        stop_ids = [tokenizer.eos_token_id, tokenizer.get_command("<|user|>"),
                    tokenizer.get_command("<|observation|>")]
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


def predict(history, prompt, max_length, top_p, temperature):
    stop = StopOnTokens()
    messages = []
    if prompt:
        messages.append({"role": "system", "content": prompt})
    for idx, (user_msg, model_msg) in enumerate(history):
        if prompt and idx == 0:
            continue
        if idx == len(history) - 1 and not model_msg:
            # messages.append({"role": "user", "content": user_msg})
            query = user_msg
            break
        if user_msg:
            messages.append({"role": "user", "content": user_msg})
        if model_msg:
            messages.append({"role": "assistant", "content": model_msg})

    model_inputs = tokenizer.build_chat_input(query, history=messages, role='user').input_ids.to(
        next(model.parameters()).device)
    streamer = TextIteratorStreamer(tokenizer, timeout=600, skip_prompt=True, skip_special_tokens=True)
    eos_token_id = [tokenizer.eos_token_id, tokenizer.get_command("<|user|>"),
                    tokenizer.get_command("<|observation|>")]
    generate_kwargs = {
        "input_ids": model_inputs,
        "streamer": streamer,
        "max_new_tokens": max_length,
        "do_sample": True,
        "top_p": top_p,
        "temperature": temperature,
        "stopping_criteria": StoppingCriteriaList([stop]),
        "repetition_penalty": 1,
        "eos_token_id": eos_token_id,
    }
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()
    for new_token in streamer:
        if new_token and '<|user|>' not in new_token:
            history[-1][1] += new_token
        yield history


with gr.Blocks() as demo:
    gr.HTML("""<h1 align="center">LongWriter Chat Demo</h1>""")
    chatbot = gr.Chatbot()

    with gr.Row():
        with gr.Column(scale=3):
            with gr.Column(scale=12):
                user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=5, container=False)
            with gr.Column(min_width=32, scale=1):
                submitBtn = gr.Button("Submit")
        with gr.Column(scale=1):
            prompt_input = gr.Textbox(show_label=False, placeholder="Prompt", lines=10, container=False)
            pBtn = gr.Button("Set Prompt")
        with gr.Column(scale=1):
            emptyBtn = gr.Button("Clear History")
            max_length = gr.Slider(0, 32768, value=32768, step=1.0, label="Maximum length(Input + Output)", interactive=True)
            top_p = gr.Slider(0, 1, value=0.8, step=0.01, label="Top P", interactive=True)
            temperature = gr.Slider(0.01, 1, value=0.6, step=0.01, label="Temperature", interactive=True)


    def user(query, history):
        return "", history + [[query, ""]]


    def set_prompt(prompt_text):
        return [[prompt_text, "成功设置prompt"]]


    pBtn.click(set_prompt, inputs=[prompt_input], outputs=chatbot)

    submitBtn.click(user, [user_input, chatbot], [user_input, chatbot], queue=False).then(
        predict, [chatbot, prompt_input, max_length, top_p, temperature], chatbot
    )
    emptyBtn.click(lambda: (None, None), None, [chatbot, prompt_input], queue=False)

demo.queue()
demo.launch(server_name="127.0.0.1", server_port=8008, inbrowser=True, share=True)
