import os
import tiktoken
import openai
import replicate as rep
import litellm
from litellm import completion, acompletion
from openai.error import OpenAIError
import config as app_config
import asyncio
from concurrent.futures import Executor, ThreadPoolExecutor

litellm.openai_key = app_config.openai_api_key
litellm.huggingface_key = app_config.huggingface_api_key
litellm.replicate_key = app_config.replicate_api_key

openai.api_key = app_config.openai_api_key
if app_config.openai_api_base is not None:
    openai.api_base = app_config.openai_api_base

## set model alias map
model_alias_map = {
    "GPT-3.5": "gpt-3.5-turbo",
    "Davinci-003": "text-davinci-003",
    "GPT-4": "gpt-4",
    "GPT-3.5-16k": "gpt-3.5-turbo-16k",
    "Mistral-7b": "replicate/mistralai/mistral-7b-instruct-v0.1:83b6a56e7c828e667f21fd596c338fd4f0039b46bcfa18d973e8e70e455fda70"
}
litellm.model_alias_map = model_alias_map

COMPLETION_OPTIONS = {
    "temperature": 0.7,
    "max_tokens": 1000,
    "top_p": 1,
    "request_timeout": 60.0,
}

OPENAI_COMPLETION_OPTIONS = {
    "temperature": 0.7,
    "max_tokens": 1000,
    "top_p": 1,
    "frequency_penalty": 0,
    "presence_penalty": 0,
    "request_timeout": 60.0,
}

class ChatGPT:
    def __init__(self, model="GPT-3.5"):
        assert model in {
            "Davinci-003", 
            "GPT-3.5", 
            "GPT-3.5-16k", 
            "GPT-4",
            "Mistral-7b",
            }, f"Unknown model: {model}"
        self.model = model

    async def send_message(self, message, dialog_messages=[], chat_mode="assistant"):
        if chat_mode not in app_config.chat_modes.keys():
            raise ValueError(f"Chat mode {chat_mode} is not supported")

        n_dialog_messages_before = len(dialog_messages)
        answer = None
        while answer is None:
            try:
                if self.model in {"GPT-3.5", "GPT-3.5-16k", "GPT-4"}:
                    messages = self._generate_prompt_messages(message, dialog_messages, chat_mode)
                    r = await acompletion(
                        model=self.model,
                        messages=messages,
                        **OPENAI_COMPLETION_OPTIONS
                    )
                    answer = r.choices[0].message["content"]
                elif self.model == "Davinci-003":
                    messages = self._generate_prompt_messages(message, dialog_messages, chat_mode)
                    r = await acompletion(
                        model=self.model,
                        messages=messages,
                        **OPENAI_COMPLETION_OPTIONS
                    )
                    answer = r.choices[0].message["content"]
                elif self.model in {"Mistral-7b"}:
                    messages = self._generate_prompt_messages(message, dialog_messages, chat_mode)
                    r = await acompletion(
                        model=self.model,
                        messages=messages,
                        **COMPLETION_OPTIONS
                    )
                    answer = r.choices[0].message["content"]
                    
                else:
                    raise ValueError(f"Unknown model: {self.model}")

                answer = self._postprocess_answer(answer)

                n_input_tokens, n_output_tokens = r["usage"]["prompt_tokens"], r["usage"]["completion_tokens"]
            except OpenAIError as e:  # too many tokens
                if len(dialog_messages) == 0:
                    raise ValueError("Dialog messages is reduced to zero, but still has too many tokens to make completion") from e

                # forget first message in dialog_messages
                dialog_messages = dialog_messages[1:]

        n_first_dialog_messages_removed = n_dialog_messages_before - len(dialog_messages)

        return answer, (n_input_tokens, n_output_tokens), n_first_dialog_messages_removed

    async def send_message_stream(self, message, dialog_messages=[], chat_mode="assistant"):
        if chat_mode not in app_config.chat_modes.keys():
            raise ValueError(f"Chat mode {chat_mode} is not supported")

        n_dialog_messages_before = len(dialog_messages)
        answer = None
        while answer is None:
            try:
                if self.model in {"GPT-3.5", "GPT-3.5-16k", "GPT-4"}:
                    messages = self._generate_prompt_messages(message, dialog_messages, chat_mode)
                    r_gen = await acompletion(
                        model=self.model,
                        messages=messages,
                        stream=True,
                        **OPENAI_COMPLETION_OPTIONS
                    )

                    answer = ""
                    async for r_item in r_gen:
                        delta = r_item.choices[0].delta
                        if "content" in delta:
                            answer += delta.content
                            n_input_tokens, n_output_tokens = self._count_tokens_from_messages(messages, answer, model=self.model)
                            n_first_dialog_messages_removed = n_dialog_messages_before - len(dialog_messages)
                            yield "not_finished", answer, (n_input_tokens, n_output_tokens), n_first_dialog_messages_removed
                
                elif self.model in {"Mistral-7b"}:
                    messages = self._generate_prompt_messages(message, dialog_messages, chat_mode)
                    r_gen = await acompletion(
                        model=self.model,
                        messages=messages,
                        stream=True,
                        **COMPLETION_OPTIONS
                    )

                    answer = ""
                    async for r_item in r_gen:
                        delta = r_item.choices[0].delta
                        if "content" in delta:
                            answer += delta.content
                            n_input_tokens, n_output_tokens = self._count_tokens_from_messages(messages, answer, model=self.model)
                            n_first_dialog_messages_removed = n_dialog_messages_before - len(dialog_messages)
                            yield "not_finished", answer, (n_input_tokens, n_output_tokens), n_first_dialog_messages_removed
                
                elif self.model == "Davinci-003":
                    messages = self._generate_prompt_messages(message, dialog_messages, chat_mode)
                    r_gen = await acompletion(
                        model=self.model,
                        messages=messages,
                        stream=True,
                        **OPENAI_COMPLETION_OPTIONS
                    )

                    answer = ""
                    async for r_item in r_gen:
                        delta = r_item.choices[0].delta
                        if "content" in delta:
                            answer += delta.content
                            n_input_tokens, n_output_tokens = self._count_tokens_from_messages(messages, answer, model=self.model)
                            n_first_dialog_messages_removed = n_dialog_messages_before - len(dialog_messages)
                            yield "not_finished", answer, (n_input_tokens, n_output_tokens), n_first_dialog_messages_removed

                answer = self._postprocess_answer(answer)

            except OpenAIError as e:  # too many tokens
                if len(dialog_messages) == 0:
                    raise e

                # forget first message in dialog_messages
                dialog_messages = dialog_messages[1:]

        yield "finished", answer, (n_input_tokens, n_output_tokens), n_first_dialog_messages_removed  # sending final answer

    def _generate_prompt(self, message, dialog_messages, chat_mode):
        prompt = app_config.chat_modes[chat_mode]["prompt_start"]
        prompt += "\n\n"

        # add chat context
        if len(dialog_messages) > 0:
            prompt += "Chat:\n"
            for dialog_message in dialog_messages:
                prompt += f"User: {dialog_message['user']}\n"
                prompt += f"Assistant: {dialog_message['bot']}\n"

        # current message
        prompt += f"User: {message}\n"
        prompt += "Assistant: "

        return prompt

    def _generate_prompt_messages(self, message, dialog_messages, chat_mode):
        prompt = app_config.chat_modes[chat_mode]["prompt_start"]

        messages = [{"role": "system", "content": prompt}]
        for dialog_message in dialog_messages:
            messages.extend(
                (
                    {"role": "user", "content": dialog_message["user"]},
                    {"role": "assistant", "content": dialog_message["bot"]},
                )
            )
        messages.append({"role": "user", "content": message})

        return messages

    def _postprocess_answer(self, answer):
        answer = answer.strip()
        return answer

    def _count_tokens_from_messages(self, messages, answer, model="GPT-3.5"):
        if model == "GPT-3.5-16k":
            encoding = tiktoken.encoding_for_model("gpt-3.5-turbo-16k")
            tokens_per_message = 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            tokens_per_name = -1  # if there's a name, the role is omitted
        elif model == "GPT-3.5" or model != "GPT-4" and model == "Mistral-7b":
            encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
            tokens_per_message = 4
            tokens_per_name = -1
        elif model == "GPT-4":
            encoding = tiktoken.encoding_for_model("gpt-4")
            tokens_per_message = 3
            tokens_per_name = 1
        else:
            raise ValueError(f"Unknown model: {model}")

        # input
        n_input_tokens = 0
        for message in messages:
            n_input_tokens += tokens_per_message
            for key, value in message.items():
                n_input_tokens += len(encoding.encode(value))
                if key == "name":
                    n_input_tokens += tokens_per_name

        n_input_tokens += 2

        # output
        n_output_tokens = 1 + len(encoding.encode(answer))

        return n_input_tokens, n_output_tokens

    def _count_tokens_from_prompt(self, prompt, answer, model="Davinci-003"):
        encoding = tiktoken.encoding_for_model(model)

        n_input_tokens = len(encoding.encode(prompt)) + 1
        n_output_tokens = len(encoding.encode(answer))

        return n_input_tokens, n_output_tokens


async def transcribe_audio(audio_file):
    r = await openai.Audio.atranscribe("whisper-1", audio_file)
    return r["text"]


async def generate_images(model, prompt, n_images=4, size="512x512", image_file=None):
    if model == "OpenAI":
        r = await openai.Image.acreate(prompt=prompt, n=n_images, size=size)
        image_urls = [item.url for item in r.data]
    elif model == "Replicate":
        os.environ["REPLICATE_API_TOKEN"] = app_config.replicate_api_key
        executor = ThreadPoolExecutor(max_workers=5)
        r = await asyncio.get_event_loop().run_in_executor(executor, lambda: replicate_run(prompt, n_images, image=image_file))

        image_urls = r
    
    return image_urls


async def is_content_acceptable(prompt):
    r = await openai.Moderation.acreate(input=prompt)
    return not all(r.results[0].categories.values())


def replicate_run(prompt, n_images=4, image=None):
    return rep.run(
        "stability-ai/sdxl:c221b2b8ef527988fb59bf24a8b97c4561f1c671f73bd389f866bfb27c061316",
        input={
            "prompt": prompt,
            "num_images": n_images,
            "image": image,
        },
    )