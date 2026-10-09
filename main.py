import os
import random
import time

import streamlit as st
from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)

load_dotenv()

MAX_RETRIES = 4
BASE_DELAY = 1.5
MAX_DELAY = 17.0
RETRYABLE = (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError)


@st.cache_resource
def get_providers():
    """Create clients only for providers with configured API keys."""
    providers = []

    if os.getenv("GEMINI_API_KEY"):
        providers.append(
            {
                "name": "gemini",
                "model": "gemini-3.8-flash",
                "client": OpenAI(
                    api_key=os.environ["GEMINI_API_KEY"],
                    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                    max_retries=0,
                    timeout=30,
                ),
            }
        )

    if os.getenv("OPENAI_API_KEY"):
        providers.append(
            {
                "name": "openai",
                "model": "gpt-4o-mini",
                "client": OpenAI(
                    api_key=os.environ["OPENAI_API_KEY"],
                    max_retries=0,
                    timeout=30,
                ),
            }
        )

    return providers


def call_with_retries(provider, **params):
    """Call one provider with exponential backoff and jitter."""
    for attempt in range(MAX_RETRIES + 1):
        try:
            return provider["client"].chat.completions.create(
                model=provider["model"], **params
            )
        except RETRYABLE:
            if attempt == MAX_RETRIES:
                raise
            delay = min(BASE_DELAY * (2**attempt), MAX_DELAY)
            time.sleep(delay + random.uniform(0, delay * 0.50))


def chat(messages):
    """Try configured providers in priority order."""
    providers = get_providers()
    if not providers:
        raise RuntimeError("Add GEMINI_API_KEY or OPENAI_API_KEY to the .env file.")

    last_error = None
    for provider in providers:
        try:
            response = call_with_retries(
                provider,
                messages=messages,
                temperature=0.9,
                top_p=0.6,
                max_tokens=600,
                stop=["END", "\n\n\n"],
            )
            return response, provider["name"]
        except Exception as error:
            last_error = error

    raise RuntimeError("All configured providers failed.") from last_error


st.set_page_config(page_title="GK Chatbot", page_icon="💬")
st.title("💬 GK Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask something..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    request_messages = [
        {"role": "system", "content": "You are a very intelligent assistant."},
        *st.session_state.messages,
    ]

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response, provider = chat(request_messages)
                answer = response.choices[0].message.content or "No response returned."
                st.markdown(answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )
                if response.usage:
                    with st.caption("Usage"):
                        st.write(
                            f"{provider} · {response.usage.total_tokens} total tokens"
                        )
            except Exception as error:
                st.error(str(error))
