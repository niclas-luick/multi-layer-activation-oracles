import os
import dotenv
from slist import Slist

import interp_tools.api_utils.client as client
import interp_tools.api_utils.shared as shared


def get_openai_caller(cache_dir: str) -> client.OpenAICaller:
    dotenv.load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    organization_id = os.getenv("OPENAI_ORGANIZATION_ID")
    if api_key is None:
        raise RuntimeError("OPENAI_API_KEY missing in environment.")
    return client.OpenAICaller(
        api_key=api_key,
        cache_path=cache_dir,
        organization=organization_id,
    )


async def run_single_prompt(
    prompt: shared.ChatHistory,
    caller: client.OpenAICaller,
    model_name: str,
    temperature: float,
    max_tokens: int,
) -> client.OpenaiResponse:
    """Run a single game and return the model's response."""
    response = await caller.call(
        prompt,
        config=shared.InferenceConfig(
            temperature=temperature, max_tokens=max_tokens, model=model_name
        ),
    )
    return response


async def run_api_model(
    model_name: str,
    prompts: list[shared.ChatHistory],
    caller: client.OpenAICaller,
    temperature: float,
    max_tokens: int,
    max_par: int,
) -> list[str]:
    """Call `model_name` once per prompt. Note: This supports caching of API requests as well - will just use cached responses from disk if they exist."""

    responses = []

    # responses = await asyncio.gather(*[run_game(game, caller, temperature, max_tokens) for game in games]
    responses = await Slist(prompts).par_map_async(
        func=lambda prompt: run_single_prompt(
            prompt, caller, model_name, temperature, max_tokens
        ),
        max_par=max_par,
        tqdm=True,
    )

    return [r.first_response for r in responses]
