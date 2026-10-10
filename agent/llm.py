"""Minimal OpenAI Responses API streaming client. Credentials come only from environment variables."""
import json
import os
import httpx


class ProviderResponseError(RuntimeError):
    def __init__(self, message, response=None):
        super().__init__(message)
        self.response = response


class ResponsesAgent:
    def __init__(self):
        self.model = os.environ.get('OPENAI_MODEL', 'gpt-6-astra')
        if not os.environ.get('OPENAI_API_KEY'):
            raise RuntimeError('Set OPENAI_API_KEY (and optionally OPENAI_MODEL, OPENAI_BASE_URL) for --live runs')

    async def respond(self, history, on_delta, *, instructions, tools):
        payload = dict(model=self.model, instructions=instructions, input=history, tools=tools,
                       store=False, stream=True, parallel_tool_calls=False,
                       include=['reasoning.encrypted_content'], reasoning={'effort': 'medium'},
                       max_output_tokens=8192, service_tier='default')
        url = os.environ.get('OPENAI_BASE_URL', 'https://api.openai.com/v1').rstrip('/') + '/responses'
        headers = {'Authorization': 'Bearer ' + os.environ['OPENAI_API_KEY']}
        async with httpx.AsyncClient(timeout=httpx.Timeout(180, connect=20)) as client:
            async with client.stream('POST', url, json=payload, headers=headers) as response:
                if response.status_code >= 400:
                    await response.aread()
                    raise RuntimeError(f'OpenAI request failed (HTTP {response.status_code})')
                async for line in response.aiter_lines():
                    if not line.startswith('data: ') or line[6:] == '[DONE]':
                        continue
                    event = json.loads(line[6:])
                    kind = event.get('type')
                    if kind == 'response.output_text.delta':
                        on_delta(event['delta'])
                    elif kind == 'response.completed':
                        result = event['response']
                        if not result.get('model', '').startswith(self.model):
                            raise ProviderResponseError('Provider returned a different model than requested', result)
                        return result
                    elif kind in ('error', 'response.failed', 'response.incomplete'):
                        raise ProviderResponseError('OpenAI response failed or exhausted its output budget', event.get('response'))
        raise RuntimeError('OpenAI stream ended without a completed response')
