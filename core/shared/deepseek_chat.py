import json
from typing import Any
from typing import Type

from agno.models.message import Message
from agno.models.openai import OpenAIChat
from agno.models.response import ModelResponse
from agno.utils.models.schema_utils import get_response_schema_for_provider
from pydantic import BaseModel


class DeepSeekStructuredOutputError(RuntimeError):
    """Raised when DeepSeek cannot produce output matching the requested schema."""


class DeepSeekChat(OpenAIChat):
    """OpenAI-compatible DeepSeek client with Pydantic structured-output support.

    DeepSeek accepts JSON mode but not OpenAI's ``json_schema`` response format.
    This adapter keeps Agno's native ``output_schema`` contract at its boundary:
    it sends ``json_object``, injects the original schema into the system message,
    validates the response, and performs one schema-only correction if needed.
    """

    name: str = 'DeepSeekChat'
    provider: str = 'DeepSeek'

    def get_request_params(self, response_format=None, tools=None, tool_choice=None, run_response=None) -> dict[str, Any]:
        params = super().get_request_params(
            response_format=None,
            tools=tools,
            tool_choice=tool_choice,
            run_response=run_response
        )

        if response_format is not None:
            params['response_format'] = {'type': 'json_object'}

        return params

    @staticmethod
    def _schema_type(response_format: Any) -> Type[BaseModel] | None:
        if isinstance(response_format, type) and issubclass(response_format, BaseModel):
            return response_format
        return None

    @staticmethod
    def _schema_instruction(schema: Type[BaseModel]) -> str:
        payload = json.dumps(
            get_response_schema_for_provider(schema, 'openai'),
            ensure_ascii=False,
            separators=(',', ':')
        )
        return (
            'Return exactly one valid JSON object that strictly conforms to the JSON Schema below. '
            'Use exactly the declared property names, include every required property, preserve all value types, '
            'and do not add Markdown or explanatory text outside the JSON object.\n'
            f'JSON Schema:\n{payload}'
        )

    @classmethod
    def _messages_with_schema(cls, messages: list[Message], schema: Type[BaseModel]) -> list[Message]:
        adapted = [message.model_copy(deep=True) for message in messages]
        instruction = cls._schema_instruction(schema)

        for message in adapted:
            if message.role in ('system', 'developer') and isinstance(message.content, str):
                message.content = f'{message.content}\n\n{instruction}'
                break
        else:
            adapted.insert(0, Message(role='system', content=instruction))

        return adapted

    @staticmethod
    def _validate_content(content: Any, schema: Type[BaseModel]) -> BaseModel:
        if isinstance(content, schema):
            return content
        if isinstance(content, str):
            payload = json.loads(content)
        else:
            payload = content

        DeepSeekChat._reject_extra_fields(
            payload,
            get_response_schema_for_provider(schema, 'openai'),
            get_response_schema_for_provider(schema, 'openai')
        )
        return schema.model_validate(payload, strict=True)

    @staticmethod
    def _reject_extra_fields(value: Any, node: dict[str, Any], root: dict[str, Any], path: str = '$') -> None:
        if '$ref' in node:
            target: Any = root
            for part in node['$ref'].removeprefix('#/').split('/'):
                target = target[part]
            DeepSeekChat._reject_extra_fields(value, target, root, path)
            return

        branches = node.get('anyOf') or node.get('oneOf')
        if branches:
            errors = []
            for branch in branches:
                try:
                    DeepSeekChat._reject_extra_fields(value, branch, root, path)
                    return
                except ValueError as error:
                    errors.append(error)
            if errors:
                raise errors[0]

        if isinstance(value, dict) and 'properties' in node:
            properties = node['properties']
            extras = sorted(set(value) - set(properties))
            if extras:
                raise ValueError(f'{path} contains unexpected properties: {extras}')
            for key, child in value.items():
                if key in properties:
                    DeepSeekChat._reject_extra_fields(child, properties[key], root, f'{path}.{key}')

        if isinstance(value, list) and isinstance(node.get('items'), dict):
            for index, item in enumerate(value):
                DeepSeekChat._reject_extra_fields(item, node['items'], root, f'{path}[{index}]')

    @classmethod
    def _correction_messages(
        cls,
        messages: list[Message],
        content: Any,
        schema: Type[BaseModel],
        error: Exception
    ) -> list[Message]:
        correction = (
            'The previous JSON did not conform to the required schema. Correct formatting and schema conformance only; '
            'preserve the analysis and meaning. Return exactly one corrected JSON object and nothing else.\n'
            f'Validation error:\n{error}\n\n{cls._schema_instruction(schema)}'
        )
        return [
            *messages,
            Message(role='assistant', content=str(content)),
            Message(role='user', content=correction)
        ]

    def invoke(
        self,
        messages,
        assistant_message,
        response_format=None,
        tools=None,
        tool_choice=None,
        run_response=None,
        compress_tool_results=False
    ) -> ModelResponse:
        schema = self._schema_type(response_format)
        adapted = self._messages_with_schema(messages, schema) if schema else messages
        response = super().invoke(
            adapted,
            assistant_message,
            response_format=response_format,
            tools=tools,
            tool_choice=tool_choice,
            run_response=run_response,
            compress_tool_results=compress_tool_results
        )

        if schema is None or response.tool_calls or response.content is None:
            return response

        try:
            self._validate_content(response.content, schema)
            return response
        except Exception as first_error:
            corrected_messages = self._correction_messages(adapted, response.content, schema, first_error)
            corrected = super().invoke(
                corrected_messages,
                Message(role='assistant'),
                response_format=response_format,
                tools=None,
                tool_choice=None,
                run_response=run_response,
                compress_tool_results=compress_tool_results
            )
            try:
                self._validate_content(corrected.content, schema)
            except Exception as second_error:
                raise DeepSeekStructuredOutputError(
                    f'DeepSeek failed schema {schema.__name__} after one correction: {second_error}'
                ) from second_error
            return corrected

    async def ainvoke(
        self,
        messages,
        assistant_message,
        response_format=None,
        tools=None,
        tool_choice=None,
        run_response=None,
        compress_tool_results=False
    ) -> ModelResponse:
        schema = self._schema_type(response_format)
        adapted = self._messages_with_schema(messages, schema) if schema else messages
        response = await super().ainvoke(
            adapted,
            assistant_message,
            response_format=response_format,
            tools=tools,
            tool_choice=tool_choice,
            run_response=run_response,
            compress_tool_results=compress_tool_results
        )

        if schema is None or response.tool_calls or response.content is None:
            return response

        try:
            self._validate_content(response.content, schema)
            return response
        except Exception as first_error:
            corrected_messages = self._correction_messages(adapted, response.content, schema, first_error)
            corrected = await super().ainvoke(
                corrected_messages,
                Message(role='assistant'),
                response_format=response_format,
                tools=None,
                tool_choice=None,
                run_response=run_response,
                compress_tool_results=compress_tool_results
            )
            try:
                self._validate_content(corrected.content, schema)
            except Exception as second_error:
                raise DeepSeekStructuredOutputError(
                    f'DeepSeek failed schema {schema.__name__} after one correction: {second_error}'
                ) from second_error
            return corrected
