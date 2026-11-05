"""Schema-only veRL tool.

veRL builds the tool list (and the schemas shown to the model in the chat
template) from ``multi_turn.tool_config_path``. Our tools are executed by
``TVCacheToolAgentLoop`` through TVCache, so the BaseTool only advertises the
schema. If it is ever executed directly, the agent loop was misconfigured.
"""

from typing import Any

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import ToolResponse


class SchemaOnlyTool(BaseTool):
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        return (
            ToolResponse(
                text=f"Tool '{self.name}' must run under TVCacheToolAgentLoop "
                "(set agent_name=tvcache_tool_agent in the dataset)."
            ),
            0.0,
            {},
        )
