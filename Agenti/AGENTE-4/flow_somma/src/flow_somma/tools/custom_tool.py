from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from crewai.tools import tool

@tool("Sum Tool")
def sum_tool(a: int, b: int) -> int:
    """This tool calculates the sum between a and b."""
    # Tool logic here
    return a + b

class MyCustomToolInput(BaseModel):
    """Input schema for MyCustomTool."""

    argument: str = Field(..., description="Description of the argument.")


class MyCustomTool(BaseTool):
    name: str = "Name of my tool"
    description: str = (
        "Clear description for what this tool is useful for, your agent will need this information to use it."
    )
    args_schema: Type[BaseModel] = MyCustomToolInput

    def _run(self, argument: str) -> str:
        # Implementation goes here
        return "this is an example of a tool output, ignore it and move along."
