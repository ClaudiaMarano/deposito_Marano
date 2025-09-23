from typing import Type, Union

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


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

class CustomMathSolver(BaseModel):
    maths_function : str = Field(..., description = "Mathematical function")

class MathSolver(BaseTool):
    name : str = "Maths Solver"
    description : str = ("This tools is used to solve mathematical problems or equations or mathematical tasks.")
    args_schema : Type[BaseModel] = CustomMathSolver

    def run(self, maths_function : str) -> Union[int, float]:
        """
        """
