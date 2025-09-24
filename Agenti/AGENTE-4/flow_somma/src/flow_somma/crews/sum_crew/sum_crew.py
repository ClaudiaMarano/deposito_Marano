from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from flow_somma.tools.custom_tool import sum_tool
import os

# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators


@CrewBase
class SumCrew:
    """Poem Crew"""

    agents: List[BaseAgent]
    tasks: List[Task]
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

 
    @agent
    def maths_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["maths_agent"],  # type: ignore[index]
            tools=[sum_tool]
        )

   
    @task
    def solve_sum_task(self) -> Task:
        return Task(
            config=self.tasks_config["solve_sum"],  # type: ignore[index]
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Sum Crew"""
        

        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
        )
