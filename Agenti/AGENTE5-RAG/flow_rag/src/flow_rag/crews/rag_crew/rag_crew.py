from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from crewai_tools import SerperDevTool


@CrewBase
class RagCrew:

    agents: List[BaseAgent]
    tasks: List[Task]

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    
    @agent
    def rag_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["rag_agent"],  
            tools=[SerperDevTool()]  # inserisci tool rag
        )


    @task
    def rag_task(self) -> Task:
        return Task(
            config=self.tasks_config["rag_task"],  
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Research Crew"""

        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
        )
