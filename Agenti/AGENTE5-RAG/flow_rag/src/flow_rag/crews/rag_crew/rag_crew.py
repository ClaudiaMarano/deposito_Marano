from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from crewai_tools import SerperDevTool
from flow_rag.tools.custom_tool import MyCustomTool
from crewai.llm import LLM
import os


@CrewBase
class RagCrew:

    agents: List[BaseAgent]
    tasks: List[Task]

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    
    @agent
    def rag_agent(self) -> Agent:
        # Configurazione Azure OpenAI LLM
        azure_llm = LLM(
            model="azure/gpt-4o",
            api_key=os.getenv("AZURE_API_KEY"),
            base_url=os.getenv("AZURE_API_BASE"),
            api_version=os.getenv("AZURE_API_VERSION", "2024-12-01-preview")
        )
        
        return Agent(
            config=self.agents_config["rag_agent"],  
            tools=[SerperDevTool(), MyCustomTool()],
            llm=azure_llm
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
