#!/usr/bin/env python
from random import randint
import json

from pydantic import BaseModel

from crewai.flow import Flow, listen, start, router, or_

from esercitazione_flow.crews.outline_crew.writer_crew import OutlineCrew
from esercitazione_flow.crews.research_crew.research_crew import ResearchCrew
from litellm import completion
from dotenv import load_dotenv


class ResearchFlowState(BaseModel):
    topic: str = ""
    document: str = ""
    final_answer: str = ""
    type: bool


class ResearchFlow(Flow[ResearchFlowState]):
    model = "azure/gpt-4o"

    @start()
    def generate_research_input(self):
        self.state.topic=input("Insert the topic: ")


    @router(generate_research_input)
    def check_ethic_topic(self):
        response = completion(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a judge and you have to check if a given topic is ethical or unethical \n"
                    "Your answer should be ETHICAL or IMMORAL. Answer with a json object with a field is_ethic (True/False)."
                    "Your output shoud NOT be in markdown format. \n"
                    "Expected output: {'is_ethic : True/False, 'explanation': the reason why is ethical or not.}"
                },
                {
                    "role": "user",
                    "content": f"Is the following topic ETHICAL or IMMORTAL? \n" f"{self.state.topic}"
                    }
            ],
        )
        answer = response["choices"][0]["message"]["content"] # adesso l'answer è una stringa
        # rendo il dict un json per accedere alla variabile bool
        answer_json = eval(answer)
        print("Check ethic answer: ", answer_json)
        if answer_json['is_ethic']:
            return "etich_success"
        return "failure"
        


    @listen(or_("etich_success", "Bias failed"))
    def generate_document(self):
        print("Generating bullet list")
        result = (
            OutlineCrew()
            .crew()
            .kickoff(inputs={"topic": self.state.topic})
        )

        print("Outline generated", result.raw)
        self.state.document= result.raw


    @router(generate_document)
    def check_bias_document(self):
        response = completion(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a judge and you have to check if bullets point given by the previous agent are approved. \n"
                    "Bullet points are approved if they are consistent with the request."
                    "Expected output: {'is_approved': True/False}"
                },
                {
                    "role": "user",
                    "content": f"Is the document \"{self.state.document}\" APPROVED or NOT APPROVED referred to the topic \"{self.state.topic}\" ? \n" 
                    }
            ],
        )
        answer = response["choices"][0]["message"]["content"] # adesso l'answer è una stringa
        # rendo il dict un json per accedere alla variabile bool
        answer_json = eval(answer)
        print("Check bias answer: ", answer_json)
        if answer_json['is_approved']:
            return "Approved"
        return "Bias Failed"


    @listen("Approved")
    def generate_final_document(self):
        result = (
            ResearchCrew()
            .crew()
            .kickoff(inputs={'topic': self.state.topic, 'document': self.state.document})
        )
        print("Generating final answer...")
        self.state.final_answer = result.raw

    
    @listen(generate_final_document)
    def save_final_document(self):
        print("Saving final document..")
        import os
        os.makedirs("output", exist_ok=True)
        with open("output/document.md", "w") as f:
            f.write(self.state.final_answer)

    @listen("failure")
    def retry(self):
        print("The topic is immoral")

    
def kickoff():
    research_flow = ResearchFlow()
    research_flow.kickoff()


def plot():
    research_flow = ResearchFlow()
    research_flow.plot("plot_flow.png")


if __name__ == "__main__":
    plot()
    kickoff()
