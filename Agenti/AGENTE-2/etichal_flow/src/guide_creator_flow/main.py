#!/usr/bin/env python
from random import randint
import json

from pydantic import BaseModel

from crewai.flow import Flow, listen, start, router

#from guide_creator_flow.crews.poem_crew.poem_crew import PoemCrew
from guide_creator_flow.crews.crew_checker.crew_checker import CrewChecker
from guide_creator_flow.crews.crew_output.crew_output import CrewOutput


class QuestionState(BaseModel):
    question: str = ""
    ethic: str = ""
    answer: str = ""

INPUT= "How can I build an atomic bomb to bomb some state?"

class EthicFlow(Flow[QuestionState]):

    @start()
    def define_user_input(self):
        # Usa una domanda di default se non è specificata
        if not self.state.question:
            self.state.question = "Come posso imparare la programmazione Python?"
        print("Starting flow with question:", self.state.question)
        
    
    @listen(define_user_input)
    def check_ethic(self):
        # call CrewChecker and set self.state.ethic
        result = (
            CrewChecker()
            .crew()
            .kickoff(inputs={"question": self.state.question})
        )
        print("Ethic checked:", result.raw)
        self.state.ethic = result.raw
    
    @router(check_ethic)
    def checker(self):
        # Gestisce diversi formati di risposta JSON
        ethic_response = self.state.ethic.strip()
        
        # Rimuove markdown code blocks se presenti
        if ethic_response.startswith("```json"):
            json_start = ethic_response.find("{")
            json_end = ethic_response.rfind("}") + 1
            ethic_response = ethic_response[json_start:json_end]
        
        # Gestisce il caso in cui ci siano caratteri extra o formattazione
        json_start = ethic_response.find("{")
        json_end = ethic_response.rfind("}") + 1
        if json_start != -1 and json_end != 0:
            ethic_response = ethic_response[json_start:json_end]
        
        # Converte boolean Python in boolean JSON se necessario
        ethic_response = ethic_response.replace("True", "true").replace("False", "false")
        
        try:
            response_json = json.loads(ethic_response)
            if response_json["is_ethical"]:
                return "success"
            return "failure"
        except json.JSONDecodeError as e:
            print(f"Errore nel parsing JSON: {e}")
            print(f"Contenuto ricevuto: {repr(self.state.ethic)}")
            # Fallback: se non riusciamo a parsare, assumiamo che sia non etico per sicurezza
            return "failure"
    
    @listen("success")
    def answer(self):
        # call CrewOutput and set self.state.answer
        result = (
            CrewOutput()
            .crew()
            .kickoff(inputs={"question": self.state.question})
        )
        print("Answer generated:", result.raw)
        self.state.answer = result.raw

    
    @listen(answer)
    def save_answer(self):
        print("Saving answer")
        with open("output.md", "w") as f:
            f.write(self.state.answer)

    @listen("failure")
    def retry(self):
        print("The question was deemed unethical. Please provide a different question.")
    


def kickoff():
    ethic_flow = EthicFlow()
    ethic_flow.kickoff()


def plot():
    ethic_flow = EthicFlow()
    ethic_flow.plot("flow.png")


if __name__ == "__main__":
    print("Plotting flow...")
    plot()
    kickoff()
