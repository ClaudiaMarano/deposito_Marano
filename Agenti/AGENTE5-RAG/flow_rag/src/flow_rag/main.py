#!/usr/bin/env python
from random import randint

from pydantic import BaseModel

from crewai.flow import Flow, listen, start, router, or_

from flow_rag.crews.rag_crew.rag_crew import RagCrew
from dotenv import load_dotenv
from litellm import completion

load_dotenv(dotenv_path="Agenti\AGENTE5-RAG\flow_rag\.env")

class RagState(BaseModel):
    query : str = ""
    topic : str = " Rivoluzione francese " # topic del documento 
    answer : str = ""
    

class RagAgentFlow(Flow[RagState]):
    model = "azure/gpt-4o"
    
    @start()
    def start_flow(self):
        print("Starting flow..")

    @listen(or_(start_flow, "relevant_failure"))
    def get_user_query(self):
        query = input(" Fammi una domanda: ")
        self.state.query = query

    
    @router(get_user_query)
    def evaluate_question(self):
        response = completion(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a judge and you have to check if the query is relevant to the context. \n"
                    "The context is the following. " f"{self.state.topic}"
                    "Expected output: {'is_relevant : True/False, 'explanation': the reason why is relevant or not.}"
                },
                {
                    "role": "user",
                    "content": f"Is the following query Relevant or Irrelevant? \n" f"{self.state.query}"
                }
            ],
        )
        answer = response["choices"][0]["message"]["content"] # adesso l'answer è una stringa
        # rendo il dict un json per accedere alla variabile bool
        answer_json = eval(answer)
        print("Check relevant answer: ", answer_json)
        if answer_json.get('is_relevant', False):
            return "relevant_success"
        print("Query is not relevant... Retry: ")    
        return "relevant_failure"


    @listen("relevant_success")
    # Se la domanda è pertinente al contesto, viene effettuata la ricerca tramite rag
    def rag_search_and_answer(self):
        print("Generating rag answer...")
        
        # Now pass both query and context to the crew
        result = (
            RagCrew()
            .crew()
            .kickoff(inputs={"query": self.state.query})
        )

        print("Rag answer: ", result.raw)
        self.state.answer = result.raw
        
        
        

    @listen(rag_search_and_answer)
    def save_answer(self):
        print("Saving answer")
        import os
        os.makedirs("output", exist_ok=True)
        with open("output/answer.md", "w") as f:
            f.write(self.state.answer)


def kickoff():
    poem_flow = RagAgentFlow()
    poem_flow.kickoff()


def plot():
    poem_flow = RagAgentFlow()
    poem_flow.plot()


if __name__ == "__main__":
    kickoff()
