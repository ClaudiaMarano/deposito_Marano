#!/usr/bin/env python
from random import randint

from pydantic import BaseModel

from crewai.flow import Flow, listen, start

from flow_somma.crews.sum_crew.sum_crew import SumCrew
from flow_somma.tools.custom_tool import sum_tool


class SumFlowState(BaseModel):
    a: int = 0
    b: int = 0  # imposta anche a float se necessario
    result: int = 0


class SumFlow(Flow[SumFlowState]):

    @start()
    def accept_int_input(self):
        while True:
            try:
                self.state.a = int(input("Inserisci il primo numero intero: "))
                break
            except ValueError:
                print("Errore: Inserisci un numero intero valido!")
        
        while True:
            try:
                self.state.b = int(input("Inserisci il secondo numero intero: "))
                break
            except ValueError:
                print("Errore: Inserisci un numero intero valido!")
        
        print(f"Numeri inseriti: a = {self.state.a}, b = {self.state.b}")
    

    @listen(accept_int_input)
    def calculate_sum(self):
        print("Calcolo la somma...")
        result = (
            SumCrew()
            .crew()
            .kickoff(inputs={"a": self.state.a, "b": self.state.b})
        )
        print("Sum computed", result.raw)
        self.state.result = int(result.raw.strip())  # Converte in intero rimuovendo spazi
        print("The sum result is", self.state.result)


def kickoff():
    sum_flow = SumFlow()
    sum_flow.kickoff()


def plot():
    sum_flow = SumFlow()
    sum_flow.plot()


if __name__ == "__main__":
    kickoff()
