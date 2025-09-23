# Creazione di Agenti con Crewai
Installazione su Windows:

```
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
uv tool install crewai
```

Creazione di una crew:
```
crewai create nome_crew
```

Installo le dipendenze del progetto:
```
crewai install
```

Per effettuare il run del progetto:
```
crewai runcrewai 
```

Creazione di un flusso:
```
crewai create flow nome_flow
```
Aggiunta di una crew al flusso:
```
crewai flow add-crew nome_crew
```
