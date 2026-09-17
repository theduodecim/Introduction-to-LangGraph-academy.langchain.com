# Module 1 — Índice de lecciones

## Lesson 1: Motivation

- [LangChain Academy - Introduction to LangGraph - Motivation.pdf](https://files.cdn.thinkific.com/file_uploads/967498/attachments/ecd/3cc/6d3/LangChain_Academy_-_Introduction_to_LangGraph_-_Motivation.pdf) *(ya descargado)*

### LangGraph FAQ

**¿Necesito usar LangChain para usar LangGraph? ¿Cuál es la diferencia?**
No. LangGraph es un framework de orquestación para sistemas agénticos complejos, más bajo nivel y controlable que los agentes de LangChain. LangChain, en cambio, provee una interfaz estándar para interactuar con modelos y otros componentes, útil para cadenas y flujos de retrieval simples.

**¿En qué se diferencia LangGraph de otros frameworks de agentes?**
Otros frameworks agénticos sirven para tareas simples y genéricas, pero no alcanzan para tareas complejas y específicas de cada empresa. LangGraph da un framework más expresivo para manejar esas tareas únicas, sin encerrar al usuario en una única arquitectura cognitiva de caja negra.

**¿LangGraph impacta el rendimiento de mi app?**
No agrega overhead al código y está diseñado específicamente pensando en workflows de streaming.

**¿LangGraph es open source? ¿Es gratis?**
Sí. Es una librería open source con licencia MIT, gratis de usar.

**¿LangSmith Deployment (antes LangGraph Platform/Cloud) es open source?**
No. [LangSmith Deployment](https://docs.langchain.com/langgraph-platform) es software propietario que eventualmente será un servicio pago para ciertos tiers de uso. Siempre van a avisar con anticipación antes de cobrar, y los primeros usuarios tendrán precios preferenciales.

**¿Cómo habilito LangSmith Deployment?**
Todos los usuarios de LangSmith en los planes Plus y Enterprise tienen acceso. Ver [docs](https://docs.langchain.com/langgraph-platform).

**¿En qué se diferencian LangGraph y LangSmith Deployment?**
LangGraph es un framework de orquestación con estado, que da mayor control a los workflows de agentes. LangSmith Deployment es un servicio para deployar y escalar aplicaciones LangGraph.

**¿Cómo encaja LangGraph en el ecosistema LangChain?**

Los frameworks open source ayudan a construir agentes:
- **LangChain**: ayuda a arrancar rápido a construir agentes, con cualquier proveedor de modelo que elijas.
- **LangGraph**: permite controlar cada paso de tu agente custom con orquestación de bajo nivel, memoria, y soporte de human-in-the-loop. Maneja tareas de larga duración con ejecución durable.

**LangSmith** es una plataforma que ayuda a los equipos de IA a usar datos de producción en vivo para testing y mejora continua. Provee:
- **Observability**: ver exactamente cómo piensa y actúa tu agente, con tracing detallado y métricas de tendencias agregadas.
- **Evaluation**: testear y puntuar el comportamiento del agente sobre datos de producción y datasets offline, para mejora continua.
- **Deployment**: shippear tu agente con un click, sobre infraestructura escalable pensada para tareas de larga duración.

---

## Lesson 2: Simple Graph

- Notebook: `simple-graph.ipynb`
- [Descargar en GitHub](https://github.com/langchain-ai/langchain-academy/blob/main/module-1/simple-graph.ipynb)
- [Ver en Google Colab](https://colab.research.google.com/github/langchain-ai/langchain-academy/blob/main/module-1/simple-graph.ipynb)

---

## Lesson 3: LangSmith Studio

- [LangSmith Studio](https://studio.langchain.com/)
- [Descargar archivos de Studio del Module 1 en GitHub](https://github.com/langchain-ai/langchain-academy/tree/main/module-1/studio)

---

## Lesson 4: Chain

- Notebook: `chain.ipynb`
- [Descargar en GitHub](https://github.com/langchain-ai/langchain-academy/blob/main/module-1/chain.ipynb)
- [Ver en Google Colab](https://colab.research.google.com/github/langchain-ai/langchain-academy/blob/main/module-1/chain.ipynb)

---

## Lesson 5: Router

- Notebook: `router.ipynb`
- [Descargar en GitHub](https://github.com/langchain-ai/langchain-academy/blob/main/module-1/router.ipynb)
- [Ver en Google Colab](https://colab.research.google.com/github/langchain-ai/langchain-academy/blob/main/module-1/router.ipynb)

---

## Lesson 6: Agent

- Notebook: `agent.ipynb`
- [Descargar en GitHub](https://github.com/langchain-ai/langchain-academy/blob/main/module-1/agent.ipynb)
- [Ver en Google Colab](https://colab.research.google.com/github/langchain-ai/langchain-academy/blob/main/module-1/agent.ipynb)

---

## Lesson 7: Agent with Memory

- Notebook: `agent-memory.ipynb`
- [Descargar en GitHub](https://github.com/langchain-ai/langchain-academy/blob/main/module-1/agent-memory.ipynb)
- [Ver en Google Colab](https://colab.research.google.com/github/langchain-ai/langchain-academy/blob/main/module-1/agent-memory.ipynb)

---

## [Opcional] Lesson 8: Deployment

> Actualmente, LangSmith Deployment solo está disponible para usuarios del plan LangSmith Plus. Esta lección es opcional.
> LangSmith Deployment antes se llamaba LangGraph Platform.

- Notebook: `deployment.ipynb`
- [Descargar en GitHub](https://github.com/langchain-ai/langchain-academy/blob/main/module-1/deployment.ipynb)
- [Ver en Google Colab](https://colab.research.google.com/github/langchain-ai/langchain-academy/blob/main/module-1/deployment.ipynb)
