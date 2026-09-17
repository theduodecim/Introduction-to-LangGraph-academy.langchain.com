# Agent memory: agregando persistencia con checkpointers

## Repaso

Ya armamos un agente que sabe:

- **`act`**: dejar que el modelo llame tools específicas.
- **`observe`**: pasarle el resultado de la tool de vuelta al modelo.
- **`reason`**: dejar que el modelo razone sobre ese resultado y decida el próximo paso (llamar otra tool o responder directo).

## Objetivo

Ahora vamos a extender ese agente agregándole **memoria**, para que pueda recordar información de invocaciones anteriores.

---

## 1. Instalación y configuración inicial

```python
%%capture --no-stderr
%pip install --quiet -U langchain_openai langchain_core langgraph langgraph-prebuilt
```

```python
import os, getpass

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("OPENAI_API_KEY")
```

```python
_set_env("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "langchain-academy"
```

Todo esto es igual a lo que veníamos haciendo: seteamos la API key y activamos el tracing con LangSmith en el proyecto `langchain-academy`.

---

## 2. El mismo agente de siempre (3 tools + assistant)

```python
from langchain_openai import ChatOpenAI

def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

# This will be a tool
def add(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b

def divide(a: int, b: int) -> float:
    """Divide a and b.

    Args:
        a: first int
        b: second int
    """
    return a / b

tools = [add, multiply, divide]
llm = ChatOpenAI(model="gpt-4o")
llm_with_tools = llm.bind_tools(tools)
```

```python
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, SystemMessage

# System message
sys_msg = SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs.")

# Node
def assistant(state: MessagesState):
   return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}
```

**En español:** esto es exactamente lo mismo que en el notebook de "agent": tres tools (`add`, `multiply`, `divide`), y un nodo `assistant` que invoca al modelo con el mensaje de sistema más el historial de la conversación.

---

## 3. Armar el grafo (igual que antes)

```python
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition, ToolNode
from IPython.display import Image, display

# Graph
builder = StateGraph(MessagesState)

# Define nodes: these do the work
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

# Define edges: these determine how the control flow moves
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
)
builder.add_edge("tools", "assistant")
react_graph = builder.compile()

# Show
display(Image(react_graph.get_graph(xray=True).draw_mermaid_png()))
```

Mismo grafo ReAct que ya conocemos: `START → assistant`, loop `assistant ↔ tools` mientras haya tool calls, y `END` cuando el modelo responde en lenguaje natural.

---

## 4. El problema: el grafo no tiene memoria entre invocaciones

Probamos una primera invocación:

```python
messages = [HumanMessage(content="Add 3 and 4.")]
messages = react_graph.invoke({"messages": messages})
for m in messages['messages']:
    m.pretty_print()
```

Salida:

```
Human Message: Add 3 and 4.
Ai Message: Tool Calls → add(a=3, b=4)
Tool Message: Name: add → 7
Ai Message: The sum of 3 and 4 is 7.
```

Ahora hacemos una **segunda invocación**, distinta, refiriéndonos al resultado anterior:

```python
messages = [HumanMessage(content="Multiply that by 2.")]
messages = react_graph.invoke({"messages": messages})
for m in messages['messages']:
    m.pretty_print()
```

Salida:

```
Human Message: Multiply that by 2.
Ai Message: Tool Calls → multiply(a=2, b=2)
Tool Message: Name: multiply → 4
Ai Message: The result of multiplying 2 by 2 is 4.
```

**¿Qué pasó?** El modelo no tiene idea de que "that" se refiere al `7` que calculamos antes. Como no sabe a qué número nos referimos, asume "2" (probablemente porque interpreta el "2" del mensaje como ambos operandos) y hace `2 × 2 = 4`, que no es lo que queríamos.

**En español, la causa raíz:** el **estado es transitorio a una sola ejecución del grafo**. Cada `invoke()` arranca con un estado nuevo y vacío. La primera invocación tuvo su propio estado (que terminó con el `7`), y la segunda invocación es una ejecución **completamente independiente**, sin ninguna conexión con la anterior. Esto limita mucho la posibilidad de tener conversaciones de varios turnos.

---

## 5. La solución: persistencia con un checkpointer

LangGraph puede usar un **checkpointer** para guardar automáticamente el estado del grafo después de cada paso. Esta capa de persistencia incorporada es lo que le da "memoria" al grafo, permitiéndole retomar desde el último estado guardado.

Uno de los checkpointers más simples es `MemorySaver`: un almacén clave-valor **en memoria** (no persiste si reiniciás el proceso, pero alcanza para entender el concepto). Solo hay que compilar el grafo pasándole ese checkpointer:

```python
from langgraph.checkpoint.memory import MemorySaver
memory = MemorySaver()
react_graph_memory = builder.compile(checkpointer=memory)
```

### Cómo funciona por dentro (la idea del "thread")

- El checkpointer escribe un **checkpoint** en cada paso del grafo. Ese checkpoint guarda el estado del grafo en ese momento, además de metadatos (por ejemplo, cuál es el próximo nodo a ejecutar) y un ID propio.
- Los checkpoints de una misma conversación se agrupan en lo que LangGraph llama un **thread** (hilo).
- Para acceder (o continuar) ese thread más adelante, alcanza con pasar el mismo `thread_id`.

**En español, en criollo:** un *thread* es básicamente el historial completo de una conversación, guardado paso a paso. Mientras uses el mismo `thread_id`, el grafo va a poder "acordarse" de todo lo que pasó antes en ese thread.

---

## 6. Probando la memoria con un `thread_id`

Definimos una config con un `thread_id` y hacemos la primera invocación:

```python
# Specify a thread
config = {"configurable": {"thread_id": "1"}}

# Specify an input
messages = [HumanMessage(content="Add 3 and 4.")]

# Run
messages = react_graph_memory.invoke({"messages": messages}, config)
for m in messages['messages']:
    m.pretty_print()
```

Salida:

```
Human Message: Add 3 and 4.
Ai Message: Tool Calls → add(a=3, b=4)
Tool Message: Name: add → 7
Ai Message: The sum of 3 and 4 is 7.
```

Ahora, la parte clave: hacemos una **segunda invocación**, pasando **el mismo `config`** (mismo `thread_id: "1"`):

```python
messages = [HumanMessage(content="Multiply that by 2.")]
messages = react_graph_memory.invoke({"messages": messages}, config)
for m in messages['messages']:
    m.pretty_print()
```

Salida:

```
Human Message: Add 3 and 4.
Ai Message: Tool Calls → add(a=3, b=4)
Tool Message: Name: add → 7
Ai Message: The sum of 3 and 4 is 7.

Human Message: Multiply that by 2.
Ai Message: Tool Calls → multiply(a=7, b=2)
Tool Message: Name: multiply → 14
Ai Message: The result of multiplying 7 by 2 is 14.
```

**En español:** esta vez sí funciona. Como usamos el mismo `thread_id`, LangGraph recupera **todo el estado previo** guardado en ese thread (la conversación completa: "Add 3 and 4" → tool call → "7" → respuesta) y le agrega el nuevo `HumanMessage` ("Multiply that by 2.") a continuación. El modelo ahora tiene todo el contexto disponible, entiende que "that" es el `7` recién calculado, y hace `multiply(7, 2) = 14`, que es lo correcto.

En resumen: al compilar el grafo con un checkpointer, cada paso queda guardado como un checkpoint dentro de un thread. Pasando el mismo `thread_id` en invocaciones futuras, el grafo retoma exactamente donde había quedado, con acceso a todo el estado anterior.

---

## 7. Viendo la memoria en Studio

El mismo código de este notebook está disponible como `agent.py` dentro de la carpeta `studio/`, registrado en `langgraph.json`.

Un detalle importante: **en Studio no hace falta pasar el checkpointer manualmente**. Esto es porque Studio está respaldado por la API de LangGraph, que empaqueta el código y trae su propia capa de persistencia (en este caso, Postgres) sin que el usuario tenga que configurar nada.

Para levantarlo localmente:

```bash
langgraph dev
```

Desde ahí, dentro del proyecto (que incluye "Simple Graph", "Router" y "Agent"), se puede abrir un thread nuevo, escribir por ejemplo `"Multiply two and three"` y ver:

1. El input del usuario.
2. Cómo el `assistant` arma un tool call estructurado (nombre de la función y argumentos) a partir del lenguaje natural.
3. Cómo ese tool call va al nodo `"tools"`, que ejecuta la función `multiply` de verdad.
4. El `ToolMessage` con el resultado (`6`), que vuelve al `assistant`.
5. La respuesta final en lenguaje natural ("el resultado de multiplicar 2 y 3 es 6"), que termina el grafo.

> Nota: al igual que en el notebook del router, la documentación aclara que la herramienta que se ve en el video (la app de escritorio) fue reemplazada por **LangSmith Studio**, que se corre localmente con `langgraph dev` y se abre desde el navegador.

---

## Resumen de conceptos clave

| Concepto | Qué es | En este ejemplo |
|---|---|---|
| **Estado transitorio** | Por defecto, el estado de un grafo solo existe durante esa ejecución (`invoke`) | Sin memoria, el agente no recordaba el `7` de la invocación anterior |
| **Checkpointer** | Componente que guarda el estado del grafo en cada paso | `MemorySaver()` |
| **Checkpoint** | Una "foto" del estado del grafo en un paso puntual, con metadatos (próximo nodo, ID, etc.) | Se genera automáticamente en cada paso |
| **Thread** | Colección de checkpoints que representan una misma conversación | Identificado por `thread_id` |
| **`config` con `thread_id`** | Forma de decirle al grafo a qué conversación (thread) pertenece esta invocación | `{"configurable": {"thread_id": "1"}}` |
| **Persistencia en Studio** | Studio no requiere pasar un checkpointer manual; usa su propia capa de persistencia (Postgres) vía la API de LangGraph | Nada que configurar del lado del usuario |

Con muy poco código (`MemorySaver` + `thread_id`) el agente pasa de ser completamente "amnésico" entre invocaciones a poder sostener una conversación de varios turnos, recordando resultados anteriores. Esta idea de persistencia es la base sobre la que se van a construir conceptos más avanzados más adelante en el curso.
