# Router: un agente simple en LangGraph

## Repaso

Ya armamos un grafo que usa `messages` como estado y un chat model con tools bindeadas. Vimos que ese grafo puede:

- Devolver un **tool call**.
- Devolver una **respuesta en lenguaje natural**.

## Objetivo

Podemos pensar a ese comportamiento como un **router**: el chat model "enruta" entre dos caminos posibles (responder directo, o llamar a una tool) según el input del usuario.

Esto es un ejemplo simple de **agente**: es el LLM quien decide el flujo de control de la aplicación, ya sea llamando a una tool o respondiendo directamente.

Para completar el router necesitamos dos ideas nuevas:

1. Agregar un **nodo** que efectivamente ejecute la tool (hasta ahora solo generábamos el tool call, pero nunca lo corríamos de verdad).
2. Agregar una **arista condicional** que mire la salida del chat model y decida: si es un tool call, ir al nodo de tools; si no, terminar (`END`).

---

## 1. Instalación

```python
%%capture --no-stderr
%pip install --quiet -U langchain_openai langchain_core langgraph langgraph-prebuilt
```

Nota el paquete nuevo acá: `langgraph-prebuilt`, que trae componentes ya armados por LangGraph (los que vamos a usar en este notebook).

---

## 2. API key del modelo

```python
import os, getpass

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("OPENAI_API_KEY")
```

**En español:** igual que en los notebooks anteriores, esto pide la `OPENAI_API_KEY` solo si todavía no está seteada como variable de entorno.

---

## 3. El modelo y la tool

Reutilizamos la misma tool de siempre, `multiply`, y se la bindeamos al modelo:

```python
from langchain_openai import ChatOpenAI

def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

llm = ChatOpenAI(model="gpt-4o")
llm_with_tools = llm.bind_tools([multiply])
```

Nada nuevo hasta acá: es lo mismo que veníamos haciendo en el notebook de "chain".

---

## 4. Los componentes nuevos: `ToolNode` y `tools_condition`

Acá está la novedad. En vez de armar a mano el nodo que ejecuta la tool y la lógica de la arista condicional, LangGraph nos da dos componentes **prearmados**:

- **`ToolNode`**: un nodo listo para usar que ejecuta una tool. Solo hay que pasarle la lista de funciones (tools) con las que va a trabajar.
- **`tools_condition`**: una arista condicional lista para usar, que mira el último mensaje del modelo:
  - Si es un **tool call** → enruta al nodo de tools.
  - Si **no** es un tool call → enruta a `END`.

```python
from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition

# Node
def tool_calling_llm(state: MessagesState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# Build graph
builder = StateGraph(MessagesState)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", ToolNode([multiply]))
builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges(
    "tool_calling_llm",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
)
builder.add_edge("tools", END)
graph = builder.compile()

# View
display(Image(graph.get_graph().draw_mermaid_png()))
```

**En español, paso a paso:**

- El nodo `tool_calling_llm` es el mismo que en el notebook anterior: invoca al modelo (con la tool bindeada) con los mensajes actuales.
- Se agrega un segundo nodo, `"tools"`, que es una instancia de `ToolNode([multiply])`: este nodo sabe cómo ejecutar la función `multiply` de verdad cuando le llega un tool call.
- El flujo queda así:
  - `START → tool_calling_llm` (arista normal, siempre se ejecuta).
  - Desde `tool_calling_llm`, se usa `tools_condition` como arista condicional: si el modelo devolvió un tool call, va a `"tools"`; si no, va directo a `END`.
  - `"tools" → END` (una vez ejecutada la tool, el grafo termina).

Esto es exactamente el patrón "router" que se describe arriba: el LLM decide el camino, y el grafo lo ejecuta.

---

## 5. Probando el router

```python
from langchain_core.messages import HumanMessage
messages = [HumanMessage(content="Hello, what is 2 multiplied by 2?")]
messages = graph.invoke({"messages": messages})
for m in messages['messages']:
    m.pretty_print()
```

Con una entrada que sí requiere la tool (como multiplicar dos números), el grafo:

1. Pasa por `tool_calling_llm`, que devuelve un tool call.
2. `tools_condition` detecta que es un tool call y enruta a `"tools"`.
3. El nodo `"tools"` ejecuta `multiply` de verdad y agrega un `ToolMessage` con el resultado al estado.
4. El grafo termina (`END`).

**En español:** a diferencia del notebook anterior, acá el tool call **no se queda solo como una intención**: el `ToolNode` realmente corre la función Python y devuelve el resultado como un mensaje más en la conversación (`ToolMessage`).

Si en cambio la entrada es algo como `"Hello world."`, que no necesita ninguna tool, el flujo es:

1. `tool_calling_llm` responde en lenguaje natural (sin tool call).
2. `tools_condition` ve que no hay tool call, así que enruta directo a `END`.
3. El usuario recibe la respuesta del modelo sin pasar por el nodo `"tools"`.

---

## 6. Viendo el router en Studio

Al igual que con el grafo simple, este router también está disponible como script Python (`router.py`) dentro de la carpeta `studio/`, registrado en `langgraph.json`.

Para levantarlo:

```bash
langgraph dev
```

Esto abre Studio en el navegador. Ahí se puede:

- Elegir el grafo `router`.
- Mandar un mensaje que **no** necesite la tool (ej: `"Hi, I'm Lance"`) y ver que responde directo, sin pasar por el nodo de tools.
- Crear un thread nuevo y mandar algo que **sí** la necesite (ej: `"Multiply 2 and 3"`) y ver visualmente:
  - El tool call generado (nombre de la función y argumentos, bien formateados).
  - Cómo el flujo pasa por el nodo `"tools"`.
  - El `ToolMessage` con el resultado final.

**En español:** Studio permite comparar de forma visual, thread por thread, los dos caminos posibles del router: respuesta directa vs. ejecución de tool, algo que es mucho más difícil de "ver" solo leyendo el output del notebook.

> Nota: la documentación actual de LangChain indica que, desde que se grabó este video, la herramienta pasó a llamarse **LangSmith Studio** y ahora se corre localmente (con `langgraph dev`) y se accede desde el navegador, en vez de usar la app de escritorio que se muestra en el video.

---

## Resumen de conceptos clave

| Concepto | Qué es | En este ejemplo |
|---|---|---|
| **Router** | Patrón donde el LLM decide el flujo: responder directo o llamar a una tool | El nodo `tool_calling_llm` |
| **`ToolNode`** | Nodo prearmado de LangGraph que ejecuta una o más tools | `ToolNode([multiply])` |
| **`tools_condition`** | Arista condicional prearmada: va a `"tools"` si hay tool call, si no va a `END` | Conecta `tool_calling_llm` con `"tools"` o `END` |
| **`ToolMessage`** | Mensaje que representa el resultado de haber ejecutado una tool | Se agrega al estado después de correr `multiply` |
| **Agente simple** | Sistema donde el LLM controla el flujo de la aplicación | Este mismo router |

Este router es la versión más básica de un **agente**: el modelo decide qué hacer, y el grafo se encarga de ejecutar esa decisión (llamar a la tool o terminar). Es la base sobre la que se construyen agentes más complejos más adelante en el curso.
