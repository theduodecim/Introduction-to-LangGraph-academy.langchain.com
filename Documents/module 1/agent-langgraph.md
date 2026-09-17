# Agent: el patrón ReAct en LangGraph

## Repaso

Ya armamos un **router**:

- El chat model decide si hace un tool call o no, según el input del usuario.
- Usamos una arista condicional para enrutar a un nodo que llama a la tool, o directamente terminar (`END`).

## Objetivo

Ahora extendemos ese router hasta convertirlo en una **arquitectura de agente genérica**.

En el router de antes, invocábamos al modelo y, si elegía llamar a una tool, le devolvíamos el `ToolMessage` (el resultado) al usuario y ahí terminaba todo.

**La idea clave de este notebook:** ¿qué pasa si, en vez de devolverle ese `ToolMessage` al usuario, se lo mandamos *de nuevo al modelo*? Así el modelo puede decidir si llama a otra tool, o si ya está en condiciones de responder directamente.

Esto es la intuición detrás de **ReAct**, una arquitectura de agente muy popular, que tiene tres componentes:

1. **Act**: dejar que el modelo llame a tools específicas.
2. **Observe**: pasarle el resultado de la tool de vuelta al modelo.
3. **Reason**: dejar que el modelo razone sobre ese resultado y decida qué hacer a continuación (llamar otra tool, o responder directamente).

Este ciclo se repite mientras el modelo siga eligiendo llamar tools, hasta que decide que ya puede responder en lenguaje natural y ahí el flujo termina. En la práctica, conviene además poner límites (por ejemplo, un límite máximo de recursión) para evitar loops infinitos.

---

## 1. Instalación

```python
%%capture --no-stderr
%pip install --quiet -U langchain_openai langchain_core langgraph langgraph-prebuilt
```

---

## 2. API key del modelo

```python
import os, getpass

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("OPENAI_API_KEY")
```

---

## 3. LangSmith (tracing)

Acá aparece algo nuevo: vamos a usar **LangSmith** para hacer *tracing*, es decir, para poder ver en detalle, paso a paso, qué hizo el agente por dentro. Los runs se van a loguear a un proyecto llamado `langchain-academy`.

```python
_set_env("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "langchain-academy"
```

**En español:** con estas tres líneas, cualquier ejecución del grafo que hagamos de acá en adelante queda registrada (loggeada) en LangSmith, bajo el proyecto `langchain-academy`. Esto no cambia el comportamiento del agente, pero nos permite después ir a la web de LangSmith y ver la traza completa de la ejecución.

---

## 4. Definir las tools

A diferencia de los notebooks anteriores (donde solo teníamos `multiply`), ahora definimos **tres tools**: sumar, multiplicar y dividir.

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

# For this ipynb we set parallel tool calling to false as math generally is done sequentially, and this time we have 3 tools that can do math
# the OpenAI model specifically defaults to parallel tool calling for efficiency, see https://python.langchain.com/docs/how_to/tool_calling_parallel/
# play around with it and see how the model behaves with math equations!
llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)
```

**En español, dos detalles importantes:**

- Ahora tenemos una lista `tools = [add, multiply, divide]`, y se la bindeamos entera al modelo con `bind_tools`.
- Se agrega `parallel_tool_calls=False`. Por defecto, algunos modelos (como los de OpenAI) intentan llamar varias tools "en paralelo" para ser más eficientes. Pero como acá las cuentas se tienen que hacer **en orden** (por ejemplo, primero sumar y recién después multiplicar el resultado), se desactiva esa opción para forzar que las llamadas sean secuenciales.

---

## 5. El nodo "assistant" y el mensaje de sistema

Se define un **mensaje de sistema** con instrucciones generales para el agente, y un nodo (`assistant`) que invoca al modelo con ese mensaje de sistema más el historial de mensajes del estado.

```python
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, SystemMessage

# System message
sys_msg = SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs.")

# Node
def assistant(state: MessagesState):
   return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}
```

**En español:** el `SystemMessage` es básicamente la "consigna general" del agente: le decimos que es un asistente útil encargado de hacer operaciones aritméticas. El nodo `assistant` arma la lista de mensajes a mandarle al modelo como `[sys_msg] + state["messages"]`, es decir, primero la instrucción del sistema, y después todo el historial de la conversación acumulado hasta ese momento.

---

## 6. Armar el grafo (con el loop de ReAct)

Usamos, igual que en el router, `MessagesState`, un nodo `Tools` (con la lista completa de tools) y `tools_condition` como arista condicional. El nodo del modelo ahora se llama `assistant` en vez de `tool_calling_llm`.

**La diferencia clave con el router:** en vez de que el nodo `"tools"` termine el grafo, lo conectamos *de nuevo* hacia `"assistant"`, formando un loop.

```python
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode
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

**En español, así queda el flujo:**

1. `START → assistant` (arista normal).
2. Desde `assistant`, `tools_condition` decide:
   - Si el último mensaje es un tool call → va a `"tools"`.
   - Si no → va a `END`.
3. **Lo nuevo:** `"tools" → "assistant"` (en vez de `"tools" → END`). Esto es lo único que se modificó respecto al router, pero cambia completamente el comportamiento: ahora, después de ejecutar una tool, el resultado vuelve al modelo para que siga razonando, en lugar de terminar ahí.
4. Este ciclo `assistant → tools → assistant → tools → ...` se repite tantas veces como el modelo decida seguir llamando tools, hasta que en algún momento responde en lenguaje natural (sin tool call) y recién ahí `tools_condition` lo manda a `END`.

---

## 7. Probando el agente

Le pedimos una cadena de tres operaciones, que necesariamente hay que resolver en orden:

```python
messages = [HumanMessage(content="Add 3 and 4. Multiply the output by 2. Divide the output by 5")]
messages = react_graph.invoke({"messages": messages})
```

```python
for m in messages['messages']:
    m.pretty_print()
```

Salida (resumida):

```
Human Message: Add 3 and 4. Multiply the output by 2. Divide the output by 5

Ai Message: Tool Calls → add(a=3, b=4)
Tool Message: Name: add → 7

Ai Message: Tool Calls → multiply(a=7, b=2)
Tool Message: Name: multiply → 14

Ai Message: Tool Calls → divide(a=14, b=5)
Tool Message: Name: divide → 2.8

Ai Message: The final result after performing the operations (3 + 4) × 2 ÷ 5 is 2.8.
```

**En español, paso a paso, lo que pasó internamente:**

1. El modelo recibe la consigna completa y decide que el primer paso es sumar: llama a `add(3, 4)`.
2. El resultado (`7`) vuelve al modelo como `ToolMessage`.
3. El modelo razona sobre ese `7` y decide multiplicarlo por 2: llama a `multiply(7, 2)`.
4. El resultado (`14`) vuelve al modelo.
5. El modelo razona sobre ese `14` y decide dividirlo por 5: llama a `divide(14, 5)`.
6. El resultado (`2.8`) vuelve al modelo.
7. Ahora el modelo ya tiene todo lo necesario, así que responde en lenguaje natural con el resultado final, **sin** generar un nuevo tool call. Ahí `tools_condition` corta el loop y el grafo termina.

Este es un ejemplo simple de **tres tool calls secuenciales** resueltos por el agente en un solo `invoke`, sin que el usuario tenga que intervenir entre paso y paso.

---

## 8. Mirar el detalle en LangSmith

LangSmith es una plataforma que da acceso a **tracing** (trazabilidad de cada paso) y también a evaluación de modelos. Acá se usa la parte de tracing para "abrir la caja" del agente.

Pasos que se muestran en el video:

1. Ir a `smith.langchain.com` e iniciar sesión.
2. Entrar a la lista de proyectos y abrir el proyecto `langchain-academy` (el mismo nombre que se configuró en `LANGSMITH_PROJECT`).
3. Ahí aparece la traza (*trace*) de la ejecución del agente que acabamos de correr desde el notebook.

Dentro de la traza se puede ver, entre otras cosas:

- Cada invocación individual al modelo (por ejemplo, la llamada inicial a `ChatOpenAI`).
- El **system prompt** exacto que se usó.
- El **input humano** que se mandó.
- Las **tres funciones** (`add`, `multiply`, `divide`) que el modelo decidió llamar, junto con sus *payloads* (argumentos).
- El punto donde se evalúa `tools_condition`, confirmando que la salida del modelo efectivamente contenía un tool call.
- Cada **nodo de tools** ejecutándose (uno por cada llamada: `add`, `multiply`, `divide`), con su resultado.
- Cómo esos resultados vuelven al modelo (`assistant`) y cómo, finalmente, se genera la respuesta en lenguaje natural.
- Metadatos útiles como **uso de tokens** y **latencia** de cada paso.

**En español:** LangSmith es un buen complemento de LangGraph Studio: mientras Studio es más visual y está pensado para interactuar con el grafo en tiempo real, LangSmith se enfoca en el detalle fino de cada ejecución ya ocurrida (qué prompt exacto se mandó, cuántos tokens gastó, cuánto tardó cada paso, etc.), algo muy útil para debuggear y para comparar distintos modelos entre sí más adelante en el curso.

---

## Resumen de conceptos clave

| Concepto | Qué es | En este ejemplo |
|---|---|---|
| **ReAct** | Arquitectura de agente: Act (llamar tool) → Observe (ver resultado) → Reason (decidir el próximo paso) | El loop `assistant ↔ tools` |
| **Diferencia con el router** | El router termina después de ejecutar la tool; el agente vuelve a mandarle el resultado al modelo | `builder.add_edge("tools", "assistant")` en vez de `"tools" → END` |
| **`parallel_tool_calls=False`** | Fuerza al modelo a llamar las tools de a una, en orden | Necesario porque las 3 operaciones son secuenciales |
| **System message** | Instrucción general que define el rol/comportamiento del agente | "You are a helpful assistant tasked with performing arithmetic..." |
| **Loop de tool calls** | El ciclo se repite mientras el modelo siga generando tool calls | Suma → multiplica → divide → respuesta final |
| **LangSmith** | Plataforma de tracing y evaluación para inspeccionar cada paso de una ejecución | Proyecto `langchain-academy` |

Con este cambio tan pequeño (una sola arista nueva, de `"tools"` de vuelta a `"assistant"`) pasamos de un router simple a un **agente genérico** capaz de resolver tareas de varios pasos por su cuenta, encadenando tool calls hasta llegar a la respuesta final.
