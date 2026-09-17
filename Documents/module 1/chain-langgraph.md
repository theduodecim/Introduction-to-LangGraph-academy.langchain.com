# Chain: mensajes, chat models y tools en LangGraph

## Repaso

Ya construimos un grafo simple con nodos, aristas normales y aristas condicionales.

## Objetivo

Ahora vamos a construir una **chain** (cadena) que combina 4 ideas:

1. Usar **chat messages** (mensajes de chat) como estado del grafo.
2. Usar **chat models** (modelos de chat) dentro de los nodos.
3. **Bindear tools** (herramientas) a nuestro chat model.
4. **Ejecutar tool calls** (llamadas a herramientas) dentro de los nodos del grafo.

Primero vamos a ver estas ideas por separado, y después las vamos a integrar todas en un grafo de LangGraph.

---

## 1. Instalación

```python
%%capture --no-stderr
%pip install --quiet -U langchain_openai langchain_core langgraph
```

---

## 2. Messages (mensajes)

Los modelos de chat interactúan mediante **mensajes**, que representan los distintos roles dentro de una conversación. LangChain soporta varios tipos:

- `HumanMessage`: mensaje del usuario/humano.
- `AIMessage`: mensaje del modelo (la IA).
- `SystemMessage`: instrucción de comportamiento para el modelo.
- `ToolMessage`: mensaje que viene de la ejecución de una tool.

Cada mensaje puede tener:
- `content`: el contenido del mensaje.
- `name` (opcional): quién es el autor del mensaje.
- `response_metadata` (opcional): metadatos, generalmente completados por el proveedor del modelo en los `AIMessage`.

Creamos una lista de mensajes simulando una conversación:

```python
from pprint import pprint
from langchain_core.messages import AIMessage, HumanMessage

messages = [AIMessage(content=f"So you said you were researching ocean mammals?", name="Model")]
messages.append(HumanMessage(content=f"Yes, that's right.", name="Lance"))
messages.append(AIMessage(content=f"Great, what would you like to learn about.", name="Model"))
messages.append(HumanMessage(content=f"I want to learn about the best place to see Orcas in the US.", name="Lance"))

for m in messages:
    m.pretty_print()
```

**En español:** esto arma una lista de mensajes alternando `AIMessage` y `HumanMessage`, simulando el ida y vuelta de una conversación entre "Model" y "Lance". `pretty_print()` simplemente los imprime en consola de forma legible.

---

## 3. Chat Models

Los modelos de chat toman una **secuencia de mensajes** como entrada. Acá se usa OpenAI a través de `ChatOpenAI`.

```python
import os, getpass

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("OPENAI_API_KEY")
```

**En español:** esta función chequea si la variable de entorno `OPENAI_API_KEY` ya está definida; si no, te la pide de forma segura (sin mostrarla en pantalla) y la guarda como variable de entorno.

Con la key configurada, se carga el modelo y se lo invoca con la lista de mensajes:

```python
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o")
result = llm.invoke(messages)
type(result)
```

El resultado es un objeto `AIMessage`. Si lo inspeccionamos:

```python
result
```

Vemos que tiene:
- `content`: el texto de respuesta del modelo.
- `response_metadata`: información adicional, como tokens usados, nombre del modelo, motivo de finalización, etc.

```python
result.response_metadata
```

**En español:** en resumen, le pasamos una lista de mensajes al modelo, y este nos devuelve un `AIMessage` con el texto generado más metadatos útiles (por ejemplo, cuántos tokens consumió la llamada).

---

## 4. Tools (herramientas)

Las **tools** son útiles cuando queremos que el modelo interactúe con sistemas externos (por ejemplo, una API), que normalmente requieren un formato de entrada específico (un *payload*) en lugar de lenguaje natural.

Cuando le "bindeamos" (asociamos) una función al modelo como tool, le damos conocimiento sobre el esquema de entrada que esa función necesita. El modelo decide, según el mensaje del usuario, si conviene llamar a esa tool, y devuelve una salida que respeta el esquema de la función.

Ejemplo con una función simple, `multiply`:

```python
def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

llm_with_tools = llm.bind_tools([multiply])
```

**En español:** `bind_tools` le "avisa" al modelo que tiene disponible la función `multiply`, incluyendo qué parámetros necesita (`a` y `b`, ambos enteros) según el docstring y las anotaciones de tipo de la función.

Si le pasamos una consulta en lenguaje natural que requiere esa función, el modelo no responde con texto, sino con un **tool call**:

```python
tool_call = llm_with_tools.invoke([HumanMessage(content=f"What is 2 multiplied by 3", name="Lance")])
```

```python
tool_call.tool_calls
```

Esto devuelve algo como:

```
[{'name': 'multiply',
  'args': {'a': 2, 'b': 3},
  'id': 'call_lBBBNo5oYpHGRqwxNaNRbsiT',
  'type': 'tool_call'}]
```

**En español:** el modelo entendió que la pregunta requiere multiplicar, y en vez de devolver texto libre, devuelve una estructura con el **nombre de la función** a llamar (`multiply`) y los **argumentos** exactos (`a: 2, b: 3`). Esto es lo que después se puede usar para ejecutar la función real.

---

## 5. Usar mensajes como el estado del grafo

Ahora integramos todo esto en LangGraph. Definimos el estado como `MessagesState`, un `TypedDict` con una sola clave, `messages`, que es una lista de mensajes:

```python
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage

class MessagesState(TypedDict):
    messages: list[AnyMessage]
```

### El problema: por defecto el estado se sobrescribe

Como vimos en el grafo simple, por defecto cada nodo **sobrescribe** el valor de la clave del estado. Pero en una conversación no queremos perder el historial: queremos **agregar** (append) cada mensaje nuevo a la lista existente, no reemplazarla.

### La solución: reducers

Un **reducer** define cómo se actualiza una clave del estado. Si no se especifica ninguno, LangGraph sobrescribe el valor (comportamiento por defecto). Para que los mensajes se acumulen, usamos el reducer ya incluido `add_messages`, anotando la clave `messages`:

```python
from typing import Annotated
from langgraph.graph.message import add_messages

class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
```

**En español:** el `Annotated[..., add_messages]` le dice a LangGraph: "cuando actualices esta clave, no sobrescribas la lista completa; en su lugar, agregá el mensaje nuevo al final de la lista existente".

### `MessagesState` ya viene predefinido

Como este patrón (una lista de mensajes con el reducer `add_messages`) es tan común, LangGraph ya trae su propia clase `MessagesState` lista para usar, así que no hace falta redefinirla a mano:

```python
from langgraph.graph import MessagesState

class MessagesState(MessagesState):
    # Podemos agregar más claves además de "messages", que ya viene incluida
    pass
```

### Viendo el reducer `add_messages` en aislado

```python
# Initial state
initial_messages = [AIMessage(content="Hello! How can I assist you?", name="Model"),
                    HumanMessage(content="I'm looking for information on marine biology.", name="Lance")
                   ]

# New message to add
new_message = AIMessage(content="Sure, I can help with that. What specifically are you interested in?", name="Model")

# Test
add_messages(initial_messages, new_message)
```

**En español:** esto confirma que `add_messages` simplemente toma la lista original y le agrega el mensaje nuevo al final, devolviendo la lista completa actualizada.

---

## 6. Armar el grafo

Con `MessagesState` definido, armamos un nodo que:
1. Toma los mensajes actuales del estado.
2. Invoca al modelo (con tools bindeadas).
3. Devuelve el resultado como un nuevo mensaje para agregar al estado.

```python
from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END
    
# Node
def tool_calling_llm(state: MessagesState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# Build graph
builder = StateGraph(MessagesState)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_edge(START, "tool_calling_llm")
builder.add_edge("tool_calling_llm", END)
graph = builder.compile()

# View
display(Image(graph.get_graph().draw_mermaid_png()))
```

**En español:** el grafo es muy simple: `START → tool_calling_llm → END`. El único nodo llama al modelo (que tiene la tool `multiply` bindeada) con los mensajes actuales del estado, y el mensaje de respuesta se agrega automáticamente gracias al reducer `add_messages`.

---

## 7. Probando el grafo

### Caso 1: mensaje que no requiere ninguna tool

```python
messages = graph.invoke({"messages": HumanMessage(content="Hello!")})
for m in messages['messages']:
    m.pretty_print()
```

Salida:

```
================================ Human Message =================================

Hello!
================================== Ai Message ==================================

Hi there! How can I assist you today?
```

**En español:** el modelo responde con lenguaje natural porque no detecta ninguna necesidad de usar la tool `multiply`.

### Caso 2: mensaje que sí requiere la tool

```python
messages = graph.invoke({"messages": HumanMessage(content="Multiply 2 and 3")})
for m in messages['messages']:
    m.pretty_print()
```

Salida:

```
================================ Human Message =================================

Multiply 2 and 3!
================================== Ai Message ==================================
Tool Calls:
  multiply (call_Er4gChFoSGzU7lsuaGzfSGTQ)
 Call ID: call_Er4gChFoSGzU7lsuaGzfSGTQ
  Args:
    a: 2
    b: 3
```

**En español:** en este caso, el modelo detecta que la consulta necesita multiplicar, así que en vez de devolver texto, devuelve un **tool call** con el nombre de la función (`multiply`) y los argumentos (`a: 2`, `b: 3`), exactamente igual a lo que vimos antes "en aislado", pero ahora corriendo dentro del grafo de LangGraph.

> Nota: en este punto el grafo **decide** que hay que llamar a la tool, pero todavía no la **ejecuta** de verdad (no hay un nodo que corra `multiply` y devuelva el resultado). Eso es el siguiente paso natural: agregar un nodo de ejecución de tools al grafo.

---

## Resumen de conceptos clave

| Concepto | Qué es | En este ejemplo |
|---|---|---|
| **Messages** | Objetos que representan turnos de una conversación (`HumanMessage`, `AIMessage`, etc.) | Conversación entre "Lance" y "Model" |
| **Chat model** | Modelo que recibe una lista de mensajes y devuelve un `AIMessage` | `ChatOpenAI(model="gpt-4o")` |
| **Tool** | Función Python que el modelo puede "elegir" llamar, con un esquema de entrada definido | `multiply(a, b)` |
| **`bind_tools`** | Le da al modelo conocimiento de qué tools tiene disponibles | `llm.bind_tools([multiply])` |
| **Tool call** | Salida del modelo que indica qué función llamar y con qué argumentos | `{'name': 'multiply', 'args': {'a': 2, 'b': 3}}` |
| **`MessagesState`** | Estado predefinido de LangGraph: una lista de mensajes con reducer `add_messages` | Estado de nuestro grafo |
| **Reducer (`add_messages`)** | Define cómo se actualiza una clave del estado; en este caso, agrega en vez de sobrescribir | Permite conservar el historial completo de la conversación |

Con esto ya tenemos los cuatro bloques que se pidieron: mensajes como estado, un chat model dentro de un nodo, una tool bindeada al modelo, y la generación de tool calls dentro del grafo. El paso siguiente lógico es agregar la **ejecución** real de esas tools.
