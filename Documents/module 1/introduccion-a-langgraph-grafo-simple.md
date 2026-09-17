# Introducción a LangGraph: El Grafo Más Simple

Este documento explica, paso a paso, cómo construir el grafo más simple posible en **LangGraph**, con el objetivo de presentar sus componentes fundamentales: **State**, **Nodes** y **Edges** (incluyendo los **conditional edges**).

## Idea general

Vamos a construir un grafo con esta estructura:

```
START → node_1 → (node_2 o node_3, decidido al azar) → END
```

- La conexión entre `START` y `node_1` es una **arista normal** (*normal edge*): siempre se ejecuta.
- La conexión desde `node_1` hacia `node_2` o `node_3` es una **arista condicional** (*conditional edge*): se elige uno de los dos caminos según una condición definida por nosotros (en este caso, una probabilidad 50/50).
- Tanto `node_2` como `node_3` terminan en `END`.

Este ejemplo es intencionalmente sencillo: no resuelve ningún problema real, sino que sirve para mostrar la mecánica básica de LangGraph antes de pasar a casos más complejos.

---

## 1. Instalación

Primero instalamos la librería `langgraph`.

```python
%%capture --no-stderr
%pip install --quiet -U langgraph
```

---

## 2. Definir el State (estado)

El **State** es el objeto que se va pasando entre los nodos y las aristas del grafo. Funciona como el "esquema de entrada" para todos los nodos y aristas.

En este ejemplo, el estado es simplemente un diccionario con una sola clave, `graph_state`, cuyo valor es un string. Usamos `TypedDict` (del módulo `typing`) para darle tipado a ese diccionario.

```python
from typing_extensions import TypedDict

class State(TypedDict):
    graph_state: str
```

**En español:** esto no es más que una forma tipada de decir "el estado va a ser un diccionario con la clave `graph_state`, y su valor va a ser siempre un texto (string)".

---

## 3. Definir los Nodes (nodos)

Los **nodos** son simplemente funciones de Python. Cada nodo:

- Recibe el `state` como primer argumento (posicional).
- Puede leer el valor actual de `graph_state` con `state['graph_state']`.
- Devuelve un nuevo valor para esa misma clave.

Por defecto, el valor que devuelve cada nodo **sobrescribe** el valor anterior del estado (esto se puede personalizar más adelante con lo que LangGraph llama *reducers*, pero no es necesario para este ejemplo).

```python
def node_1(state):
    print("---Node 1---")
    return {"graph_state": state['graph_state'] + " I am"}

def node_2(state):
    print("---Node 2---")
    return {"graph_state": state['graph_state'] + " happy!"}

def node_3(state):
    print("---Node 3---")
    return {"graph_state": state['graph_state'] + " sad!"}
```

**En español:**
- `node_1` toma el texto que ya está en `graph_state` y le agrega `" I am"`.
- `node_2` le agrega `" happy!"`.
- `node_3` le agrega `" sad!"`.

En resumen: cada nodo es una función Python muy simple que toma el estado y devuelve un estado nuevo (actualizado).

---

## 4. Definir los Edges (aristas)

Las **aristas** son las que conectan los nodos entre sí.

- **Arista normal:** se usa cuando siempre queremos ir de un nodo a otro (por ejemplo, de `START` a `node_1`).
- **Arista condicional:** se usa cuando queremos decidir dinámicamente a qué nodo ir a continuación. Se implementa como una función que devuelve el nombre del próximo nodo a visitar.

En este ejemplo, la arista condicional (`decide_mood`) recibe el estado, pero en realidad no lo usa para decidir nada (es solo un ejemplo de juguete). La decisión se toma con un número aleatorio: si es menor a 0.5, vamos a `node_2`; si no, vamos a `node_3`.

```python
import random
from typing import Literal

def decide_mood(state) -> Literal["node_2", "node_3"]:
    
    # A menudo usaremos el estado para decidir a qué nodo ir
    user_input = state['graph_state']
    
    # Aquí simplemente hacemos un 50/50 entre los nodos 2 y 3
    if random.random() < 0.5:

        # El 50% de las veces, devolvemos Node 2
        return "node_2"
    
    # El otro 50% de las veces, devolvemos Node 3
    return "node_3"
```

**En español:** `decide_mood` es la función de la arista condicional. Tira una especie de "moneda" y, según el resultado, decide si el flujo continúa hacia `node_2` o hacia `node_3`. En casos más avanzados, esta función analizaría el contenido real del `state` para tomar una decisión con sentido (por ejemplo, el resultado de un modelo, una clasificación, etc.), pero aquí es solo azar.

---

## 5. Construir el grafo

Ahora combinamos todo lo anterior usando la clase `StateGraph`:

1. Se inicializa `StateGraph` con la clase `State` definida antes.
2. Se agregan los nodos con `add_node`.
3. Se definen las conexiones (aristas):
   - Una arista normal entre `START` y `node_1`.
   - Una arista condicional entre `node_1` y (`node_2` / `node_3`), usando la función `decide_mood`.
   - Aristas normales desde `node_2` y `node_3` hacia `END`.
4. Se **compila** el grafo (esto hace validaciones básicas de la estructura).
5. Se puede **visualizar** el grafo como un diagrama Mermaid.

```python
from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END

# Build graph
builder = StateGraph(State)
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_node("node_3", node_3)

# Logic
builder.add_edge(START, "node_1")
builder.add_conditional_edges("node_1", decide_mood)
builder.add_edge("node_2", END)
builder.add_edge("node_3", END)

# Add
graph = builder.compile()

# View
display(Image(graph.get_graph().draw_mermaid_png()))
```

**En español:**
- `START` y `END` son nodos especiales: `START` indica por dónde entra la información al grafo, y `END` indica el nodo terminal (donde el grafo deja de ejecutarse).
- Al compilar el grafo, LangGraph revisa que la estructura sea válida.
- La visualización con Mermaid muestra el flujo: `START → node_1`, luego una línea punteada (que representa la arista condicional) hacia `node_2` o `node_3`, y finalmente ambos convergiendo en `END`.

---

## 6. Ejecutar el grafo (invocación)

El grafo compilado implementa el llamado **protocolo runnable**, una interfaz estándar de LangChain que incluye métodos como `invoke`.

Para ejecutar el grafo, simplemente lo invocamos con un valor inicial para el estado:

```python
graph.invoke({"graph_state": "Hi, this is Lance."})
```

**En español, lo que pasa internamente:**
1. El grafo arranca desde `START`.
2. Pasa a `node_1`, que agrega `" I am"` al estado.
3. La arista condicional (`decide_mood`) decide, al azar, si seguir hacia `node_2` o `node_3`.
4. El nodo elegido agrega `" happy!"` o `" sad!"` respectivamente.
5. El grafo llega a `END` y devuelve el estado final.

Por ejemplo, si el camino elegido fue `node_1 → node_3`, la salida sería:

```python
{'graph_state': 'Hi, this is Lance. I am sad!'}
```

Y en la consola se habría impreso:

```
---Node 1---
---Node 3---
```

Si se ejecuta varias veces, el resultado cambia cada vez (a veces "happy!", a veces "sad!"), porque la elección entre `node_2` y `node_3` es aleatoria.

### Sobre `invoke`

- `invoke` ejecuta **todo el grafo de forma sincrónica**: espera que cada paso termine antes de pasar al siguiente.
- Devuelve el **estado final**, es decir, el estado después de que se ejecutó el último nodo del camino recorrido (en el ejemplo, después de `node_3`).

---

## Resumen de conceptos clave

| Concepto | Qué es | En este ejemplo |
|---|---|---|
| **State** | El objeto (diccionario) que se pasa entre nodos y aristas | Un diccionario con una clave: `graph_state` (string) |
| **Node** | Una función Python que recibe el estado y devuelve un nuevo estado | `node_1`, `node_2`, `node_3` |
| **Normal edge** | Conexión fija entre dos nodos, siempre se recorre | `START → node_1`, `node_2 → END`, `node_3 → END` |
| **Conditional edge** | Conexión que decide dinámicamente el próximo nodo mediante una función | `decide_mood`, entre `node_1` y `node_2`/`node_3` |
| **`invoke`** | Método estándar para ejecutar el grafo de punta a punta, de forma sincrónica | `graph.invoke({"graph_state": "..."})` |

Este ejemplo, aunque simple, contiene todos los bloques básicos que se van a repetir (con más complejidad) en cualquier grafo de LangGraph: definir un estado, escribir nodos que lo transformen, y conectar esos nodos con aristas normales y condicionales.
