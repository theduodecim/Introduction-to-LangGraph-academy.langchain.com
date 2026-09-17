# Deployment: llevar el agente a producción

## Repaso

Ya llegamos a un agente con memoria que puede:

- **`act`**: dejar que el modelo llame tools específicas.
- **`observe`**: pasarle el resultado de la tool de vuelta al modelo.
- **`reason`**: razonar sobre ese resultado para decidir el próximo paso.
- **`persist state`**: usar un checkpointer en memoria para conversaciones largas, con interrupciones.

## Objetivo

Todo esto lo construimos en un notebook. La pregunta natural es: **¿y ahora cómo lo llevo a producción?** Este notebook cubre cómo desplegar el agente, tanto localmente (con Studio) como en la nube (con LangSmith / LangGraph Cloud).

---

## 1. Instalación

```python
%%capture --no-stderr
%pip install --quiet -U langgraph_sdk langchain_core
```

Acá aparece un paquete nuevo: `langgraph_sdk`, la librería que vamos a usar para interactuar con los grafos ya desplegados (sea local o en la nube).

---

## 2. Conceptos clave

Antes de tocar código, conviene tener claros estos cinco conceptos:

- **LangGraph**: la librería (Python y JavaScript) que venimos usando para armar los agentes/grafos.
- **LangGraph API**: empaqueta el código del grafo y le agrega, sobre eso, una cola de tareas (*task queue*) para manejar operaciones asíncronas, y persistencia para mantener el estado entre interacciones.
- **LangSmith Deployment** (antes llamado *LangGraph Cloud*): el servicio hosteado de la LangGraph API. Permite desplegar grafos directamente desde un repositorio de GitHub, y da monitoreo y tracing de los grafos desplegados. Cada deployment queda accesible mediante una URL propia.
- **LangSmith Studio** (antes *LangGraph Studio*): el IDE que venimos usando, que por detrás usa la API como backend. Puede correr **local** (como veníamos haciendo) o apuntar a un **deployment en la nube**.
- **LangGraph SDK**: la librería de Python para interactuar programáticamente con los grafos, ya sea que estén corriendo local o en la nube. Da una interfaz consistente: crear clientes, acceder a "assistants", manejar threads, y ejecutar runs.

**En español, la idea general:** Studio en realidad **ya venía usando la API** todo este tiempo, solo que de forma implícita. Ahora lo hacemos explícito usando el SDK directamente desde el notebook, y después mostramos que lo mismo funciona apuntando a un deployment en la nube: es la misma interfaz, solo cambia la URL.

---

## 3. Testing local: levantar Studio

Igual que en notebooks anteriores, desde la carpeta `studio/` del módulo:

```bash
langgraph dev
```

Esto debería mostrar algo así:

```
- 🚀 API: http://127.0.0.1:2024
- 🎨 Studio UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
- 📚 API Docs: http://127.0.0.1:2024/docs
```

Se abre la **Studio UI** en el navegador, apuntando a esa API local.

> Nota: como en los notebooks anteriores, la herramienta pasó a llamarse **LangSmith Studio**, corriendo localmente y accedida desde el navegador, en lugar de la app de escritorio que se muestra en el video original.

```python
if 'google.colab' in str(get_ipython()):
    raise Exception("Unfortunately LangGraph Studio is currently not supported on Google Colab")
```

**En español:** este chequeo simplemente corta la ejecución si estás en Google Colab, porque ahí no se puede correr Studio localmente.

---

## 4. Conectarse a la API local con el SDK

Acá está lo nuevo: en vez de interactuar con Studio solo desde el navegador, usamos el **SDK** desde el propio notebook para hablarle a esa misma API local.

```python
from langgraph_sdk import get_client
```

```python
# This is the URL of the local development server
URL = "http://127.0.0.1:2024"
client = get_client(url=URL)

# Search all hosted graphs
assistants = await client.assistants.search()
```

**En español:** `get_client(url=...)` crea un cliente apuntando a la URL de la API (en este caso, la que corre local gracias a `langgraph dev`). Con `client.assistants.search()` (notar el `await`, es una llamada asíncrona) obtenemos la lista de todos los grafos ("assistants") disponibles en ese servidor: `simple_graph`, `router`, `agent`, etc.

Inspeccionamos uno de ellos:

```python
assistants[-3]
```

```
{'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca',
 'graph_id': 'agent',
 'config': {},
 'metadata': {'created_by': 'system'},
 'name': 'agent',
 'created_at': '2025-03-04T22:57:28.424565+00:00',
 'updated_at': '2025-03-04T22:57:28.424565+00:00',
 'version': 1}
```

**En español:** cada "assistant" tiene un `assistant_id` propio, y un `graph_id` que indica a cuál de nuestros grafos corresponde (acá, `"agent"`, el que construimos en el notebook de memoria).

### Crear un thread y correr el agente vía SDK

```python
# We create a thread for tracking the state of our run
thread = await client.threads.create()
```

```python
from langchain_core.messages import HumanMessage

# Input
input = {"messages": [HumanMessage(content="Multiply 3 by 2.")]}

# Stream
async for chunk in client.runs.stream(
        thread['thread_id'],
        "agent",
        input=input,
        stream_mode="values",
    ):
    if chunk.data and chunk.event != "metadata":
        print(chunk.data['messages'][-1])
```

Salida (resumida, un mensaje por línea):

```
Human: Multiply 3 by 2.
Ai: (tool call) multiply(a=3, b=2)
Tool (multiply): 6
Ai: The result of multiplying 3 by 2 is 6.
```

**En español, paso a paso:**

- `client.threads.create()` crea un thread nuevo (igual concepto que vimos con `MemorySaver`, pero ahora administrado del lado de la API).
- `client.runs.stream(...)` corre el grafo `"agent"` en ese thread, con el mensaje humano como input.
- `stream_mode="values"` significa que vamos a ir recibiendo el **estado completo** del grafo después de cada paso (el streaming en detalle se ve más adelante en el curso).
- En cada `chunk`, `chunk.data['messages'][-1]` es el último mensaje agregado al estado en ese paso: primero el humano, después el tool call, después el resultado de la tool, y finalmente la respuesta en lenguaje natural.

Es exactamente el mismo comportamiento que veníamos viendo con `invoke()` en el notebook anterior, pero ahora accedido de forma remota (aunque "remota" acá siga siendo `localhost`) a través del SDK, en vez de llamar directamente a la función Python del grafo.

---

## 5. Testing con Cloud (LangSmith Deployment)

El siguiente paso es desplegar el mismo grafo en la nube. A alto nivel, los pasos son:

### a) Crear un repositorio en GitHub

- Ir a GitHub, crear un repositorio nuevo (por ejemplo, `langchain-academy`).

### b) Agregarlo como remote y subir el código

Desde la terminal, en la carpeta donde clonaste `langchain-academy` al empezar el curso:

```bash
git remote add origin https://github.com/tu-usuario/tu-repo.git
git push -u origin main
```

### c) Conectar LangSmith con el repositorio

- Entrar a [LangSmith](https://smith.langchain.com/).
- Ir a la pestaña **Deployments**.
- Click en **+ New Deployment**.
- Elegir el repositorio de GitHub recién creado.
- Apuntar el **LangGraph API config file** a uno de los `langgraph.json` dentro de las carpetas `studio` (por ejemplo, `module-1/studio/langgraph.json`).
- Configurar las API keys necesarias (se pueden copiar directamente del archivo `.env` de esa misma carpeta `studio`).

**En español:** en esencia, le estás diciendo a LangSmith "tomá este repo, fijate en este `langgraph.json` qué grafos hay definidos, y desplegalos como un servicio hosteado". Esto es básicamente lo mismo que hace `langgraph dev` localmente, pero corriendo en la infraestructura de LangSmith en vez de en tu máquina.

Una vez desplegado, el deployment queda con:

- Una **URL propia** para acceder a la API.
- **Trazas recientes** y **monitoreo** de las ejecuciones.
- Acceso directo a **LangGraph Studio**, ahora apuntando a esa URL de producción en vez de a `localhost`.

---

## 6. Conectarse al deployment en la nube con el SDK

Una vez desplegado, se puede interactuar con él exactamente igual que con la versión local, cambiando solo la URL.

Primero, la API key de LangSmith:

```python
import os, getpass

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("LANGSMITH_API_KEY")
```

Después, apuntamos el cliente a la URL del deployment (en vez de `http://127.0.0.1:2024`):

```python
# Replace this with the URL of your deployed graph
URL = "https://langchain-academy-8011c561878d50b1883f7ed11b32d720.default.us.langgraph.app"
client = get_client(url=URL)

# Search all hosted graphs
assistants = await client.assistants.search()
```

```python
# Select the agent
agent = assistants[0]
```

```python
agent
```

```
{'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca',
 'graph_id': 'agent',
 'created_at': '2024-08-23T17:58:02.722920+00:00',
 'updated_at': '2024-08-23T17:58:02.722920+00:00',
 'config': {},
 'metadata': {'created_by': 'system'}}
```

Y corremos el mismo tipo de invocación que antes:

```python
from langchain_core.messages import HumanMessage

# We create a thread for tracking the state of our run
thread = await client.threads.create()

# Input
input = {"messages": [HumanMessage(content="Multiply 3 by 2.")]}

# Stream
async for chunk in client.runs.stream(
        thread['thread_id'],
        "agent",
        input=input,
        stream_mode="values",
    ):
    if chunk.data and chunk.event != "metadata":
        print(chunk.data['messages'][-1])
```

Salida (resumida):

```
Human: Multiply 3 by 2.
Ai: (tool call) multiply(a=3, b=2)
Tool (multiply): 6
Ai: 3 multiplied by 2 equals 6.
```

**En español, el punto central de todo el notebook:** el código del lado del cliente es **prácticamente idéntico** al que usamos con el servidor local. Lo único que cambió fue el valor de `URL`: pasamos de `http://127.0.0.1:2024` (la API corriendo en tu máquina) a la URL pública del deployment en LangSmith. El SDK te da esa interfaz uniforme, sin importar si el grafo corre en tu laptop o en la nube.

---

## Resumen de conceptos clave

| Concepto | Qué es |
|---|---|
| **LangGraph** | La librería para definir los grafos/agentes (lo que veníamos usando) |
| **LangGraph API** | Empaqueta el grafo y le agrega cola de tareas + persistencia |
| **LangSmith Deployment** | Hosteo de esa API en la nube, desplegable desde un repo de GitHub, con monitoreo y tracing |
| **LangSmith Studio** | El IDE visual; puede apuntar a la API local o a un deployment en la nube |
| **LangGraph SDK** | Librería Python (`get_client`, `client.assistants`, `client.threads`, `client.runs`) para interactuar con cualquiera de los dos |
| **`assistant_id` / `graph_id`** | Identifican, respectivamente, la instancia y la definición del grafo desplegado |
| **`thread_id`** | Igual que en el notebook de memoria: agrupa el estado/historial de una conversación |
| **`stream_mode="values"`** | Devuelve el estado completo del grafo después de cada paso, a medida que se ejecuta |

En definitiva, el flujo completo del módulo termina así: un mismo grafo, escrito una sola vez, se puede probar en el notebook con `invoke()`, visualizarse con Studio local, y finalmente desplegarse a producción con un par de clicks conectando GitHub a LangSmith. El código para interactuar con él —local o en la nube— es el mismo SDK, cambiando solo la URL.
