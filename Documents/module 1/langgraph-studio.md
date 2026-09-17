# LangGraph Studio: visualizar los grafos fuera del notebook

## ¿Qué es?

Es una IDE visual para correr y depurar los grafos de LangGraph sin usar un notebook. Cada módulo del curso tiene su propia carpeta `studio/` con todo lo necesario para cargarla como proyecto.

## Estructura de la carpeta `studio`

```
studio/
├── __pycache__/
├── .langgraph_api/
├── .env            ← acá van tus API keys
├── .env.example    ← plantilla de referencia
├── agent.py        ← un grafo
├── router.py       ← otro grafo
├── simple.py       ← el grafo simple que armamos (node_1, node_2, node_3)
├── langgraph.json  ← config que le dice a Studio qué grafos cargar
└── requirements.txt
```

- `langgraph.json` define los grafos disponibles (por ejemplo "agent", "simple", "router") y sus rutas.
- El `.env` es lo único que hay que configurar a mano: copiar `.env.example` a `.env` y completar las API keys. Sin esto, Studio no puede ejecutar los nodos que llaman a un LLM.

## Cómo levantarlo

Desde adentro de la carpeta `studio/`:

```bash
langgraph dev
```

Esto levanta un servidor local y abre LangGraph Studio en el navegador, cargando automáticamente todos los grafos declarados en `langgraph.json` (incluido el `simple_graph` que construimos en el notebook anterior).

## Qué se puede hacer ahí

- **Elegir el grafo**: en el ejemplo del video, seleccionan "Simple Graph" (el de `node_1` → `node_2`/`node_3` → `END`).
- **Ingresar el estado inicial** desde un formulario visual (equivalente a `graph.invoke({"graph_state": "..."})` en el notebook), en vez de escribir código.
- **Ver la ejecución nodo por nodo**: Studio resalta qué nodo se está ejecutando en cada momento.
- **Threads**: cada corrida del grafo queda agrupada en un "thread", que guarda el historial de esa ejecución. Se puede crear un thread nuevo ("new run") para cada invocación y volver atrás para revisar corridas anteriores (por ejemplo, ver que una vez fue a `node_2` y otra vez a `node_3`, producto de la arista condicional aleatoria).
- Es una herramienta pensada sobre todo para **debugging visual**: en este grafo simple no aporta demasiado, pero se vuelve muy útil en grafos más complejos que vienen después.

## Resumen del flujo de trabajo

1. Configurar `.env` con las API keys (una sola vez por módulo).
2. Parado dentro de `studio/`, correr `langgraph dev`.
3. Se abre Studio en el navegador, se elige el grafo, se ingresa el estado inicial y se ejecuta.
4. Se inspeccionan los nodos ejecutados y el historial de threads.
