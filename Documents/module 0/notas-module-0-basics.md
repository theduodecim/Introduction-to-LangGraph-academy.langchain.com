# Module 0 — Basics: notas rápidas

## Primera llamada al modelo (vía OpenRouter)

```python
from langchain_core.messages import HumanMessage
msg = HumanMessage(content="Hello world", name="Lance")
gpt4o_chat.invoke([msg])
```

Resultado:

```
AIMessage(content='Hello! How can I help you today?', ...)
model_provider: 'openai'
model_name: 'nvidia/nemotron-3.5-lightning:free'
finish_reason: 'stop'
tool_calls: []
usage_metadata: {input_tokens: 18, output_tokens: 330, total_tokens: 348, reasoning_tokens: 321}
```

**Lectura del resultado:**
- El objeto que devuelve el modelo es siempre un `AIMessage` (no un string plano).
- `model_provider: 'openai'` no significa que uses OpenAI — es porque OpenRouter expone la misma interfaz OpenAI-compatible; el modelo real está en `model_name`.
- `reasoning_tokens: 321` sobre 330 tokens de salida: Nemotron 3.5 Lightning razona internamente incluso para un saludo simple. Con este modelo, contá con más consumo de tokens/latencia por llamada que con un modelo sin razonamiento explícito.
- `tool_calls: []` — no se usó ninguna herramienta en esta llamada (todavía no se definió ninguna).
- `cost: 0` — tier free de OpenRouter.

## Conceptos básicos del video/notebook (module-0/basics)

- **Chat models**: reciben una secuencia de *mensajes* como input y devuelven un mensaje como output. LangChain no aloja modelos propios, son todas integraciones de terceros (OpenAI, Anthropic, etc.) con una interfaz común.
- **Mensajes**: tienen un `role` (quién habla) y un `content` (el texto). `HumanMessage`, `AIMessage`, etc.
- **Atajo**: si le pasás un string directo a `.invoke()`, LangChain lo convierte automáticamente en un `HumanMessage`.
- **Parámetros estándar**:
  - `model`: nombre del modelo.
  - `temperature`: qué tan determinística es la respuesta. Bajo (ej. 0) → respuestas más factuales/consistentes. Alto (cerca de 1) → más creativo/variado.
- **Métodos comunes a todos los chat models**: `invoke()` (devuelve la respuesta completa) y `stream()` (devuelve tokens/chunks a medida que se generan).
- **Portabilidad entre proveedores**: la interfaz (`invoke`, `stream`, mensajes) es la misma sin importar el modelo — por eso pudimos cambiar de OpenAI a OpenRouter/Nemotron sin tocar la lógica del notebook, solo la config de conexión.
- **Tavily**: search tool usado en module-4 (opcional, hay alternativas como Wikipedia o Google si no querés usarlo).

## Para tener en cuenta con el modelo actual (Nemotron 3.5 Lightning free)

- Alto uso de `reasoning_tokens` → más lento/pesado que un modelo sin razonamiento en tareas triviales.
- Si en algún notebook se pone lento, pesado, o falla con tool calling: cambiar `OPENROUTER_MODEL` a `meta-llama/llama-3.3-70b-instruct:free` en el `.env` y reiniciar el kernel — sin tocar código.
