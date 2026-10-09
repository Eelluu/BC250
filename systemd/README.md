# systemd/ — units legacy (pre-v10)

Los servicios de la pila **v10** (llama-swap + bot de Discord) **no se copian desde este
directorio**: `install.sh` los genera inline en `/etc/systemd/system/`.
Esta carpeta conserva únicamente los units de la etapa anterior, por referencia histórica.

## Fuente canónica (v10)

| Servicio | Se instala en | Lo genera |
|----------|---------------|-----------|
| `queue-runner.service` (bot Discord) | `/etc/systemd/system/queue-runner.service` | `install.sh` (bloque systemd, sección "Create services") |
| `llama-swap.service` | `/etc/systemd/system/llama-swap.service` | `install.sh` |
| `queue-runner` (OpenRC) | `/etc/init.d/queue-runner` | `install.sh` (solo Alpine) |

El wrapper `/opt/netscan/bin/queue-runner-wrapper.sh` (también generado por `install.sh`)
carga `/opt/netscan/.env` con `set -a`, porque `EnvironmentFile` de systemd **no** elimina
las comillas de valores como `LLM_MODEL="gemma4-12b-q3"`.

Forma correcta de (re)instalar los servicios:

```bash
sudo ./install.sh install      # crea los units
sudo systemctl restart queue-runner
```

No copies los ficheros `.service` de esta carpeta a `/etc/systemd/system/`: los units de la
era Signal/Ollama dependen de servicios que ya no existen en v10 y comprueban el puerto
equivocado (`11434` en vez de `8080` de llama-swap).

## Qué queda aquí

- `signal-cli.service` — daemon JSON-RPC de Signal (stack anterior).
- `ollama.service`, `ollama.service.d/override.conf`, `ollama-watchdog.*`, `ollama-proxy.service`
  — stack Ollama (sustituido por llama-swap).
- `bc250-health.*` — health check del host (sigue siendo útil, pero se instala aparte).
- `queue-runner-nightly.*` — batch nocturno (anterior).
- `scripts/ollama-proxy.py` — proxy de Ollama (anterior).
