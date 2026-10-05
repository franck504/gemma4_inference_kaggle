# 🚀 Gemma 4 Inference API — Kaggle + ngrok

> **LLM inference API gratuite** basée sur **Gemma 4 E4B-IT** (Google), tournant sur **Kaggle GPU T4**, exposée publiquement via **ngrok**, compatible avec les standards **OpenAI**, **Anthropic** et un format natif.

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![Kaggle](https://img.shields.io/badge/Kaggle-GPU_T4-20BEFF?logo=kaggle)](https://kaggle.com)
[![Gemma](https://img.shields.io/badge/Gemma_4-E4B--IT-orange?logo=google)](https://ai.google.dev/gemma)
[![ngrok](https://img.shields.io/badge/ngrok-tunnel-blueviolet)](https://ngrok.com)

---

## 📋 Table des matières

- [Aperçu](#-aperçu)
- [Architecture](#-architecture)
- [Prérequis](#-prérequis)
- [Configuration ngrok (Kaggle Secrets)](#-configuration-ngrok--kaggle-secrets)
- [Lancement du notebook](#-lancement-du-notebook)
- [Endpoints API](#-endpoints-api)
- [Exemples d'utilisation](#-exemples-dutilisation)
- [Script de test](#-script-de-test)
- [Paramètres de génération](#-paramètres-de-génération)
- [Logs](#-logs)
- [Limitations](#-limitations)

---

## 🔍 Aperçu

Ce projet permet de déployer une **API d'inférence LLM** en quelques minutes, **gratuitement**, en utilisant uniquement Kaggle (GPU T4 gratuit) et un compte ngrok (tunnel HTTPS gratuit).

**Cas d'usage :** prototypage, démo, tests d'intégration, remplacement temporaire d'OpenAI/Anthropic.

### Standards supportés

| Standard | Endpoints | SDK compatible |
|----------|-----------|----------------|
| 🟢 **Natif** | `POST /generate`, `POST /generate/stream` | `requests`, `httpx`, `curl` |
| 🔵 **OpenAI** | `GET /v1/models`, `POST /v1/chat/completions` | `openai`, `LangChain`, `litellm` |
| 🟠 **Anthropic** | `POST /v1/messages` | `anthropic` |

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     KAGGLE KERNEL                        │
│                                                          │
│  ┌──────────────┐    ┌──────────────────────────────┐   │
│  │  Gemma 4     │    │  FastAPI (port 8000)          │   │
│  │  E4B-IT      │◄──►│  ├── GET  /health             │   │
│  │  4-bit NF4   │    │  ├── POST /generate           │   │
│  │  ~9 GB VRAM  │    │  ├── POST /generate/stream    │   │
│  └──────────────┘    │  ├── GET  /v1/models          │   │
│                      │  ├── POST /v1/chat/completions │   │
│                      │  └── POST /v1/messages         │   │
│                      └────────────┬─────────────────┘   │
└───────────────────────────────────┼─────────────────────┘
                                    │ ngrok tunnel
                                    ▼
                    https://XXXXX.ngrok-free.dev
                            │
                ┌───────────┼───────────┐
                ▼           ▼           ▼
           openai SDK   anthropic   curl/requests
```

---

## ✅ Prérequis

### 1. Compte Kaggle
- Créer un compte sur [kaggle.com](https://www.kaggle.com)
- **Accepter les conditions** du modèle Gemma 4 : [kaggle.com/models/google/gemma-4](https://www.kaggle.com/models/google/gemma-4) → cliquer sur **"Request Access"**
- Activer la **vérification du numéro de téléphone** (requis pour les GPU)

### 2. Compte ngrok
- Créer un compte gratuit sur [ngrok.com](https://ngrok.com)
- Récupérer votre **Authtoken** : [dashboard.ngrok.com/get-started/your-authtoken](https://dashboard.ngrok.com/get-started/your-authtoken)

---

## 🔑 Configuration ngrok — Kaggle Secrets

Le token ngrok doit être stocké dans les **Kaggle Secrets** (ne jamais le mettre en clair dans le notebook).

### Étapes :

1. Ouvrir votre notebook sur [kaggle.com](https://kaggle.com)
2. Dans le panneau de gauche → icône **"Add-ons"** → **"Secrets"** *(ou menu Settings → Secrets)*
3. Cliquer sur **"+ Add a new secret"**
4. Remplir :
   - **Label (clé)** : `NGROK_TOKEN`  ← ⚠️ Exactement ce nom, sensible à la casse
   - **Value** : votre Authtoken ngrok → [dashboard.ngrok.com/get-started/your-authtoken](https://dashboard.ngrok.com/get-started/your-authtoken)
5. Activer le toggle **"Notebook access"** → **Save**

### Vérification dans le notebook

Le notebook récupère automatiquement le secret avec :

```python
from kaggle_secrets import UserSecretsClient
NGROK_TOKEN = UserSecretsClient().get_secret("NGROK_TOKEN")
```

> ⚠️ `os.environ.get("NGROK_TOKEN")` **ne fonctionne pas** dans Kaggle — les secrets ne sont pas exposés comme variables d'environnement. Toujours utiliser `UserSecretsClient`.

---

## 🚀 Lancement du notebook

### Étape 0 — Importer le notebook dans Kaggle

Il y a **deux façons** d'utiliser ce notebook dans Kaggle :

#### Option A — Import direct du fichier `.ipynb` *(recommandé)*

1. Aller sur [kaggle.com/code](https://www.kaggle.com/code) → **"New Notebook"**
2. Une fois dans l'éditeur → menu **"File"** → **"Import Notebook"**
3. Onglet **"GitHub"** → entrer l'URL du repo :
   ```
   https://github.com/franck504/gemma4_inference_kaggle
   ```
   → Sélectionner `gemma_inference.ipynb` → **Import**

#### Option B — Clone manuel

1. Télécharger le fichier `gemma_inference.ipynb` depuis GitHub
2. Sur Kaggle → **"New Notebook"** → **"File"** → **"Upload Notebook"** → glisser le `.ipynb`

---

### Configuration Kaggle requise

Avant de lancer les cellules :

1. **Accelerator** : `Settings → Accelerator → GPU T4 x2`
2. **Internet** : activer l'accès internet dans Settings *(requis pour ngrok)*
3. **Secrets** : ajouter `NGROK_TOKEN` (voir section ci-dessus)

### Ordre d'exécution des cellules

> ⚠️ **Ne pas utiliser "Run All"** — la cellule 3 prend 5-15 min. Exécuter cellule par cellule.

| Cellule | Description | Durée estimée |
|---------|-------------|--------------|
| **1** | Installation des dépendances | ~2 min |
| **2** | Vérification GPU | ~5s |
| **3** | ⏳ Téléchargement + chargement modèle (cache après 1ère fois) | **5-15 min** |
| **4** | Test inférence locale (optionnel) | ~30s |
| **5** | 🚀 Lancement API + ngrok (tous les endpoints) | ~10s |
| **6** | Tests automatisés des 7 endpoints | ~3-5 min |
| **7** | Affichage du guide d'utilisation | ~1s |

### Sortie attendue de la cellule 5

```
✅ Modèle 'gemma-4-e4b-it' détecté — initialisation...
✅ Token ngrok depuis Kaggle Secrets
🔐 Token ngrok configuré

═══════════════════════════════════════════════════════════════════
🌐 GEMMA 4 MULTI-STANDARD API — EN LIGNE
═══════════════════════════════════════════════════════════════════
🔗 URL publique : https://XXXXX.ngrok-free.dev
📚 Swagger UI   : https://XXXXX.ngrok-free.dev/docs
───────────────────────────────────────────────────────────────────
🟢 NATIVE
   POST https://XXXXX.ngrok-free.dev/generate
   POST https://XXXXX.ngrok-free.dev/generate/stream
🔵 OPENAI COMPATIBLE
   GET  https://XXXXX.ngrok-free.dev/v1/models
   POST https://XXXXX.ngrok-free.dev/v1/chat/completions
🟠 ANTHROPIC COMPATIBLE
   POST https://XXXXX.ngrok-free.dev/v1/messages
═══════════════════════════════════════════════════════════════════

✅ Serveur démarré — logs des requêtes ci-dessous ↓
```

---

## 📡 Endpoints API

### Status

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| `GET` | `/` | Info générale + état GPU |
| `GET` | `/health` | Health check |
| `GET` | `/docs` | Documentation Swagger interactive |

### 🟢 Standard Natif

| Méthode | Endpoint | Streaming |
|---------|----------|-----------|
| `POST` | `/generate` | Non — retourne JSON complet |
| `POST` | `/generate/stream` | Oui — SSE `{token, done}` |

**Corps de requête :**
```json
{
  "prompt": "Votre question",
  "max_new_tokens": 512,
  "temperature": 0.7,
  "top_p": 0.9,
  "repetition_penalty": 1.1,
  "system_prompt": "Tu es un assistant utile."
}
```

**Réponse `/generate` :**
```json
{
  "text": "La réponse du modèle...",
  "model": "gemma-4-e4b-it",
  "prompt_tokens": 25,
  "completion_tokens": 87,
  "total_tokens": 112,
  "generation_time_s": 5.34
}
```

**Réponse `/generate/stream` (SSE) :**
```
data: {"token": "La ", "done": false}
data: {"token": "réponse", "done": false}
data: {"token": " du", "done": false}
...
data: {"token": "", "done": true, "prompt_tokens": 25, "completion_tokens": 87, "generation_time_s": 5.34}
```

---

### 🔵 Standard OpenAI

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| `GET` | `/v1/models` | Liste des modèles (compatible `client.models.list()`) |
| `POST` | `/v1/chat/completions` | Chat — `stream: true/false` |

**Format de requête identique à OpenAI :**
```json
{
  "model": "gemma-4-e4b-it",
  "messages": [
    {"role": "system", "content": "Tu es un assistant utile."},
    {"role": "user", "content": "Bonjour !"}
  ],
  "max_tokens": 512,
  "temperature": 0.7,
  "stream": false
}
```

**Réponse non-streaming :**
```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1728124800,
  "model": "gemma-4-e4b-it",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "Bonjour ! Comment puis-je vous aider ?"},
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 20, "completion_tokens": 12, "total_tokens": 32}
}
```

**Réponse streaming (SSE) :**
```
data: {"id":"chatcmpl-abc123","choices":[{"delta":{"role":"assistant","content":""},"finish_reason":null}]}
data: {"id":"chatcmpl-abc123","choices":[{"delta":{"content":"Bonjour"},"finish_reason":null}]}
data: {"id":"chatcmpl-abc123","choices":[{"delta":{},"finish_reason":"stop"}]}
data: [DONE]
```

---

### 🟠 Standard Anthropic

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| `POST` | `/v1/messages` | Messages API — `stream: true/false` |

**Format de requête identique à Anthropic :**
```json
{
  "model": "gemma-4-e4b-it",
  "system": "Tu es un assistant utile.",
  "messages": [{"role": "user", "content": "Bonjour !"}],
  "max_tokens": 512,
  "stream": false
}
```

**Réponse non-streaming :**
```json
{
  "id": "msg_abc123",
  "type": "message",
  "role": "assistant",
  "content": [{"type": "text", "text": "Bonjour ! Comment puis-je vous aider ?"}],
  "model": "gemma-4-e4b-it",
  "stop_reason": "end_turn",
  "usage": {"input_tokens": 20, "output_tokens": 12}
}
```

**Réponse streaming — événements SSE Anthropic :**
```
event: message_start
data: {"type":"message_start","message":{"id":"msg_abc123",...}}

event: content_block_start
data: {"type":"content_block_start","index":0,...}

event: content_block_delta
data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Bonjour"}}

event: message_stop
data: {"type":"message_stop"}
```

---

## 💻 Exemples d'utilisation

Remplacer `https://XXXXX.ngrok-free.dev` par votre URL ngrok affichée dans la cellule 5.

### 🟢 Natif — curl

```bash
# Non-streaming
curl -X POST https://XXXXX.ngrok-free.dev/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explique le machine learning en 3 phrases.",
    "max_new_tokens": 200,
    "temperature": 0.7
  }'

# Streaming
curl -N -X POST https://XXXXX.ngrok-free.dev/generate/stream \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Liste 5 frameworks Python.", "max_new_tokens": 300}'
```

### 🟢 Natif — Python

```python
import requests, json

API = "https://XXXXX.ngrok-free.dev"

# Non-streaming
r = requests.post(f"{API}/generate", json={
    "prompt": "C'est quoi Python ?",
    "max_new_tokens": 150,
    "system_prompt": "Réponds en français, brièvement."
})
print(r.json()["text"])

# Streaming
with requests.post(f"{API}/generate/stream",
    json={"prompt": "Explique Docker.", "max_new_tokens": 200},
    stream=True) as r:
    for line in r.iter_lines():
        if line:
            d = json.loads(line.decode()[5:])  # strip "data: "
            if not d["done"]:
                print(d["token"], end="", flush=True)
```

---

### 🔵 OpenAI SDK

```bash
pip install openai
```

```python
from openai import OpenAI

client = OpenAI(
    api_key="dummy",   # n'importe quelle valeur non vide
    base_url="https://XXXXX.ngrok-free.dev/v1"
)

# Non-streaming
response = client.chat.completions.create(
    model="gemma-4-e4b-it",
    messages=[
        {"role": "system", "content": "Tu es un assistant utile."},
        {"role": "user",   "content": "Explique le deep learning."}
    ],
    max_tokens=200,
    temperature=0.7,
)
print(response.choices[0].message.content)
print(f"Tokens: {response.usage}")

# Streaming
for chunk in client.chat.completions.create(
    model="gemma-4-e4b-it",
    messages=[{"role": "user", "content": "Bonjour !"}],
    stream=True,
):
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

### 🔵 LangChain

```bash
pip install langchain-openai
```

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

llm = ChatOpenAI(
    base_url="https://XXXXX.ngrok-free.dev/v1",
    api_key="dummy",
    model="gemma-4-e4b-it",
    temperature=0.7,
)

response = llm.invoke([
    SystemMessage(content="Tu es un expert Python."),
    HumanMessage(content="Explique les décorateurs Python.")
])
print(response.content)

# Streaming
for chunk in llm.stream("Liste 5 frameworks Python."):
    print(chunk.content, end="", flush=True)
```

### 🔵 litellm

```bash
pip install litellm
```

```python
import litellm

response = litellm.completion(
    model="openai/gemma-4-e4b-it",
    api_base="https://XXXXX.ngrok-free.dev/v1",
    api_key="dummy",
    messages=[{"role": "user", "content": "Bonjour !"}],
    max_tokens=200,
)
print(response.choices[0].message.content)
```

---

### 🟠 Anthropic SDK

```bash
pip install anthropic
```

```python
import anthropic

client = anthropic.Anthropic(
    api_key="dummy",
    base_url="https://XXXXX.ngrok-free.dev",
)

# Non-streaming
message = client.messages.create(
    model="gemma-4-e4b-it",
    max_tokens=200,
    system="Tu es un assistant utile. Réponds en français.",
    messages=[{"role": "user", "content": "C'est quoi le deep learning ?"}]
)
print(message.content[0].text)
print(f"Tokens: {message.usage}")

# Streaming
with client.messages.stream(
    model="gemma-4-e4b-it",
    max_tokens=200,
    messages=[{"role": "user", "content": "Explique Docker en 2 phrases."}]
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
```

---

## 🧪 Script de test

Le script [`test_api.sh`](./test_api.sh) teste les **7 endpoints** des 3 standards en une commande :

```bash
chmod +x test_api.sh
./test_api.sh https://XXXXX.ngrok-free.dev
```

**Résultat attendu :**
```
✅ 1/7 GET  /health
✅ 2/7 🟢  POST /generate                   → 5.6s
✅ 3/7 🟢  POST /generate/stream            → 9.6s (streaming)
✅ 4/7 🔵  POST /v1/chat/completions        → chatcmpl-xxx
✅ 5/7 🔵  POST /v1/chat/completions stream → [DONE]
✅ 6/7 🟠  POST /v1/messages               → msg_xxx
✅ 7/7 🟠  POST /v1/messages stream        → message_stop
✅ TOUS LES TESTS PASSÉS ! (7/7)
```

---

## ⚙️ Paramètres de génération

| Paramètre | Défaut | Plage | Standards |
|-----------|:------:|-------|-----------|
| `max_new_tokens` / `max_tokens` | 512 | 1–2048 | Tous |
| `temperature` | 0.7 | 0.01–2.0 | Tous |
| `top_p` | 0.9 | 0.01–1.0 | Tous |
| `repetition_penalty` | 1.1 | 1.0–2.0 | Natif, OpenAI |
| `system_prompt` | null | string | Natif |
| `system` | null | string | Anthropic |

### Conseils

| Usage | temperature | top_p |
|-------|:-----------:|:-----:|
| 🎯 Précis / factuel | 0.2 – 0.4 | 0.85 |
| ⚖️ Équilibré (défaut) | 0.6 – 0.8 | 0.90 |
| 🎨 Créatif | 1.0 – 1.3 | 0.95 |
| 💻 Code | 0.2 – 0.3 | 0.85 |

---

## 📋 Logs

Les logs sont affichés **en temps réel dans la sortie de la cellule 5** de Kaggle :

```
HH:MM:SS METHOD /endpoint                           → STATUS |    Xms | IP:x.x.x.x [Standard]
────────────────────────────────────────────────────────────────────────────────────────
12:08:01 GET    /health                              → 200 |     2ms | IP:34.71.X.X
12:08:09 POST   /generate                            → 200 |  5596ms | IP:34.71.X.X [Native]
12:08:35 POST   /v1/chat/completions                 → 200 |  7024ms | IP:34.71.X.X [OpenAI]
12:09:01 POST   /v1/messages                         → 200 |  8341ms | IP:34.71.X.X [Anthropic]
```

Chaque log indique :
- ⏰ **Heure** de la requête
- 📋 **Méthode + endpoint**
- ✅ **Status HTTP** (vert < 400, rouge ≥ 400)
- ⏱️ **Temps de réponse** en ms
- 🌐 **IP** du demandeur (vraie IP via `X-Forwarded-For` de ngrok)
- 🏷️ **Standard** utilisé (Native / OpenAI / Anthropic)

---

## ⚠️ Limitations

| Limitation | Détail |
|------------|--------|
| **Durée session** | Kaggle arrête le kernel après ~12h d'inactivité GPU |
| **URL ngrok** | Change à chaque redémarrage de la cellule 5 (plan gratuit) |
| **1 tunnel** | Le plan gratuit ngrok permet 1 seul tunnel simultané |
| **Concurrent requests** | Le modèle génère de façon séquentielle (1 requête à la fois) |
| **VRAM** | ~9 GB utilisés sur 14.6 GB disponibles (T4 x2) |
| **Vitesse** | ~7-10 tokens/s en 4-bit sur T4 |

### Redémarrer après expiration

Si la session Kaggle expire, relancer dans l'ordre :
```
Cellule 3 → Cellule 5 → (optionnel) Cellule 6
```
Les cellules 1 et 2 ne sont pas nécessaires après le premier lancement.

---

## 📁 Structure du projet

```
kaggle_LLM/Gemma_4/
├── gemma_inference.ipynb   # Notebook principal (7 cellules)
├── test_api.sh             # Script de test bash (7 endpoints)
└── README.md               # Ce fichier
```

---

## 🔗 Ressources

- [Gemma 4 sur Kaggle](https://www.kaggle.com/models/google/gemma-4)
- [Documentation FastAPI](https://fastapi.tiangolo.com)
- [ngrok Dashboard](https://dashboard.ngrok.com)
- [OpenAI SDK Python](https://github.com/openai/openai-python)
- [Anthropic SDK Python](https://github.com/anthropic/anthropic-sdk-python)
- [LangChain OpenAI](https://python.langchain.com/docs/integrations/chat/openai)
- [litellm](https://github.com/BerriAI/litellm)

---

## 📄 Licence

MIT — libre d'utilisation pour prototypage et usage personnel.
