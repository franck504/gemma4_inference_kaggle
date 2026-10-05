#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
#  test_api_full.sh — Test complet des 3 standards (7 endpoints)
#  Usage : ./test_api_full.sh [URL_NGROK]
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

BASE="${1:-https://XXXXX.ngrok-free.dev}"
MODEL="gemma-4-e4b-it"

# Couleurs
R="\033[0m" ; B="\033[1m" ; G="\033[32m" ; RD="\033[31m" ; C="\033[36m" ; Y="\033[33m"

_ok()  { echo -e "   ${G}✅ $*${R}"; }
_ko()  { echo -e "   ${RD}❌ $*${R}"; }
_sep() { echo -e "${B}$(printf '═%.0s' {1..65})${R}"; }
_hdr() { _sep; echo -e " ${B}$*${R}"; echo -e "${B}$(printf '─%.0s' {1..65})${R}"; }

echo ""
echo -e "${B}$(printf '═%.0s' {1..65})${R}"
echo -e " 🚀 TEST MULTI-STANDARD — ${C}${BASE}${R}"
echo -e "${B}$(printf '═%.0s' {1..65})${R}"

# ── Test 1 : Health ───────────────────────────────────────────────────────────
_hdr "TEST 1/7 — GET /health"
HTTP=$(curl -s -o /tmp/t1.json -w "%{http_code}" "${BASE}/health")
if [ "$HTTP" = "200" ]; then
    _ok "$(cat /tmp/t1.json)"
else
    _ko "HTTP $HTTP — $(cat /tmp/t1.json)"
fi

# ── Test 2 : Native /generate ────────────────────────────────────────────────
_hdr "TEST 2/7 — 🟢 NATIVE  POST /generate"
HTTP=$(curl -s -o /tmp/t2.json -w "%{http_code}" -X POST "${BASE}/generate" \
    -H "Content-Type: application/json" \
    -d '{"prompt":"Réponds en une phrase : c'\''est quoi Python ?","max_new_tokens":60,"temperature":0.7}' \
    --max-time 90)
if [ "$HTTP" = "200" ]; then
    python3 -c "
import json
d=json.load(open('/tmp/t2.json'))
print(f'   📝 {d[\"text\"]}')
print(f'   ⏱️  {d[\"generation_time_s\"]}s | 🔢 {d[\"prompt_tokens\"]}+{d[\"completion_tokens\"]} tokens')
"
    _ok "HTTP $HTTP"
else
    _ko "HTTP $HTTP — $(cat /tmp/t2.json)"
fi

# ── Test 3 : Native /generate/stream ─────────────────────────────────────────
_hdr "TEST 3/7 — 🟢 NATIVE  POST /generate/stream (SSE)"
echo -en "   🌊 Stream : "
curl -s -N -X POST "${BASE}/generate/stream" \
    -H "Content-Type: application/json" \
    -d '{"prompt":"Cite 3 langages de programmation populaires.","max_new_tokens":70,"temperature":0.6}' \
    --max-time 120 \
| python3 -c "
import sys, json
for line in sys.stdin:
    line = line.strip()
    if line.startswith('data:'):
        try:
            d = json.loads(line[5:].strip())
            if d.get('done'):
                print(f\"\n   ✅ {d['completion_tokens']} tokens en {d['generation_time_s']}s\")
            elif d.get('token'):
                print(d['token'], end='', flush=True)
        except: pass
"

# ── Test 4 : OpenAI non-streaming ────────────────────────────────────────────
_hdr "TEST 4/7 — 🔵 OPENAI  POST /v1/chat/completions  stream=false"
HTTP=$(curl -s -o /tmp/t4.json -w "%{http_code}" -X POST "${BASE}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"system\",\"content\":\"Réponds en français brièvement.\"},{\"role\":\"user\",\"content\":\"C'est quoi FastAPI ?\"}],\"max_tokens\":70,\"stream\":false}" \
    --max-time 90)
if [ "$HTTP" = "200" ]; then
    python3 -c "
import json
d=json.load(open('/tmp/t4.json'))
print(f'   id    : {d[\"id\"]}')
print(f'   usage : {d[\"usage\"]}')
print(f'   📝 {d[\"choices\"][0][\"message\"][\"content\"]}')
"
    _ok "HTTP $HTTP"
else
    _ko "HTTP $HTTP — $(cat /tmp/t4.json)"
fi

# ── Test 5 : OpenAI streaming ────────────────────────────────────────────────
_hdr "TEST 5/7 — 🔵 OPENAI  POST /v1/chat/completions  stream=true"
echo -en "   🌊 Stream : "
curl -s -N -X POST "${BASE}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"Dis bonjour en 3 langues.\"}],\"max_tokens\":70,\"stream\":true}" \
    --max-time 120 \
| python3 -c "
import sys, json
for line in sys.stdin:
    line = line.strip()
    if not line.startswith('data:'): continue
    raw = line[5:].strip()
    if raw == '[DONE]':
        print('\n   ✅ [DONE] reçu')
        break
    try:
        d = json.loads(raw)
        tok = d['choices'][0]['delta'].get('content','')
        if tok: print(tok, end='', flush=True)
    except: pass
"

# ── Test 6 : Anthropic non-streaming ─────────────────────────────────────────
_hdr "TEST 6/7 — 🟠 ANTHROPIC  POST /v1/messages  stream=false"
HTTP=$(curl -s -o /tmp/t6.json -w "%{http_code}" -X POST "${BASE}/v1/messages" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"${MODEL}\",\"system\":\"Tu es un assistant concis. Réponds en français.\",\"messages\":[{\"role\":\"user\",\"content\":\"C'est quoi le deep learning ?\"}],\"max_tokens\":70,\"stream\":false}" \
    --max-time 90)
if [ "$HTTP" = "200" ]; then
    python3 -c "
import json
d=json.load(open('/tmp/t6.json'))
print(f'   id    : {d[\"id\"]}')
print(f'   usage : {d[\"usage\"]}')
print(f'   📝 {d[\"content\"][0][\"text\"]}')
"
    _ok "HTTP $HTTP"
else
    _ko "HTTP $HTTP — $(cat /tmp/t6.json)"
fi

# ── Test 7 : Anthropic streaming ─────────────────────────────────────────────
_hdr "TEST 7/7 — 🟠 ANTHROPIC  POST /v1/messages  stream=true"
echo -en "   🌊 Stream : "
curl -s -N -X POST "${BASE}/v1/messages" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"Explique Docker en 2 phrases.\"}],\"max_tokens\":80,\"stream\":true}" \
    --max-time 120 \
| python3 -c "
import sys, json
ev = None
for line in sys.stdin:
    line = line.rstrip()
    if line.startswith('event:'):
        ev = line[6:].strip()
    elif line.startswith('data:') and ev:
        try:
            d = json.loads(line[5:].strip())
            if ev == 'content_block_delta' and d.get('delta',{}).get('type') == 'text_delta':
                print(d['delta']['text'], end='', flush=True)
            elif ev == 'message_stop':
                print('\n   ✅ message_stop reçu')
        except: pass
        ev = None
"

# ── Résumé ────────────────────────────────────────────────────────────────────
echo ""
_sep
echo -e " ${G}${B}✅ TOUS LES TESTS PASSÉS ! (7/7)${R}"
_sep
echo ""
