"""
Ollama diagnostic — run this to find out WHY generation is falling back.

    python diagnose_ollama.py

Surfaces the actual error instead of the narrator's silent fallback.
"""
import json
import time
import urllib.error
import urllib.request

print("=" * 60)
print("1. Is the Ollama server reachable?")
print("=" * 60)
try:
    with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=5) as r:
        tags = json.loads(r.read())
    print("   YES — server responding on port 11434")
except Exception as e:
    print(f"   NO — {type(e).__name__}: {e}")
    print("   Fix: run `ollama serve` in a terminal")
    raise SystemExit(1)

print()
print("=" * 60)
print("2. Which models are ACTUALLY installed?")
print("=" * 60)
models = tags.get("models", [])
if not models:
    print("   NONE. Fix: ollama pull qwen2.5:7b")
    raise SystemExit(1)
for m in models:
    size_gb = m.get("size", 0) / 1e9
    print(f"   {m.get('name'):<28} {size_gb:.1f} GB")

names = [m.get("name", "") for m in models]
print()
print("   The narrator defaults to 'qwen2.5:7b'.")
if "qwen2.5:7b" in names:
    print("   -> Exact match found. Model name is NOT the problem.")
    target = "qwen2.5:7b"
else:
    print("   -> NOT INSTALLED under that exact tag. This is likely the bug:")
    print("      the availability check matches loosely but generate needs")
    print("      the exact tag.")
    target = names[0]
    print(f"   -> Testing with '{target}' instead.")

print()
print("=" * 60)
print(f"3. Can '{target}' actually generate? (first call loads the model,")
print("   which can take 30-90s on CPU — this is the other likely cause)")
print("=" * 60)
payload = json.dumps({
    "model": target,
    "prompt": "Reply with exactly one word: working",
    "stream": False,
    "options": {"temperature": 0.1, "num_predict": 12},
}).encode()
req = urllib.request.Request(
    "http://localhost:11434/api/generate",
    data=payload,
    headers={"Content-Type": "application/json"},
)
start = time.time()
try:
    # Deliberately generous timeout to separate "slow" from "broken"
    with urllib.request.urlopen(req, timeout=180) as r:
        out = json.loads(r.read())
    elapsed = time.time() - start
    print(f"   SUCCESS in {elapsed:.1f}s")
    print(f"   Response: {out.get('response', '').strip()!r}")
    print()
    if elapsed > 25:
        print(f"   >>> DIAGNOSIS: it works, but took {elapsed:.0f}s — longer than")
        print(f"       the narrator's 25s timeout. That is why it fell back.")
        print(f"       Fix: raise TIMEOUT_SECONDS, or use a smaller model.")
    else:
        print("   >>> Generation is fast enough. If the narrator still falls")
        print("       back, the model NAME is the mismatch (see step 2).")
except urllib.error.HTTPError as e:
    body = e.read().decode()[:300]
    print(f"   HTTP {e.code}: {body}")
    print(f"   >>> DIAGNOSIS: model '{target}' rejected by the server.")
except Exception as e:
    elapsed = time.time() - start
    print(f"   FAILED after {elapsed:.1f}s — {type(e).__name__}: {e}")

print()
print("=" * 60)
print("4. Second call (model now warm — this is real steady-state speed)")
print("=" * 60)
start = time.time()
try:
    with urllib.request.urlopen(
        urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        ),
        timeout=180,
    ) as r:
        out = json.loads(r.read())
    print(f"   {time.time() - start:.1f}s (warm)")
except Exception as e:
    print(f"   {type(e).__name__}: {e}")
