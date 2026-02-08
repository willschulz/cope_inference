# Testing CoPE from a Client Machine

This document describes **how to test the CoPE inference API from a client machine** (e.g. `datascience.manx-celsius.ts.net`). It is intended to be copy‑paste friendly and suitable for quick sanity checks as well as basic load testing.

---

## Assumptions

* The CoPE service VM is running and reachable over **Tailscale**
* The service is listening on **port 8000**
* You are running commands from a client on the same tailnet

Service reference (example):

* Tailscale IP: `100.97.88.105`
* Hostname (MagicDNS): `coper-v1.manx-celsius.ts.net`
* Base URL: `http://100.97.88.105:8000`

---

## 1. Connectivity & Health Check

First verify that the API is reachable and that the model has loaded.

```bash
curl -sS http://100.97.88.105:8000/health
```

Expected fields in the response:

* `status: "healthy"`
* `model_loaded: true`
* `gpu_available: true`
* `gpu_name: "NVIDIA L4"`

If this fails:

* Check Tailscale connectivity (`tailscale status`)
* Ensure the service is running on the VM (`systemctl status cope-vllm`)

---

## 2. Single‑Item Classification (`/label`)

Use this endpoint for quick smoke tests.

### Example policy

```text
# Criteria

## Definition of Labels

### (P): Political

#### Includes
- Political actors, parties, elections, public policy

#### Excludes
- Personal, non‑political discussion
```

### curl test

```bash
curl -X POST http://100.97.88.105:8000/label \
  -H "Content-Type: application/json" \
  -d '{
    "policy": "# Criteria\n\n## Definition of Labels\n\n### (P): Political\n\n#### Includes\n- Political actors, parties, elections, public policy\n\n#### Excludes\n- Personal, non-political discussion",
    "content": "The governor announced a new housing bill today."
  }'
```

You should receive a JSON response containing:

* `label` (0 or 1)
* `confidence`
* `prob_positive`

---

## 3. Batch Classification (`/batch`) — Recommended

This endpoint should be used for any real workload.

### Minimal batch example

```bash
curl -X POST http://100.97.88.105:8000/batch \
  -H "Content-Type: application/json" \
  -d '{
    "policy": "# Criteria\n\n## Definition of Labels\n\n### (P): Political\n\n#### Includes\n- Political actors, parties, elections, public policy\n\n#### Excludes\n- Personal, non-political discussion",
    "items": [
      {"id": "1", "content": "I baked bread today."},
      {"id": "2", "content": "Congress passed a budget resolution."}
    ]
  }'
```

Expected output:

* `processed: 2`
* `results` array with preserved IDs

---

## 4. Python Client Example

```python
import requests

API_URL = "http://100.97.88.105:8000"

POLICY = """
# Criteria

## Definition of Labels

### (P): Political

#### Includes
- Political actors, parties, elections, public policy

#### Excludes
- Personal, non-political discussion
"""

items = [
    {"id": "a", "content": "The president gave a speech."},
    {"id": "b", "content": "I went hiking yesterday."},
]

resp = requests.post(
    f"{API_URL}/batch",
    json={"policy": POLICY, "items": items},
    timeout=120,
)
resp.raise_for_status()

result = resp.json()
print(result)
```

---

## 5. Interpreting Results

Each item returns:

* `label`: 0 or 1 (or `-1` if skipped)
* `confidence`: probability of the chosen label
* `prob_positive`: probability of label=1
* `error` (optional): e.g. prompt too long

Skipped items (`label == -1`) should be filtered and handled separately.

---

## 6. Common Failure Modes

| Symptom            | Likely Cause                  | Fix                                         |
| ------------------ | ----------------------------- | ------------------------------------------- |
| Connection refused | Service not running           | `systemctl start cope-vllm`                 |
| Timeout            | Tailscale routing or overload | Check `tailscale status`, reduce batch size |
| 503                | Model still loading           | Wait ~2–3 minutes after boot                |
| Many skipped items | Prompt too long               | Shorten policy or content                   |

---

## 7. Notes for Large Runs

* Use batches of **100–500 items**
* Use **2–4 concurrent batch requests** for throughput
* Checkpoint progress every ~10k items
* Expect ~100–120 items/sec on an L4 GPU

---

## 8. Hostname vs IP

If MagicDNS is enabled, you may use:

```text
http://coper-v1.manx-celsius.ts.net:8000
```

Otherwise, prefer the Tailscale IP for reliability.

---

End of document.
