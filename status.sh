#!/bin/bash
echo "[+] Checking vLLM endpoint availability..."
curl -s http://127.0.0.1:8000/v1/models -H "Authorization: Bearer local-dummy-key" | grep -q "Qwen2.5-72B-Instruct-AWQ" && echo "[✔] vLLM server is UP and serving Qwen." || echo "[!] vLLM server is DOWN or starting up."