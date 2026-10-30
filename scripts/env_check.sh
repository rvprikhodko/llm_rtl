#!/usr/bin/env bash
# Training environment check for Linux: native, WSL2 or inside a Docker container.
# Read-only: changes nothing on the system.
#
# Usage:
#   bash env_check.sh > env_report_linux.txt 2>&1

echo "== OS";      head -3 /etc/os-release; uname -r
if grep -qi microsoft /proc/version 2>/dev/null; then
    echo "Running under WSL"
    # The GPU driver comes from Windows; nvidia-smi and libcuda live here
    ls /usr/lib/wsl/lib 2>/dev/null | grep -E "nvidia-smi|libcuda.so" | head -3 \
        || echo "WARNING: /usr/lib/wsl/lib missing: this is WSL1 or the Windows driver lacks WSL support"
fi
if [ -f /.dockerenv ]; then echo "Running inside a Docker container"; fi

echo "== GPU";     nvidia-smi --query-gpu=name,compute_cap,memory.total,memory.used,driver_version,power.limit --format=csv
echo "== CUDA";    nvidia-smi | grep "CUDA Version"; nvcc --version 2>&1 | tail -1
ls -d /usr/local/cuda* 2>/dev/null || echo "no /usr/local/cuda*"
echo "== GCC";     gcc --version 2>&1 | head -1
echo "== Sudo";    if sudo -n true 2>/dev/null; then echo "passwordless sudo: yes"; else echo "passwordless sudo: no (may still work with a password)"; fi
echo "== Python";  python3 --version; python3 -m pip --version 2>&1; python3 -m venv --help >/dev/null 2>&1 && echo "venv: ok" || echo "venv: missing (python3-venv)"
command -v uv conda 2>/dev/null
python3 - <<'EOF' 2>&1
try:
    import torch
    print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device", torch.cuda.get_device_name(0), "cc", torch.cuda.get_device_capability(0), "bf16", torch.cuda.is_bf16_supported())
        print("arch list", torch.cuda.get_arch_list())
        free, total = torch.cuda.mem_get_info(); print(f"vram free/total GB: {free/1e9:.1f}/{total/1e9:.1f}")
except ImportError:
    print("torch not installed")
EOF
echo "== CPU/RAM"; nproc; free -h | head -2
echo "== Disk";    df -h ~ /tmp | uniq
echo "== Tools";   for t in iverilog yosys docker git; do printf "%s: " "$t"; command -v "$t" || echo "not found"; done
echo "== Net"
# -sS keeps curl quiet but prints the reason when a request fails
for url in https://huggingface.co https://pypi.org/simple/ https://files.pythonhosted.org https://download.pytorch.org/whl/; do
    printf "%s: " "$url"; curl -sS -I -m 15 -o /dev/null -w "%{http_code}\n" "$url" 2>&1
done
echo "proxy env: ${https_proxy:-${HTTPS_PROXY:-none}}"
if command -v getent >/dev/null; then getent hosts pypi.org || echo "DNS: pypi.org does not resolve"; fi
echo "(any HTTP code, even 403/404, means the host is reachable; 000 means it is not)"
