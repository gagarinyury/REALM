#!/usr/bin/env python3
"""Управление подом RunPod для стокового прогона REALM.

Ключ берётся из переменной окружения RUNPOD_API_KEY либо из dotenv-файла, путь к
которому задаёт RUNPOD_ENV_FILE (по умолчанию ./.env.local), — чтобы не плодить копии
секрета. Скрипт доработан под кастомный образ: у stanfordvl/omnigibson нет sshd,
поэтому его ставит стартовая команда, а ключ приезжает в контейнер переменной
PUBLIC_KEY из настроек аккаунта.

    python3 runpod/pod.py create [--gpu a6000] [--disk 150]
    python3 runpod/pod.py status
    python3 runpod/pod.py wait
    python3 runpod/pod.py kill     # ОБЯЗАТЕЛЬНО после работы — иначе капает оплата
"""
import json, os, sys, time, urllib.request, argparse

API = "https://api.runpod.io/graphql"
KEYFILE = os.path.expanduser(os.environ.get("RUNPOD_ENV_FILE", ".env.local"))
HERE = os.path.dirname(os.path.abspath(__file__))

GPUS = {
    "a6000": "NVIDIA RTX A6000",     # 48 ГБ, Ampere, есть RT-ядра — в списке Omniverse
    "a5000": "NVIDIA RTX A5000",     # 24 ГБ, $0.16/час — минимум под сервер π0-FAST (~13 ГБ VRAM)
    "a4500": "NVIDIA RTX A4500",
    "a40": "NVIDIA A40",
    "4090": "NVIDIA GeForce RTX 4090",
    "l40s": "NVIDIA L40S",
    "6000ada": "NVIDIA RTX 6000 Ada Generation",
}

# Образ авторов REALM: .docker/realm.Dockerfile строка 1. Это OmniGibson 1.1.1 —
# та версия, на которой получен опубликованный baseline 0.61.
IMAGES = {
    "stock": "stanfordvl/omnigibson:1.1.1",        # авторский стенд, baseline 0.61
    "391": "stanfordvl/behavior:3.9.1",            # версия, на которую портировали мы
    # Только сервер политики: симуляция MolmoSpaces идёт на evox2 (CPU), поэтому
    # тяжёлый образ OmniGibson здесь не нужен. Тег — из штатного шаблона RunPod
    # runpod-torch-v280, то есть заведомо существующий.
    "torch": "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404",
}
IMAGE = IMAGES["stock"]

# ENTRYPOINT образа — micromamba run -n omnigibson, поэтому команда выполнится внутри env.
START = (
    "bash -c 'apt-get update -qq && apt-get install -y -qq openssh-server rsync curl && "
    "mkdir -p /run/sshd /root/.ssh && echo \"$PUBLIC_KEY\" > /root/.ssh/authorized_keys && "
    "chmod 700 /root/.ssh && chmod 600 /root/.ssh/authorized_keys && "
    "sed -i \"s/^#*PermitRootLogin.*/PermitRootLogin prohibit-password/\" /etc/ssh/sshd_config && "
    "/usr/sbin/sshd -D -p 22'"
)


def key():
    if os.environ.get("RUNPOD_API_KEY"):
        return os.environ["RUNPOD_API_KEY"]
    if os.path.exists(KEYFILE):
        for line in open(KEYFILE):
            if line.startswith("RUNPOD_API_KEY="):
                return line.split("=", 1)[1].strip()
    sys.exit(f"нет RUNPOD_API_KEY: ни в окружении, ни в {KEYFILE}")


def gql(q):
    req = urllib.request.Request(
        f"{API}?api_key={key()}",
        data=json.dumps({"query": q}).encode(),
        headers={"Content-Type": "application/json", "User-Agent": "curl/8.4.0"},
    )
    d = json.load(urllib.request.urlopen(req))
    if "errors" in d:
        sys.exit("RunPod: " + json.dumps(d["errors"])[:600])
    return d["data"]


def create(gpu, disk, container_disk, image=IMAGE, min_vcpu=8, min_ram=32):
    gid = GPUS[gpu]
    args = json.dumps(START)[1:-1]     # экранирование под строку в GraphQL
    q = f'''mutation {{ podFindAndDeployOnDemand(input: {{
        cloudType: ALL, gpuCount: 1,
        volumeInGb: {disk}, containerDiskInGb: {container_disk},
        minVcpuCount: {min_vcpu}, minMemoryInGb: {min_ram},
        gpuTypeId: "{gid}", name: "realm-stock",
        imageName: "{image}",
        dockerArgs: "{args}", ports: "22/tcp", volumeMountPath: "/workspace",
        startSsh: true
    }}) {{ id imageName machineId costPerHr }} }}'''
    p = gql(q)["podFindAndDeployOnDemand"]
    if not p:
        sys.exit(f"не удалось получить {gpu} — нет свободных машин, попробуй другую карту")
    print(f"под создан: {p['id']}  ${p['costPerHr']}/час")
    open(os.path.join(HERE, ".pod_id"), "w").write(p["id"])
    return p["id"]


def pod_id():
    f = os.path.join(HERE, ".pod_id")
    if not os.path.exists(f):
        sys.exit("нет активного пода (.pod_id отсутствует)")
    return open(f).read().strip()


def status(quiet=False):
    pid = pod_id()
    d = gql(f'query {{ pod(input:{{podId:"{pid}"}}) {{ id name costPerHr desiredStatus '
            f'runtime {{ uptimeInSeconds ports {{ ip publicPort privatePort type }} }} }} }}')["pod"]
    if not quiet:
        rt = d.get("runtime") or {}
        up = rt.get("uptimeInSeconds")
        print(f"под {d['id']}  статус={d['desiredStatus']}  ${d['costPerHr']}/час"
              + (f"  аптайм={up//60} мин  списано≈${d['costPerHr']*up/3600:.2f}" if up else "  (поднимается)"))
        for p in (rt.get("ports") or []):
            if p["privatePort"] == 22:
                print(f"  SSH: ssh root@{p['ip']} -p {p['publicPort']}")
    return d


def wait_ssh(timeout=1800):
    """Ждёт SSH и печатает, сколько времени ушло — это и есть замер скачивания образа."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        d = status(quiet=True)
        for p in ((d.get("runtime") or {}).get("ports") or []):
            if p["privatePort"] == 22 and p.get("ip"):
                el = int(time.time() - t0)
                print(f"SSH готов за {el // 60} мин {el % 60} с: root@{p['ip']} -p {p['publicPort']}")
                open(os.path.join(HERE, ".pod_ssh"), "w").write(f"{p['ip']} {p['publicPort']}\n")
                return p["ip"], p["publicPort"]
        print(f"  ... {int(time.time() - t0)} с, статус={d['desiredStatus']}", flush=True)
        time.sleep(15)
    sys.exit("под не поднял SSH за отведённое время")


def kill():
    pid = pod_id()
    gql(f'mutation {{ podTerminate(input:{{podId:"{pid}"}}) }}')
    os.remove(os.path.join(HERE, ".pod_id"))
    print(f"под {pid} уничтожен, оплата остановлена")


def balance():
    m = gql("query { myself { clientBalance currentSpendPerHr } }")["myself"]
    print(f"баланс ${m['clientBalance']}  расход ${m['currentSpendPerHr']}/час")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["create", "status", "wait", "kill", "balance"])
    ap.add_argument("--gpu", default="a6000", choices=list(GPUS))
    ap.add_argument("--disk", type=int, default=150)
    ap.add_argument("--container-disk", type=int, default=60)
    ap.add_argument("--image", default="stock", choices=list(IMAGES))
    ap.add_argument("--min-vcpu", type=int, default=8)
    ap.add_argument("--min-ram", type=int, default=32)
    a = ap.parse_args()
    if a.cmd == "create":
        create(a.gpu, a.disk, a.container_disk, IMAGES[a.image], a.min_vcpu, a.min_ram); wait_ssh()
    elif a.cmd == "status":  status()
    elif a.cmd == "wait":    wait_ssh()
    elif a.cmd == "kill":    kill()
    elif a.cmd == "balance": balance()
