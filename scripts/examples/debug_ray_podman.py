"""GPU node: does podman-hpc run hang when called from a Ray actor (like the agent loop)?"""
import asyncio, os, ray

IMG = "tb1/hello-world:d28711d"

async def start_file(name):
    import tempfile
    with tempfile.TemporaryFile() as fh:
        p = await asyncio.create_subprocess_exec(
            "podman-hpc", "run", "-d", "--name", name, "--entrypoint=", "--pull=never", "--stop-timeout=0", IMG, "sleep", "infinity",
            stdin=asyncio.subprocess.DEVNULL, stdout=fh, stderr=fh, start_new_session=True)
        try:
            await asyncio.wait_for(p.wait(), 60)
        except asyncio.TimeoutError:
            return f"{name}: HUNG >60s (file)"
        fh.seek(0)
        return f"{name}: rc={p.returncode} (file) {fh.read().decode()[-80:].strip()}"

async def start(name):
    p = await asyncio.create_subprocess_exec(
        "podman-hpc", "run", "-d", "--name", name, "--entrypoint=", "--pull=never", "--stop-timeout=0", IMG, "sleep", "infinity",
        stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, start_new_session=True)
    try:
        out, _ = await asyncio.wait_for(p.communicate(), 60)
        return f"{name}: rc={p.returncode} {out.decode()[-200:].strip()}"
    except asyncio.TimeoutError:
        p.kill() if p.returncode is None else None
        return f"{name}: HUNG >60s"

@ray.remote(num_cpus=1)
class A:
    def env(self):
        return {k: v for k, v in os.environ.items() if any(s in k for s in ("XDG", "TMP", "HOME", "CUDA", "NVIDIA", "PODMAN", "CONTAINER", "RAY_", "PATH", "SLURM_JOB_ID", "USER"))}
    async def gofile(self, names):
        return await asyncio.gather(*(start_file(n) for n in names))
    async def go(self, names):
        return await asyncio.gather(*(start(n) for n in names))

ray.init(num_cpus=8, include_dashboard=False)
a = A.remote()
env = ray.get(a.env.remote())
for k in ("XDG_RUNTIME_DIR", "TMPDIR", "HOME"):
    print(f"SHELL {k}={os.environ.get(k)}")
print("\n".join(ray.get(a.gofile.remote(["ray-2", "ray-3", "ray-4", "ray-5"]))))
print("\n".join(ray.get(a.go.remote(["ray-1"]))))
