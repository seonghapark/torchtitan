"""Aurora/Intel XPU 환경을 위한 MPI + mpi4py 분산 초기화 유틸.

이 파일은 torchtitan.models.gemma.train 에 있는 분산 부트스트랩 핵심만
독립적으로 꺼내 정리한 모듈이다.

주요 목적
- torchrun 없이 mpiexec/PALS/PMI로 실행할 때 RANK/WORLD_SIZE/LOCAL_RANK 보정
- 모든 rank가 동일한 MASTER_ADDR 을 공유하도록 mpi4py 브로드캐스트
- oneCCL 초기화에 필요한 CCL_LOCAL_* 환경변수 보정
- XPU 백엔드 자동 선택 (xccl 우선, oneCCL ccl 차선)

빠른 사용 예시
1) 환경변수만 먼저 보정하고 싶은 경우
   bootstrap_dist_env()

2) 환경변수 보정 + 백엔드 선택 + process group 초기화까지 한 번에
   device, rank, world_size = setup_distributed()

3) 이미 torch.distributed.init_process_group 를 직접 호출하는 코드가 있는 경우
   bootstrap_dist_env()
   backend, device = pick_backend_and_device()
   dist.init_process_group(backend=backend)
"""

# pyright: reportMissingImports=false

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.distributed as dist

# torchrun 과 MPI 계열 런처의 환경변수 이름 차이를 흡수하기 위한 lookup table
_RANK_ENV_KEYS = ("RANK", "PMI_RANK", "PALS_RANKID", "OMPI_COMM_WORLD_RANK")
_SIZE_ENV_KEYS = (
    "WORLD_SIZE",
    "PMI_SIZE",
    "PALS_NRANKS",
    "OMPI_COMM_WORLD_SIZE",
)
_LOCAL_RANK_ENV_KEYS = (
    "LOCAL_RANK",
    "PALS_LOCAL_RANKID",
    "PMI_LOCAL_RANK",
    "MPI_LOCALRANKID",
    "OMPI_COMM_WORLD_LOCAL_RANK",
)


def first_env(keys: tuple[str, ...]) -> Optional[str]:
    """keys 순서대로 환경변수를 조회해 첫 번째 유효 값을 반환한다.

    하는 일
    - 런처마다 환경변수 이름이 다를 때 우선순위를 둔 통합 조회

    사용법
    - rank 후보 목록에서 찾기: first_env(_RANK_ENV_KEYS)
    - world size 후보 목록에서 찾기: first_env(_SIZE_ENV_KEYS)
    """
    for key in keys:
        value = os.environ.get(key)
        if value is not None and value != "":
            return value
    return None


def set_master_addr_via_mpi(comm=None) -> None:
    """rank 0 hostname 을 MPI 브로드캐스트로 공유해 MASTER_ADDR 를 설정한다.

    하는 일
    - 멀티노드 mpiexec 실행에서 모든 rank 가 같은 MASTER_ADDR 을 쓰게 보장
    - mpi4py 가 없으면 단일 노드 fallback 으로 로컬 hostname 사용

    사용법
    - 보통 직접 호출하지 말고 bootstrap_dist_env 에 맡기면 된다.
    - 이미 MPI.COMM_WORLD 를 갖고 있으면 comm 인자로 전달 가능.
    """
    if comm is None:
        try:
            from mpi4py import MPI  # type: ignore
        except ImportError:
            import socket

            os.environ.setdefault("MASTER_ADDR", socket.gethostname())
            return
        comm = MPI.COMM_WORLD
    else:
        from mpi4py import MPI  # type: ignore  # noqa: F401

    hostname = None
    if comm.Get_rank() == 0:
        import socket

        hostname = socket.gethostname()

    master = comm.bcast(hostname, root=0)
    os.environ["MASTER_ADDR"] = str(master)


def bootstrap_ccl_env(mpi_comm=None) -> None:
    """oneCCL fallback hang 예방을 위해 CCL_LOCAL_* 환경변수를 보정한다.

    하는 일
    - CCL_LOCAL_RANK, CCL_LOCAL_IDX 를 LOCAL_RANK 기준으로 설정
    - CCL_LOCAL_SIZE, CCL_LOCAL_COUNT 를 런처 env 또는 mpi4py 로 계산해 설정

    왜 필요한가
    - Aurora 에서 CCL_LOCAL_* 가 비어 있으면 oneCCL 이 ATL probing 으로
      내려가며 초기화 지연/행이 발생할 수 있다.

    사용법
    - 보통 bootstrap_dist_env 내부에서 자동 호출되므로 별도 호출 불필요.
    """
    local_rank = os.environ["LOCAL_RANK"]
    os.environ.setdefault("CCL_LOCAL_RANK", local_rank)
    os.environ.setdefault("CCL_LOCAL_IDX", local_rank)

    local_size = (
        os.environ.get("PALS_LOCAL_SIZE")
        or os.environ.get("MPI_LOCALNRANKS")
        or os.environ.get("OMPI_COMM_WORLD_LOCAL_SIZE")
    )

    if local_size is None and mpi_comm is not None:
        try:
            from mpi4py import MPI  # type: ignore

            node_comm = mpi_comm.Split_type(MPI.COMM_TYPE_SHARED)
            local_size = str(node_comm.Get_size())
        except Exception:
            pass

    if local_size:
        os.environ.setdefault("CCL_LOCAL_SIZE", local_size)
        os.environ.setdefault("CCL_LOCAL_COUNT", local_size)


def bootstrap_dist_env() -> None:
    """RANK/WORLD_SIZE/LOCAL_RANK/MASTER_ADDR/MASTER_PORT 를 보정한다.

    하는 일
    1) torchrun 이 이미 넣은 RANK/WORLD_SIZE 가 있으면 그대로 사용
    2) 없으면 mpi4py COMM_WORLD 에서 rank/size 획득
    3) 그것도 없으면 PMI/PALS/OMPI 환경변수 사용
    4) 최종 fallback 은 단일 프로세스(0/1)
    5) MASTER_ADDR, MASTER_PORT, oneCCL env 보정

    사용법
    - dist.init_process_group 전에 반드시 한 번 호출한다.
    """
    torchrun_ok = "RANK" in os.environ and "WORLD_SIZE" in os.environ

    rank_val: Optional[str] = None
    size_val: Optional[str] = None
    mpi_comm = None

    if not torchrun_ok:
        try:
            from mpi4py import MPI  # type: ignore

            mpi_comm = MPI.COMM_WORLD
            rank_val = str(mpi_comm.Get_rank())
            size_val = str(mpi_comm.Get_size())
        except ImportError:
            pass

        if rank_val is None:
            rank_val = first_env(_RANK_ENV_KEYS)
        if size_val is None:
            size_val = first_env(_SIZE_ENV_KEYS)

        if rank_val is None or size_val is None:
            rank_val = "0"
            size_val = "1"

        os.environ["RANK"] = rank_val
        os.environ["WORLD_SIZE"] = size_val

    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = first_env(_LOCAL_RANK_ENV_KEYS) or "0"

    if "MASTER_ADDR" not in os.environ:
        set_master_addr_via_mpi(mpi_comm)

    os.environ.setdefault("MASTER_PORT", "29500")

    bootstrap_ccl_env(mpi_comm)


def pick_xpu_backend() -> str:
    """Intel XPU 에서 사용할 distributed backend 이름을 선택한다.

    선택 우선순위
    1) TORCH_XPU_BACKEND 환경변수 (수동 override)
    2) torch.distributed.is_backend_available("xccl") 가능 시 xccl
    3) oneccl_bindings_for_pytorch import 가능 시 ccl

    사용법
    - XPU 환경에서 backend 문자열이 필요할 때 호출
    - 일반적으로 pick_backend_and_device 에서 자동 사용됨
    """
    override = os.environ.get("TORCH_XPU_BACKEND")
    if override:
        return override

    try:
        from torch.distributed import is_backend_available  # type: ignore

        if is_backend_available("xccl"):
            return "xccl"
    except (ImportError, AttributeError):
        pass

    try:
        import oneccl_bindings_for_pytorch  # type: ignore  # noqa: F401

        return "ccl"
    except ImportError:
        pass

    raise RuntimeError(
        "XPU 사용 가능하지만 등록된 distributed backend 를 찾지 못했습니다.\n"
        "해결 방법:\n"
        "  - xccl 지원 PyTorch XPU 빌드 사용, 또는\n"
        "  - oneccl_bind_pt 설치, 또는\n"
        "  - TORCH_XPU_BACKEND 환경변수로 강제 지정"
    )


def pick_backend_and_device() -> tuple[str, torch.device]:
    """현재 노드에서 backend 와 rank-local device 를 함께 선택한다.

    선택 순서
    - XPU 우선 (Aurora/Intel)
    - 그다음 CUDA
    - 마지막 CPU(gloo)

    사용법
    - backend, device = pick_backend_and_device()
    - dist.init_process_group(backend=backend)
    """
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.set_device(local_rank)
        backend = pick_xpu_backend()
        return backend, torch.device(f"xpu:{local_rank}")

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        return "nccl", torch.device(f"cuda:{local_rank}")

    return "gloo", torch.device("cpu")


def setup_distributed() -> tuple[torch.device, int, int]:
    """분산 실행에 필요한 환경/백엔드/PG 초기화를 한 번에 수행한다.

    하는 일
    - bootstrap_dist_env 호출
    - backend/device 선택
    - 필요 시 dist.init_process_group 호출
    - 최종 device, rank, world_size 반환

    사용법
    - device, rank, world_size = setup_distributed()
    - 이후 모델/데이터 로더 구성 시 rank/world_size 사용
    """
    bootstrap_dist_env()
    backend, device = pick_backend_and_device()

    if not dist.is_initialized():
        dist.init_process_group(backend=backend)

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    return device, rank, world_size


def is_rank0() -> bool:
    """현재 프로세스가 rank 0 인지 확인한다.

    사용법
    - rank 0 에서만 로그/체크포인트 저장을 수행할 때 사용
    """
    return (not dist.is_initialized()) or dist.get_rank() == 0


__all__ = [
    "bootstrap_dist_env",
    "bootstrap_ccl_env",
    "first_env",
    "is_rank0",
    "pick_backend_and_device",
    "pick_xpu_backend",
    "set_master_addr_via_mpi",
    "setup_distributed",
]
