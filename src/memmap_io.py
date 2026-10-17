"""Channel-by-channel NumPy memmap I/O to limit peak RAM."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import numpy as np
from numpy.lib.format import dtype_to_descr, write_array_header_1_0

CancelCheck = Callable[[], None] | None


def open_writable_memmap(path: Path, shape: tuple[int, ...], dtype: np.dtype) -> np.memmap:
    """Create a writable ``.npy`` memmap without zero-filling the payload.

    Recordings of a few gigabytes otherwise pay a full extra write just to
    extend the file. The file is sparse when the OS allows it, so unread
    regions stay unallocated (HP/LP caches grow only as channels are filled).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    dtype = np.dtype(dtype)
    shape = tuple(int(dim) for dim in shape)
    nbytes = int(np.prod(shape, dtype=np.int64)) * int(dtype.itemsize)
    with open(path, "wb") as handle:
        write_array_header_1_0(
            handle,
            {
                "descr": dtype_to_descr(dtype),
                "fortran_order": False,
                "shape": shape,
            },
        )
        offset = int(handle.tell())
    _resize_file(path, offset + nbytes)
    return np.memmap(path, dtype=dtype, mode="r+", offset=offset, shape=shape, order="C")


def close_memmap(array: np.ndarray | None) -> None:
    """Release the OS mapping so Windows can delete or reopen the file."""
    if array is None:
        return
    mmap = getattr(array, "_mmap", None)
    if mmap is None:
        return
    try:
        mmap.close()
    except Exception:
        pass


def _resize_file(path: Path, size: int) -> None:
    """Extend ``path`` to ``size`` bytes, sparse when the filesystem allows."""
    if size < 0:
        raise ValueError("file size must be >= 0")
    if sys.platform == "win32":
        try:
            _resize_sparse_windows(path, size)
            return
        except OSError:
            pass
    with open(path, "r+b") as handle:
        if size == 0:
            handle.truncate(0)
            return
        handle.seek(size - 1)
        handle.write(b"\0")


def _resize_sparse_windows(path: Path, size: int) -> None:
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    generic_read = 0x80000000
    generic_write = 0x40000000
    share = 0x1 | 0x2 | 0x4
    open_existing = 3
    file_attribute_normal = 0x80
    fsctl_set_sparse = 0x000900C4

    kernel32.CreateFileW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    kernel32.CreateFileW.restype = wintypes.HANDLE
    kernel32.DeviceIoControl.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
        ctypes.c_void_p,
    ]
    kernel32.DeviceIoControl.restype = wintypes.BOOL
    kernel32.SetFilePointerEx.argtypes = [
        wintypes.HANDLE,
        ctypes.c_longlong,
        ctypes.POINTER(ctypes.c_longlong),
        wintypes.DWORD,
    ]
    kernel32.SetFilePointerEx.restype = wintypes.BOOL
    kernel32.SetEndOfFile.argtypes = [wintypes.HANDLE]
    kernel32.SetEndOfFile.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    handle = kernel32.CreateFileW(
        str(path),
        generic_read | generic_write,
        share,
        None,
        open_existing,
        file_attribute_normal,
        None,
    )
    invalid = wintypes.HANDLE(-1).value
    if handle == invalid or handle is None:
        raise OSError(ctypes.get_last_error(), f"CreateFileW failed for {path}")
    try:
        returned = wintypes.DWORD(0)
        if not kernel32.DeviceIoControl(
            handle,
            fsctl_set_sparse,
            None,
            0,
            None,
            0,
            ctypes.byref(returned),
            None,
        ):
            raise OSError(ctypes.get_last_error(), "FSCTL_SET_SPARSE failed")
        position = ctypes.c_longlong()
        if not kernel32.SetFilePointerEx(handle, ctypes.c_longlong(int(size)), ctypes.byref(position), 0):
            raise OSError(ctypes.get_last_error(), "SetFilePointerEx failed")
        if not kernel32.SetEndOfFile(handle):
            raise OSError(ctypes.get_last_error(), "SetEndOfFile failed")
    finally:
        kernel32.CloseHandle(handle)


def write_2d_channelwise(
    source: np.ndarray,
    path: Path,
    *,
    dtype: np.dtype = np.dtype(np.float32),
    cancel_check: CancelCheck = None,
    progress_every: int = 0,
    on_channel: Callable[[int, int], None] | None = None,
) -> Path:
    """Copy a 2D array to disk one channel at a time (shape preserved)."""
    arr = np.asarray(source)
    if arr.ndim != 2:
        raise ValueError("write_2d_channelwise expects a 2D array.")
    n_ch, n_samp = int(arr.shape[0]), int(arr.shape[1])
    mm = open_writable_memmap(path, (n_ch, n_samp), dtype)
    try:
        for ch in range(n_ch):
            if cancel_check is not None:
                cancel_check()
            mm[ch] = np.asarray(arr[ch], dtype=dtype)
            if on_channel is not None:
                on_channel(ch, n_ch)
            elif progress_every > 0 and (ch + 1) % progress_every == 0:
                print(f"  memmap {path.name}: channel {ch + 1}/{n_ch}")
        mm.flush()
    finally:
        del mm
    return path


def load_readonly_memmap(path: Path) -> np.memmap:
    return np.load(path, mmap_mode="r")
