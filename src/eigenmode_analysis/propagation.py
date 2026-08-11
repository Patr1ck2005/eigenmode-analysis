import numpy as np

def _ensure_xy_axes(m1, m2, A):
    m1 = np.asarray(m1).ravel()  # x 轴（列数）
    m2 = np.asarray(m2).ravel()  # y 轴（行数）
    A = np.asarray(A)
    if A.shape != (m2.size, m1.size):
        # 尝试按 (ny, nx) 重塑
        try:
            A = A.reshape(m2.size, m1.size)
        except Exception:
            raise ValueError("A 的形状应为 (len(m2), len(m1))")
    # 简单检查均匀采样
    if m1.size>2 and not np.allclose(np.diff(m1), np.diff(m1)[0]):
        raise ValueError("m1 不是均匀采样，简略模块不支持（请先重采样）。")
    if m2.size>2 and not np.allclose(np.diff(m2), np.diff(m2)[0]):
        raise ValueError("m2 不是均匀采样，简略模块不支持（请先重采样）。")
    return m1, m2, A

def angular_spectrum_propagate(
    m1, m2, A0, z, *,
    k0=1.0,
    include_evanescent=True,
    k_scale="normalized_to_k0",
    time_sign=-1,   # 新增：时间因子符号；-1 表示 e^{-iωt}（默认），+1 表示 e^{+iωt}
):
    """
    time_sign: 控制时间因子 e^{±iωt} 的符号。
        -1: 使用 e^{-iωt} 约定（默认），沿 +z 传播用 exp(+i kz z)
        +1: 使用 e^{+iωt} 约定，沿 +z 传播用 exp(-i kz z)
    """
    import numpy as np

    m1, m2, A0 = _ensure_xy_axes(m1, m2, A0)
    ny, nx = A0.shape

    # 频域坐标 -> 物理波数
    if k_scale == "normalized_to_k0":
        kx = k0 * m1
        ky = k0 * m2
    elif k_scale == "raw":
        kx, ky = m1, m2
    else:
        raise ValueError("k_scale 只能是 'normalized_to_k0' 或 'raw'")

    KX, KY = np.meshgrid(kx, ky, indexing='xy')
    kxy2 = KX**2 + KY**2
    kz = np.zeros_like(KX, dtype=np.complex128)
    mask_prop = kxy2 <= (k0**2 + 0.0)

    kz[mask_prop] = np.sqrt(k0**2 - kxy2[mask_prop])
    if include_evanescent:
        kz[~mask_prop] = 1j * np.sqrt(kxy2[~mask_prop] - k0**2)
    else:
        A0 = A0.copy()
        A0[~mask_prop] = 0.0

    # —— 根据时间符号决定传播因子 ——
    if time_sign not in (+1, -1):
        raise ValueError("time_sign 只能取 +1 (e^{+iωt}) 或 -1 (e^{-iωt})")

    Az = A0 * np.exp(-1j * time_sign * kz * z)
    # 说明：
    #  time_sign = -1 → exp(+i kz z)  (e^{-iωt} 约定)
    #  time_sign = +1 → exp(-i kz z)  (e^{+iωt} 约定)

    # IFFT 回到实空间
    Af = np.fft.ifftshift(Az)
    Exy = np.fft.fftshift(np.fft.ifft2(Af, norm='ortho'))

    dkx = np.abs(kx[1] - kx[0]) if nx > 1 else 1.0
    dky = np.abs(ky[1] - ky[0]) if ny > 1 else 1.0
    dx = 2*np.pi / (nx * dkx)
    dy = 2*np.pi / (ny * dky)
    x = (np.arange(nx) - nx//2) * dx
    y = (np.arange(ny) - ny//2) * dy

    return Az, (x, y, Exy)


def angular_spectrum_xz_slice(
    m1, m2, A0, z_list, *, k0=1.0, y0=0.0,
    include_evanescent=True, k_scale="normalized_to_k0", time_sign=-1,
):
    """
    生成 XZ 纵截面（取 y≈y0 的切片）。
    返回: x, z, Exz  (Exz 形状 = [len(z), len(x)]，每一行是该 z 的 y=y0 截面)
    """
    z_arr = np.asarray(z_list).ravel()
    # 先算一次拿到坐标
    _, (x, y, Exy0) = angular_spectrum_propagate(
        m1, m2, A0, z_arr[0], k0=k0,
        include_evanescent=include_evanescent, k_scale=k_scale, time_sign=time_sign
    )
    iy = np.argmin(np.abs(y - y0))
    Exz = np.empty((z_arr.size, x.size), dtype=np.complex128)
    Exz[0, :] = Exy0[iy, :]

    for i in range(1, z_arr.size):
        _, (x, y, Exy) = angular_spectrum_propagate(
            m1, m2, A0, z_arr[i], k0=k0,
            include_evanescent=include_evanescent, k_scale=k_scale, time_sign=time_sign
        )
        Exz[i, :] = Exy[iy, :]

    return x, z_arr, Exz
