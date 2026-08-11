# -*- coding: utf-8 -*-
import numpy as np
from dataclasses import dataclass
from typing import Literal, Optional, Dict, Tuple
from scipy.optimize import curve_fit



# =========================
# 模型：gamma 用 FWHM 表示
# =========================
def lorentzian(x, A, x0, gamma, y0):
    """Lorentzian with gamma as FWHM."""
    g2 = (0.5 * gamma) ** 2
    return y0 + A * g2 / ((x - x0) ** 2 + g2)

def fano(x, A, x0, gamma, q, y0):
    """Fano lineshape with gamma as FWHM. eps = 2*(x-x0)/gamma."""
    eps = 2.0 * (x - x0) / gamma
    return y0 + A * ((q + eps) ** 2) / (1.0 + eps ** 2)


# =========================
# 结果结构
# =========================
@dataclass
class FitResult:
    model: Literal["lorentzian", "fano"]
    x_fit: np.ndarray           # 按 x 排序后的点（与输入 x 等长）
    y_fit: np.ndarray           # 拟合曲线 y(model, x_fit, *p)
    params: Dict[str, float]    # 拟合参数
    perr: Dict[str, float]      # 参数标准差（来自协方差对角线）
    pcov: np.ndarray            # 协方差矩阵
    rss: float                  # 残差平方和
    aic: float                  # AIC
    bic: float                  # BIC


# =========================
# 初值与预处理
# =========================
def _robust_initial_guess(x: np.ndarray, y: np.ndarray, model: str):
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]

    # 背景：两端中位数
    edge_n = max(3, len(y_sorted) // 20)
    y0 = float(np.median(np.r_[y_sorted[:edge_n], y_sorted[-edge_n:]]))

    # 峰位与幅度
    i_max = int(np.argmax(y_sorted))
    x0 = float(x_sorted[i_max])
    A_guess = float(np.max(y_sorted) - y0)

    # FWHM 粗估
    half = y0 + 0.5 * A_guess
    above = np.where(y_sorted >= half)[0]
    if len(above) >= 2:
        gamma = float(x_sorted[above[-1]] - x_sorted[above[0]])
        if gamma <= 0:
            gamma = float((np.max(x_sorted) - np.min(x_sorted)) / 10.0)
    else:
        gamma = float((np.max(x_sorted) - np.min(x_sorted)) / 10.0)

    if model == "lorentzian":
        p0 = [A_guess if A_guess != 0 else 1.0, x0, max(gamma, 1e-6), y0]
        bounds = ([-np.inf, np.min(x_sorted), 1e-9, -np.inf],
                  [ np.inf, np.max(x_sorted),  np.inf,  np.inf])
        names = ["A", "x0", "gamma", "y0"]
    else:  # fano
        q_guess = 1.0 * np.sign(A_guess if A_guess != 0 else 1.0)
        p0 = [A_guess if A_guess != 0 else 1.0, x0, max(gamma, 1e-6), q_guess, y0]
        bounds = ([-np.inf, np.min(x_sorted), 1e-9, -np.inf, -np.inf],
                  [ np.inf, np.max(x_sorted),  np.inf,  np.inf,  np.inf])
        names = ["A", "x0", "gamma", "q", "y0"]

    return x_sorted, y_sorted, p0, bounds, names


def _finalize_result(
    model: str,
    names,
    popt,
    pcov,
    x_fit_resid,    # 用于计算残差/信息准则的 x（通常为拟合时的排序样本）
    y_fit_resid,    # 与 x_fit_resid 对应的 y
    fun,
    x_out=None      # 用于输出的平滑网格；若为 None 则沿用 x_fit_resid
) -> FitResult:
    # 用拟合区的样本计算残差与信息准则（不受平滑网格影响）
    y_model_resid = fun(x_fit_resid, *popt)
    resid = y_fit_resid - y_model_resid
    rss = float(np.sum(resid ** 2))
    n = len(y_fit_resid)
    k = len(popt)
    rss_safe = rss if rss > 0 else 1e-24
    aic = n * np.log(rss_safe / n) + 2 * k
    bic = n * np.log(rss_safe / n) + k * np.log(n)

    # 输出曲线：若提供 x_out，用它生成更平滑的曲线；否则沿用拟合样本
    if x_out is None:
        x_out = x_fit_resid
    y_out = fun(x_out, *popt)

    perr_vals = np.sqrt(np.maximum(np.diag(pcov), 0.0)) if pcov is not None else np.full(k, np.nan)
    params = {k_: float(v) for k_, v in zip(names, popt)}
    perr = {k_: float(e) for k_, e in zip(names, perr_vals)}

    return FitResult(
        model=model,
        x_fit=x_out,
        y_fit=y_out,
        params=params,
        perr=perr,
        pcov=pcov,
        rss=rss,
        aic=float(aic),
        bic=float(bic),
    )



# =========================
# 核心拟合 API
# =========================
def fit_lineshape(
    x: np.ndarray,
    y: np.ndarray,
    model: Literal["lorentzian", "fano"] = "lorentzian",
    sigma: Optional[np.ndarray] = None,
    absolute_sigma: bool = False,
    maxfev: int = 100000,
    fit_range: Optional[Tuple[float, float]] = None,
    # ---- 新增：控制输出平滑度 ----
    output_samples: Optional[int] = None,
    output_range: Optional[Tuple[float, float]] = None,
) -> FitResult:
    """
    对给定 (x, y) 进行 Lorentzian 或 Fano 拟合。
    参数
    ----
    fit_range : (xmin, xmax)，仅在该区间内取点进行拟合。
    output_samples : 若提供，则在指定范围内用等距网格输出更平滑的曲线。
    output_range   : 与 output_samples 搭配。未提供时，若有 fit_range 则用它，否则用 (min(x), max(x)).
    """
    # 1) 选择拟合数据
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    if fit_range is not None:
        xmin, xmax = fit_range
        mask = (x >= xmin) & (x <= xmax)
        if not np.any(mask):
            raise ValueError("No data points found within fit_range")
        x_used, y_used = x[mask], y[mask]
        if sigma is not None:
            sigma = np.asarray(sigma).ravel()[mask]
    else:
        x_used, y_used = x, y

    # 2) 准备模型与初值
    model = model.lower()
    assert model in {"lorentzian", "fano"}
    x_sorted, y_sorted, p0, bounds, names = _robust_initial_guess(x_used, y_used, model)
    fun = lorentzian if model == "lorentzian" else fano

    # 3) 拟合
    popt, pcov = curve_fit(
        fun,
        x_sorted,
        y_sorted,
        p0=p0,
        bounds=bounds,
        sigma=sigma,
        absolute_sigma=absolute_sigma,
        maxfev=maxfev,
    )

    # 4) 生成输出网格（平滑）
    x_out = None
    if output_samples is not None:
        if output_range is None:
            # 采纳你的建议：优先用 fit_range，否则用 (min(x), max(x))
            xmin_out, xmax_out = fit_range if fit_range is not None else (float(np.min(x)), float(np.max(x)))
        else:
            xmin_out, xmax_out = float(output_range[0]), float(output_range[1])
            if xmin_out > xmax_out:
                xmin_out, xmax_out = xmax_out, xmin_out
        x_out = np.linspace(xmin_out, xmax_out, int(output_samples))

    # 5) 打包结果（RSS/AIC/BIC 基于拟合样本；输出曲线基于 x_out）
    return _finalize_result(model, names, popt, pcov, x_sorted, y_sorted, fun, x_out=x_out)



def fit_both_and_compare(
    x: np.ndarray,
    y: np.ndarray,
    criterion: Literal["rss", "aic", "bic"] = "aic",
    fit_range: Optional[Tuple[float, float]] = None,
    # ---- 新增：平滑输出参数，传给子函数 ----
    output_samples: Optional[int] = None,
    output_range: Optional[Tuple[float, float]] = None,
) -> Tuple[FitResult, FitResult, FitResult]:
    """同时拟合 Lorentzian 与 Fano，在 fit_range 内比较两者。"""
    res_l = fit_lineshape(
        x, y, model="lorentzian", fit_range=fit_range,
        output_samples=output_samples, output_range=output_range
    )
    res_f = fit_lineshape(
        x, y, model="fano", fit_range=fit_range,
        output_samples=output_samples, output_range=output_range
    )

    def score(res: FitResult):
        if criterion == "rss":
            return res.rss
        elif criterion == "bic":
            return res.bic
        else:
            return res.aic

    best = res_l if score(res_l) <= score(res_f) else res_f
    return res_l, res_f, best
