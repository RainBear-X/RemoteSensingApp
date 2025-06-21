# -*- coding: utf-8 -*-
"""
Band Math & Spectral Indices
----------------------------

Created : 2025-06-17
Author  : 张子涵
Version : 0.1.0
"""
from __future__ import annotations
import ast, math
import numpy as np


def ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    return (nir - red) / (nir + red + 1e-12)


def ndwi(nir: np.ndarray, green: np.ndarray) -> np.ndarray:
    return (green - nir) / (green + nir + 1e-12)


def _safe_eval(expr: str, **bands: np.ndarray) -> np.ndarray:
    """Safely evaluate a math expression using provided band arrays.

    Only a subset of Python syntax is allowed to avoid security risks.
    """
    allowed = (
        ast.Expression, ast.BinOp, ast.UnaryOp,
        ast.Num, ast.Constant, ast.Name,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod,
        ast.UAdd, ast.USub, ast.Load
    )
    tree = ast.parse(expr, mode="eval")
    for n in ast.walk(tree):
        if not isinstance(n, allowed):
            raise ValueError(f"Unsafe element: {ast.dump(n)}")
    return eval(compile(tree, "<expr>", "eval"),
                {"np": np, "math": math}, bands)


def custom_expression(expr: str, *band_list: np.ndarray) -> np.ndarray:
    bands = {f"B{i+1}": b for i, b in enumerate(band_list)}
    return _safe_eval(expr, **bands)
