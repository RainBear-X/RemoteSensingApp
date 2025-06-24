#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Top-level package for the Remote Sensing Processor."""

import sys
from pathlib import Path

# 保证无论从哪里执行脚本, 都能正确导入 ``src`` 包
_package_dir = Path(__file__).resolve()
_root_dir = _package_dir.parent
if str(_root_dir) not in sys.path:
    sys.path.insert(0, str(_root_dir))

