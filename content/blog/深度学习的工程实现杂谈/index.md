+++
# Content Identity
title = "深度学习的工程实现杂谈"
description = "记录在深度学习项目搭建和工程实现过程中遇到的具体问题和解决技巧，以字典形式不断更新，为未来项目提供参考和团队成员知识共享的平台。"
summary = "以往的文章大多侧重于理论探讨和学术分析，而这篇文章的重点则是记录我们在实际工程实现过程中遇到的具体问题和解决技巧。本文将不断更新，以字典的形式记录在项目搭建和实现过程中遇到的各种值得记录的问题。"
# Authoring
author = "Chandery"
date = "2024-12-16T10:18:59+08:00"
lastmod = "2024-12-16T10:18:59+08:00"
license = "CC-BY"

# Organization
categories = ["Deep Learning"]
tags = ["技术相关", "深度学习", "工程实现", "Python", "杂谈"]
## Series
# series = ""
# parts = ""
# weight = 1

# Display
featured = false
recommended = true
thumbnail = "https://imgs-chan-1329526870.cos.ap-beijing.myqcloud.com/format%2Cwebp"

# Publication Control
draft = false
layout = "page"

# Advanced SEO
seo_type = "BlogPosting"
seo_image = "https://imgs-chan-1329526870.cos.ap-beijing.myqcloud.com/format%2Cwebp"
twitter_username = ""
+++

# 写在前面

以往的文章大多侧重于理论探讨和学术分析，而这篇文章的重点则是记录我们在实际工程实现过程中遇到的具体问题和解决技巧。通过这些记录，我们希望能够为未来的项目提供有价值的参考材料，使得在面对类似挑战时能够更加高效地找到解决方案。此外，这篇文章也将作为一个实用指南，便于团队成员和其他读者在需要时进行查阅，以汲取经验和启发。

本文将不断更新，以字典的形式记录在项目搭建和实现过程中遇到的各种值得记录的问题。这将为我们提供一个详细的参考，以帮助解决未来可能遇到的类似问题，同时也为团队协作提供一个知识共享的平台。

# 关于工程文件夹中的自定义包文件

> 2024.12.16

正式的工程文件往往是模块化的

```python
ProjectDir/
├── scripts/
│   ├── config/
│   │   └── config.yaml
│   ├── train.py
│   └── ...
├── models/
│   ├── model.py
│   └── ...
├── utils/
│   ├── metrics.py
│   └── ...
└── datautils/
    ├── dataloader.py
    ├── dataprocess.py
    └── ...
```

这种设计方法即增加了项目的可读性，也便于代码维护。

但是单纯的目录设计会让模块的导入出现问题

例如：如果想在./scripts/train.py中导入./models/model.py中的模型类，直接使用 `from models.model import MLP`会报错

这是因为scripts和models是同级目录，import的机制导致无法识别不在当前文件同级及子级目录中的模块（除非import对象是一个包文件）

**因此解决办法就是手动把每个模块目录变成一个包文件**

首先在每个模块目录下增加一个 `__init__.py`文件，这样让python认为这个目录是一个包文件，这使得目录中的模块可以被导入。

```python
ProjectDir/
├── scripts/
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── config.yaml
│   ├── train.py
│   └── ...
├── models/
│   ├── __init__.py
│   ├── model1.py
│   └── ...
├── utils/
│   ├── __init__.py
│   ├── metrics.py
│   └── ...
└── datautils/
    ├── __init__.py
    ├── dataloader.py
    ├── dataprocess.py
    └── ...
```

但是光这样是不够的，现在运行仍然会报错。

查阅资料，发现python import的逻辑为：

> 1. **内置模块**：Python首先检查内置模块（即用C语言编写并在Python解释器中编译的模块，直接内置在Python中的模块，如 `sys`, `os` 等）。
> 2. **当前工作目录**：如果你在一个脚本中执行 `import` 语句，Python会在当前工作目录中查找模块。
> 3. **`PYTHONPATH` 环境变量**：这是一个环境变量，其中可以包含多个目录路径。Python会在这些路径中查找模块。如果存在，`PYTHONPATH` 中的路径将会被添加到 `sys.path` 中。
> 4. **第三方库的目录（`site-packages`）**：通常情况下，Python会查找安装在 `site-packages` 目录中的第三方库。这是通过包管理工具如 `pip` 安装的包所在的默认位置。
> 5. **系统默认路径**：在Python的安装目录中，会有一些默认的路径，通常包括标准库。

显然，工作目录并不在这些路径中。

为了跑起来，我们大可以直接 `export PYTHONPATH=/path/to/your/modules:$PYTHONPATH`，但是这往往是绝对路径，不具有迁移性。

最好的处理办法是在import文件之前在内置模块 `sys.path`中手动加入工作目录

```python
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__)), os.path.pardir))

import {packages}
from models.model import MLP
```

*注意，设置的位置一定要在import自定义包的前面*

**这样的做法既满足了要求又满足迁移性，是完美的解决办法**

