+++
# Content Identity
title = "MultiEpochDataloader和Validation优化"
description = "探讨使用MultiEpochDataloader和Lightning Validation策略优化训练过程，解决不同Epoch之间数据准备的时间间隔问题，实现Epoch之间的无缝衔接。"
summary = "不同的Epoch之间的数据准备总是会有间隔，这耗费了不必要的时间。并且搭配使用Lightning时，如果不合理的使用Validation，这个问题会更加显著。本文主要讨论使用MultiEpochDataloader和Lightning Validation策略解决这个问题，实测Epoch之间无缝衔接。"
# Authoring
author = "Chandery"
date = "2025-03-26T08:39:54+08:00"
lastmod = "2025-03-26T08:39:54+08:00"
license = "CC-BY"

# Organization
categories = ["Deep Learning"]
tags = ["技术相关", "深度学习", "PyTorch", "Lightning", "优化"]
## Series
# series = ""
# parts = ""
# weight = 1

# Display
featured = false
recommended = true
thumbnail = "https://imgs-chan-1329526870.cos.ap-beijing.myqcloud.com/1533221234868.jpeg"

# Publication Control
draft = false
layout = "page"

# Advanced SEO
seo_type = "BlogPosting"
seo_image = "https://imgs-chan-1329526870.cos.ap-beijing.myqcloud.com/1533221234868.jpeg"
twitter_username = ""
+++

## Intro

不同的Epoch之间的数据准备总是会有间隔，这耗费了不必要的时间。并且搭配使用Lightning时，如果不合理的使用Validation，这个问题会更加显著。这里主要讨论使用MultiEpochDataloader和Lightning Validation策略解决这个问题，实测Epoch之间无缝衔接。

## MultiEpochDataloader

Torch.utils.data.Dataloader在处理数据的时候，每个epoch都会重置，感知最明显的就是shuffle=True的时候，每次给出的顺序是随机的。但是这个重置过程频繁地在内存，显存之间调度会耗费很多时间。我们可以使用MultiEpochDataloader代替它

```python
class MultiEpochsDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)
```

MultiEpochDataloader在Dataloader的基础上，固定batch_sampler（可以看到_RepeatSampler中固定了sampler）。这样Epoch之间就不需要进行重置，节省了Epoch之间采样器生成的时间。

主要的问题在于distribution：

- 在使用Lightning自动进行分布式训练的时候，会报batch_sampler不具有分布式的能力的错误。这方面解决起来不容易，对底层代码不够熟悉的我暂时难以解决。需要不断精进。



## Validation

Validation往往不是每一轮都要进行，或许每10轮、20轮进行一次。

一般的处理方法是

```python
if current_epoch % interval == 0:
  for batch in val_dataloader:
    # valid
```

在Lightning中，我们定义Validation_step进行自定义valid操作。

但是Lightning的特性是对于每个batch都会调用一次Validation_step方法，相当于

```python
for batch in val_dataloader:
	if current_epoch % interval == 0:
    # valid
```

这显然是不优的，会白白浪费很多数据准备的时间。

强如Lightning不可能不考虑这个问题，于是翻阅文档，我们找到了trainer的参数：

- val_check_interval
- check_val_every_n_epoch

前者是基于step的，后者是基于epoch的，按需取用。

具体的

```python
trainer = pl.Trainer(..., check_val_every_n_epoch = 10)
```

表示每10轮进行一次validation，这时候在不需要Vlid的轮次，Lightning根本不会调用到validation_step方法，也就不会去准备valid所需的数据，实现轮次间的无缝衔接。

