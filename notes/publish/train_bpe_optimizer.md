> 写在前面，如果对 cs336_assignment1_basics.pdf 理解有疑问的，可以参考 [[assigment1_overview&bpe_basics]] 我对文档的翻译（部分解释细节）

虽然，基础版 BPE 分词器相对朴素版已经快了很多了，但是，回顾下 [[train_bpe#训练结果&分析]] ：
总耗时：1686 秒（约 28 分钟）

---

## 回顾 profile

profile 里有几列，先搞清楚含义：

|列名|含义|
|---|---|
|`ncalls`|这个函数被调用了多少次|
|`tottime`|函数**自身**花了多少秒（不含它调用的子函数）|
|`cumtime`|函数**累计**花了多少秒（含子函数）|

**详细解读**：

```
总耗时：1686 秒（约 28 分钟，你说的 17 分钟可能是另一次跑的）

train()          cumtime=1686s  ← 整个训练
  pretokenize()  cumtime= 619s  ← 预分词，占 37%
    .read()               68s   ← 读文件
    utf_8_decode          32s   ← 解码
    re.split              23s   ← 切 special token
    finditer            2717700次，每次都重新编译正则！← 你的 bug
    _compile            2717701次，29s              ← 就是这里
    .group()       93s           ← regex 匹配本身

  train() 自身   tottime= 416s  ← BPE 合并循环
    any()          cumtime= 441s，被调 5.8亿次！  ← 最大瓶颈
    <genexpr>      tottime= 243s，被调 18亿次！   ← any 里面的生成器
    max()           cumtime= 161s，被调 9746次     ← 每轮都扫全表
    <lambda>        tottime=  72s，被调 3.7亿次    ← max 的 key 函数
    len()           tottime=  49s，被调 5.9亿次    ← any 里的 len
```

## 优化：

[[train_bpe#优化优先级]] 分析了优化方向：
1. **`pair_to_words` 索引**——消灭 5.84 亿次 `any` 调用，预期节省 400+ 秒
2. **预编译正则**——消灭 271 万次重复编译，预期节省 30 秒
3. **堆优化 max**——预期节省 100+ 秒，但实现稍复杂
上述三个优化方向都是提升性能，我们还面临一个问题，就是语料文件过大时是无法一次性加载到内存的，而且，文档也给了一个多进程并行的优化建议。不过，并行也只能用在预分词阶段，也是在分块读取文件的一个附属优化，提升的重点还是上面耗时长的地方。因此，我们优化的优先级如下：

第一步：pair_to_words 索引      → 消灭合并循环里的 any/genexpr
第二步：预编译正则              → 消灭 pretokenize 里的重复编译  
第三步：pretokenize 多进程并行  → 在前两步完成后，pretokenize 才值得并行
第四步：堆优化 max             → 进一步压缩合并循环

---

### 1.预编译正则  
这个方向在优先级上并不靠前，先做的原因就是这个操作简单，所以先做。

**问题所在**：[[train_bpe]] 版本每次调用 `re.finditer(self.pattern, chunk)` 时，`regex` 库都要把那个字符串 pattern 重新编译一次。这个函数被调用了 **271 万次**，每次都编译，浪费了 29 秒。

**compile 原理**：正则编译 = 把字符串变成内部的有限状态机，是相对耗时的操作。编译一次、复用多次才是正确用法。

**修改方法**：使用前提前编译一次，之后用编译好的对象调用。

```python
# 修改前：每次调用都传字符串给 re.finditer，内部重新编译
for match in re.finditer(self.pattern, chunk):

# 修改后：只需改两处

# 1. 在 pretokenize 里加一行
compiled_pattern = re.compile(self.pattern)  # ← 新增这一行

# 2. 在 pretokenize 里替换调用方式
# 修改前：
for match in re.finditer(self.pattern, chunk):
# 修改后：
for match in self.compiled_pattern.finditer(chunk):
```

同样，`re.split` 也应该预编译：

```python
# pretokenize 里，修改前：
if special_tokens:
    escape_special_tokens = "|".join(re.escape(t) for t in special_tokens)
    chunks = re.split(escape_special_tokens, text)

# 修改后（在 train() 开始处编译一次，传给 pretokenize）：
if special_tokens:
    escape_special_tokens = "|".join(re.escape(t) for t in special_tokens)
    special_pattern = re.compile(escape_special_tokens)  # 编译一次
    chunks = special_pattern.split(text)                 # 用对象调用
```

**验证方法**：修改后重新 profile，看 `_compile` 的 `ncalls` 从 271 万次降到个位数。

---

### 2. pair_to_words 索引（关键，消灭 any）

这是最重要的优化，需要讲清楚**为什么** `any` 会被调用 5.8 亿次。

**当前代码逻辑**：

```python
# 每次合并（共 9743 次）都要做：
for word in word_encodings:           # 遍历所有词（约 6 万个词）
    old_encoding = word_encodings[word]
    if not any(                       # 检查这个词里有没有 merge_pair
        (old_encoding[i], old_encoding[i+1]) == merge_pair
        for i in range(len(old_encoding) - 1)
    ):
        continue
```

**问题**：每次合并，都要遍历**所有 6 万个词**来找 " 哪些词含有这个 pair"。但实际上含有这个 pair 的词可能只有几百个。像大海捞针，而且每次都重新捞。

**9743 次合并 × 6 万个词 = 5.8 亿次 `any` 调用**，这就是 profile 里看到的数字。

**解决思路**：建一个反向索引，直接记录 " 每个 pair 出现在哪些词里 "。

```
pair_to_words[(97, 98)] = {"ab", "abc", "cab"}
               ↑ pair       ↑ 所有包含这个 pair 的词
```

有了这个索引，找 " 哪些词含有 merge_pair" 就从 O(所有词数量) 变成 O(1) 的直接查找。

**完整的修改方案**（最小改动原则）：

```python
def train(self, input_path, vocab_size, special_tokens):
    # ... 初始化词汇表，pretokenize，word_encodings 的代码不变 ...

    # ── 新增：建立反向索引 ──
    # pair_to_words[pair] = set of words that contain this pair
    # 和 pair_counts 同步维护
    from collections import defaultdict
    pair_to_words: Dict[Tuple[int, int], set] = defaultdict(set)
    for word, encoding in word_encodings.items():
        for i in range(len(encoding) - 1):
            pair_to_words[(encoding[i], encoding[i+1])].add(word)

    num_merges = vocab_size - size
    pair_counts = defaultdict(int, self.count_pairs(word_counts, word_encodings))

    for merge_idx in range(num_merges):
        if not pair_counts:
            break

        merge_pair = max(
            pair_counts,
            key=lambda x: (pair_counts[x], self.vocab[x[0]], self.vocab[x[1]])
        )

        token_id = size

        # ── 关键改动：不再遍历所有词，直接查索引 ──
        affected_words = pair_to_words.get(merge_pair, set()).copy()
        # 注意必须 .copy()，因为下面会修改 pair_to_words[merge_pair]

        for word in affected_words:          # 只遍历受影响的词
            old_encoding = word_encodings[word]
            new_encoding = self.merge_encoding(old_encoding, merge_pair, token_id)
            cnt = word_counts[word]

            # 更新 pair_counts：减旧的，加新的
            for i in range(len(old_encoding) - 1):
                pair = (old_encoding[i], old_encoding[i+1])
                pair_counts[pair] -= cnt
                pair_to_words[pair].discard(word)   # ← 同步更新反向索引

            for i in range(len(new_encoding) - 1):
                pair = (new_encoding[i], new_encoding[i+1])
                pair_counts[pair] += cnt
                pair_to_words[pair].add(word)       # ← 同步更新反向索引

            word_encodings[word] = new_encoding

        # 清理
        del_keys = [k for k, v in pair_counts.items() if v <= 0]
        for k in del_keys:
            del pair_counts[k]
            if k in pair_to_words:
                del pair_to_words[k]

        self.merges.append((self.vocab[merge_pair[0]], self.vocab[merge_pair[1]]))
        self.vocab[token_id] = self.vocab[merge_pair[0]] + self.vocab[merge_pair[1]]
        size += 1

    return self.vocab, self.merges
```

**验证方法**：

```python
# 想要验证可以在 train() 循环里加一行临时打印，观察优化效果
affected_words = pair_to_words.get(merge_pair, set()).copy()
if merge_idx < 5:  # 只打印前5次合并
    print(f"合并 {merge_idx}: pair={merge_pair}, "
          f"受影响词数={len(affected_words)}, "
          f"总词数={len(word_encodings)}")
# 期望看到：受影响词数 远小于 总词数
```

期望输出类似：

```
合并 0: pair=(115, 116), 受影响词数=312, 总词数=62748
合并 1: pair=(97, 110),  受影响词数=891, 总词数=62748
```

---
## 优化后
- 训练中
还是单线程，只有一个 cpu 在跑满，其它空闲，内存依然拉满。

![](https://raw.githubusercontent.com/Emma-uestc/cs336-assignment1/main/notes/_assets/Pasted-image-20260518221314.png)

- 结果概览
![](https://raw.githubusercontent.com/Emma-uestc/cs336-assignment1/main/notes/_assets/Pasted-image-20260518223107.png)

## 性能分析（优化版）

![](https://raw.githubusercontent.com/Emma-uestc/cs336-assignment1/main/notes/_assets/Pasted-image-20260518230905.png)


![](https://raw.githubusercontent.com/Emma-uestc/cs336-assignment1/main/notes/_assets/Pasted-image-20260518231122.png)

---
### profile 数据解读

#### 现状诊断

两步优化效果符合预期：`any/genexpr` 从 441 秒归零，`_compile` 从 271 万次降到 2 次。总耗时从 1687 秒降到 757 秒，快了 2.2 倍。瓶颈已经**彻底换人**了：

```
pretokenize   569s   占 75%   ← 新的第一杀手
max()         156s   占 21%   ← 第二
lambda key     71s   （max 的组成部分）
```

`max()` 和 `lambda` 加起来 227 秒，是合并循环的主体，但这两个无法独立于 `pretokenize` 优化，原因是：

**`pretokenize` 569 秒里，真正做正则匹配的只占一小部分**。看这几行：

```
pretokenize 自身 tottime = 365s   ← Python 层的循环开销（defaultdict 写入等）
.group()    tottime =  92s        ← regex 匹配结果提取，被调 5.36亿次
utf_8_decode              42s     ← 文件解码
.read()                   32s     ← 磁盘读取  
re.split                  34s     ← special token 切分
```

5.36 亿次 `.group()` 调用是 regex 在工作，这部分无法消除。但 365 秒的 Python 层 `tottime` 和 34 秒的 `re.split` 可以通过**多进程并行**线性压缩。

---
#### 对比  
> 还是把 profile 数据给了 AI，让其给出”看图说话“的分析方式
![](https://raw.githubusercontent.com/Emma-uestc/cs336-assignment1/main/notes/_assets/Pasted-image-20260519105704.png)

### 下一步优化方向：多进程并行 pretokenize

现在应该没有使用单核机器的，像我的是机器 16 核，`pretokenize` 天然可并行（每个 chunk 独立），理论上可以把 569 秒压到 `569 / 16 ≈ 36 秒`。加上合并循环的 227 秒，总耗时目标 **~270 秒（约 4.5 分钟）**。

由于我是在 WSL2 中跑的，本身也有一些 的进程启动开销和内存压力，实际大概能到 **5~8 分钟**，不过已经远低于作业要求的 30 分钟上限。至于 2 分钟目标，需要再加堆优化（替换 `max()` 操作），我打算做完并行后再看新 profile。

具体要改的代码只有两个函数，`pretokenize` 拆成 " 分块函数 + 单块处理函数 + 主调函数 " 三部分，改动量不算大。

---
具体要改的代码只有两个函数，`pretokenize` 拆成 " 分块函数 + 单块处理函数 + 主调函数 " 三部分，改动量不大。
你想现在开始写，还是先对上面的分析有疑问？
两步优化效果符合预期：`any/genexpr` 从 441 秒归零，`_compile` 从 271 万次降到 2 次。总耗时从 1687 秒降到 757 秒，快了 2.2 倍。

但现在瓶颈已经**彻底换人**了：

```
pretokenize   569s   占 75%   ← 新的第一杀手
max()         156s   占 21%   ← 第二
lambda key     71s   （max 的组成部分）
```

`max()` 和 `lambda` 加起来 227 秒，是合并循环的主体，但这两个无法独立于 `pretokenize` 优化，原因是：

**`pretokenize` 569 秒里，真正做正则匹配的只占一小部分**。看这几行：

```
pretokenize 自身 tottime = 365s   ← Python 层的循环开销（defaultdict 写入等）
.group()    tottime =  92s        ← regex 匹配结果提取，被调 5.36亿次
utf_8_decode              42s     ← 文件解码
.read()                   32s     ← 磁盘读取  
re.split                  34s     ← special token 切分
```

5.36 亿次 `.group()` 调用说明 regex 在认真工作，这部分无法消除。但 365 秒的 Python 层 `tottime` 和 34 秒的 `re.split` 可以通过**多进程并行**线性压缩。

---



![](https://raw.githubusercontent.com/Emma-uestc/cs336-assignment1/main/notes/_assets/Pasted-image-20260520212400.png)


