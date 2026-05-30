作业 1 的最终目标是实现一个语言模型 

# 1. 作业概览  

本作业的内容是从零开始实现训练一个标准的 Transformer 模型所需所有组件。
## 1.1 将要实现哪些组件  

1. BPE 分词器 (Section 2)
2. Transformer 模型 (Section 3)
3. 交叉损失函数和 AdamW 优化器 (Section 4)
4. 迭代训练过程，支持序列化加载模型和优化器状态 (Section 4)

## 1.2 需要运行哪些组件  

1.  BPE 分词器 在 TinyStories 数据集上的训练
2. 使用训练好的分词器对数据集进行编码
3. 使用 TinyStories 数据集训练 Transformer 模型
4. 使用训练的 Transformer 模型生成和评估困惑度
5. 在 OpenWebText 数据集上训练模型，并将困惑度提交到排行榜

## 1.3 你可以使用哪些

每一个组件都从零开始构建，不能使用 `torch.nn`, `torch.nn.functional`, `otorch.optim`,以及除了以下组件，其它都不能使用：
- `torch.nn.Parameter`
- 包括 `torch.nn` 类中的，如 `Module`, `ModuleList`, `Sequential` 等
- `torch.optim.Optimizer` 基类 

## 1.4 AI 工具使用声明

- 使用 AI Agent 时确保包含了 AGENTS.md 文件，通过向 AI 聊天机器人提供提示词时也应该包含该文件
- 强烈建议禁用 AI 自动补全功能
- 全部关于使用 AI 工具的策略请看 [这篇文档](https://docs.google.com/document/d/1SZAlExB1qAc9izHt54gwunNpjKE6wXb8Y7yA_e-baK8/edit?tab=t.0#heading=h.n5d3vcbxn33y)

## 1.5 代码长什么样

代码的 git 仓库地址：[https://github.com/stanford-cs336/assignment1-basics](https://github.com/stanford-cs336/assignment1-basics)

```bash
tree -L 2
.
└── assignment1-basics
    ├── AGENTS.md
    ├── CHANGELOG.md
    ├── CLAUDE.md
    ├── LICENSE
    ├── README.md
    ├── cs336_assignment1_basics.pdf
    ├── cs336_basics
    ├── make_submission.sh
    ├── pyproject.toml
    ├── tests
    └── uv.lock
```

1. `cs336_basics/` 所有代码都写在这个目录下，当前这里没有任何代码，你怎么从零实现都可以
2. adapters.py
3. test_*.py

## 1.6 How to submit


## 1.7 如何获取数据集  

作业 1 将会用到两个预处理过的数据集：TinyStories 数据集和 OpenWebText 数据集，`README.md` 文档中有具体的下载方式。
如果你没有足够算力，这部分还提供了一些 tips 可以参考。

# 2. BPE 分词器

作业的第一部分是实现一个 BPE 分词器。具体来说，就算使用字节序列表示任意 Unicode）字符串，然后基于该字节序列训练我们的 BPE 分词器，最后使用我们自己训练的 BPE 分词器将文本编码为 token，用于语言建模。

## 2.1 标准 Unicode 

Unicode 编码是将字符映射为整数：

```pyhon
>>> ord('牛')
29275
>>> chr(29275)
'牛'
```

 Problem (unicode1):  Understanding Unicode

(a):What Unicode character does `chr(0)`  return?  
 A: chr(0) returns a **Null character**(often referred to as NULL), which corresponds to the Unicode code point U+0000.  
**核心知识点**：`chr(0)` 对应的是 Unicode 码位 U+0000，称为 Null Character。
**注意**: 你可以会疑惑，你在终端中执行 `chr(0)` 返回的不是空白

```python
>>> chr(0)
'\x00'
```

这是因为 Python 在交互式环境中，你输入 `chr(0)`，它会调用该对象的 `__repr__()` 方法，返回一个字符串，看起来像一个合法的 Python 表达式，通常是给开发者看的，你试着打印该字符看下，你会得到如下输出：

```python
>>> chr(0)
'\x00'
>>> print('\x00')

>>> print(chr(0))

>>> 
```

这时返回的就是空白，因为 print 方法默认调用 `str()`，可以理解为这是给人类看的。

(b):How does the character's string representation(__repr__) diff from its printed representation?  

A: Its string representation (__repr__) displays the esacped sequence `'\x00' `for debugging purposes whereas its printed representaion renders the actual invisible control character resulting no visible output.  

好吧，这道题，我们上面已经解释过了。
**核心知识点**：repr 显示转义符 \x00 以便调试，print 渲染实际字符（不可见）。

(c): What happens when this character occurs in text?  

A: It behaves as a valid, invisible character within the string and does not terminate the string (unlike in C languages), allowing subsequent text to be processed and displayed normally.  

**核心知识点（重点）**：

在 C 语言中，`\0` 是字符串结束符（**Null Terminator**）。如果你在 C 语言里写 `"test\0string"`，打印出来只有 `"test"`，后面的会被丢弃。

但在 Python 中，字符串是确定的长度（Length-prefixed），`chr(0)` 只是一个普通的字符。所以 `print("..."+chr(0)+"...")` 不会截断字符串，后面的内容依然会被打印出来。

## 2.2 Unicode 编码

尽管标准 Unicode 定义了从字符到码点（整数序列）的映射，但是直接使用其训练分词器仍不是明智的选择。 因为 Unicode 码点有 15 万 + 字符，这是一个庞大且稀疏的词表库。因此，我们使用 Unicode 编码将字符映射为字节序列。Unicode 标准本身定义了三种编码：UTF-8、UTF-16 和 UTF-32，其中 UTF-8 是互联网上的主流编码（占所有网页的 98% 以上）。

Python 中使用 `encode()` 函数将 Unicode 字符串编码为 UTF-8，使用 `list ` 等方法获取 `bytes ` 对象并对其迭代使用 `encode`。最后，使用 ` decode()` 函数将 UTF-8 字节序列解码为 Unicode 字符串。
- 看下面的例子

```python
# 定义包含中英文和日文的字符串
test_string = "hello! こんにちは!"

# 使用 UTF-8 编码将其转换为字节流
utf8_encoded = test_string.encode("utf-8")

# 打印字节流对象及其类型
print(utf8_encoded)
# 输出: b'hello! \xe3\x81\x93\xe3\x82\x93\xe3\x81\xab\xe3\x81\xa1\xe3\x81\xaf!'
print(type(utf8_encoded))
# 输出: <class 'bytes'>

# 获取编码后字符串的字节值（0-255 之间的整数列表）
byte_values = list(utf8_encoded)
print(byte_values)
# 输出: [104, 101, 108, 108, 111, 33, 32, 227, 129, 147, 227, 130, 147, 227, 129, 171, 227, 129, 161, 227, 129, 175, 33]

# 核心对比：字符长度 vs 字节长度
print(len(test_string))   # 输出: 13
print(len(utf8_encoded))  # 输出: 23

# 将字节流重新解码回字符串
print(utf8_encoded.decode("utf-8"))
# 输出: hello! こんにちは!
```

---

###  深入理解 UTF-8 编码逻辑

这个示例最核心的价值在于展示了 **Unicode 字符**与 **UTF-8 字节**之间不是“一对一”的关系。

#### **为什么长度不一样？ (13 vs 23)**

- **英文字符与符号**：`h`, `e`, `l`, `l`, `o`, `!`, （空格）以及末尾的 `!` 都是标准 ASCII 字符。在 UTF-8 中，它们只占用 **1 个字节**。
    
    - 例如：`h` 的 ASCII 码是 104，对应列表里的第一个数。
        
- **日文字符 (平假名)**：`こ`, `ん`, `に`, `ち`, `は`。在 UTF-8 编码下，大多数常用汉字和日文假名通常占用 **3 个字节**。
    
    - 计算逻辑：$7 \text{ 个 ASCII 字符} \times 1 + 5 \text{ 个日文字符} \times 3 + 1 \text{ 个符号} \times 1 = 23 \text{ 字节}$。
        
#### **字节值的含义**

当你运行 `list(utf8_encoded)` 时，你看到的是真实的**二进制数据**在内存中的整数表现：

- **前 7 位**：`104, 101, 108, 108, 111, 33, 32` 直接对应 `hello!` 。
    
- **中间部分**：从 `227` 开始的序列，每三个数代表一个日文字符。例如 `227, 129, 147` 合起来在解码时才会被识别为 `こ`。
    

---

### 3. 关键概念区分：`str` vs `bytes`

在 Python 3 中，这种区分非常严格：

1. **`str` 对象 (test_string)**：是抽象的文本。它在内存中以 Unicode 点阵存储。你调用 `len()` 得到的是**字符数**。
    
2. **`bytes` 对象 (utf8_encoded)**：是原始的 8 位二进制数据。它是给机器看、存入硬盘或通过网络传输的。你调用 `len()` 得到的是**物理字节数**。
    
> **小贴士**：这也就是为什么在读取网页或文件时，如果编码设置错误（比如用 `GBK` 去解 `UTF-8` 的字节流），就会出现乱码。因为解码器在尝试将那些“三字节组合”强行对应到错误的字符表上。

````ad-note
title: Problem (unicode2): Unicode Encodings

(a)What are some reasons to prefer training our tokenizer on UTF-8 encoded bytes, rather than UTF-16 or UTF-32? It may be helpful to compare the output of these encodings for various input strings.

UTF-8 is preferred because it is variable-length and space-efficient, representing common ASCII characters with just one byte, whereas UTF-16 and UTF-32 use fixed larger sizes (2 or 4 bytes) that introduce excessive null-byte padding for the same text.


(b)Consider the following (incorrect) function, which is intended to decode a UTF-8 byte string into a Unicode string. Why is this function incorrect? Provide an example of an input byte string that yields incorrect results.

```Python

def decode_utf8_bytes_to_str_wrong(bytestring: bytes):

    return "".join([bytes([b]).decode("utf-8") for b in bytestring])

```

![](https://raw.githubusercontent.com/Emma-uestc/cs336-assignment1/main/images/Pasted-image-20260512142337.png)

```bash
>>> decode_utf8_bytes_to_str_wrong("你".encode("utf-8"))
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "<stdin>", line 2, in decode_utf8_bytes_to_str_wrong
UnicodeDecodeError: 'utf-8' codec can't decode byte 0xe4 in position 0: unexpected end of data
```

可以看出，对于英文字符（占用一个字节）可以正常解码，但是对于一个中文字符占用 3 个字节的解码错误。

```Python

>>> test_string = "你"

>>> print(test_string.encode("utf-8"))

b'\xe4\xbd\xa0'

```

我们的 `decode_utf8_bytes_to_str_wrong` 是逐字节的解码，而 ' 你 ' 是使用三个字节表示的，因此当它遇到第一个字节 `0xe4` 时无法解码出对应的字符。
````

## 2.3 Subword Tokenization

虽然上述 bytes 层级的分词器可以解决 word 层级的分词器面临的有些 word 不在词汇表中的问题，但是将文本拆分到 bytes 层级将会导致输入序列过于冗长，如一个长度为 10 个 word 的字符串，拆分到 bytes 级别，输入序列长度远远大于 10.这会严重降低训练的效率。

子词分词就是介于 word 层级分词器与 bytes 层级分词器之间的折中方案。bytes 层级分词器的词表大小是 256([0,255])，子词分词器通过牺牲词汇表规模来换取对输入字节序列更佳的压缩效果。例如，如果字节序列 `b'the'` 在我们的原始文本训练数据中频繁出现，那么在词汇表中为其分配一个条目，就能将这个由 3 个 token 组成的序列简化为单个 token。

如何选择子词呢？先后由 R. Sennrich et al.P. Gage 等提出了 字节对（Byte Pair Encoding,BPE） 算法。这是一种压缩算法，它通过不断迭代将出现频率最高的字节对替换（“合并”）为一个新的、未使用的索引。需要注意的是，该算法会向词汇表中添加子词 token，以最大限度地提高输入序列的压缩率——如果某个词在输入文本中出现足够多次，它将被表示为一个子词单元。

通过 BPE 构建词汇表的子词分词器通常被称为 BPE 分词器。在本作业中，我们将实现一个字节级 BPE 分词器，其词汇表项为字节或合并的字节序列，这在处理词汇表外词汇和保持输入序列长度可控方面兼顾了二者优势。构建 BPE 分词器词汇表的过程被称为“训练”BPE 分词器。

以上关于分词器的演进和详细对比见 [[Lecture 01#3.1. 分词技术的演进过程与缺陷改进方案]]

## 2.4 BPE Tokenizer Training

BPE 分词器训练主要包括以下几步。

### 词表初始化

分词器的词汇表是字节串与整数 ID 的一对一映射关系。由于我们正在训练一个字节级别的 BPE 分词器，因此初始词汇表就是所有字节的集合。由于字节值共有 256 种可能，因此初始词汇表的大小为 256。

>如何理解这里的字节串 (bytestring) token
>首先，token 的本质是词汇表里的一个基本单位
> 那什么是字节串呢？它的单位不是文本字符，是二进制数据，
> 还记得我们前面的例子吗
> 对于英文来说，`b'a'` 既是一个字符也是一个字节 既是一个字符也是一个字节;
> 对于中文 `b'\xe4\xbd\xa0'`（" 你 "）的第一个字节，它本身没有任何意义，但在分词器看来，它就是一个字节串 token，在分词器训练过程中，算法发现 `b'\xe4\xbd\xa0'` 这三个字节经常出现在一起，于是经过多次合并，最终就创建了一个新的 token,这个 token 的 ID 是 1024（假设），我们的词汇表就是这样不断扩展的。

### 预分词

按理来说，一旦我们有了词汇表（我们的初始词汇表），原则上就可以开始对文本进行统计，对出现频率最高的相邻字节对进行合并。但是，这种方法的计算成本非常高，因为每次合并都需要对语料库进行一次完整的遍历（时间复杂度非常高）。除此之外，如果直接对语料库进行合并操作，可能会产生仅仅因为标点不同而不同的 token（例如，dog! 与 dog.），尽管这些 token 语义上高度相似，但就会映射为不同的 ID

>为什么说这种直接开始的计算成本非常高呢？我们需要从数据结构和处理流程两个维度来拆解。
>### 1. 基础时间复杂度分析
>我们假设语料库总共有 $N$ 个字符（或初始字节），词表大小（Vocabulary Size）目标是增加 $M$ 个合并项（Merges）。
>按照我们上面的逻辑，每一次合并（Merge）的逻辑如下：
>   1. **扫描统计**：遍历当前语料库，统计所有相邻对（Pairs）的频率。
 >    - **复杂度：$O(N)$**
 >   
>   2. **寻找最频项**：从统计结果中找到频率最高的 Pair（例如 `(b'h', b'e')`）。
 >   
      - **复杂度：$O(V)$**，其中 $V$ 是当前不同 Pair 的数量，通常 $V \ll N$。
 >       
>   3. **执行替换**：再次遍历语料库，将所有的 `(b'h', b'e')` 替换为新的 Token ID（例如 `256`）。
  >  
>     - **复杂度：$O(N)$**
    因为要进行 $M$ 次合并，所以总的时间复杂度是：
>
>$$O(M \times N)$$
在现代大模型（LLM）的背景下，这个数字是非常恐怖的：
>
> - **$N$ (语料规模)**：现在的预训练语料通常以 **Terabytes** 计，对应的字节数 $N$ 在 $10^{12}$ 级别。
  >  
>- **$M$ (合并次数)**：常见的词表大小（如 GPT-4, Llama 3）在 32k 到 128k 之间。
    这意味着如果你每合并一个 Token 都要扫一遍万亿级的字节流，计算量将无法承受。
---

为了避免上面的情况，我们先对语料库进行预分词。可以理解为先对语料库做一个粗粒度的分词，有助于统计字符对出现的频率。比如，单词“text”在预分词统计中出现了 10 次，当我们再统计字节对 `(b't', b'e')`）的频率时，可以直接将其计数增加 10，而无需再遍历整个语料库。不过，记住，我们训练的是一个字节级的 BPE 模型，因此每个预分词都表示为一串 UTF-8 字节。

R. Sennrich 等提出的原始 BPE 实现通过简单地按空格分割（即 `s.split(" ")`）来进行预分词。这种方法在基于 SentencePiece 的分词器中仍然存在（例如 Llama 1 和 2 的分词器）。 大多数现代分词器采用基于正则表达式的预分词器，这一做法源自 GPT-2；A. Radford 等人的研究 。我们将使用原始正则表达式的一种稍加优化的形式，该版本取自 github.com/openai/tiktoken/pull/234/files
`>>> PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""`

使用的是 `regex` 库，而不是 Python 自带的 ` re`,看下面的例子做个简单的了解

```Python
>>> # requires `regex` package

>>> import regex as re

>>> re.findall(PAT, "some text that i'll pre-tokenize")

['some', ' text', ' that', ' i', "'ll", ' pre', '-', 'tokenize']
```

### 执行 BPE 合并

经过前面的操作，现在已经得到了预分词的 token 以及对应的 UTF-8 字节序。现在可以执行合并操作了（即训练 BPE 分词器）。从宏观层面看，BPE 算法迭代的统计每一对字节对出现的频率，并找出频率最高的进行合并，假设经过一轮迭代找到的是（“A”、“B”），执行合并得到新的 token 是“AB”，对应的 ID 是 256，将这个新的词汇假如到词汇表中，因此词汇表大小就是初始词汇表大小（本例中 256）加上合并操作执行的次数。为了提高 BPE 训练的效率，我们不考虑跨越预分词边界的字节对，另外，在合并过程中，如果出现了相同频率的字节对，选择字典序更大的字节对合并，比如，例如，若配对 (“A”, “B”)、(“A”, “C”)、(“B”, “ZZ”) 和 (“BA”, “A”) 的频率均最高，则我们会合并 (“BA”, “A”)

```python
>>> max([("A", "B"), ("A", "C"), ("B", "ZZ"), ("BA", "A")])

('BA', 'A')
```

### 特殊 token

通常，某些字符串（例如 <|endoftext|>）用于编码元数据（例如文档之间的分界）。因此，在对文本进行编码时，这些被视为特殊 token，这些 token 不能被拆分多个 token（即始终保留位单个 token）。例如，标志文档结束的字符串 <|endoftext|> 不能被拆分，因为我们需要根据这个判断文档是否结束，这些特殊 token 添加到词汇表中，具有固定的 ID。

R. Sennrich 等人的算法 1 [3] 包含了一个低效的 BPE 分词器训练实现（基本上遵循了我们上面概述的步骤）。作为第一个练习，实现并测试该函数以检验你的理解可能会有所帮助。

````ad-note
title:Example (bpe_example): BPE training example
---
以下是 R. Sennrich 等[3]提供的一个示例。考虑一个由以下文本组成的语料库：
```text
low low low low low
lower lower widest widest widest
newest newest newest newest newest newest
```
特殊token是 `<|endoftext|>` 

**词汇表**
初始词汇表是256个字节再加上特殊token，特殊token的ID就设为256
所以，初始词表：
词表 = `{0: b'\x00', ..., 255: b'\xff'}` + 特殊Token
vocab = `{0: b'\x00', ..., 255: b'\xff', b'<|endoftext|>': 256}`
但是，这个初始化过程不是我们手动的，而是使用Python 代码帮我们实现的
```python
# 初始化 256 个基础字节
vocab = {bytes([i]): i for i in range(256)}
# 添加特殊 Token
vocab[b'<|endoftext|>'] = 256
```

**预分词**
为简化起见，并以便于关注合并过程，本例中我们假设预分词仅按空格进行分割。经过预分词和计数后，最终得到频率表。
`{low: 5, lower: 2, widest: 3, newest: 6}`

将其表示为 `dict[tuple[bytes, ...], int]` 会比较方便，例如 `{(l, o, w): 5, ...}`。请注意，在 Python 中，即使是一个字节也是 `bytes` 对象。Python 中没有专门表示单个字节的 `byte` 类型，就像没有专门表示单个字符的 `char` 类型一样。
所以，实际预分词后应该是：

**合并**
这是一个迭代循环过程：
```
**`while` (当前词表大小 < 目标词表大小):**

	1. **全局扫描统计：** 统计当前语料库中，所有相邻的两个元素组成的“对”（Pair）出现的总次数。
	    
	    - 例如：统计 `('e', 's')` 出现了 9 次，`('s', 't')` 出现了 9 次。
	        
	2. **寻找最优对：** 找到出现次数最高的那一组。如果次数相同，则按字母顺序（字典序）选一个。
	    
	    - 例如：选中了 `('s', 't')`。
	        
	3. **执行合并（Merge）：** 遍历整个语料库，将所有连续出现的 `s` 和 `t` 替换成一个新的整体 `st`。
	    
	    - 旧：`('w', 'i', 'd', 'e', 's', 't')`
	        
	    - 新：`('w', 'i', 'd', 'e', 'st')`
	        
	4. **更新词表：** 将这个新合成的 `st` 加入词表。
	    
	5. **循环往复：** 词表计数加 1，回到第一步重新统计，直到达到你预设的词表规模。  
```


**“真相”补充**

虽然上面的字符合并（如 `s` + `t` = `st`）非常直观，但在实际的大模型，为了处理世界上所有的语言且不让初始词表爆炸，底层操作是这样的：

- **处理对象是字节（Bytes）：** 算法最初面对的不是字符 `a, b, c`，而是 0-255 的 **字节数值**。
  对文本encoding 后是[bytes: List[int]]，比如上面的文本应该是：
  `{'low': [108, 111, 119], ' low': [32, 108, 111, 119], ' lower': [32, 108, 111, 119, 101, 114], ' widest': [32, 119, 105, 100, 101, 115, 116], ' newest': [32, 110, 101, 119, 101, 115, 116]}`
  
- **合并的是整数值：** 实际操作中，计算机会统计 `[101, 115]`（即 `e` 和 `s` 的 UTF-8 编码）出现的频率，并将它们合并为一个新的数字（如 `256`）。
    
- **为什么要这样做？** 因为字节是计算机的通用语言。用字节作为起点，BPE 就可以用一个很小的基础词表（256个）来编码世界上任何语言的文本，而不会遇到“未知字符”（Unknown Token）的问题。
    
**一句话总结：** BPE 就是通过不断合并最常出现的“邻居”，把零散的**零件（字节）**组装成**零件包（子词）**，最后组装成成品（单词）的过程。
````

## 2.5 训练 BPE 分词器

理解了 BPE 训练逻辑，下面开始写代码，先在 TinyStories 数据集上训练，关于数据集的下载详情请见

### 预分词并行化

在训练 BPE 过程中，你应该会发现，预分词是整个过程的主要瓶颈。但是我们可以使用 `multiprocessing` 库对预分词部分进行并行处理，从而加快预分词速度。具体来说，文档建议在预分词的并行实现中，将语料库进行分块处理，同时确保分块边界位于某个特殊 token 的开头。

可以参考 `assignment1-basics/cs336_basics/pretokenization_example.py` 实现通过对文本分块并行的方式提高预分词效率。

这种分块方式始终有效，因为我们绝不会跨文档边界进行合并。就本次作业而言，你可以始终采用这种方式进行分割。无需担心收到一个非常大的语料库却不包含 <|endoftext|> 标签的边界情况。

### 预分词前移除特殊 token

在预分词这步中，切记在训练预分词时使用正则表达式（`re.finditer`）将特殊 token 从语料库（或者分块语料文本，如果你对文本做了分块处理，以便并行训练）中移除。并且一定在特殊 token 处进行切分。比如有这样一段语料（或者分块语料）`[Doc 1]<|endoftext|>[Doc 2]`，一定要在 `<|endoftext|>` 处进行切分，然后分别对 `[Doc 1]` 和 `[Doc 2]` 执行预训练。这样才能避免跨文档边界合并。换句话说，特殊 token 定义了训练时的硬分割边界，它们本身不应计入合并次数。可以使用 `re.split` 方法实现切分，使用 `"|".join(special_tokens)` 将所有的特殊 token 拼接起来作为分隔符（需谨慎使用 re.escape，因为特殊 token 中可能包含 `|`）。测试用例 `test_train_bpe_special_tokens` 将对此进行验证。

````ad-note
title:这段内容想表达的就是
||作用|
|---|---|
|特殊token|定义**硬性分割边界**，两侧文本不能跨越它合并|
|特殊token本身|**不参与**合并计数，不贡献BPE频率统计|


**一句话总结就是：特殊 token 是 " 防火墙 "：把语料切成独立片段分别处理，自己却不进入 BPE 训练。**

````

### 优化合并操作

在上文的简化示例中，BPE 训练的朴素实现速度较慢，因为每次合并时，它都会遍历所有字节对找出出现频率最高的字节对。然而，每次合并后唯一发生变化的字节对计数，只有那些与合并后的字节对重叠的字节对。因此，可以通过为所有字节对建立计数索引并增量更新这些计数，而不是显式遍历每个字节对来计算其出现频率，从而提高 BPE 训练的速度。通过这种缓存机制可以显著提升速度，但需要注意的是，BPE 训练中的合并部分在 Python 中无法并行化。

````ad-note
title:朴素方法效率分析&改进
1. 为什么慢
   回顾下主循环 while 中的逻辑，首先，全局扫描统计：
   按照朴素方法，没合并一次就先做做一个全局扫描统计相邻字节对频率，效率极其低下（$O(N^2)$ 级别的复杂度）。
2. 
   每次合并后，其实**绝大多数组合频率并没有发生变化，只有与合并目标相邻的“邻居”受到了影响**。
   我们用前面出现过的单词 `widest` 举个具体的例子：

- **合并前：** 字母序列是 `w, i, d, e, s, t`。 相邻的对有：`(w,i)`, `(i,d)`, `(d,e)`, `(e,s)`, `(s,t)`。
    
- **动作：** 我们决定把最频繁的 `s` 和 `t` 合并成 `st`。
    
- **合并后：** 字母序列变成了 `w, i, d, e, st`。 新的相邻对变成了：`(w,i)`, `(i,d)`, `(d,e)`, `(e, st)`。

**发现了吗？** 受影响的仅仅是参与合并的字符及其前后的邻居：

- **消失的对：** `(e,s)` 和 `(s,t)` 不复存在了，因为它们被合并打断了。
    
- **新增的对：** 产生了一个新的对 `(e, st)`。
    
- **完全没变的对：** 前面的 `(w,i)`, `(i,d)`, `(d,e)` 根本没有受到任何影响！
  
3. 如何优化呢？
  ** 缓存与增量更新**，就是文档中说的 caching procedure / indexing
  - 首先，老老实实从头到尾扫描一遍，建立一个所有相邻字节对统计频率的大字典
  - 之后每次扫描不再扫描全文，而是在这个大字典里做加减法
    - 扣除那些因为合并而被破坏频率值的旧字节对（比如 `(e,s)` 频率减1）
    - 增加因为合并而产生的新频率对（比如 `(e, st)` 频率加1）。
  这种只更新局部变化的方式，训练速度会有指数级的提升。   
````

文档给了我们两个工程操作 Tips

````ad-tip
title: Tips1:使用性能分析工具
不要想当然的靠猜去优化代码。而是应该用 Python 的 `cProfile` 或 `py-spy` 等工具跑一下，让工具告诉你到底是代码的哪一行占用了 90% 的时间，然后优化那里。
````

````ad-tip
title: Tips2:使用小数据集测试&调试
写好代码后，千万别直接用整个TinyStories数据集去跑测试，那样一旦卡住或者报错，你可能等半天什么也得不到。应该先切一块几万篇的小数据集（Debug dataset）来跑通流程、测试优化效果，确认跑得飞快且没有 Bug 了，再用全量数据去炼丹。其实，课程提供了抽样的小数据集。
```bash
find ./* -name "*sample*"
./tests/fixtures/tinystories_sample_5M.txt
./tests/fixtures/tinystories_sample.txt
```

按照指导下载的完整的数据集分别是用来训练和验证的。

| 数据集                          | 规模       | 用途        |
| ---------------------------- | -------- | --------- |
| TinyStoriesV2-GPT4-train.txt | 2.12M 文档 | 最终训练      |
| TinyStoriesV2-GPT4-valid.txt | 22K 文档   | **验证** |

````

看下我们要做的第一个代码任务吧。

````ad-seealso
title:Problem (`train_bpe`): BPE Tokenizer Training
有以上分析，BPE 分词器的训练至少需要包含以下参数：
- **Input**：
  **`input_path`**:`str`,用来训练BPE分词器的文件路径
  `vocab_size`: `int`,一个正整数，定义最终词汇表大小（包含初始基础词汇，特殊token,不断迭代合并产生的token的总和）
  `special_tokens`:`list[str]`,需要加入到词汇表中的字符串列表。训练过程中的合并操作硬边界，不允许跨越这个边界进行合并操作，不参与训练。
- **Output**：
  `vocab`:` dict[int, bytes]`，token 词汇表，token 及其对应的 ID
  `merges`: `list[tuple[bytes, bytes]]`，合并后的字节对列表。
  
  文档中虽然说实现一个BPE分词器函数，由以上分析和功能模块化，我们显然应该实现一个BPE training 类，按照上述讨论的步骤，不同的函数实现不同的功能。
关于这里的实现，可以参考其它BPE 分词训练的系列文章。
````

---

````ad-seealso
title:Problem (train_bpe_tinystories): BPE Training on TinyStories
(a)在TinyStories数据集上训练BPE分词器。
- vocabulary  size 最大设置为10,000
- 不要忘记将special token 加入词表
- 返回的词表和merge 规则持久化到磁盘
- 记录训练时长和内存指标（建议观察CPU和内存指标）
资源配置要求：≤ 30 minutes (no GPUs), ≤ 30 GB RAM

(b) BPE分词器训练过程中，哪个部分花费时间最长。
预分词花费时间最长。
关于以上两题分析请看[[train_bpe#训练结果&分析]]
````

````ad-seealso
title:Problem (`train_bpe_expts_owt`):  BPE Training on OpenWebText
(a) 在数据集 OpenWebText 上训练 BPE 分词器，要求：
`vocab_size` 最大32,000
`vocab`,`merges` 持久化到磁盘
资源限制：≤ 12 hours (no GPUs), ≤ 100 GB RAM
(b) 与在TinyStories数据集上的训练结果对比
````

## 2.6 BPE Tokenizer: Encoding and Decoding

通过前面的 BPE 分词器工作，我们完成了**训练**分词器的工作，从原始文本语料库中**学习** BPE 的合并规则（merges）和词表（vocab）。得到了：

- `vocab: dict[int, bytes]` — 词表
- `merges: list[tuple[bytes, bytes]]` — 合并规则列表
接下来，我们要使用我们训练得到的词表和合并规则进行编码/解码工作。
> 可以类比为：前面的工作是在 " 制造工具 "，产出的是规则/参数，而不是对任何文本的实际处理。
> 接下来要做的部分** 这是实现一个 `Tokenizer` **类**，它加载已经训练好的 vocab 和 merges，然后用它们对**任意新文本**进行编码和解码。

### 2.6.1 对文本编码

BPE 对文本进行编码的过程与我们训练 BPE 词汇表的方式如出一辙。该过程主要包括以下几个主要步骤。

- **步骤 1：预分词**
首先对序列进行预分词，并将每个预分词表示为 UTF-8 字节，与我们在 BPE 训练中的做法相同。然后，对这些经过预分词得到的字节序列合并为 vocab 中存在的元素。合并操作是独立进行的（不跨预分词边界进行合并）。

---
关于这里的预分词与 BPE 分词器训练时预分词有些区别

|BPE 训练时|encode 时|
|---|---|---|
|目的|统计词频|切分文本做 merge|
|special token|切掉不要，只是边界|切掉但要**保留**，需要转成 id|
|是否需要括号|不需要|需要|

---


**2. `(...)` — 正则表达式的捕获组，这才是保留分隔符的关键**

```python
re.split("(A|B)", "xAyBz")   # 有括号：['x', 'A', 'y', 'B', 'z']  保留分隔符
re.split("A|B",   "xAyBz")   # 无括号：['x', 'y', 'z']            丢弃分隔符
```

---


- **步骤 2： 使用 `merges`**
使用前面训练得到的 vocab 和 merges,利用正则 pre-tokenize 新文本，切成 pre-tokens

````ad-seealso
title: Example (bpe_encoding):  BPE encoding example
输入字符串：'the cat ate'
vocab:`{0: b' ', 1: b'a', 2: b'c', 3: b'e', 4: b'h', 5: b't', 6: b'th', 7: b' c', 8: b' a', 9: b'the', 10: b' at'},
merges:[(b't', b'h'), (b' ', b'c'), (b' ', b'a'), (b'th', b'e'), (b' a', 10 b't')]`
首选，对输入字符串进行pre-tokenize,得到 `['the', ' cat', ' ate']`
然后，对每个pre-token 根据`merges` 进行合并。
第一个 pre-token 'the' 表示为 `[b't', b'h', b'e']`，我们查找 merges,找到了`(b't', b'h')`,对'the'的pre-token 进一步合并为 `[b'th', b'e']`,然后，继续找，在`merges` 中又找到了`(b'th', b'e')`，进一步合并为 `[b'the']`,以此类推，直到在`merges` 中找不到可再合并的bytes,然后，我们查找`[b'the']` 的ID 为 `9`.
`' cat'` 和 `' ate'`重复上述过程，`' cat'` 表示为`[7, 1, 5]`，`' ate'`表示为`[10, 3]`，所以，输入字符串最终表示为 `[9, 7, 1, 5, 10, 3]`.
````

- **特殊 token**
  分词器必须正确处理用户定义的特殊 token
- **内存管理**
  当文件较大时，无法一次性加载到内存，必须分块处理。这样内存复杂度始终为 O(1),而不是 O(N)。这个过程，我们需要确保单词不会跨越分块边界，否则得到的词法分析结果将会出现问题。
---
### **2.6.2 Decoding text ** 

decode 就是将整型序列解码为纯文本。最简单的做法就是查询 vocab，将 ID 对应的字节序列进行组合拼接到一起，然后将字节解码为 Unicode 字符串。不过，要注意的是，输入的 ID 不一定能映射为一个有效的 Unicode 字符串（因为用户可以输入任意整数 ID 序列）。如果输入 token 的 ID 无法映射为有效的 Unicode 字符串，应该将格式错误的字节替换为官方的 Unicode 替换符 U+FFFD。`bytes.decode` 的 ` errors` 参数控制 Unicode 解码错误的处理方式，使用 `errors='replace'` 将自动将格式错误的数据替换为替换标记。

````ad-seealso
title:Problem (tokenizer):  Implementing the tokenizer (15 points)
实现一个 `Tokenizer` 类，根据给定的 vocab 和 merges,将文本编码为整型序列，将整型序列解码为纯文本。需要支持用户提供的特殊 token。如果这些特殊 token 不在 vocab 中，将其添加到 vocab 中。建议提供以下接口。
`def __init__(self, vocab, merges, special_tokens=None)`:根据给定的词汇表、merges 列表以及特殊 token 列表（可选的）构建一个分词器。该函数应接受以下参数：
```python
vocab: dict[int, bytes]
merges: list[tuple[bytes, bytes]]
special_tokens: list[str] | None = None
```

`def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None)`:类方法
该方法根据序列化的词汇表、合并列表（格式与 BPE 分词器训练代码输出一致）以及特殊词汇列表（可选的），构建并返回一个 Tokenizer。
该方法应接受以下附加参数：
```python
vocab_filepath: str  
merges_filepath: str  
special_tokens: list[str] | None = None
```
`def encode(self, text: str) -> list[int]`：将输入文本编码为整数 ID 序列。

`def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]`：给定一个可迭代对象，对象元素是字符串类型（例如 Python 文件句柄），返回一个生成器，生成 token ID，注意这里，不能一次将所有文本内容加载到内存（因为大文件无法一次性加载到内存），所以使用 yield 生成分块，逐块调用 `encode()`方法。
`def decode(self, ids: list[int]) -> str`：将 token ID 解码为文本。
errors='replace')`。
实现后，修改 `adapters.get_tokenizer`，执行 `uv run pytest tests/test_tokenizer.py `测试 `Tokenizer` 类
````

## 2.7 Experiments  

````ad-seealso
title: Problem (tokenizer_experiments):  Experiments with tokenizers
(a)从 TinyStories 和 OpenWebText 中抽样 10 份文档。使用你之前训练好的
TinyStories 和 OpenWebText 分词器（词汇量分别为 10K 和 32K），
将这些抽取的文档编码为整数 ID。每个分词器的压缩比是多少？
（字节数/token 数）？

(b)如果使用 TinyStories 分词器对 OpenWebText 抽样文本进行编码，会发生什么？
比较压缩率，并/或定性描述会发生什么。

(c) 估算你的分词器的吞吐量（例如，以字节/秒为单位）。然后将 Pile 数据集（825GB 文本）进行分词需要多长时间？

(d)使用在 TinyStories 和 OpenWebText 数据集上训练得到的分词器，将相应的训练集和
开发集编码为一组整数词标 ID。我们稍后将利用这些 ID 来训练我们的语言模型。建议将token ID 序列化为数据类型为 uint16 的 NumPy 数组。为什么 uint16 是合适的选择？

````

# 3 Transformer Language Model Architecture


这个简单，查 vocab 反向映射，拼接 bytes 后 `decode('utf-8', 