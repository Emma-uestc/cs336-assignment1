> 写在前面，如果对 cs336_assignment1_basics.pdf 理解有疑问的，可以参考 [[assigment1_overview&bpe_basics]] 我对文档的翻译（部分解释细节）
## 环境配置

工欲善其事必先利其器，所以写代码前需要先配置环境。课程已经为我们提供了 uv 配置，我们只需要按照以下步骤执行

1. 进入作业目录并同步依赖

这一步会下载并安装所有依赖（包括 PyTorch）。因为文件较大，请耐心等待直到进度条跑完。

```bash

cd assignment1-basics

uv sync
```

![](https://emma-uestc.github.io/cs336-assignment1/notes/_assets/Pasted-image-20260514110651.png)

2. 激活虚拟环境

`uv` 会在当前目录下创建一个 `.venv` 文件夹。你需要激活它，才能让终端使用刚才安装的 PyTorch。

```bash

source .venv/bin/activate

```

*(激活成功后，你的命令行提示符前面通常会出现 `(assignment1-basics)` 或 `(.venv)` 字样)*

3. 验证环境（关键）

执行下面这行代码，检查 PyTorch 是否安装成功，以及是否能识别你的 RTX 2060 显卡。

```bash

python -c "import torch; print(f'Torch Version: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'Device Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

```

**预期输出：**

如果一切顺利，你应该能看到类似这样的输出：

*   Torch Version: 2.6.0...

*   CUDA Available: **True**

*   Device Name: NVIDIA GeForce RTX 2060...

![](https://emma-uestc.github.io/cs336-assignment1/notes/_assets/Pasted-image-20260514111345.png)

运行代码可以先 `source` 激活环境，然后在该环境内执行 `python <python_file_path>`，

当然，你也可以像 `README` 文档中提到的那样需要运行代码时 `uv run <python_file_path>`
这是 `uv` 的“懒加载”模式：

- 如果环境不存在 → 自动创建
- 如果依赖没装 → 自动安装
- 如果 lock 文件变化 → 自动同步
- 然后直接运行脚本

## 下载数据

``` sh

mkdir -p data

cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz

gunzip owt_train.txt.gz

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz

gunzip owt_valid.txt.gz  

cd ..

```

## Train BPE 实现

按照 [[assigment1_overview&bpe_basics#2.4 BPE Tokenizer Training]] 分析知，我们的训练逻辑如下：

```python
class BPETrainer_Naive:

    def __init__(self):

        self.pattern =  r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        self.vocab: Dict[int, bytes] = {}

        self.merges: List[Tuple[bytes, bytes]] = []  

    def train(self,

        input_path: str,

        vocab_size: int) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:

        """

        Args:

            input_path: str, the path of the input file

            vocab_size: int, the size of the vocabulary

        Returns:

            vocabulary: Dict[int, bytes], the vocabulary of the BPE tokenizer

        """

        # TODO 1. Initialize the vocabulary

        # TODO 2. Pre-tokenize the text

        # TODO 3. Initialize the word encodings

        # TODO 4. BPE training loop
```

### **初始化词汇表**
初始化基础词汇表 `[0,255]`

```PYTHON
# 使用一个全局变量 N_BYTES
N_BYTES = 256
        # basic vocabulary [0,255]

        self.vocab = {i: bytes([i]) for i in range(N_BYTES)}

        # add special tokens to vocabulary

        size = N_BYTES

        for token in special_tokens:

            self.vocab[size] = token.encode('utf-8')

            size += 1
```

### **预分词实现**

```python
    def pretokenize(self, input_path: str, special_tokens: List[str] = ["<|endoftext|>"]) -> Dict[str, int]:

        """
        Args:

            input_path: str, the path of the input file

            special_token: List[str]

        Returns:
            word_counts: Dict[str, int], the frequency of each word, exclude special tokens

        """

        word_counts = defaultdict(int)

        with open(input_path, 'r', encoding='utf-8') as f:

            text = f.read()

        if special_tokens:

            escape_special_tokens = "|".join(re.escape(t) for t in special_tokens)

            chunks = re.split(escape_special_tokens, text)

        else:

            chunks = [text]

        for chunk in chunks:

            for match in re.finditer(self.pattern, chunk):

                word_counts[match.group(0)] += 1

        return dict(word_counts)
```

测试验证预分词正确性：

```python
if __name__ == "__main__":

    input_path = 'tests/fixtures/tinystories_sample.txt'

    special_tokens = ["<|endoftext|>"]

    if not os.path.exists(input_path):

        raise FileNotFoundError(f'Input file not found: {input_path}')

    else:

        print(f'Testint pretokenize ....')

        trainer = BPETrainer_Naive()

  

        word_counts = trainer.pretokenize(input_path, special_tokens)

        print(f'\nFound {len(word_counts)} unique words.')

  

        sorted_words_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

        print(f'Top 10 most frequent words:')

        for word, count in sorted_words_counts[:10]:

            print(f'{word}: {count}')
```

查看结果

```bash
Testint pretokenize ....

Found 274 unique words.
Top 10 most frequent words:
.: 64
,: 39
 the: 28

: 25
 a: 24
 and: 22
 was: 21
 to: 21
 with: 12
 his: 11
```

## 初始化 word encodings

```python
        # TODO 2. Pre-tokenize the text

        word_counts = self.pretokenize(input_path, special_tokens)

        # TODO 3. Initialize the word encodings

        word_encodings = {word: list(word.encode('utf-8')) for word in word_counts}
```

![](https://emma-uestc.github.io/cs336-assignment1/notes/_assets/Pasted-image-20260515140052.png)
## 迭代训练   
到了我们的核心训练环节了，由 [[assigment1_overview&bpe_basics#2.4 BPE Tokenizer Training]] 分析，该循环分为以下步骤：
a. 建立一个大字典，统计相邻字节对频率
遍历 encoding，建立一个字节对频率统计字典
我们写一个统计字节对的函数，`count_pairs`
**Input**
`word_counts`: `dict[str, int]`，预分词的结果，
`word_encodings`: `dict[str, List[int]]`, 初始的 word encoding
Output
pair_counts: dict[tuple[int,int], int]: 相邻字节对频率

```python
    def count_pairs(self,

        word_counts: Dict[str, int],

        word_encodings: Dict[str, List[int]]

        ) -> Dict[Tuple[int, int], int]:

        """

        Args:

            word_counts: the frequency of each word got from pretokenize

            word_encodings: the initial encoding

        Returns:

            pair_counts: the bytes pair counts

        """

        pair_counts = defaultdict(int)
        for word, count in word_counts.items():

            encoding = word_encodings[word]

            for i in range(len(encoding) - 1):

                pair = (encoding[i], encoding[i+1])

                pair_counts[pair] += count

        return dict(pair_counts)
```

![](https://emma-uestc.github.io/cs336-assignment1/notes/_assets/Pasted-image-20260515140355.png)

b.迭代合并
  1. 寻找出现最频繁的字节对
  2. 合并为新的 token
  3. 对所有包含这个字节对的 encoding 进行 merge 操作，更新 word_encodings
  4. 更新 vocab
  由于，后面还会对 merge 操作进行优化，为了统一，我们还是采用模块化设计，这里也将 merge 操作使用函数实现。  

朴素版的完整实现如下：

```python
"""

Implement a naïve BPE trainer

"""
N_BYTES = 256

vocab_size = 1024

from ctypes import sizeof

import regex as re

from collections import defaultdict

from typing import Dict, List, Tuple

import os

  
class BPETrainer_Naive:

    def __init__(self):

        self.pattern =  r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        self.vocab: Dict[int, bytes] = {}

        self.merges: List[Tuple[bytes, bytes]] = []
 

    def train(self,

        input_path: str,

        vocab_size: int,

        special_tokens: List[str]

        ) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:

        """

        Args:

            input_path: str, the path of the input file

            vocab_size: int, the size of the vocabulary

        Returns:

            vocabulary: Dict[int, bytes], the vocabulary of the BPE tokenizer

        """

        # TODO 1. Initialize the vocabulary

        # basic vocabulary [0,255]

        self.vocab = {i: bytes([i]) for i in range(N_BYTES)}

        # add special tokens to vocabulary

        size = N_BYTES

        for token in special_tokens:

            self.vocab[size] = token.encode('utf-8')

            size += 1

        # TODO 2. Pre-tokenize the text

        word_counts = self.pretokenize(input_path, special_tokens)

        # TODO 3. Initialize the word encodings

        word_encodings = {word: list(word.encode('utf-8')) for word in word_counts}

        # TODO 4. BPE training loop

        num_merges = vocab_size - size

        for merge_idx in range(num_merges):

            # a. find the max count pair to be merged

            pair_counts = self.count_pairs(word_counts, word_encodings)

            if not pair_counts:

                print('No more pairs to merge, quit')

                break

            merge_pair = max(

                pair_counts,

                key=lambda x: (pair_counts[x], self.vocab[x[0]], self.vocab[x[1]])

            )

            max_count = pair_counts[merge_pair]

            # b. merge and update the word encodings

            token_id = size

            for word in word_encodings:

                new_encoding = self.merge_encoding(word_encodings[word], merge_pair, token_id)

                word_encodings[word] = new_encoding

            # c. update the merges

            self.merges.append((self.vocab[merge_pair[0]], self.vocab[merge_pair[1]]))

            # d. update vocabulary

            self.vocab[token_id] = self.vocab[merge_pair[0]] + self.vocab[merge_pair[1]]

            size += 1

  

        return self.vocab, self.merges

  

    def merge_encoding(self,

        encoding: List[int],

        merge_pair: Tuple[int, int],

        token_id: int

        ) -> List[int]:

  

        new_encoding = []

        p0, p1 = merge_pair

  

        i = 0

        while i < len(encoding):

            if i < len(encoding) - 1 and encoding[i] == p0 and encoding[i+1] == p1:

                new_encoding.append(token_id)

                i += 2

            else:

                new_encoding.append(encoding[i])

                i += 1

        return new_encoding

  

    def count_pairs(self,

        word_counts: Dict[str, int],

        word_encodings: Dict[str, List[int]]

        ) -> Dict[Tuple[int, int], int]:

        """

        Args:

            word_counts: the frequency of each word got from pretokenize

            word_encodings: the initial encoding

        Returns:

            pair_counts: the bytes pair counts

        """

        pair_counts = defaultdict(int)

  

        for word, count in word_counts.items():

            encoding = word_encodings[word]

            for i in range(len(encoding) - 1):

                pair = (encoding[i], encoding[i+1])

                pair_counts[pair] += count

        return dict(pair_counts)

  
  

    def pretokenize(self, input_path: str, special_tokens: List[str] = ["<|endoftext|>"]) -> Dict[str, int]:

        """

        Args:

            input_path: str, the path of the input file

            special_token: List[str]

        Returns:

            word_counts: Dict[str, int], the frequency of each word, exclude special tokens

        """

  

        word_counts = defaultdict(int)

  

        with open(input_path, 'r', encoding='utf-8') as f:

            text = f.read()

  

        if special_tokens:

            escape_special_tokens = "|".join(re.escape(t) for t in special_tokens)

            chunks = re.split(escape_special_tokens, text)

        else:

            chunks = [text]

        for chunk in chunks:

            for match in re.finditer(self.pattern, chunk):

                word_counts[match.group(0)] += 1

        return dict(word_counts)

```

![](https://emma-uestc.github.io/cs336-assignment1/notes/_assets/Pasted-image-20260515153124.png)
框的是 ID 为 256 的特殊 token.
![](https://emma-uestc.github.io/cs336-assignment1/notes/_assets/Pasted-image-20260515153652.png)
### **测试**  
修改 `adapters.run_train_bpe`，添加我们的实现，然后执行 `uv run pytest tests/test_train_bpe.py`
符合预期，时间复杂度高，超时,也是我们接下来要优化的目标

```bash
_____________________________________________________________________________ test_train_bpe_speed ______________________________________________________________________________

    def test_train_bpe_speed():
        """
        Ensure that BPE training is relatively efficient by measuring training
        time on this small dataset and throwing an error if it takes more than 1.5 seconds.
        This is a pretty generous upper-bound, it takes 0.38 seconds with the
        reference implementation on my laptop. In contrast, the toy implementation
        takes around 3 seconds.
        """
        input_path = FIXTURES_PATH / "corpus.en"
        start_time = time.time()
        _, _ = run_train_bpe(
            input_path=input_path,
            vocab_size=500,
            special_tokens=["<|endoftext|>"],
        )
        end_time = time.time()
>       assert end_time - start_time < 1.5
E       assert (1778838328.7780395 - 1778838326.5302143) < 1.5

tests/test_train_bpe.py:24: AssertionError
============================================================================ short test summary info ============================================================================
FAILED tests/test_train_bpe.py::test_train_bpe_speed - assert (1778838328.7780395 - 1778838326.5302143) < 1.5
```

![](https://emma-uestc.github.io/cs336-assignment1/notes/_assets/Pasted-image-20260515175324.png)