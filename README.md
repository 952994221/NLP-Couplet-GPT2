# 基于GPT-2的自动对对联器

## 项目文件简介

```c
config // 模型参数的文件夹
data // 训练数据文件夹
model // 预训练和生成模型文件夹
tokenizations // tokenizer文件夹，包括默认的Bert Tokenizer，分词版Bert Tokenizer以及BPE Tokenizer。
vocab // 字典文件夹
generate.py // 生成脚本
preprocess.py // 数据预处理脚本
train.py // 训练脚本
```

## 训练数据预处理

[couplet-dataset](https://github.com/wb14123/couplet-dataset)已经有开源的对联数据集，包含了77万条对联数据，其主要的数据文件位于`in.txt`和`out.txt`两个文件，前者是上联，后者是下联，数据集的格式如下：

```c
// in.txt
晚 风 摇 树 树 还 挺 
愿 景 天 成 无 墨 迹 
丹 枫 江 冷 人 初 去 
忽 忽 几 晨 昏 ， 离 别 间 之 ， 疾 病 间 之 ， 不 及 终 年 同 静 好 
闲 来 野 钓 人 稀 处 
...
```

```c
// out.txt
晨 露 润 花 花 更 红 
万 方 乐 奏 有 于 阗 
绿 柳 堤 新 燕 复 来 
茕 茕 小 儿 女 ， 孱 羸 若 此 ， 娇 憨 若 此 ， 更 烦 二 老 费 精 神 
兴 起 高 歌 酒 醉 中 
...
```

为了方便模型的训练，把`in.txt`和`out.txt`去除不可显示字符后合并得到数据集`train.txt`，其每一行为一个对联，上下联之间以`-`号分隔。注意到`train.txt`中有一些低俗和敏感的内容：

```c
// train.txt的结构也是如下所示
胡言乱语神经病-婢膝奴颜马屁精
新梅潜志三春意-老骥卧槽千里情
妓女出门访情人，来时万福，去时万福-龙王二诏求直谏，龟也得言，鳖也得言
新年移步向花海-美日躬迎胡锦涛
```

使用敏感词库`badwords.txt`对数据集进行过滤得到训练数据集

由于GPT-2是以字为单位预测的，因此需要把文本转换成数字序列，之后GPT-2模型生成预测的数字序列再转换成文字

```python
"晚风摇树树还挺"
[3241, 7599, 3031, 3409, 3409, 6820, 2923]
```

利用[Hugging Face](https://github.com/huggingface/transformers)的`BERT`预训练模型`bert-base-chinese`的字典`vocab.txt`，字典大小`21128`，同时使用它们提供的`BertTokenizer`进行文字和数字序列的相互转换：

```c
原始文字序列："晚风摇树树还挺-晨露润花花更红"
转换后的数字序列：[3241, 7599, 3031, 3409, 3409, 6820, 2923, 118, 3247, 7463, 3883, 5709, 5709, 3291, 5273, 102]
// 转换后的数字序列的最后一个多出来的102是BertTokenizer为每一段文本生成的分隔符
```

将`train.txt`的数据切分成100份`tokenized_train_i.txt`，分别对每份进行训练

## GPT-2模型简介

GPT-2是一个在海量数据集上训练的基于Transformer架构的巨大模型

### Transformer架构

Transformer是一种基于自注意力机制的Seq2Seq模型，由于Seq2Seq模型最常用在机器翻译上，因此可以用机器翻译为例子来理解Transformer架构。

#### Seq2Seq模型

不管是英文、法文还是中文，一个自然语言的句子基本上可以被视为一个有时间顺序的序列数据。而我们知道RNN很适合用来处理有时间关系的序列数据，给定一个向量序列，RNN就是输出一个一样长度的向量作为输出，这种以RNN为基础的Encoder-Decoder架构就是Seq2Seq模型。

我们把来源语言以及目标语言的句子都视为一个独立的序列以后，机器翻译事实上就是一个序列生成任务：对一个输入序列（来源语言）做些有意义的转换与处理以后，输出一个新的序列（目标语言）


Encoder跟Decoder是各自独立的RNN，Encoder把输入的句子逐词做处理最后得到的隐状态向量交给Decoder来生成目标语言。其原理在于两个句子虽然语言不同，但因为它们有相同的语义，Encoder在将整个来源语言的句子转换为一个嵌入空间中的向量后，Decoder能利用隐含在该向量中的语义信息来重新生成具有相同意思的目的语言的句子，这样的模型就像是在模拟人类做翻译的两个主要过程：

- Encoder解译来源文字的文意
- Decoder重新编译该文意至目标语言

#### Seq2Seq模型+注意力机制

基本的Seq2Seq模型里的一个重要假设是Encoder能把输入句子的语义全都压缩成一个固定维度的语义向量，之后Decoder只要利用该向量里的信息就能重新生成具有相同意义，但不同语言的句子。但很显然只有一个向量的时候，是不太可能把一个很长的句子的所有语义压缩进去的，因此需要引入注意力机制：

> 与其只把Encoder处理完句子产生的最后「一个」向量交给Decoder并要求其从中提取整句信息，不如将Encoder在处理每个词汇后所生成的「所有」输出向量都交给Decoder，让Decoder自己决定在生成新序列的时候要把「注意」放在Encoder的哪些输出向量上面。

以上就是注意力机制的中心思想：提供更多信息给Decoder，并令其自行学会该怎么提取信息。


Decoder自行学会该怎么提取信息的过程，就是其在生成序列的每个元素时都能动态地考虑自己要看哪些Encoder生成的向量，并决定从中提取多少信息。我们可以通过下图简单理解：

![](https://leemeng.tw/images/transformer/attention_mechanism_luong.jpg)

左右两边分别是Encoder与Decoder ，纵轴则是多层的神经网络层，Encoder一共会生成4个隐状态向量，在Decoder的每个序列生成时，其会进行如下过程：

1. 使用Decoder当前的红色隐状态向量$h_t$和Encoder所有蓝色隐状态向量$h_s$做比较，利用$score$函式计算出$h_t$对每个$h_s$的注意程度
2. 以此注意程度为权重，加权平均所有Encoder的隐状态向量$h_s$以取得上下文向量$context\ vector$
3. 将此上下文向量与Decoder的隐状态向量结合成一个注意向量$attention\ vector$，并作为该时间的输出
4. 该注意向量会作为Decoder下个时间点的输入

即如下公式

~~由于github不支持公式，后面的公式只能以源码显示了~~

<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{align}&space;&&space;\alpha_{ts}=\frac&space;{exp(score(\boldsymbol&space;h_t,\boldsymbol&space;{\overline&space;h_s}))}{\sum_{s'=1}^S&space;exp(score(\boldsymbol&space;h_t,\boldsymbol&space;{\overline&space;h_s})}\qquad&space;[Attention\&space;weight]&space;\\&space;&&space;\boldsymbol&space;c_t=\sum_s&space;\alpha_{ts}&space;\boldsymbol&space;{\overline&space;h_s}&space;\qquad&space;[Context\&space;vector]&space;\\&space;&&space;\boldsymbol&space;\alpha_t=f(\boldsymbol&space;c_t,\boldsymbol&space;h_t)=tanh(\boldsymbol&space;W_c[\boldsymbol&space;c_t;\boldsymbol&space;h_t])&space;\qquad&space;[Attention\&space;vector]&space;\end{align}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{align}&space;&&space;\alpha_{ts}=\frac&space;{exp(score(\boldsymbol&space;h_t,\boldsymbol&space;{\overline&space;h_s}))}{\sum_{s'=1}^S&space;exp(score(\boldsymbol&space;h_t,\boldsymbol&space;{\overline&space;h_s})}\qquad&space;[Attention\&space;weight]&space;\\&space;&&space;\boldsymbol&space;c_t=\sum_s&space;\alpha_{ts}&space;\boldsymbol&space;{\overline&space;h_s}&space;\qquad&space;[Context\&space;vector]&space;\\&space;&&space;\boldsymbol&space;\alpha_t=f(\boldsymbol&space;c_t,\boldsymbol&space;h_t)=tanh(\boldsymbol&space;W_c[\boldsymbol&space;c_t;\boldsymbol&space;h_t])&space;\qquad&space;[Attention\&space;vector]&space;\end{align}" title="\begin{align} & \alpha_{ts}=\frac {exp(score(\boldsymbol h_t,\boldsymbol {\overline h_s}))}{\sum_{s'=1}^S exp(score(\boldsymbol h_t,\boldsymbol {\overline h_s})}\qquad [Attention\ weight] \\ & \boldsymbol c_t=\sum_s \alpha_{ts} \boldsymbol {\overline h_s} \qquad [Context\ vector] \\ & \boldsymbol \alpha_t=f(\boldsymbol c_t,\boldsymbol h_t)=tanh(\boldsymbol W_c[\boldsymbol c_t;\boldsymbol h_t]) \qquad [Attention\ vector] \end{align}" /></a>

#### Transformer

Transformer是基于自注意力机制的Seq2Seq模型。在上一节中我们仍然使用RNN对序列处理，这就导致无法有效地平行运算，要得到某个时间点的隐状态向量必须计算其之前的所有向量，例如有以下输入序列：

```
[a1, a2, a3, a4]
```

要获得最后一个时间点`a4`的输出向量`b4`必须把整个序列都计算一遍才行。

Transformer参考注意力机制提出了自注意力机制，使得其既可以处理序列数据，也可以平行运算。自注意机制利用矩阵运算在RNN的一个时间点内就能计算出所有`bi`，且每个`bi`都包含了整个输入序列的信息。现在我们来看下Transformer的架构，其Encoder和Decoder都是由多个block组成的，Encoder的输出会作为Decoder的输入，如下图所示：

![img](http://jalammar.github.io/images/t/The_transformer_encoder_decoder_stack.png)

Encoder-Decoder结构如下图所示

![img](http://jalammar.github.io/images/t/Transformer_decoder.png)

在上图中我们可以看到，Decoder利用注意力机制关注Encoder的输出序列（Encoder-Decoder Attention），而Encoder和Decoder各自利用自注意力机制关注自己处理的序列（Self-Attention），无法平行运算的RNN完全消失。我们可以对Transformer的自注意机制总结如下：

- Encoder在生成输入元素的隐状态向量时关注自己序列中的其他元素，从中获得上下文信息
- Decoder在生成输出元素的隐状态向量时关注Encoder的输出序列，从中获得上下文信息
- Decoder在生成输出元素的隐状态向量时关注自己序列中的其他元素，从中获得上下文信息

**Self-Attention**

Self-Attention的计算公式如下，$Z$代表一个特征矩阵
$$
Z=Attention(Q,K,V)=softmax(\frac{QK^T}{\sqrt d_k})V
$$
在Self-Attention中，每个单词有三个不同的向量：查询向量（$Q$），键向量（$K$），值向量（$V$），它们分别由嵌入词向量$X$与三个权值矩阵$W^Q,W^K,W^V$相乘得到，它们的作用如下：

1. 查询向量：当前单词的查询向量被用来和其它单词的键向量相乘，从而得到其它词相对于当前词的注意力得分。
2. 键向量：键向量就像是序列中每个单词的标签，它是搜索相关单词时用来匹配的对象。
3. 值向量：值向量是单词真正的表征，当计算出注意力得分后，使用值向量进行加权求和得到能代表当前位置上下文的向量。

![img](http://jalammar.github.io/images/t/transformer_self_attention_vectors.png)

在$Attention$的计算公式中，整个过程如下：

1. 将每个输入单词转换为嵌入向量$X$
2. 根据$X$和$W^Q,W^K,W^V$得到$q,k,v$三个向量
3. 为每个向量计算一个score：$score(q \cdot k)$，并归一化即除以$\sqrt d_k$
4. 对score加上激活函数$softmax$，得到其它词相对于当前词的注意力得分
5. $softmax$再点乘向量$v$，得到需要从每个单词中获取的信息
6. 所有向量$v$相加得到向量$z=\sum v$

实际计算时是以矩阵形式计算的：

![img](http://jalammar.github.io/images/t/self-attention-matrix-calculation-2.png)

对比之前的Seq2Seq模型+注意力机制的注意力计算公式，我们可以看到之前的三个公式合并成了一个公式，不再有RNN的平行计算的问题。

**Encoder-Decoder Attention**

Encoder-Decoder Attention的计算过程和Self-Attention完全一样，只不过$Q$来自上一个Decoder的输出，$K,V$来自Encoder的输出。

**Feed Forward**

Feed Forward模块有两层，第一层的激活函数是$ReLU$，第二层是一个线性激活函数，该模块的计算公式可以表示为：
$$
FFN(Z)=max(0,ZW_1+b_1)W_2+b_2
$$
**Position Embedding**

在上面的模型中，Transformer并没有处理序列的位置信息，换句话它只是一个功能更强大的词袋模型而已，无论句子的结构怎么打乱得到的结果是相同的，因此Transformer还引入了位置编码，计算公式如下：
$$
PE(pos,2=\sin (\frac{pos}{10000^{\frac {2i} {d_{model}}}}) \\
PE(pos,2i+1)=\cos (\frac{pos}{10000^{\frac {2i} {d_{model}}}})
$$

### GPT-2

在理解了Transformer架构后，GPT-2模型就非常好理解了，因为GPT-2模型只使用了Transformer架构的Decoder部分，其训练目标为一般的语言模型，通过前文预测下个字。

![img](https://pic3.zhimg.com/80/v2-32335efcbb995a12c1da12d0edeccc5e_720w.jpg)



此外原来Transformer的Decoder中的Self-Attention变成了Masked Self-Attention：

![img](https://pic3.zhimg.com/80/v2-19720b1c70a294558dc9456477156b06_720w.jpg)

两者的区别在于Self-Attention模块允许一个单词看到序列的全部单词，而Masked Self-Attention模块只允许单词看到其左侧的单词信息，这也是符合GPT-2的训练目标的：通过前文预测下个字。

![img](https://pic4.zhimg.com/80/v2-8649a1552b21ee3c283c2952649ec64b_720w.jpg)

当最后一个Decoder产生输出后，输出向量乘上嵌入矩阵可以得到词汇表中每个单词对应的注意力得分，之后对于词汇表中得分较高的一部分单词（通过top-k参数设置单词数量），将它们的得分作为概率从整个单词列表中进行抽样，最终得到输出的单词。

![img](https://pic4.zhimg.com/80/v2-0871db72d018b09d71d90b31c2d1362f_720w.jpg)

### 模型训练

可以直接使用`transformers`模型库中封装好的`GPT2LMHeadModel`预训练模型，在此基础上使用对联数据集进行fine-tune训练

`Tokenizer`和字典使用数据预处理部分提到的`BertTokenizer`以及对应字典`vocab.txt`

## 训练结果

`generate.py`已经配置好参数，直接使用`python generate.py`运行即可。

选取测试数据`data/test/in.txt`中的部分上联对模型测试，每个上联生成5个下联，所得结果如下：

<table>
    <tr>
        <th>上联</th>
        <th>生成的下联</th>
        <th>实际下联</th>
    </tr>
    <tr>
        <td rowspan=5>腾飞上铁，锐意改革谋发展，勇当千里马</td>
        <td>高奏凯歌，倾情频亮续创新，争当领头羊</td>
        <td rowspan=5>和谐南供，安全送电保畅通，争做领头羊</td>
    </tr>
    <tr>
        <td>实干惠民，倾心从小创和谐，再做领头羊</td>
    </tr>
    <tr>
        <td>崛起扬帆，同心圆梦创和谐，再做领头羊</td>
    </tr>
    <tr>
        <td>开放下交，倾心献力倡文明，争做领头羊</td>
    </tr>
    <tr>
        <td>奋起中华，倾情学术建和谐，争当领头羊</td>
    </tr>
    <tr>
        <td rowspan=5>兴漫香江红袖舞</td>
        <td>龙腾粤海碧云飞</td>
        <td rowspan=5>廉生清远浊泉吟</td>
    </tr>
    <tr>
        <td>魂迷野径绿林稀</td>
    </tr>
    <tr>
        <td>魂迷宝石白莲陶</td>
    </tr>
    <tr>
        <td>寒生宝殿紫霞腾</td>
    </tr>
    <tr>
        <td>魂牵故里彩云飞</td>
    </tr>
        </tr>
    <tr>
        <td rowspan=5>煦煦春风，吹暖五湖四海</td>
        <td>浩浩碧波，荡涤万水千山</td>
        <td rowspan=5>霏霏细雨，润滋万户千家</td>
    </tr>
    <tr>
        <td>盈盈秋水，映红万户千家</td>
    </tr>
    <tr>
        <td>婷婷娇女，妆成万水千山</td>
    </tr>
    <tr>
        <td>啾啾燕语，唤回万户千家</td>
    </tr>
    <tr>
        <td>盈盈笑语，映红万紫千红</td>
    </tr>
    </tr>
        </tr>
        <tr>
        <td rowspan=5>共工触山天裂崩，乾坤竞赌</td>
        <td>武警报海底扬帆，日月争光</td>
        <td rowspan=5>女娲炼石宇凝淬，灵曜决押</td>
    </tr>
    <tr>
        <td>众智脱日月平息，今古一家</td>
    </tr>
    <tr>
        <td>武警获日犬声寂，日月双伤</td>
    </tr>
    <tr>
        <td>夺江川水日缸混，昼夜兴豪</td>
    </tr>
    <tr>
        <td>众志成钢铁焕彩，生死繁荣</td>
    </tr>
        <tr>
        <td rowspan=5>奥运会树新风，五环焕彩中国印</td>
        <td>鸟巢唱和凤城，四海欢歌奥运情</td>
        <td rowspan=5>长白山迎圣火，万众放歌阿里郎</td>
    </tr>
    <tr>
        <td>鸟巢欣闻声誉，四海欢歌奥运歌</td>
    </tr>
    <tr>
        <td>北京城演今事，四海欢呼古贝春</td>
    </tr>
    <tr>
        <td>北京城擂盛会，四海欢歌迎客情</td>
    </tr>
    <tr>
        <td>神州天享文运，四海欢歌少年情</td>
    </tr>
        <tr>
        <td rowspan=5>官廉政善江河丽</td>
        <td>民富国强气象新</td>
        <td rowspan=5>国泰民安岁月新</td>
    </tr>
    <tr>
        <td>国富民强日月新</td>
    </tr>
    <tr>
        <td>国富民强气象新</td>
    </tr>
    <tr>
        <td>民富邦兴日月新</td>
    </tr>
    <tr>
        <td>国富民强气象新</td>
    </tr>
        <tr>
        <td rowspan=5>旗展太空，九天圆梦</td>
        <td>党伸正气，四海同心</td>
        <td rowspan=5>功成奥运，四海同春</td>
    </tr>
    <tr>
        <td>心怀大宇，双手推标</td>
    </tr>
    <tr>
        <td>功成奥运，七载摘金</td>
    </tr>
    <tr>
        <td>心怀玉宇，一醉成诗</td>
    </tr>
    <tr>
        <td>声出东海，八月兴潮</td>
    </tr>
        <tr>
        <td rowspan=5>春风送喜财入户</td>
        <td>瑞雪迎新喜临门</td>
        <td rowspan=5>岁月更新福满门</td>
    </tr>
    <tr>
        <td>岁月迎新福临门</td>
    </tr>
    <tr>
        <td>岁月迎新梦飞天</td>
    </tr>
    <tr>
        <td>岁月迎新福临门</td>
    </tr>
    <tr>
        <td>岁月迎新福临门</td>
    </tr>
</table>

