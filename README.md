# TE-Dataset：Toxic Euphemism Dataset
# Dataset Details
## Data Collection and Annotaions
- To address the lack of a standard harmful euphemism dataset, we selected Bilibili as the target platform because harmful euphemisms spread quickly and widely on the site.
- To construct our corpus, we developed an efficient web scraper using the Scrapy framework to collect video comments.
- To collect different harmful euphemisms on social media platforms, the dataset needed to cover a wide range of topics. We manually selected videos from various topics as seed videos and used the video IDs to collect corresponding comments through the web scraper.
​- To ensure the diversity and balance of the comments, the seed videos were selected from different channels on Bilibili, and the seed keywords were manually gathered based on Bilibili's search engine.
- The popularity of the videos was assessed using metrics such as views, likes, donations, comments, and shares.
- The initial seed keywords covered six topics: "Race"(“种族”), "Sexism"(“色情”), "General"(“日常”), "Gender"(“性别”), "LGBT"(“性少数群体”), and "Region"(“地区”).

- The details of the TE-dataset fields:

| Dataset fields | Description | 
|-------|---------|
| topic (话题类型) | The topic type of the comment |
| content (评论内容) | The content of the comment |
| toxic (有害标签) | The comment is toxic euphemism or not | 
| length (句子长度) | The comment length | 
| euphemism (有害委婉术语) | The toxic euphemism term in the comment |

- The samples of TE-dataset:

| topic | content | toxic | length | euphemism |
|-------|---------|-------|--------|-----------|
| sexism | 金针菇都没你细！（Enoki mushrooms are not as thin as yours!） | 1 | 8 | 金针菇（Needle mushroom） |
| gender | 你把舔狗龟男放到这个行列里可还行，自娱自乐真好玩。（Well, you put the turtle licking guy in that category, and it's fun to amuse yourself.） | 1 | 25 | 龟男(Gui Nan) |
| lgbt | 这个鸡佬，喜欢精神胜利，是一个辣鸡，真是笑死人!（This chick, who likes spiritual victory, is a spicy chicken, which is hilarious!） | 1 | 23 | 鸡佬(Ji Lao) |


- Summary of the TE-dataset：

| **Topics** | **PTE Terms Number** | **Tox.** | **N-Tox.** | **Total Number** | **Average Length** | **Minimum Length** |
|--------|------------|--------|------------|--------------|----------------|--------------|
| “种族” (Race)| 244 | 2,649    | 2,381    | 5,030      | 36.2   | 4.0  |
| “色情” (Sexism) | 134 | 4,054   | 2,020    | 6,074      | 26.7  | 2.0   |
| “日常” (General)| 57 | 593    | 560      | 1,153      | 78.8 | 2.0  |
| “性别” (Gender)| 157 | 1,319  | 1,299    | 2,618      | 36.7  | 5.0   |
| “性少数群体” (LGBT)| 96 | 887   | 921      | 1,808      | 46.3 | 4.0  |
| “地区” (Region)| 81  | 1,066  | 1,222    | 2,288      | 11.2 | 3.0  |
| **Total** | 424 | 10,568 | 8,403    | 18,971     | 33.7 | 3.33  |

## USE
- dataset download
> Method 1：download files directly
- access the Link（https://github.com/yiyepianzhounc/TED-SCL）
- click "Raw" and download （The type of files is ".csv"，and the encoding type is "utf-8"）

> Method 2：download by "Git" command

- git clone https://github.com/yiyepianzhounc/TED-SCL.git

```python
    import pandas as pd
    df = pd.read_csv("path/to/dataset.csv")
```
- train.csv: for training
- eval.csv: for evaluating
- test.csv: for testing
- eu-pairs: Annotated dictionary of toxic euphemism terms
- Note：Please read the training, test, or validation datasets first, then load the mapping dictionary of harmful euphemisms and their corresponding harmful meaning target words from the eu-pairs. After that, concatenate the dictionary to the previously loaded dataframe and proceed with further data operations.

## Citation：
article：A Toxic Euphemism Detection Framework for Online Social Network Based on Semantic Contrastive Learning and Dual Channel Knowledge Augmentation

## Contact：
- Corresponding author
Professor Haizhou Wang：
E-mail：whzh.nc@scu.edu.cn

- First Author
Gang Zhou
E-mail：1207848988@qq.com

## Explanations and Issues

To prevent privacy leakage, the following processing steps have been applied to this dataset before release:

- Special entity have been deleted or anonymized;
- Personally Identifiable Information (PII) that could be used to identify individuals (including IDs or nicknames on social media platforms) has been masked;
- The release of this dataset is for research purposes only and should not be used for any inappropriate applications;
- Please sign the "Authorization for Use of the Dataset."

## 数据集详细信息 （Chinese Version）

## 数据集如何采集的？如何标注的？
- 为了解决缺乏标准有害委婉语数据集的问题，我们选择了哔哩哔哩作为目标平台，因为该网站上有害委婉语的传播迅速且广泛。​
- 为了构建我们的语料库，我们开发了一个高效的网页爬虫，使用Scrapy框架收集视频评论。​
- 为了研究社交媒体平台上不同的有害委婉语，数据集需要涵盖广泛的话题。我们手动选择了多种话题的视频作为种子视频，并通过每个视频的ID使用网络爬虫获取相应的评论。
​- 为了确保评论的多样性和平衡性，种子视频从哔哩哔哩的不同频道中选择，基于哔哩哔哩搜索引擎的种子关键词。​
- 视频的受欢迎程度通过观看次数、点赞数、打赏数、评论数和分享数等指标进行评估。​
- 初始的种子关键词包括六个话题： “种族” (Race), “色情” (Sexism), “日常” (General), “性别” (Gender), “性少数群体” (LGBT), and “地区” (Region)。

- 数据集字段的基本情况说明：topic,content,toxic,length,euphemism

| Dataset fields | Description | 
|-------|---------|
| topic(话题类型) | 该评论的话题类型 |
| content(评论内容) | 该评论的内容 |
| toxic(有害标签) | 该评论是否为有害委婉语 | 
| length(句子长度) | 该评论的句子长度 | 
| euphemism(委婉术语) | 该评论包含的有害委婉术语|

- 数据集的示例数据：

| topic | content | toxic | length | euphemism |
|-------|---------|-------|--------|-----------|
| sexism | 金针菇都没你细！（Enoki mushrooms are not as thin as yours!） | 1 | 8 | 金针菇（Needle mushroom） |
| gender | 你把舔狗龟男放到这个行列里可还行，自娱自乐真好玩。（Well, you put the turtle licking guy in that category, and it's fun to amuse yourself.） | 1 | 25 | 龟男(Gui Nan) |
| lgbt | 这个鸡佬，喜欢精神胜利，是一个辣鸡，真是笑死人!（This chick, who likes spiritual victory, is a spicy chicken, which is hilarious!） | 1 | 23 | 鸡佬(Ji Lao) |


- 数据集的详细信息统计：

| **Topics** | **PTE Terms Number** | **Tox.** | **N-Tox.** | **Total Number** | **Average Length** | **Minimum Length** |
|--------|------------|--------|------------|--------------|----------------|--------------|
| “种族” (Race)| 244 | 2,649    | 2,381    | 5,030      | 36.2   | 4.0  |
| “色情” (Sexism) | 134 | 4,054   | 2,020    | 6,074      | 26.7  | 2.0   |
| “日常” (General)| 57 | 593    | 560      | 1,153      | 78.8 | 2.0  |
| “性别” (Gender)| 157 | 1,319  | 1,299    | 2,618      | 36.7  | 5.0   |
| “性少数群体” (LGBT)| 96 | 887   | 921      | 1,808      | 46.3 | 4.0  |
| “地区” (Region)| 81  | 1,066  | 1,222    | 2,288      | 11.2 | 3.0  |
| **Total** | 424 | 10,568 | 8,403    | 18,971     | 33.7 | 3.33  |

TE-Dataset中有害委婉语评论的平均长度和最小长度分别为33.7和3.33，​TE-Dataset中PTET的总数为424。

## 如何使用该数据集？
- 数据集下载
> 方法 1：直接下载单个文件
- 访问仓库主页（https://github.com/yiyepianzhounc/TED-SCL）
- 找到数据集文件点击下载 （数据集以.csv文件形式存在，编码方式为utf-8）
- 找点击文件名称进入文件详情页，点击右上角的 Download（或 Raw）按钮直接下载
   
> 方法 2：下载整个仓库（推荐）- 需要安装 Git。

- git clone https://github.com/yiyepianzhounc/TED-SCL.git
- 数据集导入
导入方式取决于数据格式和目标工具（如 Python、Excel 等）。以下是常见格式的示例：

```python
    import pandas as pd
    df = pd.read_csv("path/to/dataset.csv")
```
- 数据集使用
   train.csv用于模型训练，eval.csv用于模型评估，test.csv用于模型测试
   注意：请读取完训练集、测试集或验证集后，再读取eu-pairs中有害委婉语与有害含义目标词的映射字典，然后拼接到原来读取到的dataframel，再进行进一步的数据操作。

## 该数据集支持的论文发表情况：
article：A Toxic Euphemism Detection Framework for Online Social Network Based on Semantic Contrastive Learning and Dual Channel Knowledge Augmentation

## 联系方式：
- Corresponding author
Professor Haizhou Wang：
E-mail：whzh.nc@scu.edu.cn

- First Author
Gang Zhou
E-mail：1207848988@qq.com

## 为避免隐私泄露，本数据集在发布前已经进行了以下处理：
- 对特殊词汇进行了删除或匿名化；
- 屏蔽了能够定位识别个人身份的PII信息（包括：社交网络平台中的ID或昵称信息）

## 本数据集的发布仅用于研究用途，请勿用于其他不当用途；

## 在使用该数据集时，请读者注意以下事项：
- 签订《Authorization for Use of the Dataset》，严格确保不用于商业用途或其他风险。
