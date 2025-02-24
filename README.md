TE-Dataset(Toxic Euphemism Dataset)

1、数据集如何采集的？如何标注的？
为了解决缺乏标准有害委婉语数据集的问题，我们选择了哔哩哔哩作为目标平台，因为该网站上有害委婉语的传播迅速且广泛。​
为了构建我们的语料库，我们开发了一个高效的网页爬虫，使用Scrapy框架收集视频评论。​
为了研究社交媒体平台上不同的有害委婉语，数据集需要涵盖广泛的话题。我们手动选择了多种话题的视频作为种子视频，并通过每个视频的ID使用网络爬虫获取相应的评论。
​为了确保评论的多样性和平衡性，种子视频从哔哩哔哩的不同频道中选择，基于哔哩哔哩搜索引擎的种子关键词。​
视频的受欢迎程度通过观看次数、点赞数、打赏数、评论数和分享数等指标进行评估。​
初始的种子关键词包括六个话题： “种族” (Race), “色情” (Sexism), “日常” (General), “性别” (Gender), “性少数群体” (LGBT), and “地区” (Region)。

2、数据集的示例数据：
TE-Dataset中有害委婉语评论的平均长度和最小长度分别为33.7和3.33，​TE-Dataset中PTET的总数为424。
数据集字段的基本情况说明：topic,content,toxic,length,euphemism
| Dataset fields | summarize | 
|-------|---------|
| topic(话题类型) | 该评论的话题类型是什么 |
| content(评论内容) | 该评论的内容是什么 |
| toxic(有害标签) | 该评论是否为有害委婉语 | 
| length(句子长度) | 该评论的句子长度 | 
| euphemism(委婉术语) | 该评论包含的有害委婉术语是什么？|

| topic | content | toxic | length | euphemism |
|-------|---------|-------|--------|-----------|
| sexism | 金针菇都没你细！（Enoki mushrooms are not as thin as yours!） | 1 | 8 | 金针菇 |
| gender | 你把舔狗龟男放到这个行列里可还行，自娱自乐真好玩。（Well, you put the turtle licking guy in that category, and it's fun to amuse yourself.） | 1 | 25 | 龟男 |
| lgbt | 这个鸡佬，喜欢精神胜利，是一个辣鸡，真是笑死人!（This chick, who likes spiritual victory, is a spicy chicken, which is hilarious!） | 1 | 23 | 鸡佬 |

3、如何使用该数据集？
1）数据集下载：
2）数据集导入：
3）数据集统计：


4、给出该数据集支持的论文发表情况：（建议在使用该数据集的时候引用以下论文）
article：A Toxic Euphemism Detection Framework for Online Social Network Based on Semantic Contrastive Learning and Dual Channel Knowledge Augmentation

5、联系方式（可长期使用的邮箱）：
whzh.nc@scu.edu.cn

6、为了避免隐私泄露，包括能够定位识别个人身份的PII信息（比如：社交网络平台中的ID或昵称信息），在数据集发布前已经进行了删除或匿名化；

7、本数据集的发布在理论上不存在其他安全风险或伦理问题，但是该数据集仅用于研究用途；

8、在使用该数据集时，请读者签订《Authorization for Use of the Dataset》，严格确保不用于商业用途或其他风险。
