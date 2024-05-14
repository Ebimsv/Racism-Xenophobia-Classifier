![alt text](https://github.com/Ebimsv/Racism-Xenophobia-Classifier/blob/main/pics/Logo.gif)

# About This Project
This project is a step-by-step guide on building a Racism-Xenophobia Classifier using PyTorch. It aims to provide a comprehensive understanding of the process involved in developing a model and its applications.

# Step 1: Accurate and concise definition of the problem
The Racism-Xenophobia-Classifier repository is a machine learning project focused on developing a classifier to detect instances of racism and xenophobia in English sentences. This project aims to provide a robust and accurate tool for identifying and categorizing text based on the presence of racism and xenophobic content.

The Racism-Xenophobia-Classifier project has diverse real-world applications. It can be employed for content moderation on social media platforms, aiding sentiment analysis by identifying racism and xenophobia, monitoring public opinion on these issues, supporting research and studies on societal attitudes, informing policy development, and serving as an educational tool for fostering inclusivity. Overall, the classifier contributes to creating safer online spaces and promoting understanding and respect in society.

# Step 2: Data Collection for Racism-Xenophobia-Classifier

## Data Collection Overview
In the data collection phase of the Racism-Xenophobia-Classifier project, the goal is to gather a diverse and representative dataset of English sentences labeled with instances of racism and xenophobia. This dataset will serve as the foundation for training and evaluating the classifier.

### Sampling Methods

![alt text](https://github.com/Ebimsv/Racism-Xenophobia-Classifier/blob/main/pics/sampling.png)
(https://github.com/Ebimsv/Torch-Linguist/blob/main/pics/RNN.png)

Sampling methods can be utilized during data collection to ensure the dataset captures a wide range of examples and maintains a balanced representation. Here are a few scenarios where sampling methods can be beneficial:

<details>
  <summary><b>1. Random Sampling</b></summary><br/>
Random sampling involves selecting data points from a larger pool without any specific pattern or bias. It ensures a diverse representation of text by capturing a wide range of examples. For the Racism-Xenophobia-Classifier project, random sampling can be used to collect sentences from various sources to avoid favoring specific contexts or demographics.  

**Advantages**
- Easy to implement.
- Each member of the population has an equal chance of being chosen.
- Free from bias.

**Disadvantages**
- If the sampling frame is large random sampling may be impractical.
- A complete list of the population may not be available.
- Minority subgroups within the population may not be present in sample.
</details>

<details>
  <summary><b>2. Stratified Sampling</b></summary><br/>
The population is divided into subgroups (strata) based on specific characteristics, such as age, gender or race. Within the strata random sampling is used to choose the sample. In the context of the Racism-Xenophobia-Classifier project, stratified sampling can be used to ensure proportional representation of different types of racism and xenophobia, such as racial slurs, discriminatory remarks, or xenophobic comments.

![alt text](https://github.com/Ebimsv/Racism-Xenophobia-Classifier/blob/main/pics/stratified-sampling.png)

**Advantages**
- Strata can be proportionally represented in the final sample.
- It is easy to compare subgroups.

**Disadvantages**
- Information must be gathered before being able to divide the population into subgroups.

#### Worked Example
A school of 1000 students are classified as follows:

57 % Brunette,  
29 % Redhead,  
14 % Blonde.  

Find a stratified sample of 200 students for this population.

**Solution:**  
Suppose we are interested in how each of these groups will react to this statement: everyone in this school has an equal chance of success. Relying on a random sample may under-represent the minority populations of the school (people with blonde hair). By grouping our population by hair colour, we can choose a sample ensuring each group is represented according to its proportion of the population. So 57 % of the sample should be brunette, 29 % should be redhead and 14 % blonde. Within each group (strata) you select your sample randomly. As our sample consists of 200 people, 114 should be brunette, 58 should be redhead and 28 should be blonde.
</details>

<details>
  <summary><b>3. Clustered Sampling</b></summary><br/>
Clustered sampling involves dividing the population into clusters or groups, and then randomly selecting clusters for data collection. In the Racism-Xenophobia-Classifier project, clustered sampling can be used to select specific online communities, forums, or news articles that are more likely to contain instances of racism and xenophobia, ensuring a more focused collection of relevant data.

![alt text](https://github.com/Ebimsv/Racism-Xenophobia-Classifier/blob/main/pics/clustered-sampling.png)

**Advantages**
- Cuts down the cost and time by collecting data from only a limited number of groups.  
- Can show grouped variations.  

**Disadvantages**
- It is not a genuine random sample.  
- The sample size is smaller and from thus the sample is likely to be less representative of the population.  

**Example**  
The children in a classroom are divided up depending on which table they sit at. A sample can be obtained from this classroom by choosing **n** number of tables to represent the class.
</details>

<details>
  <summary><b>4. Convenience Sampling</b></summary><br/>
Convenience sampling involves collecting data from readily available sources or individuals that are easily accessible. In the context of the Racism-Xenophobia-Classifier project, convenience sampling may involve collecting data from social media platforms, online forums, or public discussions where instances of racism and xenophobia are frequently observed.
</details>

By applying these data collection methods to the Racism-Xenophobia-Classifier project, we can gather a diverse and representative dataset that covers various types of racism and xenophobia, captures informative examples, and avoids biases or limited perspectives.

### Collecting and Organizing Data

Here are the steps of collecting data for the Racism-Xenophobia-Classifier project:

Data collection is the initial phase where textual content related to racism and xenophobia is gathered from various sources. The sources can be diverse, including social media platforms, online forums, news articles, blogs, and more, depending on the objectives of the text classification project. The goal is to compile a dataset that is representative of different types of racist and xenophobic statements, as well as non-racist and non-xenophobic content.

**Source Diversity**: It's important to collect data from a variety of sources to ensure the dataset covers a wide range of linguistic styles, formats, and contexts in which racism and xenophobia can manifest. This diversity helps in building a robust model capable of accurately understanding and classifying racist and xenophobic texts across different scenarios, such as online discussions, news reports, and personal narratives.  

**Domain-Specific Data**: While the project focuses on detecting racism and xenophobia in general, it may be beneficial to gather data from specific domains where these issues are prevalent, such as political discourse, social commentary, or historical accounts. This ensures that the model is trained on language and terminology specific to these domains, enhancing its accuracy and relevance in identifying racist and xenophobic statements in those contexts.

Here are different types of data collection methods commonly used:

1. **Surveys**: Surveys involve collecting data through a set of structured questions administered to individuals or groups. They can be conducted through various mediums such as online forms, telephone interviews, or in-person interviews. Surveys provide quantitative or qualitative information depending on the type of questions asked.

2. **Interviews**: Interviews involve direct interaction with individuals or groups to gather information. They can be structured (where specific questions are asked) or unstructured (more conversational), and can be conducted face-to-face, over the phone, or through video calls. Interviews provide in-depth insights and allow for follow-up questions.

3. **Observations**: Observations involve systematically watching and recording behaviors, events, or phenomena. Researchers may be passive observers, simply recording what they see, or they may engage in participant observation, actively participating in the observed activities. Observations can provide rich contextual information but may be influenced by the observer's presence.

4. **Experiments**: Experiments involve manipulating variables to study cause-and-effect relationships. Data is collected under controlled conditions, often with a control group for comparison. Experiments are commonly used in scientific research to establish causal relationships between variables.

5. **Existing Data Analysis**: Involves using pre-existing data collected for other purposes. This can include analyzing publicly available datasets, using data collected by government agencies or research institutions, or utilizing archival data. It provides a cost-effective way to answer research questions without collecting new data.

6. **Case Studies**: Case studies involve in-depth and holistic analysis of a particular individual, group, organization, or phenomenon. They typically involve multiple data collection methods, such as interviews, observations, and document analysis. Case studies provide detailed insights into specific contexts or situations.

7. **Document Analysis**: Document analysis involves the systematic examination of written, visual, or audio materials, such as **reports**, **articles**, **speeches**, or **social media** content. Document analysis is often combined with other methods for a comprehensive understanding.

8. **Ethnography**: Ethnography involves immersing oneself in a particular cultural or social group to understand their behavior, beliefs, and practices. It typically involves participant observation, interviews, and document analysis. Ethnography provides in-depth, context-rich insights into the studied group's perspectives and experiences.

This table shows a summary about mentioned methods:


| Method                | When to Use                                          | How to Collect Data                                                                 |
|-----------------------|------------------------------------------------------|------------------------------------------------------------------------------------|
| Surveys               | To gather information from a large sample             | Administer structured questionnaires to individuals or groups                         |
| Interviews            | To obtain in-depth insights or personal experiences   | Conduct direct interactions with individuals or groups, using structured or unstructured questioning |
| Observations          | To study behaviors or events in natural settings      | Systematically watch and record behaviors, events, or phenomena                      |
| Experiments           | To establish cause-and-effect relationships          | Manipulate variables under controlled conditions and collect data accordingly         |
| Existing Data Analysis| When relevant data already exists for analysis        | Analyze pre-existing data from public sources, research institutions, or archives    |
| Case Studies          | To deeply examine specific individuals or situations | Conduct extensive analysis and investigation of individuals, groups, or phenomena     |
| Document Analysis     | To analyze written, visual, or audio materials        | Examine reports, articles, or social media content for relevant information          |
| Ethnography           | To understand behavior and beliefs in a cultural group| Immerse oneself in the cultural or social group, observe, and interact with participants |

### Useful Datasets for Racism and Xenophobia Detection
In this section, I present information on datasets that have been used for *hate speech* detection or related concepts such as *cyberbullying*, *abusive language*, *online harassment*, among others, to make it easier for researchers to obtain datasets.
Even when there are several social media platforms to get data from, the construction of a balanced labeled dataset is a costly task in time and effort, and it is still a problem for the researchers in the area. Although most of the below-listed datasets are not explicitly available, some of them can be obtained from the authors if requested.


### English
|  No|           Datasets                    (Link to paper)          |          Objects          |             Size             |                                                   Available                                                    |                                                Labels                                                 | Comment
| :----:| :--------------------------: | :-----------------------: | :--------------------------: | :------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------: |:----------------------------------------------------------------------------------------------------: |
|1|     [Dinakar et al., 2011](https://ie.technion.ac.il/~roiri/papers/3841-16937-1-PB.pdf)     |     YouTube Comments      |             6000             |                                                       -                                                         |                                 Sexuality, Race, Culture, Intelligence                                  |
|2|    [Dadvar and Jong, 2012](https://ris.utwente.nl/ws/portalfiles/portal/5512243/DIR12_reviewed04.pdf)     |       Myspace Posts       |             2200             |                                                       -                                                      |                                         Bullying, Non Bullying                                         |
|3|      [Huang et al., 2014](https://wp.comminfo.rutgers.edu/vsingh/wp-content/uploads/sites/110/2016/10/p3-huang-1.pdf)      |          Tweets           |             4865             |                                                       -                                                       |                                         Bullying, Non Bullying                                         |
|4|  [Hosseinmardi et al., 2015](https://www.cs.colorado.edu/~rhan/Papers/socinfo2015_labeled.pdf)   | Instagram Media Sessions  |             998              |                                                       -                                                          |                                         bullying, Non bullying                                         |
|5|    [Waseem and Hovy, 2016](https://www.aclweb.org/anthology/N16-2013.pdf)     |          Tweets           |            16914             |                             [Download](https://github.com/zeerakw/hatespeech)                             |                                             Racist, Sexist, Either                                          |
|6|         [Waseem, 2016](https://www.aclweb.org/anthology/W16-5618.pdf)         |          Tweets           |             6909             |                             [Download](https://github.com/zeerakw/hatespeech)                             |                                      Racist, Sexist, Either,Both                                       |
|7|     [Nobata et al., 2016](http://www.yichang-cs.com/yahoo/WWW16_Abusivedetection.pdf)      |      Yahoo Comments       |             2000             |                                                       -                                                       |                                             Abusive, Clean                                             |
|8|    [Chatzakou et al., 2017](https://arxiv.org/abs/1702.06877)    |       Twitter Users       |             9484             |                                                       -                                                       |                                       Aggressor, Bully, Spammer                                        |
|9|    [Davidson et al., 2017](https://arxiv.org/pdf/1703.04009.pdf)     |          Tweets           |            24802            | [Download](https://github.com/t-davidson/hate-speech-and-offensive-language/blob/master/data/labeled_data.csv)    |                                    hate\_speech, offensive, neither                                    |
|10|     [Golbeck et al., 2017](http://www.cs.umd.edu/~golbeck/papers/trolling.pdf)     |          Tweets           |            35000             |                                                       -                                                       |                                       Harassing, Non Harassing                                        |
|11|      [Wulczyn et al. 2017](http://papers.www2017.com.au.s3-website-ap-southeast-2.amazonaws.com/proceedings/p1391.pd)   |     Wikipedia Comments     |             100000             |                                                       [Download](figshare.com/articles/Wikipedia_Detox_Data/4054689t)                                                        |                                   Personal Attacks
|12| [Tahmasbi and Rastegari, 2018](https://dl.acm.org/doi/10.1145/3290838) |          Tweets           |            12837            |                                                       -                                                       |                                         Bullying, Non Bullying                                         |
|13|    [Anzovino et al., 2018](https://link.springer.com/chapter/10.1007/978-3-319-91947-8_6)     |          Tweets           |             4454             |                                                       -                                                       | Discredit, Stereotype, Objectification, Sexual_Harassment, Threats of Violence, Dominance, Dearailingy |
|14|     [Founta et al., 2018](https://datalab.csd.auth.gr/wp-content/uploads/publications/17909-77948-1-PB.pdf)      |          Tweets           |            80000             |          [Download](https://dataverse.mpi-sws.org/dataset.xhtml?persistentId=doi:10.5072/FK2/ZDTEMN)          |                                      Hate Speech, Offensive, None                                      |
|15|     [Gibert et al., 2018](https://www.aclweb.org/anthology/W18-5102.pdf)      | Sentences from Stormfront |            10568             |                       [Download](https://github.com/aitor-garcia-p/hate-speech-dataset)                       |                                 Hate Speech, Non Hate Speech                                      |
|16|       [SemEval19, 2019](https://www.aclweb.org/anthology/S19-2007.pdf)        |          Tweets           |             9000             |                                                       [Request Link](http://hatespeech.di.unito.it/hateval.html)                                                       |                                          Hate speech, Non Hate Speech                                      |
|17|      [OLID 2019](https://www.aclweb.org/anthology/N19-1144.pdf)    |     Tweets     |             14100             |                                                       [Download](https://competitions.codalab.org/competitions/20011#participate)                                                        |                Offensive, Non Offensive
|18|      [TREC2 2020](https://arxiv.org/ftp/arxiv/papers/2003/2003.07428.pdf)    |     Messages (Twitter,Facebook,Youtube)     |             4,263             |                                                       [Request Form](https://docs.google.com/forms/d/e/1FAIpQLSesLjGKLQlE3dmQNZUEl5QJVno7NngeLTP9XvIMCvpZu7sXNg/viewform)    |   Misogynous (GEN,NGEN), AGGRESSION LEVEL(OAG, CAG, NAG) | Data GeoLocated India
|19|      [meTooMA 2020](https://ojs.aaai.org/index.php/ICWSM/article/view/7292/7146)    |     Tweets     |            9,973             |                                                       [Download](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/JN4EYU)    |   Hate Speech (Directed, Generalized), Relevance (0,1), STANCE (Support, Opposition, Neither) | Data GeoLocated India, Australia, Kenya, Iran, UK


### Multilingual (Parallel Data)
|  No|           Datasets                    (Link to paper)          |          Objects          |             Size             |                                                   Available                                                    |   Language    |                                                 Labels                                                 |
| :----:| :--------------------------: | :-----------------------: | :--------------------------: | :------------------------------------------------------------------------------------------------------------: | :-----------: | :----------------------------------------------------------------------------------------------------: |
|1|   [XHate 999](https://www.repository.cam.ac.uk/handle/1810/315111)    |Tweets from previous published English datasets and translated to 5 languages|   600 (x  6 languages)|                                                       [Download](https://github.com/codogogo/xhate)  |English, German, Russian, Croatian, Albanian, Turkish  | sexism, racism, toxicity, hatefulness, aggression, attack, cyberbullying, misogyny, obscenity, threats, and insults.


and this is another links for finding related dataset:

| Dataset Name                                 | Description                                                   | Language | Classes     | Source          | Download |
| -------------------------------------------- | --------------------------------------------------------------| ---------| ------------| ----------------|----------|
| HateEval | Annotated tweets for hate speech and offensive language. | English | (women or immigrants) is hateful or not hateful | Twitter | https://competitions.codalab.org/competitions/19935 |

| Wikipedia Talk Labels | User comments from Wikipedia talk pages annotated for toxicity.| English | toxic or healthy | Wikipedia | https://figshare.com/articles/dataset/Wikipedia_Talk_Labels_Toxicity/4563973/2 | 

| Online Harassment Dataset (Wikimedia)| User comments from Wikimedia platforms annotated for harassment.| English | bullying or not | https://www.kaggle.com/datasets/saurabhshahane/cyberbullying-dataset | 
              |
| Cyberbullying Dataset | The data contain text and labeled as bullying or not. | English | Kaggle, Twitter, Wikipedia Talk | https://www.kaggle.com/datasets/saurabhshahane/cyberbullying-dataset |
| Hate Speech and Offensive Language Dataset | The text is classified as: hate-speech, offensive, and neither| English  |0 - hate speech 1 - offensive language 2 - neither| Twitter | https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset/data |

# Step 3: Advancements and types of Language Models:  

## Different types of language models:

# Step 4: Implementation of the selected method
## Dataset

### Prepare and preprocess data 

# References

1. https://www.ncl.ac.uk/webtemplate/ask-assets/external/maths-resources/statistics/sampling/types-of-sampling.html
2. https://github.com/aymeam/Datasets-for-Hate-Speech-Detection/blob/master/README.md



