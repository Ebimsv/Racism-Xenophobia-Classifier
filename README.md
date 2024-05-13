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

## Worked Example
A school of 1000 students are classified as follows:

57 % Brunette,  
29 % Redhead,  
14 % Blonde.  

Find a stratified sample of 200 students for this population.

**Solution**  
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
  <summary><b>5. Convenience Sampling</b></summary><br/>
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

In the data collection phase of the Racism-Xenophobia-Classifier project, we will employ a combination of surveys, interviews, and document analysis to gather data on real-life experiences and perceptions of racism and xenophobia among diverse individuals.


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

# Step 3: Advancements and types of Language Models:  

## Different types of language models:
The research of LM has received extensive attention in the literature, which can be divided into four major development stages:

<details>
  <summary><b>1. Statistical language models (SLM)</b></summary><br/>
</details>

<details>
  <summary><b>2. Neural language models (NLM)</b></summary><br/>
</details> 

## Different training approaches of Language model:

<details>
  <summary><b>1. Causal Language Models (e.g., GPT-3)</b></summary><br/>
</details>

<details>
  <summary><b>2. Masked Language Models (e.g., BERT)</b></summary><br/>
</details>

<details>
  <summary><b>3. Sequence-to-Sequence Models (e.g., T5)</b></summary><br/>
</details>

<details>
  <summary><b>What's the difference between Causal Language Modeling and Masked Language Modeling?</b></summary><br/>
  
</details>

## Different Types of Models for Language Modeling  

Language modeling involves building models that can generate or predict sequences of words or characters. 
Here are some different types of models commonly used for language modeling:  

<details>
  <summary><b>1. N-gram Language Models</b></summary><br/>
</details>

<details>
  <summary><b>2. Recurrent Neural Network (RNN)</b></summary><br/>

**Advantages of RNNs**:

  1. Ability to capture sequential dependencies and context.
  2. Flexibility in handling variable-length input and output sequences.
  3. Suitable for tasks such as text generation, speech recognition, and language translation.

**Disadvantages of RNNs**:

  1. Difficulty in learning long-term dependencies due to the vanishing/exploding gradient problem.
  2. Limited contextual understanding of complex linguistic structures.
  3. Sequential nature limits parallelization, leading to slower processing times.   

![alt text](https://github.com/Ebimsv/Torch-Linguist/blob/main/pics/RNN.png)


</details>

<details>
  <summary><b>3. Long Short-Term Memory (LSTM)</b></summary><br/>
</details>

<details>
  <summary><b>4. Gated Recurrent Unit (GRU)</b></summary><br/>
</details>

<details>
  <summary><b>5. comparing RNN, LSTM, and GRU</b></summary><br/>
</details>

<details>
  <summary><b>6. Transformer models</b></summary><br/>
</details>


# Step 3: Choose the appropriate method: Language Modeling with Embedding Layer and LSTM

## This is the diagram of proposed model  

![alt text](https://github.com/Ebimsv/Torch-Linguist/blob/main/pics/LM.png)

# Step 4: Implementation of the selected method
## Dataset

### Prepare and preprocess data 

<details>
  <summary><b>1. Download WikiText-2 dataset</b></summary><br/>
</details>

<details>
  <summary><b>Tokenize data, building and saving vocabulary </b></summary><br/>
</details>

### Exploratory Data Analysis (EDA)

<details>
  <summary><b>1. Analyzing Mean Sentence Length in Wikitext-2 </b></summary><br/>
![alt text](https://github.com/Ebimsv/Torch-Linguist/blob/main/pics/EDA-mean-sentences.png)
  </details>

<details>
<summary><b>2. Custom dataset</b></summary><br/>
</details>

## Model

<details>
<summary><b>Custom PyTorch Language Model with Flexible Embedding Options</b></summary><br/>
</details>

# References

1. https://www.ncl.ac.uk/webtemplate/ask-assets/external/maths-resources/statistics/sampling/types-of-sampling.html



