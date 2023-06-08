# UHDP
# Uncovering the Hidden Dimensions of Psychometrics – Report Repo

# Analysis Report

## Analysis 1

### Cross-semantic LLMs stability

#### 1: Database

IPIP Scales Database

36 scales composed of 3805 items organized in scales and subscales, which we will call instruments and dimensions from now on.

#### 2: Extraction

1: Extracted the embeddings of each item of the questionnaire with 19 LLMs

LLMs:
- all-MiniLM-L6-v2
- all-MiniLM-L12-v2
- multi-qa-distilbert-cos-v1
- paraphrase-mutilingual-mpnet-base-v2
- multi-qa-MiniLM-L6-cos-v1
- all-mpnet-base-v2
- all-distrilroberta-v1
- paraphrase-multilingual-MiniLM-L12-vs
- paraphrase-albert-small-v2
- multi-qa-mpnet-base-dot-v1
- paraphrase-MiniLM-L3-v2
- distiluse-base-multilingual-cased-v2
- distiluse-base-multilingual-cased-v1
- GPT-3
- bert-base-uncased
- xlm-roberta-base
- GPT-2
- microsoft deberta-base

We chose these models for different reasons: some were already implemented in sentence-transformers, and some were commonly used for embeddings purposes or were validated by the community. We also chose 5 models from huggingface disparetely (disparately?) between publishers (Google, Microsoft, OpenAI ; OPT from Meta had critical issues during the extractions).

2: Computed the semantic similarity with cosine similarity between each embedding LLM's matrix:
      $M[N items ; Length Model Embedding]$

      Similarities
      
      $\text{{cos\_sim}}(V_i, V_j) = \frac{1}{N M_i M_j} \sum_{k=1}^{N} \sum_{l=1}^{M_j} \text{{cos\_sim}}(v_{ik}, v_{jl})$

3: From this similarity matrix we performed an Exploratory Factor Analysis with PCA that outputed 9 factors to determine how many factors (or family of LLM) we had. We defined 2 families that seem to differ only by their model sizes.

We defined the number of factors over a scree plot. We defined that we would only keep factors that had an eigenvalue superior or equal to 1. We then compared the models apparented to each factor qualitatively by searching for their model sizes on the internet.

4: Our analysis used the Frobenius norm for comparing distance between matrices due to its advantageous properties. Primarily, the Frobenius norm is not subject to rotational issues. Given that we are working with high-dimensional semantic embeddings, the nature of which does not afford us complete insight into their behaviors, we wanted to mitigate any complications from rotational variance. The use of the Frobenius norm allows us to safely compare these embeddings, which helps to ensure the robustness of our findings. This was an appropriate choice in our context, providing an efficient and reliable solution for the comparative analysis of the semantic space structured by these embeddings.
      

## Analysis 2

### Semantic Network Algorithm for Psychometric Equilibrium (SNAPE)

This algorithm was used to balance the way we measure psychological dimensions. We wanted each item to be semantically neither too close nor too far from the other ones. We want to have semantic equilibrium between the items to be able to grasp the maximum amount of information possible from the participants.

For this, we take some subsets of questionnaires, their dimension, which are meant to measure only one kind of trait or state (like Anxiety, Emotionality, Warmth, Extraversion, Self-esteem etc.).

We then compute the similarity matrix of each dimension from the embeddings of the items.

With this similarity matrix, we can define a network, with nodes as the items and the edges as the cosine similarity of their embeddings.

We only used GPT-3's embeddings for this analysis as:  it was the largest model, the model trained until convergence (compared to Bloom for example), and we could easily compute embeddings for the new items, without overloading the memory of the machine we worked on, given that we made API calls for the embeddings generation.

Before trying to "balance" the networks, we had to find a formula that defined each network's equilibrium:

      $NE = (\frac{1}{M})^2 \times \frac{1}{1+SD}$

Where M is the amount of Modules defined by the Louvain community detection algorithm and SD is the standard deviation of all the semantic similarities between the items; We can see in the equation that the amount of modules is heavily penalized as well as a high standard deviation. We hope that by penalizing these variables, we can obtain a network that has one single module with homogeneous similarities.

We wanted to penalize modularity heavily as we were trying to optimize sub-scales, which are meant to measure only one psychological dimension or factor. A high modularity (amount of modules) in a network would theoretically impact the measure of a psychological dimension.

We chose the Louvain detection algorithm for its ability to rapidly iterate, and its suitability for handling weighted graphs (where the nodes aren't "existent" vs. non-existent), and the networks we aimed to study didn't have loops, which the Louvain algorithm can accomodate.

We had some major issues concerning the amount of modules of the networks in the IPIP database. So we decided to iteratively calculate the Network Equilibrium of every dimension of the IPIP database (as long as the dimension had at least 5 items) by modifying the resolution of the Louvain Algorithm at each iteration. We selected the resolution that gave us the most variance of modules for each questionnaire (0.798).

Then, we selected the dimensions that had a Network Equilibrium between .8 and .2. We selected those threshold over a qualitative analysis of the distributions: succinctly put, smaller NE values did tend to represent smaller dimensions and a "leap" was observed in the distribution. We explained it because of the exponential impact the networks modularity has on the NE equation. We did not choose networks with a value over .8 since they were already well balanced.

Because the network equilibrium formula penalizes the modules more, we ended up with networks that had a NEs between .2 and .25. We then randomly selected 10 networks in this set of 84 networks (to limit the financial cost of computation).

For each item of the 10 selected dimensions, we asked GPT-4 to generate 10 paraphrases. Giving it the item and the dimension the item should measure. We specifically prompted the model not to change the meaning of the sentence in its context. We manually checked the LLM answers to assure the meanings of the items weren't deviating from the original meaning.

For each item, and its paraphrase, we computed the semantic similarity between the original item and the paraphrases. To give us a vector of items ordered by their similarity from the original item. A higher value meant a higher similarity with the original item.

We then performed the network balance phasis using a Tree-structured Parzen Estimator algorithm for optimizing the selected items. For each iteration, the algorithm selected a floating number between 0 and 1 as the item similarity from the original network for each item. And the "loss" function was the Network Equilibrium.

We found that we could optimize every dimension from a NE of ~.2 to a NE of ~.95. We tested the hypothesis that SNAPE was useful with a t-test between the initial NEs and the final NEs distributions – on 10 dimensions. The null hypothesis (H0) was that SNAPE did not significantly change the Network Equilibriums of the sub-scales and our alternate hypothesis (H1) was that SNAPE did significantly change the Network Equilibriums of the sub-scales: (t=-2.46, p=.036, df=9). We can conclude that SNAPE is useful and efficient to re-arrange the balance of semantic networks.

We finally checked if the final items were significantly changed from the original items with a proportion z-test, chosen because we were only comparing binary data frequencies. The hypotheses were: H0: The rate of change in items was not impacted by the SNAPE algorithm ; H1: The rate of changes in items was impacted by the SNAPE algorithm: (z=10.835, p=.000). Results indicate that nearly all of the items were changed by SNAPE.

#### Anticipated bias of SNAPE

SNAPE demonstrates potential for optimizing semantic networks, but its potential biases should be considered. The main issue arises from its propensity to favor semantically similar items, which could potentially narrow the psychological range captured by the original sub-scale.

The algorithm might oversimplify psychological constructs that are inherently diverse, conflating distinct elements and losing nuanced details. Semantic closeness doesn't always translate to psychological similarity, and this mismatch could introduce bias.

To mitigate this, we could modify the Network Equilibrium (NE) formula, possibly by adjusting the weighting of the edges, to balance both the modules and the psychological breadth.

Despite these potential biases, SNAPE holds promise. Recognizing these challenges informs continuous refinement of the method, aiming for more balanced and accurate semantic networks.

## Conclusions

In conclusion, the exploration into Cross-semantic Language Learning Models (LLMs) stability and the development of the Semantic Network Algorithm for Psychometric Equilibrium (SNAPE) offers an innovative approach to the balance and measurement of psychological dimensions. The research has demonstrated that various LLMs can effectively extract embeddings for semantic similarity analysis, and the introduction of SNAPE has the potential to significantly improve the balance of psychological dimension networks. However, further refinements are necessary to counteract potential biases and enhance the algorithm's effectiveness. Overall, this research contributes valuable insights into the use of machine learning in psychological assessment, presenting promising avenues for future work. Further studies should focus on enhancing the reliability of SNAPE, investigating its potential biases, and broadening its applicability in various research and practical contexts.

## Future work

We aim to measure some participant's answers to some sets of these sub-scales to measure how much the semantic similarity between items is related to the correlations between participant's answers to the sub-scales.

Should these turn out to be related, the SNAPE algorithm could definitely be a tool for accelerating the process of designing scales in psychology.
