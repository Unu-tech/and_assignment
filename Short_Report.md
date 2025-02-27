# Task 1

Since my research is related to self-supervised learning, this task was quite interesting for me.

## Planned steps (ended up doing them)

1. Quickly get background information from HuggingFace. 
    1. Since versions of ERNIE introduces various pre-training strategies, their description will be collected. 
    1. Unusual (hyper)parameters will be listed for ease of use. Since training is required, optimization related parameters will be more detailed.
    1. Model sizes and available languages will be listed.
1. Skim the papers of each version, focusing on key innovations and context of the improvement.
    1. For example, phrase-masking encourages the model to learn combined representations of phrases but it's dependent on lexical analysis tools.  
1. List requirements for datasets and tasks. 
    1. For example, the task need to be compatible with the pre-training task (MLM). Also, the dataset has to contain a separate relationship graph such as that of wikipedia because of entity-level masking. 
1. Based on above, choose a dataset and task with desired properties.

## Relevant notes from decision logs

1. <details>
    <summary> From application perspective </summary>

    - There are 3 main versions of ERNIE, mostly in Chinese.
    - On huggingface, only ERNIE 2.0 (103M params which is close to BERT base) has English version. Since I want to use pretrained model, I'll use this.
    - In ERNIE 1.0, the researchers propose masking related entities (based on additional knowledge graph data) and phrases (based on lexical tools) jointly. By predicting the whole chunk of such related tokens, the model is encouraged to encode higher-level representation.
    - In ERNIE 2.0, introduces a very simple continual pre-training framework which ensures none of the pretext tasks are ignored by the model. But more importantly to this experiment, they also propose several pre-training tasks which will be listed below.
    - In ERNIE 3.0, it seems their main contribution is adapting previously introduced innovations to large-scale models (10B params). For now, this paper is skipped.
   </details>
1. <details>
    <summary> From research perspective </summary>

    From self-supervised learning perspective, choice of pretext task needs to be compatible with all meaningful downstream tasks. For instance, contrastive learning from image domain injects various invariant properties depending on the data augmentation. Both intuition and empirical findings show that shift, scale, or rotation invariances are compatible with most image-related downstream tasks.

    List of pretext tasks:
    - Masked Language Modeling
        - Randomly masks tokens in input text and trains model to predict the original tokens.
        - Introduces inductive bias for understanding contextual word representations.
    - Word-aware Pre-training Tasks
        - Knowledge Masking Task
            - Masks entire phrases and named entities rather than individual tokens.
            - Introduces inductive bias for understanding high-level semantic units and knowledge integration.
        - Capitalization Prediction Task
            - Predicts whether words are capitalized or not.
            - Introduces inductive bias for understanding entity recognition cues.
        - Token-Document Relation Prediction Task
            - Predicts whether a token in one segment appears in other segments of the document.
            - Introduces inductive bias for identifying key words and main topics.
    - Structure-aware Pre-training Tasks
        - Sentence Reordering Task
            - Shuffles paragraph segments and trains model to reconstruct original order.
            - According to the authors, this is for understanding relationships among sentences. But in my opinion, this is teaching permutation invariance (i.e. similar representation when the inputs are permuted).
        - Sentence Distance Task
            - Classifies sentence pairs based on proximity (adjacent, same document, different documents).
            - Similar to above.
    - Semantic-aware Pre-training Tasks
        - Discourse Relation Task
            - Predicts semantic or rhetorical relations between sentence pairs. For example, transition word "But" implies the sentences have opposite meaning.
            - Introduces inductive bias for understanding higher-level reasoning and argument structure.
        - IR Relevance Task
            - Classifies query-title pairs based on search relevance (clicked, shown but not clicked, irrelevant). Based on data collected from commercial search engine.
            - Introduces inductive bias for understanding information retrieval patterns and user intent alignment.

    Note that NER task, for example, is directly addressed by knowledge masking pretext task. In order to keep the experiment interesting, a downstream task of text classification was selected. Although not directly addressed, structure-aware and semantic-aware tasks may show improvements.
   </details>

# Task 2

I thought this task was related to task 1 and selected an experiment that's related to ERNIE. 

## About experiment and dataset

The reasoning to use IMDB dataset was as follows:

1. Text/sequence classification task was chosen.
    1. It was the most affordable task.
    1. Linear probing comparison was not done in the original paper. In my opinion, representation quality should not show improvement for this downstream task. First, the pre-text tasks mainly directly address token classification, or language generation related downstream tasks. So I wanted to test this hypothesis.
    1. But one potential source of improvement may be the difference between datasets. ERNIE uses Reddit and Discovery data in addition.
1. IMDB dataset intro:
    1. The target is a binary label indicating the review is positive or negative. This is very easy task.
    1. Very balanced dataset which makes my life easier. Also, there are only 50k total samples which is cheap and enough/suitable for linear probing.
    1. Reasonable choice for comparing two models. For instance, choosing a Reddit-related dataset would give unfair advantage to ERNIE.
1. Unfortunately, there's no feature-engineering and pre-processing. Pre-trained tokenizer automatically handle everything and I only had to set a maximum sequence length. This pads the sequence with zeros and masks the attention to padded regions of the sequence.
1. The default test split of the dataset was further split into 10k (val) and 15k (test) sized datasets. 

## Architecture for training

1. Tokenization: Pretrained models already include cleaning operations.
1. <details>
    <summary> Get intermediate outputs from pre-trained model: </summary>
    Denote the input as $x \in \mathbb{R}^{\text{seq len}\times\text{embed dim}}$ and transformer encoder layers as $l^{(i)}$. For clarity, the output of the encoder with 12 layers becomes: $T_{enc}(x)=l^{(12)}\left(l^{(11)}\left(\ldots l^{(1)}\left(x\right)\right)\right)$

    The intermediate outputs $h^{(i)}$ are defined as $h^{(i)}=l^{(i)}\left(l^{(i-1)}\left(\ldots l^{(1)}\left(x\right)\right)\right)$. 

    Next, the input of linear heads $z^{(i)}$ are a concatenation of hidden states of class token and mean of other tokens. For clarity, $z^{(i)} = [cls^{(i)\top}, mean^{(i)\top}]^\top$ where
    - $cls^{(i)} = h^{(i)}_{1, \bullet}$
    - $mean^{(i)} = \frac{1}{\text{seq len}-1} \mathbf{1}^\top_{\text{seq len}-1} h^{(i)}_{2\colon, \bullet}$

   </details>
1. <details>
    <summary> Train a linear head for each layer: </summary>
    Denote the linear heads as $W^{(i)} \in \mathbb{R}^{2\text{embed dim}}$. The loss is calculated as

    $$Loss= \frac{1}{12} \sum_{i=1}^{12}\text{BCE}\left(W^{(i)\top}z^{(i)}, y\right)$$

   </details>
1. Then AdamW optimizer with CosineAnnealingWarmRestart learning scheduler is used for optimization.
1. No tuning was necessary as we are training only linear heads. For evaluation, loss, accuracy, and aucroc scores are reported.

## About deployed service

Using Flask, a very simple service was created which has:

1. One endpoint `/comparison`. It takes a posted raw json and returns the response which contains both model's (BERT/ERNIE+linear head) output probabilities. Note that this endpoint cannot be accessed from browser (no get).
1. Landing page which includes some explanation and browser access. Interaction here just accesses `/comparison`.

### How to interact

I left the VM running so you can interact whenever you like. But please contact me when I can stop the service.

#### Web access

You can go to the landing page at: [http://35.225.152.179:5000](http://35.225.152.179:5000)

#### Using curl

```bash
curl -X POST http://35.225.152.179:5000/comparison -H "Content-Type: application/json" -d '{"text": "this works somewhat fine"}'

```

## Challenges encountered

In short, just getting used to some tools slowed me down. While some issues are elaborated in decision logs, here's summary:

1. Getting used to GCP: While I used GCP VMs, admin-related things were new to me. This includes, setting up roles, firewall rules, and researching efficiency of engines. But most significant issue was availability issue on GCP's end.
1. Getting used to huggingface: My experience is almost exclusively about designing an architecture so more application-related side was new to me. This includes, familiarizing with capabilities, getting used to docs and important paths, as well as dealing with bugs. 
1. Learning about Flask and HTML: Regarding this point, I had only limited experience, using the most primitive tools (using socket for micro-service, making static html page).
