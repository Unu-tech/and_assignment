# Decision logs

1. <details>
    <summary> Seems there are two ERNIE models. We assume the task concerns Baidu's paper. </summary>

    With published versions on HuggingFace, it seems ERNIE by Baidu is much more popular. Following is the source paper and short note on each.
    - [Baidu's paper](https://arxiv.org/pdf/1904.09223v1): About token-level knowledge integration through a masking strategy that concerns entities and phrases. By using prior knowledge graph of entity relation (e.g. wikipedia) and lexical analysis tools, entity-level and phrase-level texts are masked in chunks. 
    - [Huawei's paper](https://arxiv.org/pdf/1905.07129v1): About fusing entity relationship knowledge graphs through specialized architecture. Compared to above, this method 1) uses separate embedding for entities 2) adds architectural components to fuse entities and tokens 3) does not consider phrases.
   </details>
1. <details>
    <summary> Will run text classification experiments </summary>

    Firstly, it's the cheapest and only affordable ERNIE-related task. Secondly, papers of all versions of ERNIE are limited to fine-tuning and does not run linear probing evaluation. Although I am not very familiar with language domain, linear probing has unique importance when arbitrary downstream classification tasks are concerned ([1] and [2]). Specifically, it's compatible with the learning pressure of cross-entropy-based losses which minimizes unnecessary information loss in features due to fine-tuning task.

    For linear probing, we freeze the pre-trained model. For each input, we collect the intermediate outputs of pre-selected layers. Lastly, a linear layer is trained for each intermediate layer. A same procedure can be found in BERT paper ([3]).

    [1] Alain, G., & Bengio, Y. (2018). Understanding intermediate layers using linear classifier probes. arXiv [Stat.ML]. Retrieved from [http://arxiv.org/abs/1610.01644](http://arxiv.org/abs/1610.01644)
    
    [2] Tomihari, A., & Sato, I. (2024). Understanding Linear Probing then Fine-tuning Language Models from NTK Perspective. arXiv [Cs.LG]. Retrieved from [http://arxiv.org/abs/2405.16747](http://arxiv.org/abs/2405.16747)

    [3] Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. arXiv [Cs.CL]. Retrieved from [http://arxiv.org/abs/1810.04805](http://arxiv.org/abs/1810.04805)
    
   </details>
1. <details>
    <summary> Some notes on ERNIE versions </summary>

    - There are 3 main versions of ERNIE, mostly in Chinese.
    - On huggingface, only ERNIE 2.0 (103M params) has English version. Since I want to use pretrained model, I'll use this.
    - In ERNIE 1.0, the researchers propose masking related entities (based on additional knowledge graph data) and phrases (based on lexical tools) jointly. By predicting the whole chunk of such related tokens, the model is encouraged to encode higher-level representation.
    - In ERNIE 2.0, introduces a very simple continual pre-training framework which ensures none of the pretext tasks are ignored by the model. But more importantly to this experiment, they also propose several pre-training tasks which will be listed below.
    - In ERNIE 3.0, it seems their main contribution is adapting previously introduced innovations to large-scale models (10B params). For now, this paper is skipped.
   </details>
1. <details>
    <summary> ERNIE pretext tasks and justification of the experiment </summary>

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
1. <details>
    <summary> IMDb dataset will be used </summary>

    This dataset was chosen because:
    - Simple binary classification
    - Only 50k samples which makes evaluation cheap
    - Balanced labels
   </details>
1. <details>
    <summary> Simple service design </summary>

    The service will take a single sentence, tokenize it, and report the output probabilities of all linear heads (corresponding to intermediate layers). It will load checkpoint and host from the same machine.

    I had prior experience with only very minimalist module (socket) so I just chose the simplest looking approach which was Flask.

    To make sure I meet the task requirement, I have /comparison endpoint which can be accessed with curl command to post raw json and get results.

   </details>

# Progress notes (encountered issues)

1. <details>
    <summary> Notes will be minimal </summary>

    I messed up and ended up working on applying ERNIE methods to my research. Now due to time constraints (6 hours until EOD), I'll mainly work on implementation and come back to notes.
   </details>
1. <details>
    <summary> I may have misunderstood </summary>

    In retrospect, it seems task 2 was not related to ERNIE... I will continue as is because 1) I'm way behind schedule 2) I've already invested too much time into ERNIE considerations. But this choice means I won't be able to effectively showcase data pre-processing skills which maybe can be tested during interview. 
   </details>
1. <details>
    <summary> GCP compute engines with T4, L2, or A100 GPUs are all unavailable </summary>

    It's been 3 hours and many tries. I'll create initial buggy code and come back to training.
   </details>
1. <details>
    <summary> Cannot use GCP at all... </summary>

    It's been a day and 58 tries across many machines and regions. At this point, I'm thinking about figuring out which aws engine to use...

   I saw some discussions saying unavailability is for cheap VMs or claiming some regions have more GPUs. I tried changing everything, including escalating to expensive VMs and not including GPUs. Still, no luck...

   ![GCP_bad](https://github.com/user-attachments/assets/57844874-7fee-4241-bf45-b170f6bc8389)

   </details>
1. Systematically testing different regions worked and GCP is on
1. <details>
    <summary> Model training code is finished </summary>

    Since finally obtaining a VM this morning, the followings are done:

    - Repository setup (flat layout, simplest management but only supports dev environment).
    - Model, data, training codes.
    - Debugging everything and now training has started.
   </details>
1. <details>
    <summary> Service initial version is finished </summary>

    I also added a landing page which accesses /comparison endpoint and displays the outputs in a table.

   </details>
