# Planned steps

1. Quickly get background information from HuggingFace. 
    1. Since versions of ERNIE introduces various pre-training strategies, their description will be collected. 
    1. Unusual (hyper)parameters will be listed for ease of use. Since training is required, optimization related parameters will be more detailed.
1. Skim the papers of each version, focusing on key innovations and context of the improvement.
    1. For example, phrase-masking encourages the model to learn combined representations of phrases but it's dependent on lexical analysis tools.  
1. List requirements for datasets and tasks. 
    1. For example, the task need to be compatible with the pre-training task (MLM). Also, the dataset has to contain a separate relationship graph such as that of wikipedia because of entity-level masking. 
1. Based on above, choose a dataset and task with desired properties.

# Decision logs and study notes

1. <details>
    <summary> Seems there are two ERNIE models. We assume the task concerns Baidu's paper. </summary>

    With published versions on HuggingFace, it seems ERNIE by Baidu is much more popular. Following is the source paper and short note on each.
    - [Baidu's paper](https://arxiv.org/pdf/1904.09223v1): About token-level knowledge integration through a masking strategy that concerns entities and phrases. By using prior knowledge graph of entity relation (e.g. wikipedia) and lexical analysis tools, entity-level and phrase-level texts are masked in chunks. 
    - [Huawei's paper](https://arxiv.org/pdf/1905.07129v1): About fusing entity relationship knowledge graphs through specialized architecture. Compared to above, this method 1) uses separate embedding for entities 2) adds architectural components to fuse entities and tokens 3) does not consider phrases.
   </details>
