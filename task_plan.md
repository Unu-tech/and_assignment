# Planned steps

1. Quickly get background information from HuggingFace. 
    1. Since versions of ERNIE introduces various pre-training strategies, their description will be collected. 
    1. Unusual (hyper)parameters will be listed for ease of use. Since training is required, optimization related parameters will be more detailed.
    1. Model sizes and available languages will be listed.
1. Skim the papers of each version, focusing on key innovations and context of the improvement.
    1. For example, phrase-masking encourages the model to learn combined representations of phrases but it's dependent on lexical analysis tools.  
1. List requirements for datasets and tasks. 
    1. For example, the task need to be compatible with the pre-training task (MLM). Also, the dataset has to contain a separate relationship graph such as that of wikipedia because of entity-level masking. 
1. Based on above, choose a dataset and task with desired properties.
