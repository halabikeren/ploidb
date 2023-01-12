# PloiDB data processing and ploidy inference pipeline

This repository consists of the pipeline responsible for producing the of PloiDB plidy inferences. In also includes additional data processing and tree reconstruction scripts.

To execute the inference scheme on a given dataset, use the exec_ploidb_pipeline script with the following arguments:

```
    --counts_path PATH              chromosome counts file path  [required]
    --tree_path PATH                path to the tree file  [required]
    --output_dir PATH               directory to create the chromevol input in
                                    [required]
    --log_path PATH                 path to log file of the script  [required]
    --taxonomic_classification_path TEXT
                                    path to data file with taxonomic
                                    classification of members in the counts and
                                    tree data
    --ploidy_classification_path TEXT
                                    path to write the ploidy classification to
    --optimize_thresholds BOOLEAN   indicator weather thresholds should be
                                    optimized based on simulations
    --diploidy_threshold FLOAT RANGE
                                    threshold between 0 and 1 for the frequency
                                    of polyploidy support across mappings for
                                    taxa to be deemed as diploids  [0<=x<=1]
    --polyploidy_threshold FLOAT RANGE
                                    threshold between 0 and 1 for the frequency
                                    of polyploidy support across mappings for
                                    taxa to be deemed as polyploids  [0<=x<=1]
    --allow_base_num_parameter BOOLEAN
                                    indicator if we allow the selected model to
                                    include base number parameter or not
    --use_model_selection BOOLEAN   indicator if we allow the selected model to
                                    include base number parameter or not
    --help                          Show this message and exit.
```

 ## The output of the pipeline is created in <output_dir> and includes the following:
  1. **model_selection** - a folder consisting of the resluts of chromevol model fitting to the data with different sets of parameters
  2. **stochastic_mappings.zip** - a zip consisting of the sampled stochastic mappings, based on the all the different chromevol models, except for the gain_loss model which does not account for polyplodizations.
  3. **simulations.zip** - a zip consisting of the simulated datasets, based on the all the different chromevol models, except for the gain_loss model which does not account for polyplodizations.
  4. **ploidy.csv** - a file with the plidy inference data
  5. **classified_tree.nwk, classified_tree.phyloxml** - files of trees with the chromosome numbers and classifications of tip taxa in newick and phyloxml formats, respectively
