# Word sense disambiguation task for Norwegian Bokmål

In this section, attempts of adaptation of various approaches to solving the WSD problem were made.


At the moment: trying to run the solution from [this article](https://arxiv.org/pdf/2009.11795.pdf) for dataset prepared from the Norwegian Wordnet dataset. Files with adapted code (some changes of [original code](https://github.com/BPYap/BERT-WSD)) were uploaded.

Statistics for Dataset can be found in the file by the [link](https://colab.research.google.com/drive/1GIfCwGhKFSmQCa9P_EAx1uL7xzNTTjW0?usp=sharing). 


Link to the article|Comments|
|---|---|
[Adapting BERT for Word Sense Disambiguation with Gloss Selection Objective and Example Sentences](https://github.com/BPYap/BERT-WSD) | the code was adapted for Norwegian Bokmol|


Model|Tests|Metric|Scores on validation set
|---|---|---|---|
Norbert2	|all synsets 8 batch	|F1-score weighted |0.5302670004171881
Norbert2	|all synsets 32 batch	|F1-score weighted|0.5248100112931152
Norbert2	|no compounds 8 batch	|F1-score weighted|0.7951152866046483
Norbert2	|no compounds 32 batch	|F1-score weighted|0.8012218035059085
Norbert2	|no min 5 8 batch	|F1-score weighted|0.7928724499329288
Norbert2	|no min 5 32 batch |F1-score weighted	|0.79396340167299
Mbert	|all synsets 8 batch |F1-score weighted	|0.5289802118337789
Mbert	|all synsets 32 batch	|F1-score weighted|0.5327130906918142
Mbert	|no compounds 8 batch	|F1-score weighted|0.552393844100389
Mbert	|no compounds 32 batch|F1-score weighted|	 0.5771749770185315
Mbert	|no min 5 8 batch	|F1-score weighted| 0.47706196242781607
Mbert	|no min 5 32 batch |F1-score weighted|	0.7776396419923625
