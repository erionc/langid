# Language identification methods

Language Identification is a task within NLP (Natural Language Processing) that automates the process of identifying the natural language a piece of text (e.g., a document) is written in. This is usually a requirement for solving other NLP tasks relating to the identified language. Same as most other NLP tasks, language identification is commonly solved today with data-driven statistical and neural methods that require datasets to work. Some public datasets for language identification are:

1. Language Detection dataset of 10337 paragraphs from 17 natural languages [1].

2. Language Identification dataset of 90k text passages from 20 natural languages [2]. 

3. Wiki-40B dataset of 40 billion characters covering 40+ natural languages [3]. 

4. LID dataset of 121 lines of text, covering 201 natural languages [4]

There are different approaches to solving the task. One approach is to consider it as a text categorization task and use traditional machine learning classifiers which take little time to train. Another approach is to use multilingual PLMs (Pretrained Language Models) like BERT [5] and fine-tune them on a multilingual dataset like the ones listed above. Another approach is to utilize NLP libraries which provide ready-to-use language identification functions running on models under the hood. The examples below illustrate these approaches with Python code. For simplicity, only the first dataset listed above is used.


## Machine learning models

Data-driven methods based on traditional machine learning algorithms are still effective for solving various NLP tasks. Some of the popular methods for classification include Support Vector Machine, Logistic Regression, Naive Bayes, K-nearest Neightbors and Decition Trees. There are also ensemble methods like Random Forest, Gradient Boosting etc. which combine results from different learning models. The script ml.py implements some of these methods and can be run with the following command:

```
python ml.py -c <method>
```

The options to choose are ["lr", "mnb", "svm", "knn", "dt", "stck", "rf", "ada", "gb", "xgb"]. Most of the models are trained within one minute. The language classification accuracy ranges from 85 % to 96 % which may be acceptable in some scenarios.  


### Remarks

* Running these methods does not require a computing system eqquiped with a GPU.

* The bag-of-words representation that these methods use is sparse and wastes memory. 

* The feature selection and combination process can improve performance of these methods, but it requires expertise and time. 

* Since these models have relatively small capacity in number of parameters, they are usually prone to overfitting. 

* These methods do not yield significant performance improvements when trained with more data.

* This approach to solving the language identification task is limited by the languages that are supported in the training dataset. 


## Pretrained language models

PLMs or LLMs (Large Language Models) represent new NLP direction which has become dominant in the last five years. They are based on Transformer [6] encoder and/or decoder architectures pretrained with large amounts of unlabeled texts in one or many natural languages. Fine-tuning the PLMs on specific tasks with labeled data creates very high-performing models for many NLP tasks. The script plm.py fine-tunes Multilingual BERT and can be run with the following command: 

```
python plm.py
```

It takes about 25 minutes to run the script with the given hyperparameter setup in a computer system eqquiped with a RTX 3080 mobile 16 GB GPU. The obtained accuracy gets over 90 % just after the second epoch of training. 


### Remarks

* These methods require computing systems of high capacities which are eqquiped with GPUs or TPUs. Deep learning frameworks such as Pytorch are also required.  

* The PLMs work with a limited number of sequence lengths (512 in plm.py) which limits their applicability in some tasks and datasets.

* Selecting the PLM to fine-tune has a significant impact on performance. The PLM should have been pretrained with all the languages that are covered in the fine-tuning dataset. Typically, larger PLMs provide higher performance but do also require more computation.  

* The number of training epochs has a significant impact on performance. Many epochs can lead to overfitting and waste computation. Few of them may not provide acceptable performance.  

* This approach to solving the language identification task is limited by the languages that are supported in the training dataset. 


## Libraries for language identification

Today there are various libraries in high-level programming languages like Python that can be easily used for language identification with just a few lines of code. They offer functions that implement large models trained with huge amounts of texts, supporting hundreds of natural languages. An example is LangID that has been trained on 1629 natural languages [7]. The biggest issue with language identification task remains the support of low-resource or underrepresented languages.  


## References

[1] https://www.kaggle.com/datasets/basilb2s/language-detection

[2] https://huggingface.co/datasets/papluca/language-identification

[3] https://www.tensorflow.org/datasets/catalog/wiki40b

[4] https://aclanthology.org/2023.acl-short.75.pdf

[5] https://aclanthology.org/N19-1423

[6] https://arxiv.org/abs/1706.03762

[7] https://aclanthology.org/2020.coling-main.579