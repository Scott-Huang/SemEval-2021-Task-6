# SemEval-2021 Task 6: Detection of Persuasive Techniques in Texts and Images
The code is for subtask 1 of [SemEval-2021 Task 6](https://propaganda.math.unipd.it/semeval2021task6/index.html).

## Installation
Run `pip install -r requirements.txt` to install necessary packages to run predictions. 

And the packages used for preprocessing is in `preprocess_requirements.txt`. The preprocessed data is already provided so it is not necessary to install them.

The code is developed in Windows, but it should be able to run on most operating system.

## Usage
Run the following code to make prediction:
``python algorithm.run <model> <out_path> [--augment] [--epoch EPOCH]``
The model can be either `baseline` or `entity`, and the entity model currently does not support using augmented data. Use the help message to see the specified details.

After the model makes prediction, run
``python algorithm.util <in_path> <out_path>`` to convert the generated file into the format supported by the scorer. The usage of the scorer is documented [here](https://github.com/di-dimitrov/SEMEVAL-2021-task6-corpus).

### Preprocessing
All preprocessed data is already provided.

#### Entity Linking
The code is quite incomplete and it is in `script.ipynb`. Please be sure check the [ERINE repo](https://github.com/thunlp/ERNIE) to download the pre-trained knowledge embedding and model, and their example usage of how to retrieve all entities. 

It also requires a GCUBE_TOKEN for using `tagme`, be sure to apply one before using it. 

#### Data Augmentation
The code uses PPDB data to get synonyms. Please download the English L-sized all-paraphrase data `ppdb-2.0-l-all` [here](http://paraphrase.org/#/download) and put it into the project directory.

Then run the following code to make data augmentation on the training data:
``python algorithm.preprocess <n> <n2> <out_path>``
where `n` is the number of augmented sentences of each original text and `n2` is the number of augmented sentences of each text with rare labels.
