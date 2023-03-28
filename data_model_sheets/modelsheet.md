# Model card for "add model name here"

Sections and prompts from the [model cards paper](https://arxiv.org/abs/1810.03993), v2.

Jump to section:

- [Model details](#model-details)
- [Intended use](#intended-use)
- [Factors](#factors)
- [Metrics](#metrics)
- [Evaluation data](#evaluation-data)
- [Training data](#training-data)
- [Quantitative analyses](#quantitative-analyses)
- [Ethical considerations](#ethical-considerations)
- [Caveats and recommendations](#caveats-and-recommendations)

## Model details

_Basic information about the model._

Review section 4.1 of the [model cards paper](https://arxiv.org/abs/1810.03993).

- Person or organization developing model: our research team at MIT, University of Toronto, and the Schwartz Reisman Institute for Technology and Society (contact Aparna Balagopalan, aparnab@mit.edu)
- Model type: Computer Vision (ResNet-50), and Natural Language Processing (Transformer-based ALBERT, and RoBERTa)
- Information about training algorithms, parameters, fairness constraints or other applied
  approaches, and features: We have provided this information in the Materials and Methods section (previously in the Appendix). Specifically:
			- Data Source and Sample Selection: Section 4.2 and Section A.2
			- How we collected the labels: Section 4.1-4.5 and Section A.1-A.2
			- ML Model training (and detailed result tables, hyperparameter settings, design choices, etc.): Section 4.6 and Section A.9
- Paper or other resource for more information: TBD
- Citation details: TBD
- License: TBD
- Where to send questions or comments about the model: Please contact Aparna Balagopalan (aparnab@mit.edu)

## Intended use


Review section 4.2 of the [model cards paper](https://arxiv.org/abs/1810.03993).

### Primary intended uses
To test the hypothesis that the judgment process involved in data labeling significantly affects the labels collected, and models trained on these labels. Specifically, while judging norm violations.

### Primary intended users
Researchers in the space of machine learning, and ethical computing.

### Out-of-scope use cases
Deploying model to judge norms. All the normative codes in our study are fictional, and we do not suggest that application of such codes are always appropriate in the real world. Instead, we use these hypothetical but realistic codes to illustrate our findings.

## Factors

Review section 4.3 of the [model cards paper](https://arxiv.org/abs/1810.03993).

### Relevant factors
All labels collected and models trained may inherently reflect the social biases exhibited by data annotators. User discretion is advised.


### Evaluation factors
All models were evaluated against the labels collected. External validation has not been performed, and we highlight that all the normative codes in our study are fictional, and we do not suggest that application of such codes are always appropriate in the real world.


## Metrics


Review section 4.4 of the [model cards paper](https://arxiv.org/abs/1810.03993).

### Model performance measures
Model performance is measured using accuracy, F1-score, False Positive Rate, and False Negative Rate at a specific threshold. We also provide threshold-independent score of AUPRC.

### Decision thresholds
Unless specified otherwise, a threshold of 0.5 is used.

### Approaches to uncertainty and variability
We replicate results across different random initializations, and report 95% Confidence Intervals for performance

## Evaluation data


Review section 4.5 of the [model cards paper](https://arxiv.org/abs/1810.03993).

### Datasets
All models were evaluated against the labels collected. External validation has not been performed, and we highlight that all the normative codes in our study are fictional, and we do not suggest that application of such codes are always appropriate in the real world.


### Motivation
Our motivation was to test the hypothesis that the judgment process involved in data labeling significantly affects the labels collected, and models trained on these labels. Specifically, while judging norm violations.


### Preprocessing
We normalize and standardize data using norms from ImageNet. In text datasets, tokenization and padding is performed using Hugginface tokenizers corresponding to each language model.

## Training data

Review section 4.6 of the [model cards paper](https://arxiv.org/abs/1810.03993).


## Ethical considerations


Review section 4.8 of the [model cards paper](https://arxiv.org/abs/1810.03993).

### Data
All the model data was curated and annotated carefully, by ensuring that annotators are paid at least 12 USD/hr. This project was approved by the University of Toronto's Institutional Research Ethics Board (Protocol\#00037283).


### Human life
The annotator's viewpoint is considered throughout data labelling. Further, we highlight that the data and models created have been used to test the specific research hypotheses and caution against real-world deployment.

### Mitigations
We make available train/validation/test data splits as well as full model train/validation/testing details. However, we caution against real-world deployment without rigorous external validation.

### Risks and harms
We highlight that the data and models created have been used to test the specific research hypotheses and caution against real-world deployment.

### Use cases
- Understanding impact of judgment processes on data labeling and model creation.
- Testing impact of data labels on prediction outcomes related to norm violations.
- Comparing design choices and modelling choices on final model performance.

