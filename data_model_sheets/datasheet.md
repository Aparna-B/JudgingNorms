# Datasheet for dataset "add dataset name here"

Questions from the [Datasheets for Datasets](https://arxiv.org/abs/1803.09010) paper, v7.

Jump to section:

- [Motivation](#motivation)
- [Composition](#composition)
- [Collection process](#collection-process)
- [Preprocessing/cleaning/labeling](#preprocessingcleaninglabeling)
- [Uses](#uses)
- [Distribution](#distribution)
- [Maintenance](#maintenance)

## Motivation

### For what purpose was the dataset created? 

We acquire normative and descriptive labels in four stylized settings to test for differences in how humans judge facts and norms. Three of these settings concern judgments about images, and the fourth about short text samples. We construct fictional rules governing four settings: \emph{Clothing} mimics a dress code for clothing worn in an office or school setting, \emph{Meal} mimics a policy for meals served in schools, \emph{Pet} mimics a pet code for dogs permitted in apartment buildings, and \emph{Comment} mimics guidelines for comments posted in online forums.

### Who created the dataset (e.g., which team, research group) and on behalf of which entity (e.g., company, institution, organization)?

The data collection was performed by Aparna Balagopalan and David H. Yang on behalf of the whole research team (consisting of members from MIT, the University of Toronto, and the Schwartz Reisman Institute for Technology and Society). Some part of data curation was performed by research assistants hired at the Schwartz Reisman Institute for Technology and Society.

### Who funded the creation of the dataset? 
We acknowledge funding provided by the Schwartz Reisman Institute for Technology and Society for dataset curation, creation, and analysis.


## Composition

### What do the instances that comprise the dataset represent (e.g., documents, photos, people, countries)?
All datasets contain images curated from publicly-available datasets. The three image-based datasets contain images of clothing, food, and dogs.
The comment dataset contains public comments posted on the CivilComments platform (sampled from a large, publicly-available dataset).
Each dataset only contains a single instance type (e.g., image or text) and three corresponding labels.

### How many instances are there in total (of each type, if appropriate)?
There are 2000 unique instances in each dataset.

### Does the dataset contain all possible instances or is it a sample (not necessarily random) of instances from a larger set?
The dataset is a subset of existing large datasets (often curated from multiple sources). The samples were curated and selected such that the specific questions we asked to annotators (e.g., "is this dog large-sized?") could reasonably be inferred. As a result, the datasets are not random or representative subsets of the larger set they are sampled from. We point interested readers to the Data Curation section of our manuscript for more details. 

### What data does each instance consist of? 

Each dataset contains unprocessed text or images, and four binary labels. Specifically, the four labels indicate the presence of three factual features in the image, as well as a binary value indicating whther the corresponding rule or norm is violated. Each unique instance is annotated by up to 20 annotators. We provide the full, non-aggregated dataset.


### Is there a label or target associated with each instance?

Yes, there are four labels or targets associated with each instance (same as above).

### Is any information missing from individual instances?

In the Clothing dataset, faces have been blurred out to protect privacy. Additionally, the demographic information of annotators is available only for a subset of annotators (since this was collected as a pre- or post- survey).

### Are relationships between individual instances made explicit (e.g., users’ movie ratings, social network links)?

Yes, the instance ID is provided in the dataset.

### Are there recommended data splits (e.g., training, development/validation, testing)?

We experimented with a train/val/test split, which we provide details of.

### Are there any errors, sources of noise, or redundancies in the dataset?

We attempted to ensure a diverse collection of images, since diversity of images would be a factor that influences differences between conditions. However, while we checked that no exact duplicates (with image hashing functions) were present in the dataset, perceptually similar images/text samples do exist (as they do in the original datasets the objects were sampled from). We also note that there are a few images sampled from "Pixaby" in the Meal dataset that have identical visual content but differing aspect ratios or image hashes. We estimate this to be <=9 images, and have verified that removing these images does not cause the results to vary via a robustess check.

### Is the dataset self-contained, or does it link to or otherwise rely on external resources (e.g., websites, tweets, other datasets)?

The dataset relies on instances (and hence IDs) from other larger datasets. Additionally, we also use additional annotations available in the CivilComments dataset for some experiments.

### Does the dataset contain data that might be considered confidential (e.g., data that is protected by legal privilege or by doctor-patient confidentiality, data that includes the content of individuals’ non-public communications)?

All data is curated from publicly-available datasets.

### Does the dataset contain data that, if viewed directly, might be offensive, insulting, threatening, or might otherwise cause anxiety?

The Comment dataset contains posts/comments that might be offensive, insulting, threatening, or might otherwise cause anxiety. We ensured that we added a "CONTENT WARNING" while collecting the labels for the same.

### Does the dataset relate to people? 

The dataset does not directly relate to people, but is situated in social/normative situations. Additionally, all the data has been annotated by annotators recruited on Amazon Mechanical Turk.

### Does the dataset identify any subpopulations (e.g., by age, gender)?

In the Comment dataset, identity mentions for each comment are provided (by linking to annotations in CivilComments).

### Is it possible to identify individuals (i.e., one or more natural persons), either directly or indirectly (i.e., in combination with other data) from the dataset?

No, all annotator details are anonymized (implemented by default on Amazon Mechanical Turk).

### Does the dataset contain data that might be considered sensitive in any way (e.g., data that reveals racial or ethnic origins, sexual orientations, religious beliefs, political opinions or union memberships, or locations; financial or health data; biometric or genetic data; forms of government identification, such as social security numbers; criminal history)?

The Comment dataset contains posts/comments that might be offensive, insulting, threatening, or might otherwise cause anxiety. We ensured that we added a "CONTENT WARNING" while collecting the labels for the same. Additionally, the annotations or labels themselves may exhibit social biases. 


## Collection process


### How was the data associated with each instance acquired?

The raw instances to label are directly observable, and sampled from existing large datasets. These instances were then labeled or annotated by participants recruited on Amazon Mechanical Turk. The data was validated by ensuring participants completed attention checks correctly in most cases.


### What mechanisms or procedures were used to collect the data (e.g., hardware apparatus or sensor, manual human curation, software program, software API)?

We collect data using Amazon Mechanical Turk Requester interface. All attention check completions were verified programatically, and jobs then accepted. 

### If the dataset is a sample from a larger set, what was the sampling strategy (e.g., deterministic, probabilistic with specific sampling probabilities)?

For the Clothing dataset, we ran a pilot attribute labeling study for 4000 images with the instruction "Answer yes/no questions about some relevant attributes of the clothing shown" for obtaining noisy labels for our three factual features. These labels were then used to sample 2000 images for {labeling}. For the Meal and Pet dataset, we explicitly tried to sample more contentious or ambiguous violations of the code/policy since this is where we expect to see the most difference. We also included objects with clear violations and non-violations of the code/policy for the purpose of training machine learning models, and hence maintaining balance in classes. Note that the size/aspect ratio of the images often varied in the Meal and Pet datasets---both within and across datasets the objects were sampled from. Additionally, we attempted to ensure a diverse collection of images, since diversity of images would be a factor that influences differences between conditions. However, while we checked that no exact duplicates were present in the dataset, perceptually similar images/text samples do exist (as they do in the original datasets the objects were sampled from). We note that there are a few images sampled from "Pixaby" in the Meal dataset that have identical visual content but differing aspect ratios or image hashes. We estimate the proportion of this to  $\approx$ 0.45\% of the full dataset (i.e., $\leq 9$ duplicate images).

For the Comment dataset, we used the labels indicating toxicity of a comment present in the CivilComments dataset (using their definition of "toxicity”, identity attacks, obscenity, insults, and threats) to perform a more controlled and precise sampling. We first filtered the dataset by removing long comments. Then, we selected 70\% text samples for which the comment -- each with continuous toxicity and factual feature labels between 0 to 1 -- had high contentiousness (following the definition set a priori), and 30\% low contentiousness. In each of these cases, we ensured that an equal proportion of each factual feature was present in our dataset (e.g., $\approx$ 33\% text samples using obscene language). We also tried to ensure that labels covered the full range of possible factual feature values (0 to 1) in order to build a balanced dataset. This required adjusting the sampling procedure throughout data collection. In all cases, we shuffled the order of the objects randomly after sampling to avoid any determinism in order.


### Who was involved in the data collection process (e.g., students, crowdworkers, contractors) and how were they compensated (e.g., how much were crowdworkers paid)?

Students and research assistants were involved in data curation and selection, funded by MIT and/or the Schwartz Reisman Institute for Technology and Society. Crowdworkers were involved in data annotation, and were compensated with the aim to pay at least 12 USD/hr.

### Over what timeframe was the data collected?

Majority of the data crowd-sourcing was completed between May 2020-July 2022. Pilot experiments started in 2019.

### Were any ethical review processes conducted (e.g., by an institutional review board)?

This project was approved by the University of Toronto's Institutional Research Ethics Board (Protocol\#00037283).

### Does the dataset relate to people?

The data is annotated by participants recruited on Amazon Mechanical Turk. Further, the labels collected involve judgments of the presence of factual features/rule violations situated in social contexts. 

### Did you collect the data from the individuals in question directly, or obtain it via third parties or other sources (e.g., websites)?
We crowd-sourced data from participants via Amazon Mechanical Turk. 

### Were the individuals in question notified about the data collection?

We provided detailed instructions (and hence goal for data collection) in the annotation prompts.

### Did the individuals in question consent to the collection and use of their data?

The annotators voluntarily accept the HIT, and submit the HIT, as per standard recruitment and consent-based workflows on Amazon Mechanical Turk.

### If consent was obtained, were the consenting individuals provided with a mechanism to revoke their consent in the future or for certain uses?

Annotators have the mechanism to email the data collectors ("Requesters") with questions or requests.

### Has an analysis of the potential impact of the dataset and its use on data subjects (e.g., a data protection impact analysis) been conducted?

We do not release any personally-identifying data of the data subjects in any case. As such, we do not anticipate any adverse impacts.


## Preprocessing/cleaning/labeling


### Was any preprocessing/cleaning/labeling of the data done (e.g., discretization or bucketing, tokenization, part-of-speech tagging, SIFT feature extraction, removal of instances, processing of missing values)?

We provide the full, non-aggregated dataset. In the normative setting in each case, we removed the datapoints where participants failed an implicit attention-check as a data cleaning step.

### Was the “raw” data saved in addition to the preprocessed/cleaned/labeled data (e.g., to support unanticipated future uses)?

Yes, the "raw" data is saved, and remains available on secure local/remote servers.

### Is the software used to preprocess/clean/label the instances available?

Yes, we release the code.

### Any other comments?

## Uses

### Has the dataset been used for any tasks already?

The dataset has been used to test the hypothesis: that the judgment process involved in data labeling significantly affects the labels collected. Specifically, while judging norm violations.

### Is there a repository that links to any or all papers or systems that use the dataset?

Yes, this [Github repository] (https://github.com/Aparna-B/JudgingNorms)

### What (other) tasks could the dataset be used for?

The dataset can be used for detecting the presence of factual features in images/texts.

### Is there anything about the composition of the dataset or the way it was collected and preprocessed/cleaned/labeled that might impact future uses?

All labels collected may inherently reflect the social biases exhibited by data annotators. User discretion is advised.

### Are there tasks for which the dataset should not be used?

All the normative codes in our study are fictional, and we do not suggest that application of such codes are always appropriate in the real world. Instead, we use these hypothetical but realistic codes to illustrate our findings.


## Distribution

### Will the dataset be distributed to third parties outside of the entity (e.g., company, institution, organization) on behalf of which the dataset was created? 

In some cases, the individual data instances cannot be re-distributed. 

### How will the dataset will be distributed (e.g., tarball on website, API, GitHub)?

The data will be posted on the [Github repository] (https://github.com/Aparna-B/JudgingNorms).

### When will the dataset be distributed?

The dataset will be distributed upon manuscript notification, and all licenses being made available.

### Will the dataset be distributed under a copyright or other intellectual property (IP) license, and/or under applicable terms of use (ToU)?

The dataset (specifically the labels) will be distributed under license allowing re-distribution and reuse.

### Have any third parties imposed IP-based or other restrictions on the data associated with the instances?

No, no IP-based restritcions are imposed.

### Do any export controls or other regulatory restrictions apply to the dataset or to individual instances?

No, no regulatory restrictions apply to the labels. Individual restrictions may apply for the instances themselves (as they are sampled from existing large datasets).

## Maintenance

The data will be supported/hosted on the [Github repository] (https://github.com/Aparna-B/JudgingNorms), and maintained by the research team members.

For any questions or requests, please contact aparnab@mit.edu.


