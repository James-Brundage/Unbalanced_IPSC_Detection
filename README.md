# Unbalanced_IPSC_Detection

## Suggested Pipeline

#### 1. Labelling (Currently)

**Owner:** Hillary/Crew + Other labs

Labelling process by hand in mini analysis. This should give a .csv of each recording with all the true positive peaks. 

**To do:** Create example files of what things should look like. 

#### 2. Labelling (Own Pipeline)
**Owner:** Colin

> Colin: Consider making a Python script that can replicate some on mini-analysis labeling on raw data (to a degree)

Build labelling tool 

#### 3. Creation of Training Dataset
**Owner:** James

Take the raw recording, use the find peaks scipy tool to find all possible peaks. Use the true positives to create the true and false positive labelled dataset. This dataset should be pretty imbalanced, because we will not use any filtering or preprocessing to enhance the find peaks tool like was done in the first paper. 

**To do:** Script this process using the initial data given by Hillary in the first paper and the already existing functions. 

#### 4. Prep Dataset for Experiments
**Owner:** Colin or James

First we need to determine which method of approaching unbalanced datasets or data augmentation is best to deal with the unbalanced problem with this dataset. This can likely be done by testing methods on the original data, and comparing model performance. Once this is done, we can write a script that takes a full dataset and performs the necessary balancing, splitting and standardization. 

**To do:** Balancing experiments on original data, script the balancing, splitting and standardization. 

> Colin: Random sampling method to create 50/50 true/false positive split. James has script that grabs traces of each true/false positive. 

> Colin: Run the random sampling outputs through XGBoost model. 

#### Training/Validation
**Owner:** Eliza/James

Perform model training and validation. We need to determine which models to test, what metrics we care about, and ensure the experiments are optimized for the correct machine. 

**To do:** Determine models, script experimental design, ensure correct values are saved along with model format. 

Putting all the scripts together – repurpose old code from previous models. 
Write a script that will regenerate the figures that we made for the previous paper. 

