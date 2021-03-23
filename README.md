# Disaster Response Pipeline Project

## Table of Contents

1. [Poject](#project)
2. [Instructions](#instructions)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors](#licensing)
6. [Repository](#repository)

## Project <a name="project"></a>

This project is part of the Udacity Nanodegree Data Scientist course.
The build model anlyses text messages send during disasters and categorize them to allow sending to an appropriate disaster relief agency.

## Instructions <a name="instructions"></a>

The code runs with Python versions 3.*.<br>
Libraries used in this project are given in requirements.txt

1. Install the packages with:<br>
    `pip install -r requirements.txt`

2. These steps are only required when ran for the first time or a new dataset is supplied.<br>
    a. Run data_processing.py in data's directory with required files as arguments:<br>
      `python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db`
	
    b. Run train_classifier.py in models's directory with required files as arguments:<br>
      `python train_classifier.py ../data/DisasterResponse.db classifier.pkl`

3. The provided flask app can be started in the app's directory via:<br>
    `python run.py`

4. Go to localhost to view the webpage:<br>
    http://0.0.0.0:3001/


## Files <a name="files"></a>

The run.py file loads the database and model. With those it creates the plots and user input classification function for the html files. 
Finally the host for the webpage (currently localhost) is started. 

HTML files:<br>
The master.html file uses the provided data to supply the UI for the classification tool / input line. A user classification request is then processed via the go.html extension. <br>
Below the classification input line two plots are shown. The first one gives an overview of the amount of training messages per category. The second shows the prediction accuracy and f1-score of the classifier on a separate test dataset. 


For first model and database creation the following files need to be ran as stated in [Instructions](#instructions).<br>
With this steps the classifier can also be trained on a new dataset. Therefore the two csv files in the 'data' folder need to be replaced. The data structure of the csv files need to be maintained.

Data Cleaning via process_data.py:<br>
The process_data.py file loads the csv data, cleans it and saves it to a sqlite database. 
For cleaning, the target data is split to separate columns, duplicates and rows with target values not zero or one are removed. 

Random Forest Classifier created via train_classifier.py:<br>
The cleaned data is tokenized and split into training and test dataset. With the training dataset a pipline of a CountVectorizer, TfidfTranformer and a RandomForestClassifier is setup. The hyper parameters of the pipline are tuned via a GridSearchCV parameter search.<br>
For model quality assessment a classification report showing accuracy, precision, recall and f1-score is generated.<br>
The model and classification_report is stored to supply for run.py/html files.


## Results <a name="results"></a>

The available messages are unevenly distributed over the given target categories. For sure that makes it hard to predict positive matches when there are only a few examples in the training set.<br>
However for some categories like 'water' the prediction is much better than eg  'infrastructure related' which has a similar amount of samples. When looking at the according messages it looks as keywords like water, food, shelter work well, but if it is more from the context or words like thirsty are used instead of water the current classifier does not capture it.

Either additional text about disasters could be used for vocabulary improvement and word relationships or a different classifier could improve this. (next iteration.)

## Licensing, Authors <a name="licensing"></a>

The data was provided by [Figure Eight](https://appen.com/) within the Udacity Nanodegree Data Scientist course as a modules project task.

## Repository <a name="repository"></a>

All files required to run the program as per [Instructions](#instructions) are stored in the github repository:<br>
https://github.com/mhoenick/drpp.git
