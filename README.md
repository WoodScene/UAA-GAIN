# UAA-GAIN
The source code for Towards Sustainable Compressive Population Health: A GAN-based Year-By-Year Imputation Method    
Thank you for your interest in our work, we have uploaded the Section 4 DATA OBSERVATION AND INTUITION and all the code for the model here.

## Requirements
Install python, tensorflow. We use Python 3.7, Tensorflow 1.14.0.

## Data preparation
The three open-world Chronic Diseases Prevalence Datasets can be downloaded at:
* UK-Obesity Dataset: https://digital.nhs.uk/data-and-information/publications/statistical/quality-and-outcomes-framework-achievement-prevalence-and-exceptions-data
* US-Hypertension Dataset: https://www.cdc.gov/500cities/
* Taiwan-Diabetes Dataset: https://dep.mohw.gov.tw/DOS/cp-2519-3480-113.html

The "DATA" folder contains the downloaded raw data set and the pre-processed normalised data.

## Run
All the hyper-parameters and steps are included in the ./EXPERIMENTS/UAA-GAIN/main.py file, you can run it directly.

All other baseline methods are also in the "EXPERIMENTS" folder.

## Complete experimental results
Due to space limitations in the paper, it is not possible to show the results of the experiments for all years.
Here we show all the results of the experiments in graphs：

* UK-Obesity results：

<img src="https://github.com/WoodScene/Paper_pictures/blob/main/KDD2021/UK_RMSE.png" width="480" height="360"/><br/>
<img src="https://github.com/WoodScene/Paper_pictures/blob/main/KDD2021/UK_MAPE.png" width="480" height="360"/><br/>

* US-Hypertension results：

<img src="https://github.com/WoodScene/Paper_pictures/blob/main/KDD2021/US_RMSE.png" width="480" height="360"/><br/>
<img src="https://github.com/WoodScene/Paper_pictures/blob/main/KDD2021/US_MAPE.png" width="480" height="360"/><br/>


* Taiwan-Diabetes results：

<img src="https://github.com/WoodScene/Paper_pictures/blob/main/KDD2021/TAIWAN_RMSE.png" width="480" height="360"/><br/>
<img src="https://github.com/WoodScene/Paper_pictures/blob/main/KDD2021/TAIWAN_MAPE.png" width="480" height="360"/><br/>



