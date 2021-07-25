Hello, this is the final project in Computational Learning Final Project course.
The authors of this project are Bar Vered - 206201030 and Dor Avrahami - 315528794.
This project was done entirely by us, using the article that we chose.

This project is based on SAINT: Improved Neural Networks for Tabular Data via Row Attention and Contrastive Pre-Training
by Gowathami Somepalli, Micha Goldblum, Avi Schwartzschild and C. Bayan Bruss. All credits regarding the original code,
concepts and copyrights goes to them.

GitHub link - https://github.com/eurthokhcr/Final-Project

This rar archive contains:

 * Computational Learning Final Project - The report file containing all explanations and graphs.
 * Friedmann Test - an excel file containing all the calculations that were necessary for Stage 5.
 * stats - an excel file with 3 sheets for each algorithm containing the requested parameters.
 * README - this file which explains the project structure and how to run the program.
 
 
Requirements:
 * Python 3.8 (For RandomForest specifically python 3.7 is enough).
 * The following packages for the SAINTS editions: tensorflow, sklearn, SQLAlchemy, sklearn, numpy, pandas, Keras, imblearn, pytorch.
 * The following packages for RandomForest: sklearn, imblearn, numpy, pandas.
 
 
 
After doing all requirements, this is how to run RandomForest in RF Folder:
 
 * Cloning the git.
 * Referring to main.py file in RF Folder, lines 115 and lines 116 have parameters named 'dataset_name', 'DATASET_PATH',
   please write a dataset out of the 20 possible ones and in DATASET_PATH please write a valid path on your computer.
 * After changing the parameters above, you can run main.py file in RF Folder. The results will be added to a newely created 
   excel file, right now named '20.xlsx' but you can change it also in line 110.

This is how to run SAINT/Improved_SAINT in SAINT/Improved_SAINT. Folder:
 * Cloning the git.
 * Referring to line 712 and changing the STRING inside perform_nested_cv(STRING) to the dataset you want to analyze.
 * A folder named data with the dataset you want to analyze must exist under SAINT/Improved_SAINT.
 * After doing the stages above, you can run main.py file in SAINT/Improved_SAINT Folder. The results will be added to a newely created
   excel file, right now named '1.xlsx' but you can change it also in line 708.
   
   
Please notice that SAINT requires multi pre processing for each dataset inserted. This is why we created preprocessing for
20 datasets that were requested as part of the assignments in the final project. This means you can run the following datasets:
 * 1995_income
 * Arcene
 * abalon
 * acute-nephritis
 * Arrhythmia
 * Bank
 * cloud
 * Blastchar
 * Churn_Modeling
 * Creditcard - Cannot be cloned to git due to size, you can write 'creditcard' and SAINT will automatically download it. RandomForest needs manual download.
 * Forest - Cannot be cloned to git due to size, you can write 'forest' and SAINT will automatically download it. RandomForest needs manual download.
 * Htru2
 * Kdd99 - Cannot be cloned to git due to size, you can write 'kdd99' and SAINT will automatically download it. RandomForest needs manual download.
 * libras
 * Autos
 * Baseball
 * Blood
 * kidney
 * Qsar_bio
 * spambase
 
Please also notice that RandomForest runs take a medium amount of time (few minutes) but SAINT models can take a lot of time
depending on dataset. Small datasets such as 1995_income can take less than 10 minutes but kdd99 or creditcard can take a day.
 
These datasets are available in the github link.


   
