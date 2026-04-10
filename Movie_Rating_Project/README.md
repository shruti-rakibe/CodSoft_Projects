Movie Rating Prediction

Project Overview:
This project uses Machine Learning to predict the rating of a movie based on features such as genre, director, actors, duration, and year of release.
Movie rating prediction helps understand how different factors influence a movie's success and audience response.
This project demonstrates data preprocessing, feature selection, and regression modeling using Python.

Dataset Information:
The dataset contains information about movies and their ratings.

Column	              Description
Name	              Movie name
Year	              Release year
Duration	              Movie length (minutes)
Genre	              Type of movie (Action, Drama, Comedy, etc.)
Rating	              IMDb rating (Target variable)
Votes	              Number of user votes
Director	              Movie director
Actor 1	              Lead actor
Actor 2	              Supporting actor
Actor 3	              Supporting actor

Steps Performed:
1.Import dataset
2.Data cleaning
3.Handle missing values
4.Convert categorical values into numeric form
5.Split dataset into training and testing sets
6.Train machine learning model
7.Predict movie ratings
8.Evaluate model performance

Machine Learning Algorithm Used:
Regression algorithms are used because rating is a numeric value.

Result:
The Movie Rating Prediction model was successfully trained using machine learning algorithms. 
After preprocessing the dataset and selecting important features such as genre, director, cast, and release year, 
the model was able to predict movie ratings with good accuracy. 
Evaluation metrics like MAE, MSE, RMSE, and R² score indicate that the model performs effectively in estimating 
movie ratings based on historical data.

How to Run the Project:
Step 1: Install libraries
pip install pandas numpy matplotlib seaborn scikit-learn
Step 2: Run python file
python movie_model.py

Technologies Used:
Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
