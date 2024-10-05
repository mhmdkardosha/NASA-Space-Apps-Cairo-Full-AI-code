# ExoBot

This repository contains all the files and the steps I took while developing our AI Spacebot.

The steps are the following:
- I downloaded NASA's Planetary System Composite data from [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/).
- I extracted the features description from this [site](https://exoplanetarchive.ipac.caltech.edu/docs/API_PS_columns.html) and scraped it using ```Beautiful Soup``` Library.
- I manipulated the data set and the description tables properly to make it ready to feed it to the model.
- I used Gemini API to get my model.
- I began with some exploring, trying the model and testing it.
- I made in the last step a ```streamlit``` web application and deployed my model on it.
