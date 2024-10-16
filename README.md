# YouTube Video Engagement Prediction with Metadata
A complete Data Science Tutorial - Follow the Notebooks for a tour of the basics
---

**Project Overview:**

This project is designed to predict engagement (likes, comments, and shares) on electric scooter-related YouTube videos using various metadata attributes like video title, description, view count, and more. The project is broken down into four notebooks that guide you through the process of data collection, preprocessing, model training, and prediction. 

This project aims to provide hands-on experience with real-world machine learning tasks including data extraction, feature engineering, model building, and API deployment. The project follows a structured flow, allowing students to learn step by step while working on realistic data problems.

---

**Learning Objectives:**

1. Understand how to collect and preprocess YouTube metadata for machine learning tasks.
2. Learn how to perform feature engineering, including handling text data and numeric features.
3. Build and evaluate classification models to predict video engagement based on likes.
4. Deploy a machine learning model using FastAPI and understand how to consume the API for predictions.
5. Develop a deeper understanding of how to integrate cloud services (e.g., S3) into machine learning workflows.

---

### **Notebook Overview**

1. **Notebook 1: Data Collection (YouTube API & Metadata Extraction)**
   - **Objective:** Learn how to gather metadata from the YouTube API using video URLs or video IDs.
   - **What to Expect:** 
     - Setup of the YouTube Data API.
     - Extraction of relevant metadata (e.g., title, description, view count, like count, etc.).
     - Storing the data in a pandas DataFrame for further processing.
   - **Key Skills Learned:**
     - Working with APIs.
     - Collecting real-world data using API requests.
     - Storing and processing JSON responses in Python.

2. **Notebook 2: Data Preprocessing & Feature Engineering**
   - **Objective:** Understand how to preprocess and transform the raw data into a form suitable for machine learning models.
   - **What to Expect:**
     - Handling missing values, outliers, and invalid data.
     - Text processing for tags and descriptions (tokenization, TF-IDF).
     - Creating new features from video metadata (e.g., duration, category encoding).
     - Data normalization and scaling.
   - **Key Skills Learned:**
     - Feature engineering and extraction from text and numeric fields.
     - Preprocessing techniques including handling missing data and normalization.
     - Preparing data for machine learning models.

3. **Notebook 3: Model Training and Evaluation**
   - **Objective:** Build and evaluate a Random Forest classifier to predict video engagement categories.
   - **What to Expect:**
     - Defining the target variable by creating thresholds based on video likes (low, medium, high engagement).
     - Training the RandomForestClassifier on the transformed data.
     - Evaluating the model’s performance using metrics such as accuracy, precision, and recall.
   - **Key Skills Learned:**
     - Supervised learning (classification) with RandomForest.
     - Model evaluation using cross-validation and confusion matrices.
     - Hyperparameter tuning for better performance.

4. **Notebook 4: Deploying the Model with FastAPI**
   - **Objective:** Deploy the trained model using FastAPI and interact with it through an API.
   - **What to Expect:**
     - Saving the trained model using `joblib` and storing it in an S3 bucket (optional).
     - Setting up a FastAPI server that allows for predictions based on new input data.
     - Writing a function to fetch the model and make predictions via an API endpoint.
     - Testing the FastAPI server with a simple client.
   - **Key Skills Learned:**
     - Building and deploying machine learning models as RESTful APIs.
     - Working with FastAPI and handling API requests.
     - Loading models and predicting in a cloud-based environment.

---

### **Expected Folder Structure:**

```
.
│
├── Notebook1.ipynb
├── Notebook2.ipynb
├── Notebook3.ipynb
├── Notebook4.ipynb
├── model.pkl                # Trained machine learning model saved as a .pkl file.
├── requirements.txt         # Dependencies required to run the project.
└── README.md                # Project overview and learning objectives (this file).
```

---

### **Dependencies:**

1. **Python Libraries**:
   - `pandas`
   - `numpy`
   - `scikit-learn`
   - `requests`
   - `joblib`
   - `FastAPI`
   - `uvicorn`
   - `boto3` (optional, for interacting with S3)

2. **APIs**:
   - YouTube Data API (API key needed).

3. **Cloud Services (Optional)**:
   - AWS S3 for storing models and input data.

---

### **Running the Notebooks:**

1. Clone the project repository and navigate to the project folder.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open the notebooks sequentially in Jupyter or any compatible environment (e.g., Google Colab).

---

### **FastAPI Instructions:**

- To run the FastAPI server (Notebook 4), use the following command:
  ```bash
  uvicorn app:app --reload --host 0.0.0.0 --port 5000
  ```

- After the server is running, you can test predictions by sending a POST request to the `/predict` endpoint using the input data from the preprocessed dataset.

---

### **Next Steps for Learners:**

1. Experiment with different feature engineering techniques to improve the model's accuracy.
2. Explore other machine learning algorithms such as Gradient Boosting, or even deep learning models.
3. Enhance the FastAPI application to include model retraining or support for batch predictions.

By the end of these notebooks, you will have built a fully functional machine learning model that predicts video engagement and deployed it as a web service using FastAPI!

## Author
Name: Turyal Neeshat
Contact: [tneeshat@outlook.com]