//Create a virtual environment and install the necessary packages:

python -m venv venv
source venv/bin/activate   # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt

// TRain the Model : Run the feature_extraction.py script to extract features and train the model. This script will save the trained model as Forest_model.pkl in the backend directory.

//Run backend server : Navigate to the backend directory and start the Flask server

python server.py

//Open the frontend/index.html file in your web browser.
//You can use a local server to serve the HTML file, such as the Live Server extension in VS Code or any simple HTTP server.

//Record and Predict : Use the web interface to record audio and get predictions from the trained model.

