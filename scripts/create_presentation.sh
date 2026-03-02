#!/bin/bash

rm -rf presentations/*.py presentations/requirements.txt presentations/README.md

echo "Creating presentation directory and copying relevant files..."
cp agents/orchestrator.py presentations/orchestrator.py
cp frontend/streamlit_app.py presentations/streamlit_app.py
cp ./requirements.txt presentations/requirements.txt
cp ./README.md presentations/README.md
echo "Presentation files created in the 'presentations' directory."

# To run the presentation, navigate to the 'presentations' directory and execute:
# cd presentations
# pip install -r requirements.txt
# streamlit run streamlit_app.py

echo "Presentation setup complete. Please navigate to the 'presentations' directory and follow the instructions in the README.md to run the presentation."
echo "Note: Ensure you have Streamlit installed and properly set up in your Python environment to run the presentation."
echo "Deploy on streamlit cloud: https://docs.streamlit.io/streamlit-cloud/get-started/deploy-an-app"
