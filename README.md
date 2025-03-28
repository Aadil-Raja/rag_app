
git clone https://github.com/your-username/rag-app.git
cd rag-app
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
cd app
uvicorn main:app --reload
