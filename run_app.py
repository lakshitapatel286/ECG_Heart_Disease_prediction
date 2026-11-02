"""Start the Flask web app (run main.py first so that a trained model exists)."""
from app.app import app

if __name__ == "__main__":
    app.run(debug=False, port=5000)
