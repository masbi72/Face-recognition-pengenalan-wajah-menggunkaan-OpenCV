from flask import Flask, render_template
from routes.compare import compare_bp
from routes.capture import capture_bp
from routes.capture_dataset import capture_dataset_bp
from routes.gui_debug import gui_debug_bp

app = Flask(__name__)
app.register_blueprint(compare_bp)
app.register_blueprint(capture_bp)
app.register_blueprint(capture_dataset_bp)
app.register_blueprint(gui_debug_bp)

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
