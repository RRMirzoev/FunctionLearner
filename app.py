from flask import Flask, render_template, request
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
import io

import rpn
import learner

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/plot", methods=["POST"])
def plot():
    fun_str = request.form.get("function")
    a_str = request.form.get("a")
    b_str = request.form.get("b")

    try:
        fun_rpn = []
        if not rpn.tokenizer(fun_str, fun_rpn):
            raise ValueError("Invalid function.")

        a, b = float(a_str), float(b_str)
        if a >= b:
            raise ValueError("Invalid interval.")

        x = np.linspace(a, b, 1000)
        y = rpn.value(fun_rpn, x)

        model = learner.NeuralNetwork()
        model.train(x, y)
        plt.plot(x, y, label="True Function Values")
        plt.plot(x, model.predict(x), label="Neural Network Prediction", linestyle="--")
        plt.legend()
        plt.grid(True)

        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format="png")
        img_buffer.seek(0)
        plt.close()

        plot_data = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
        plot_url = f"data:image/png;base64,{plot_data}"

        return render_template('index.html', plot_url=plot_url, MSE=model.MSE)
    except ValueError as e:
        return render_template("index.html", error=str(e))

if __name__ == "__main__":
    app.run(debug=True)
