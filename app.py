from flask import render_template, request, Flask
import pandas as pd
import pickle  # Importing pickle to load the model

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/', methods=['POST'])
def predict():
    a = request.form.get("AverageRainingDays")
    b = request.form.get("clonesize")
    c = request.form.get("AverageOfLowerTRange")
    d = request.form.get("AverageOfUpperTRange")
    e = request.form.get("honeybee")
    f = request.form.get("osmia")
    g = request.form.get("bumbles")
    h = request.form.get("andrena")

    # Convert the input values to numeric types
    try:
        a = float(a)
        b = float(b)
        c = float(c)
        d = float(d)
        e = float(e)
        f = float(f)
        g = float(g)
        h = float(h)
    except ValueError:
        return render_template("index.html", msg="Invalid input. Please enter numeric values.")

    # Commenting out xgboost regressor
    # clf = xgb.XGBRegressor()
    # clf.load_model('crop2.pkl')

    # Load the model using pickle
    with open('crop2.pkl', 'rb') as file:
        clf = pickle.load(file)

    x = pd.DataFrame([[a, b, c, d, e, f, g, h]], columns=['AverageRainingDays', 'clonesize', 'AverageOfLowerTRange',
                                                          'AverageOfUpperTRange', 'honeybee', 'osmia', 'bumbles', 'andrena'])

    prediction = clf.predict(x)[0]
    return render_template("index.html", msg=f"Predicted Crop Yield: {prediction:.2f}")

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
