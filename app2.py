from flask import Flask, request, redirect, url_for, render_template, json, render_template_string, jsonify

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def control_panel():
    print('request.form:', request.form)
    
    if request.method == 'POST':
        if request.form.get('button') == 'button-play':
            print("play button pressed")

        elif request.form.get('button') == 'button-exit':
            print("exit button pressed")
        
        else:
            for i in range(1,5):
                if request.form.get(f'slide{i}'):
                    volume = request.form.get(f'slide{i}')
                    print('volume:', volume)
                    #return jsonify({'volume': volume})
                    return json.dumps({'volume': volume})

    print('render')
    return render_template('slider.html')

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)