###############################################
#          Import some packages               #
###############################################
from flask import Flask, render_template,request
from flask_bootstrap import Bootstrap
from flask_nav import Nav
from flask_nav.elements import *
from dominate.tags import img
from forms import ContactForm
import pandas as pd
import joblib
import xgboost as xgb
import ast
from fonctions import get_newData_processed
import numpy as np
###############################################
#          Define flask app                   #
###############################################
global clv_estimation
clv_estimation=0
############################################################
###############.       Add à Logo       ####################
############################################################
logo = img(src='./static/img/logo.png', height="50", width="50", style="margin-top:-15px")
############################################################
###############     Create a NavBar     ####################
############################################################
topbar = Navbar(logo,
                View('newPrediction', 'get_new_prediction'),
                )

# registers the "top" menubar
nav = Nav()
nav.register_element('top', topbar)

app = Flask(__name__)
app.secret_key = 'KKSJBKhjahdoazijdazio99ksd;'
Bootstrap(app)

@app.route('/newPrediction', methods=["GET","POST"])
def get_new_prediction():
    ###############################################################
    # Load model
    ###############################################################
    path = './customerLifetimeValue/finalized_model_CLV.sav'
    loaded_model = joblib.load(path)
    ###############################################################
    # Create forms to introduce data to be predicted
    ###############################################################
    form = ContactForm()
    ###############################################################
    # We use post(), so the data isn't passed directly through the         
    #URL. You'll need to fetch attributes from the user input
    ###############################################################
    if request.method == 'POST':
        data = request.form["data"]
        data = ast.literal_eval(data)
        res = pd.DataFrame(data, index=range(1))
        ###########################################################  
        # Treat new data like train data
        ###########################################################
        res= get_newData_processed(res)
        ###########################################################  
        # predict results
        ###########################################################     
        result = pd.DataFrame({'pred':loaded_model.predict(res)})
        return(render_template('new_prediction.html', form=form, clv_estimation=np.round(result.values[0][0])))
        ############################################################
        # else, we return web page to introduce data to be predicted       
        ###############################################################    
    else:
        return(render_template('new_prediction.html', form=form, clv_estimation=0))



nav.init_app(app)

###############################################
#                Run app                      #
###############################################
if __name__ == '__main__':
    app.run(debug=True)
