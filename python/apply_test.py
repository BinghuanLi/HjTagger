import pickle
import sys, os, subprocess
import optparse
import xgboost as xgb
import numpy as np
from sklearn.model_selection import KFold, train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.feature_selection import RFECV, RFE
from functools import partial
import json


rng = np.random.RandomState(31337)
exec(open("../python/load_data.py").read())

### optparse ####
usage = 'usage: %prog [options]'
parser = optparse.OptionParser(usage)
parser.add_option('-y', '--year',        dest='year'  ,      help='the data taking year',      default='2018',        type='string')
parser.add_option('-o', '--out',        dest='outputDir'  ,      help='the output base dir',      default='../outputs/test',        type='string')

(opt, args) = parser.parse_args()

year = opt.year 
outputDir = opt.outputDir 

outputPlotDir = outputDir + "/plots"
outputWgtDir = outputDir + "/weights"

print(" output plots saved in: ", outputPlotDir)
print(" output weights saved in: ", outputWgtDir)

if not os.path.exists(outputPlotDir):
    os.popen("mkdir -p "+outputPlotDir)

if not os.path.exists(outputWgtDir):
    os.popen("mkdir -p "+outputWgtDir)

##############
## options ##
##############
FeatureSelection = False
RFESelection = False
n_features = 13
GridSearch = False
postFix = "alljets" 
tagger = "ttWbkg"
####################################################################################################
## load input variables
with open('../scripts/input_variable_list.json') as json_file: 
#with open('../scripts/input_variable_RFE_list.json') as json_file: 
#with open('../scripts/input_ttV_event_variables_list.json') as json_file: 
    variable_dict = json.load(json_file)
    variable_list = [c for c in variable_dict.keys() if variable_dict[c]==1]
    variables=variable_list

## Load data
inputPath = "/home/binghuan/Work/TTHLep/TTHLep_RunII/ttH_hjtagger_xgboost/data/"
specs = ["Jet25_isToptag","run","ls","nEvent","DataEra"]
#data=load_data_2017(inputPath, variables,"Jet25_isToptag<0.5") # select only jets not tagged by TopTaggers 
data_test=load_data_events_2017(inputPath,variables,specs,year,False) # select all events 
data_test = data_test.fillna(0.)
#**********************

######################################
### load models
######################################
models = ["{}/{}_{}_2016".format(outputWgtDir, tagger, postFix), "{}/{}_{}_2017".format(outputWgtDir, tagger, postFix), "{}/{}_{}_2018".format(outputWgtDir, tagger, postFix) ] # model names

print (variables)
print (data_test.columns.values.tolist())

fig, ax = plt.subplots(figsize=(6, 6))
for model in models:
    clf = pickle.load(open("{}.pkl".format(model), "rb"))
    print ("XGBoost model {}".format(model))
    proba = clf.predict_proba(data_test[variables].values)
    dataT = group_event(data_test, proba[:,1])
    fpr, tpr, thresholds = roc_curve(dataT["target"], dataT["y_predict"].values,
        sample_weight=(dataT["totalWeight"].astype(np.float64)) )
    test_auc = auc(fpr, tpr, reorder = True)
    nS = len(dataT.iloc[dataT.target.values == 1])
    nB = len(dataT.iloc[dataT.target.values == 0])
    print ("test per-event len of sig, bkg : ", nS, nB) 
    print("XGBoost test per-event auc - {}".format(test_auc))
    
    tag = model[model.rfind("/")+1:]
    ## ROC curve
    ax.plot(fpr, tpr, lw=1, label='%s (area = %0.3f)'%(tag, test_auc))

ax.set_ylim([0.0,1.0])
ax.set_xlim([0.0,1.0])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.legend(loc="lower right")
ax.grid()
fig.savefig("{}/roc_eval_test-per-event_{}.png".format(outputPlotDir, year))
fig.savefig("{}/roc_eval_test-per-event_{}.pdf".format(outputPlotDir, year))
    
