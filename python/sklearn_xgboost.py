'''
  This script train/valid and produce plots
  ../examples/sklearn_Xgboost_csv_evtLevel_ttH_Tallinn.py (developped by Alexandra from TLL group) and ../examples/sklearn_examples.py (example in XgBoost documentations by Jamie Hall ) are the starting point of this script
  Modified by Binghuan Li    --  3 Dec 2018 
'''

import sys, os, subprocess
import optparse
import pickle
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
outputLogDir = outputDir + "/logs"

print(" output plots saved in: ", outputPlotDir)
print(" output weights saved in: ", outputWgtDir)
print(" output logs saved in: ", outputLogDir)

if not os.path.exists(outputPlotDir):
    os.popen("mkdir -p "+outputPlotDir)

if not os.path.exists(outputWgtDir):
    os.popen("mkdir -p "+outputWgtDir)

if not os.path.exists(outputLogDir):
    os.popen("mkdir -p "+outputLogDir)

##############
## options ##
##############
FeatureSelection = False
RFESelection = False
n_features = 13
GridSearch = False
postFix = "alljets" 
tagger = "ttWbkg"
ROC_test = False

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
data=load_data_2017(inputPath, variables, specs, year, False) # select all jets 
#**********************

data = data.fillna(0.)
#data.to_csv("data.csv")

#################################################################################
### Plot histograms of training variables


nbins=50
colorFast='r'
colorFastT='b'
colorFull='g'
hist_params = {'normed': True, 'histtype': 'bar', 'fill': False , 'lw':5}
#plt.figure(figsize=(60, 60))
labelBKG = "Background"
printmin=True
BDTvariables=variables
print (BDTvariables)

if (not FeatureSelection) and ( not GridSearch):
    make_plots(BDTvariables,nbins,
    data.ix[data.target.values == 0],labelBKG, colorFast,
    data.ix[data.target.values == 1],'Signal', colorFastT,
    "{}/{}_feature_distribution{}_{}".format(outputPlotDir,tagger,postFix,year),
    printmin
    )


#########################################################################################
## split dataset
traindataset, valdataset  = train_test_split(data, test_size=0.5, random_state=rng)
## to GridSearchCV the test_size should not be smaller than 0.4 == it is used for cross validation!
## to final BDT fit test_size can go down to 0.1 without sign of overtraining
#############################################################################################

print(" multiclass classification")

'''
print("Parameter optimization")
y = boston['target']
X = boston['data']
xgb_model = xgb.XGBRegressor()
clf = GridSearchCV(xgb_model,
                   {'max_depth': [2,4,6],
                    'n_estimators': [50,100,200]}, verbose=1)
clf.fit(X,y)
print(clf.best_score_)
print(clf.best_params_)

'''

############################
##### Book and Train Classifier
############################

# Early-stopping
nS = len(traindataset.ix[(traindataset.target.values == 1)])
nB = len(traindataset.ix[(traindataset.target.values == 0)])
print ("length of sig, bkg used in train: ", nS, nB, " scale_pos_weight ", float(nB)/float(nS) )

# search and save parameters
if GridSearch :
    # http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
    param_grid = {
                'n_estimators': [ 100, 200, 500],
                'min_child_weight': [ 1, 10, 50],
                'max_depth': [ 2, 3, 4],
                'learning_rate': [0.01, 0.1, 0.3]
                }
    scoring = "roc_auc"
    early_stopping_rounds = 150 # Will train until validation_0-auc hasn't improved in 100 rounds.
    cv=3
    cls = xgb.XGBClassifier(scale_pos_weight = float(nB)/float(nS))
    saveopt = "{}/GridSearch_GSCV_{}.log".format(outputLogDir,year)
    file = open(saveopt,"w")
    print ("opt being saved on ", saveopt)
    #file.write("Date: "+ str(time.asctime( time.localtime(time.time()) ))+"\n")
    file.write(str(variables)+"\n")
    result_grid = val_tune_rf(cls,
        traindataset[variables].values, traindataset["target"].astype(np.bool),traindataset["totalWeight"].astype(np.float64),
        valdataset[variables].values, valdataset["target"].astype(np.bool), valdataset["totalWeight"].astype(np.float64),
        param_grid, file)
    #file.write(result_grid)
    #file.write("Date: "+ str(time.asctime( time.localtime(time.time()) ))+"\n")
    file.close()
    print ("opt saved on ", saveopt)



# recursive elimination of features with cross validation
if FeatureSelection :
    # https://www.kaggle.com/mithrillion/a-few-python-tricks-to-mod-sklearn
    # trick to add sample weight
    if not RFESelection:
        print (" REFCV feature selections ")
        CLS = xgb.XGBClassifier(scale_pos_weight = float(nB)/float(nS))
        CLS.fit = partial(CLS.fit, sample_weight=(traindataset["totalWeight"].astype(np.float64)))
        selector = RFECV(CLS, step=1, cv=3)
        selector = selector.fit(
                traindataset[variables].values,
                traindataset.target.astype(np.bool)
        )
        saveFS = "{}/FeatureSelection_RFECV_{}.log".format(outputLogDir, year)
        file = open(saveFS,"w")
        file.write("Optimal number of features : %d" % selector.n_features_ + "\n")
        file.write(str({c: r for c, r in zip(traindataset.columns, selector.ranking_)}))
        print (" feature selection ranking ")
        print (variables)
        print (selector.ranking_)
    
        # Plot number of features VS. cross-validation scores
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlabel("Number of features selected")
        ax.set_ylabel("Cross validation score (nb of correct classifications)")
        ax.plot(range(1, len(selector.grid_scores_) + 1), selector.grid_scores_)
        fig.savefig("{}/Hj_feature_selection_{}.png".format(outputPlotDir, year))
        fig.savefig("{}/Hj_feature_selection_{}.pdf".format(outputPlotDir, year))
    
    else:
        print (" REF feature selections ")
        CLS = xgb.XGBClassifier(scale_pos_weight = float(nB)/float(nS))
        CLS.fit = partial(CLS.fit, sample_weight=(traindataset["totalWeight"].astype(np.float64)))
        selector = RFE(CLS, step=1, n_features_to_select = n_features)
        selector = selector.fit(
                traindataset[variables].values,
                traindataset.target.astype(np.bool)
        )
        saveFS = "{}/FeatureSelection_RFE_{}.log".format(outputLogDir, year)
        file = open(saveFS,"w")
        file.write(" number of features : %d" % n_features + "\n")
        features_dict = {c: r for c, r in zip(traindataset.columns, selector.ranking_)}
        with open('../scripts/input_variable_RFE_list.json', 'w') as fp:
            json.dump(features_dict, fp)
        file.write(str(features_dict))
        print (" feature selection ranking ")
        print (variables)
        print (selector.ranking_)
    


if (not GridSearch) and (not FeatureSelection):
    
    param_dist = {
                'n_estimators': 100,
                'min_child_weight': 1,
                'max_depth': 3,
                'learning_rate': 0.1
                }
    clf = xgb.XGBClassifier(scale_pos_weight = float(nB)/float(nS), **param_dist)
    
    clf.fit(
        traindataset[variables].values,
        traindataset.target.astype(np.bool),
        sample_weight=(traindataset["totalWeight"].astype(np.float64)),
        # more diagnosis, in case
        eval_set=[(traindataset[variables].values,  traindataset.target.astype(np.bool),traindataset["totalWeight"].astype(np.float64)),
        (valdataset[variables].values,  valdataset.target.astype(np.bool), valdataset["totalWeight"].astype(np.float64))],
        verbose=True,eval_metric="auc"
        )
    
    
    # The sklearn API models are picklable
    print("Pickling sklearn API models")
    # must open in binary format to pickle
    pickle.dump(clf, open("{}/{}_{}_{}.pkl".format(outputWgtDir, tagger, postFix, year), "wb"))
    clf2 = pickle.load(open("{}/{}_{}_{}.pkl".format(outputWgtDir, tagger, postFix, year), "rb"))
    print(np.allclose(clf.predict(valdataset[variables].values), clf2.predict(valdataset[variables].values)))
    
    
    # Plot ROC curve
    print (variables)
    print (traindataset[variables].columns.values.tolist())
    print ("XGBoost trained")
    proba = clf.predict_proba(traindataset[variables].values )
    # print (proba)
    fpr, tpr, thresholds = roc_curve(traindataset["target"], proba[:,1],
        sample_weight=(traindataset["totalWeight"].astype(np.float64)) )
    train_auc = auc(fpr, tpr, reorder = True)
    print("XGBoost train set auc - {}".format(train_auc))
    probaT = clf.predict_proba(valdataset[variables].values )
    fprt, tprt, thresholds = roc_curve(valdataset["target"], probaT[:,1], sample_weight=(valdataset["totalWeight"].astype(np.float64))  )
    test_auct = auc(fprt, tprt, reorder = True)
    print("XGBoost test set auc - {}".format(test_auct))
    fig, ax = plt.subplots(figsize=(6, 6))
    ## ROC curve
    #ax.plot(fprf, tprf, lw=1, label='GB train (area = %0.3f)'%(train_aucf))
    #ax.plot(fprtf, tprtf, lw=1, label='GB test (area = %0.3f)'%(test_auctf))
    ax.plot(fpr, tpr, lw=1, label='XGB train (area = %0.3f)'%(train_auc))
    ax.plot(fprt, tprt, lw=1, label='XGB test (area = %0.3f)'%(test_auct))
    #ax.plot(fprc, tprc, lw=1, label='CB train (area = %0.3f)'%(train_aucc))
    #ax.plot(fprtight, tprtight, lw=1, label='XGB test - tight ID (area = %0.3f)'%(test_auctight))
    #ax.plot(fprtightF, tprtightF, lw=1, label='XGB test - Fullsim All (area = %0.3f)'%(test_auctightF))
    ax.set_ylim([0.0,1.0])
    ax.set_xlim([0.0,1.0])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc="lower right")
    ax.grid()
    #fig.savefig("{}/{}_{}_{}_{}_roc.png".format(channel,bdtType,trainvar,str(len(trainVars(False))),hyppar))
    #fig.savefig("{}/{}_{}_{}_{}_roc.pdf".format(channel,bdtType,trainvar,str(len(trainVars(False))),hyppar))
    fig.savefig("{}/{}_roc{}_{}.png".format(outputPlotDir, tagger,postFix, year))
    fig.savefig("{}/{}_roc{}_{}.pdf".format(outputPlotDir, tagger,postFix, year))
    
    ###########################################################################
    ## feature importance plot
    ###########################################################################
    fig, ax = plt.subplots()
    f_score_dict =clf.get_booster().get_fscore()
    print(" f_score_dict ")
    print( f_score_dict )
    f_score_dict = {variables[int(k[1:])] : v for k,v in f_score_dict.items()}
    feat_imp = pd.Series(f_score_dict).sort_values(ascending=True)
    feat_imp.plot(kind='barh', title='Feature Importances')
    fig.tight_layout()
    fig.savefig("{}/{}_feature_importance{}_{}.png".format(outputPlotDir, tagger,postFix, year))
    fig.savefig("{}/{}_feature_importance{}_{}.pdf".format(outputPlotDir, tagger,postFix, year))
    
    
    ###########################################################################
    # plot correlation matrix
    ###########################################################################
    if 1>0: # FIXME
        for ii in [1,2] :
            if ii == 1 :
                datad=traindataset.loc[traindataset["target"].values == 1]
                label="signal"
            else :
                datad=traindataset.loc[traindataset["target"].values == 0]
                label="BKG"
            datacorr = datad[variables].astype(float)
            print (label)
            correlations = datacorr.corr()
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111)
            cax = ax.matshow(correlations, vmin=-1, vmax=1)
            ticks = np.arange(0,len(variables),1)
            plt.rc('axes', labelsize=8)
            ax.set_title(label)
            ax.xaxis.set_ticks_position('bottom')
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.set_xticklabels(variables,rotation=-90)
            ax.set_yticklabels(variables)
            fig.colorbar(cax)
            fig.tight_layout()
            #plt.subplots_adjust(left=0.9, right=0.9, top=0.9, bottom=0.1)
            fig.savefig("{}/{}_feature_{}_correlation{}_{}.png".format(outputPlotDir, tagger,label,postFix, year))
            fig.savefig("{}/{}_feature_{}_correlation{}_{}.pdf".format(outputPlotDir, tagger,label,postFix, year))
            ax.clear()
    
    
    ###########################################################################
    # plot probability distribution and do KS-test
    ###########################################################################
    if 1>0: # FIXME
        make_ks_plot(traindataset["target"], proba[:,1], valdataset["target"], probaT[:,1], plotname="{}/{}_ksTest{}_{}".format(outputPlotDir,tagger,postFix, year))


    ###########################################################
    # application on per-event level
    ########################################################### 
    if 1>0: # FIXME
        dataT = group_event(traindataset,proba[:,1])
        nS = len(dataT.iloc[dataT.target.values == 1])
        nB = len(dataT.iloc[dataT.target.values == 0])
        print ("traindataset per-event len of sig, bkg : ", nS, nB) 
        FPR, TPR, thresholds = roc_curve(dataT["target"], dataT["y_predict"].values,
            sample_weight=(dataT["totalWeight"].astype(np.float64)) )
        train_auc = auc(FPR, TPR, reorder = True)
        print("XGBoost per-event train set auc - {}".format(train_auc))
        #dataT.to_csv("dataT.csv",columns=["run","ls","nEvent","y_predict","target"])

        dataV = group_event(valdataset,probaT[:,1])
        FPRV, TPRV, thresholds = roc_curve(dataV["target"], dataV["y_predict"].values,
            sample_weight=(dataV["totalWeight"].astype(np.float64)) )
        test_auc = auc(FPRV, TPRV, reorder = True)
        nS = len(dataV.iloc[dataV.target.values == 1])
        nB = len(dataV.iloc[dataV.target.values == 0])
        print ("testdataset per-event len of sig, bkg : ", nS, nB) 
        print("XGBoost per-event test set auc - {}".format(test_auc))
        #dataV.to_csv("dataV.csv",columns=["run","ls","nEvent","y_predict","target"])
        fig, ax = plt.subplots(figsize=(6, 6))
        ## ROC curve
        ax.plot(FPR, TPR, lw=1, label='XGB train (area = %0.3f)'%(train_auc))
        ax.plot(FPRV, TPRV, lw=1, label='XGB test (area = %0.3f)'%(test_auc))
        ax.set_ylim([0.0,1.0])
        ax.set_xlim([0.0,1.0])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend(loc="lower right")
        ax.grid()
        fig.savefig("{}/{}_roc_perEvent_{}.png".format(outputPlotDir, tagger,postFix, year))
        fig.savefig("{}/{}_roc_perEvent_{}.pdf".format(outputPlotDir, tagger,postFix, year))
    
