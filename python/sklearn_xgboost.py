'''
  This script train/valid and produce plots
  ../examples/sklearn_Xgboost_csv_evtLevel_ttH_Tallinn.py (developped by Alexandra from TLL group) and ../examples/sklearn_examples.py (example in XgBoost documentations by Jamie Hall ) are the starting point of this script
  Modified by Binghuan Li    --  3 Dec 2018 
'''

import pickle
import xgboost as xgb
import numpy as np
from sklearn.model_selection import KFold, train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.metrics import roc_curve, auc
from sklearn.feature_selection import RFECV, RFE
from functools import partial
import json

rng = np.random.RandomState(31337)

execfile("../python/load_data.py")

##############
## options ##
##############
FeatureSelection = False
RFESelection = False
n_features = 10
GridSearch = False


####################################################################################################
## load input variables
with open('../scripts/input_variable_RFE_WOwgt_list.json') as json_file: 
    variable_dict = json.load(json_file)
    #variable_list = [c for c in variable_dict.keys() if variable_dict[c]=="1"]
    variable_list = [c for c in variable_dict.keys() if variable_dict[c]==1]
    variables=variable_list

## Load data
data=load_data_2017(inputPath, variables,"Jet25_isToptag<0.5") # select only jets not tagged by TopTaggers 
#data=load_data_2017(inputPath, variables,False) # select all jets 
#**********************

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
    "Hj_tagger_feature_distribution",
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
print "length of sig, bkg used in train: ", nS, nB, " scale_pos_weight ", nB/nS 

# search and save parameters
if GridSearch :
    # http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
    param_grid = {
                'n_estimators': [ 500, 1000],
                'min_child_weight': [10, 100],
                'max_depth': [ 3, 4],
                'learning_rate': [0.01, 0.1]
                #'n_estimators': [ 500 ]
                }
    scoring = "roc_auc"
    early_stopping_rounds = 150 # Will train until validation_0-auc hasn't improved in 100 rounds.
    cv=3
    cls = xgb.XGBClassifier(scale_pos_weight = nB/nS)
    saveopt = "GridSearch_GSCV.log"
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
        CLS = xgb.XGBClassifier(scale_pos_weight = nB/nS)
        CLS.fit = partial(CLS.fit, sample_weight=(traindataset["totalWeight"].astype(np.float64)))
        selector = RFECV(CLS, step=1, cv=3)
        selector = selector.fit(
                traindataset[variables].values,
                traindataset.target.astype(np.bool)
        )
        saveFS = "FeatureSelection_RFECV.log"
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
        fig.savefig("Hj_feature_selection.png")
        fig.savefig("Hj_feature_selection.pdf")
    
    else:
        print (" REF feature selections ")
        CLS = xgb.XGBClassifier(scale_pos_weight = nB/nS)
        CLS.fit = partial(CLS.fit, sample_weight=(traindataset["totalWeight"].astype(np.float64)))
        selector = RFE(CLS, step=1, n_features_to_select = n_features)
        selector = selector.fit(
                traindataset[variables].values,
                traindataset.target.astype(np.bool)
        )
        saveFS = "FeatureSelection_RFE.log"
        file = open(saveFS,"w")
        file.write(" number of features : %d" % n_features + "\n")
        features_dict = {c: r for c, r in zip(traindataset.columns, selector.ranking_)}
        with open('../scripts/input_variable_RFE_WOwgt_list.json', 'w') as fp:
            json.dump(features_dict, fp)
        file.write(str(features_dict))
        print (" feature selection ranking ")
        print (variables)
        print (selector.ranking_)
    


if (not GridSearch) and (not FeatureSelection):
    
    clf = xgb.XGBClassifier(scale_pos_weight = nB/nS)
    
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
    pickle.dump(clf, open("Hjtagger.pkl", "wb"))
    clf2 = pickle.load(open("Hjtagger.pkl", "rb"))
    print(np.allclose(clf.predict(valdataset[variables].values), clf2.predict(valdataset[variables].values)))
    
    
    # Plot ROC curve
    print variables
    print traindataset[variables].columns.values.tolist()
    print ("XGBoost trained")
    proba = clf.predict_proba(traindataset[variables].values )
    fpr, tpr, thresholds = roc_curve(traindataset["target"], proba[:,1],
        sample_weight=(traindataset["totalWeight"].astype(np.float64)) )
    train_auc = auc(fpr, tpr)
    print("XGBoost train set auc - {}".format(train_auc))
    probaT = clf.predict_proba(valdataset[variables].values )
    fprt, tprt, thresholds = roc_curve(valdataset["target"], probaT[:,1], sample_weight=(valdataset["totalWeight"].astype(np.float64))  )
    test_auct = auc(fprt, tprt)
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
    fig.savefig("Hj_tagger_roc.png")
    fig.savefig("Hj_tagger_roc.pdf")
    
    ###########################################################################
    ## feature importance plot
    ###########################################################################
    fig, ax = plt.subplots()
    f_score_dict =clf.booster().get_fscore()
    print(" f_score_dict ")
    print( f_score_dict )
    f_score_dict = {variables[int(k[1:])] : v for k,v in f_score_dict.items()}
    feat_imp = pd.Series(f_score_dict).sort_values(ascending=True)
    feat_imp.plot(kind='barh', title='Feature Importances')
    fig.tight_layout()
    fig.savefig("Hj_tagger_feature_importance.png")
    fig.savefig("Hj_tagger_feature_importance.pdf")
    
    
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
            fig.savefig("Hj_tagger_feature_{}_correlation.png".format(label))
            fig.savefig("Hj_tagger_feature_{}_correlation.pdf".format(label))
            ax.clear()
    
    
    ###########################################################################
    # plot probability distribution and do KS-test
    ###########################################################################
    if 1>0: # FIXME
        make_ks_plot(traindataset["target"], proba[:,1], valdataset["target"], probaT[:,1])
    
