'''
  This script load data from root and return a DataFrame
  Original script in ../examples/data_manager_Tallin.py is developped by Alexandra from TLL group
  Modified by Binghuan Li    --  1 Dec 2018 
'''

import sys , time
import itertools as it
import numpy as np
import pandas as pd
from root_numpy import root2array, tree2array 
from numpy.lib.recfunctions import append_fields
from itertools import product
from ROOT.Math import PtEtaPhiEVector,VectorUtil
import ROOT
import math , array
from random import randint
from scipy import stats
from matplotlib import pyplot as plt

inputPath = "../data/"
#variables = ["Jet_pt","Jet_qg","Jet_lepdrmin","Jet_lepdrmax","Jet_bDiscriminator"]#"EvtWeight","EvtWgtOVnJet"]
variables = ["Jet25_pt","Jet25_qg","Jet25_lepdrmin","Jet25_lepdrmax","Jet25_bDiscriminator"]#"EvtWeight","EvtWgtOVnJet25"]

#load_data_2017(inputPath, variables, False)  # select all jets
def load_data_2017(inputPath,variables,criteria) :
    print variables
    my_cols_list=variables+['proces', 'key', 'target',"totalWeight"]
    data = pd.DataFrame(columns=my_cols_list)
    #keys=['TTH_hww_HJet','TTW_NJet']
    keys=['TTH_HJet','TTV_NJet']
    print keys
    for key in keys :
        print key
        if 'ttH' in key or 'TTH' in key:
                sampleName='ttHNonbb'
        if 'ttW' in key or 'TTW' in key:
                sampleName='ttWJets'
        if 'ttV' in key or 'TTV' in key:
                sampleName='ttV'
        if 'HJet' in key:
                target=1
        if 'NJet' in key:
                target=0
        
        inputTree = 'syncTree'
        try: tfile = ROOT.TFile(inputPath+"/"+key+".root")
        except : 
            print " file "+ inputPath+"/"+key+".root deosn't exits "
            continue
        try: tree = tfile.Get(inputTree)
        except : 
            print inputTree + " deosn't exists in " + inputPath+"/"+key+".root"
            continue
        if tree is not None :
            try: chunk_arr = tree2array(tree, variables, criteria) #,  start=start, stop = stop)
            except : continue
            else :
                chunk_df = pd.DataFrame(chunk_arr, columns=variables)
                #print (len(chunk_df))
                #print (chunk_df.columns.tolist())
                chunk_df['proces']=sampleName
                chunk_df['key']=key
                chunk_df['target']=target
                # set weight to 1 
                chunk_df['totalWeight']=1
                # set negativ to zero to keep continous distribution
                chunk_df=chunk_df.clip_lower(0) 
                data=data.append(chunk_df, ignore_index=True)
        tfile.Close()
        if len(data) == 0 : continue
        nS = len(data.ix[(data.target.values == 1) & (data.key.values==key) ])
        nB = len(data.ix[(data.target.values == 0) & (data.key.values==key) ])
        print key,"length of sig, bkg: ", nS, nB , data.ix[ (data.key.values==key)]["totalWeight"].sum(), data.ix[(data.key.values==key)]["totalWeight"].sum()
        nNW = len(data.ix[(data["totalWeight"].values < 0) & (data.key.values==key) ])
        print key, "events with -ve weights", nNW
    print (data.columns.values.tolist())
    n = len(data)
    nS = len(data.ix[data.target.values == 1])
    nB = len(data.ix[data.target.values == 0])
    print " length of sig, bkg: ", nS, nB
    return data


def val_tune_rf(estimator,x_train,y_train, w_train, x_val,y_val, w_val, params, fileToWrite):
    ''' 
    tune parameters 
    '''
    from sklearn.model_selection import ParameterGrid
    from sklearn.metrics import roc_auc_score
    params_list = list(ParameterGrid(params))
    #print params_list
    #print y_val
    results = []
    for param in params_list:
        print ("Date: ", time.asctime( time.localtime(time.time()) ))
        print '=========  ',param
        estimator.set_params(**param)
        estimator.fit(x_train,y_train, sample_weight=w_train)
        preds_prob = estimator.predict_proba(x_val)
        preds_prob_train = estimator.predict_proba(x_train)
        print preds_prob
        print preds_prob[:,1]
        result = roc_auc_score(y_val,preds_prob[:,1], sample_weight = w_val)
        print 'roc_auc_score : %f'%result
        results.append((param,result))
        fileToWrite.write(str(param)+"\n")
        fileToWrite.write(str(roc_auc_score(y_val,preds_prob[:,1],sample_weight=w_val))+" "+str(roc_auc_score(y_train,preds_prob_train[:,1],sample_weight = w_train)))
        fileToWrite.write("\n")
        print ("Date: ", time.asctime( time.localtime(time.time()) ))
    results.sort(key=lambda k: k[1])
    #print results
    #print results[-1]
    return results
    
def make_plots(
    featuresToPlot,nbin,
    data1,label1,color1,
    data2,label2,color2,
    plotname,
    printmin
    ) :
    '''
    plot and save histograms
    '''
    print (len(featuresToPlot), featuresToPlot)
    hist_params = {'normed': True, 'histtype': 'bar', 'fill': True , 'lw':3}
    sizeArray=int(math.sqrt(len(featuresToPlot))) if math.sqrt(len(featuresToPlot)) % int(math.sqrt(len(featuresToPlot))) == 0 else int(math.sqrt(len(featuresToPlot)))+1
    drawStatErr=True
    plt.figure(figsize=(4*sizeArray, 4*sizeArray))
    for n, feature in enumerate(featuresToPlot):
        # add sub plot on our figure
        plt.subplot(sizeArray, sizeArray, n+1)
        # define range for histograms by cutting 1% of data from both ends
        min_value, max_value = np.percentile(data1[feature], [0.0, 99])
        min_value2, max_value2 = np.percentile(data2[feature], [0.0, 99])
        if printmin : print (min_value, max_value,feature)
        values1, bins, _ = plt.hist(data1[feature].values, weights= data1["totalWeight"].values.astype(np.float64) ,
                                   #range=(max(min(min_value,min_value2),0),  max(max_value,max_value2)), #  0.5 ),#
                                   range=(min(min_value,min_value2),  max(max_value,max_value2)), #  0.5 ),#
                                   bins=nbin, edgecolor=color1, color=color1, alpha = 0.4,
                                   label=label1, **hist_params )
        if drawStatErr:
            normed = sum(data1[feature].values)
            mid = 0.5*(bins[1:] + bins[:-1])
            err=np.sqrt(values1*normed)/normed # denominator is because plot is normalized
            plt.errorbar(mid, values1, yerr=err, fmt='none', color= color1, ecolor= color1, edgecolor=color1, lw=2)
        if 1>0 : #'gen' not in feature:
            values2, bins, _ = plt.hist(data2[feature].values, weights= data2["totalWeight"].values.astype(np.float64) ,
                                   #range=(max(min(min_value,min_value2),0),  max(max_value,max_value2)), # 0.5 ),#
                                   range=(min(min_value,min_value2),  max(max_value,max_value2)), # 0.5 ),#
                                   bins=nbin, edgecolor=color2, color=color2, alpha = 0.3,
                                   label=label2, **hist_params)
        if drawStatErr :
            normed = sum(data2[feature].values)
            mid = 0.5*(bins[1:] + bins[:-1])
            err=np.sqrt(values2*normed)/normed # denominator is because plot is normalized
            plt.errorbar(mid, values2, yerr=err, fmt='none', color= color2, ecolor= color2, edgecolor=color2, lw=2)
        #areaSig = sum(np.diff(bins)*values)
        #print areaBKG, " ",areaBKG2 ," ",areaSig
        plt.ylim(ymin=0.00001)
        if n == len(featuresToPlot)-1 : plt.legend(loc='best')
        plt.xlabel(feature)
        #plt.xscale('log')
        #plt.yscale('log')
    plt.ylim(ymin=0)
    plt.savefig(plotname)
    plt.clf()

def make_ks_plot(y_train, train_proba, y_test, test_proba, bins=30, fig_sz=(10, 8)):
    '''
        OUTPUT: outputs KS test/train overtraining plots for classifier output
        INPUTS:
        y_train - Series with outputs of model
        train_proba - np.ndarray from sklearn predict_praba(). Same shape as y_train. 0-1 probabilities from model.
        y_test - Series with outputs of model
        test_proba - np.ndarray from sklearn predict_praba(). Same shape as y_test. 0-1 probabilities from model.
        bins - number of bins for viz. Default 30.
        label_col_name - name of y-label. Change to whatever your model has it named. Default 'label'.
        fig_sz - change to True in order to get larger outputs. Default False.
    '''
    
    train = pd.DataFrame(y_train, columns=["target"])
    test = pd.DataFrame(y_test, columns=["target"])
    train["probability"] = train_proba
    test["probability"] = test_proba
                        
    decisions = []
    for df in [train, test]:
        d1 = df['probability'][df["target"] == 1]
        d2 = df['probability'][df["target"] == 0]
        decisions += [d1, d2]

    low = min(np.min(d) for d in decisions)
    high = max(np.max(d) for d in decisions)
    low_high = (low,high)

    fig = plt.figure(figsize=fig_sz)

    train_pos = plt.hist(decisions[0],
       color='b', alpha=0.5, range=low_high, bins=bins,
       histtype='stepfilled', normed=True, 
       label='sig train')

    train_neg = plt.hist(decisions[1],
       color='r', alpha=0.5, range=low_high, bins=bins,
       histtype='stepfilled', normed=True,
       label='bkg train')

    hist, bins = np.histogram(decisions[2],
          bins=bins, range=low_high, normed=True)
    scale = len(decisions[2]) / sum(hist)
    err = np.sqrt(hist * scale) / scale

    width = (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    test_pos = plt.errorbar(center, hist, yerr=err, fmt='o', c='b', label='sig test')

    hist, bins = np.histogram(decisions[3],
          bins=bins, range=low_high, normed=True)
    scale = len(decisions[2]) / sum(hist)
    err = np.sqrt(hist * scale) / scale

    test_neg = plt.errorbar(center, hist, yerr=err, fmt='o', c='r', label='bkg test')

    # get the KS score
    # If the K-S statistic is small or the p-value is high, then we cannot reject the hypothesis that the distributions of the two samples are the same.
    ks_sig = stats.ks_2samp(decisions[0], decisions[2])
    ks_bkg = stats.ks_2samp(decisions[1], decisions[3])

    plt.xlabel("Classifier Output", fontsize=12)
    plt.ylabel("Normalized Units", fontsize=12)

    plt.xlim(0, 1)
    plt.ylim(bottom=0)
    plt.plot([], [], ' ', label='Sig(Bkg) KS test p-value :'+str(round(ks_sig[1],2))+'('+str(round(ks_bkg[1],2))+')')
    plt.legend(loc='best', fontsize=12)
    plt.savefig("Hj_tagger_KS_test.png")
    plt.close()


