import FWCore.ParameterSet.Config as cms
from time import time,ctime
import sys,os
execfile("../examples/tree_convert_pkl2xml.py")

import sklearn
from collections import OrderedDict
from sklearn.externals import joblib
#print('The scikit-learn version is {}.'.format(sklearn.__version__))
import pandas
#print('The pandas version is {}.'.format(pandas.__version__))
import cPickle as pickle
#print('The pickle version is {}.'.format(pickle.__version__))
import numpy as np
#print('The numpy version is {}.'.format(np.__version__))
sys.path.insert(0, '/cvmfs/cms.cern.ch/slc6_amd64_gcc530/external/py2-pippkgs_depscipy/3.0-njopjo7/lib/python2.7/site-packages')
import xgboost as xgb
#print('The xgb version is {}.'.format(xgb.__version__))
import subprocess
from sklearn.externals import joblib
from itertools import izip

inputFile_Dir = "/afs/cern.ch/work/b/binghuan/private/TTHLep2017/CMSSW_9_4_8/src/HjTagger/python/"
inputFile_Name = "XgBoost_wgtOVnjet_WTtoptag_HWWvsTTW/Hjtagger_wgtOVnjet.pkl"
inputFile = os.path.join(inputFile_Dir, inputFile_Name)
workingDir = os.getcwd()
outputFile_Dir = os.path.join(workingDir, "")
#outputFile_Name = inputFile_Name[0:-4]
outputFile_Name = "TrainMVA_wgtOVnjet_WTtoptag_HWWvsTTW_XGBoost.weights"
outputFile = os.path.join(outputFile_Dir, "%s%s" %(outputFile_Name,".xml"))

#features=["Jet25_pt","Jet25_qg","Jet25_lepdrmin","Jet25_lepdrmax","Jet25_bDiscriminator"]
features=['Jet25_bDiscriminator', 'Jet25_lepdrmin', 'Jet25_lepptratiomax', 'Jet25_pt', 'Jet25_lepptratiomin', 'Jet25_nonjdr', 'Jet25_lepdrmax', 'Jet25_bjdr', 'Jet25_qg', 'Jet25_nonjdilepptratio']

def mul():
    print 'Today is',ctime(time()), 'All python libraries we need loaded goodHTT'
    new_dict = OrderedDict([('Jet25_pt' , 33.410943),
                ('Jet25_qg' , 0.172003),
                ('Jet25_lepdrmin' , 0.9027),
                ('Jet25_lepdrmax' , 1.987),
                ('Jet25_bDiscriminator' , 0.177587),
                ('Jet25_lepptratiomax' , 1.515),
                ('Jet25_lepptratiomin' , 1.33),
                ('Jet25_nonjdr' , 0.6),
                ('Jet25_bjdr' , 0.7),
                ('Jet25_nonjdilepptratio' , 0.85)
                ])
    #print "new-dict =", new_dict
    data = pandas.DataFrame(columns=list(new_dict.keys()))
    data= data.append(new_dict, ignore_index=True)
    result=-20
    fileOpen = None
    try:
        fileOpen = open(inputFile, 'rb')
    except IOError as e:
        print('Couldnt open or write to file (%s).' % e)
    else:
        print ('file opened')
        try:
            pkldata = pickle.load(fileOpen)
        except :
            print('Oops!',sys.exc_info()[0],'occured.')
        else:
            print ('pkl loaded')
            #proba = pkldata.predict_proba(data[data.columns.values.tolist()].values  )
            proba = pkldata.predict_proba([[ new_dict[feature] for feature in features]])
            print "proba= ",proba
            result = proba[:,1][0]
            print ('predict BDT to one event',result)

            bdt = BDTxgboost(pkldata, features, ["Background", "Signal"])
            bdt.to_tmva(outputFile)
            print "xml file is created with name : ", outputFile
            #test_eval = bdt.eval([0.410943, 0.172003, 39.9027, 74.987, 0.177587, 0.864957, 31.3425])
            test_eval = bdt.eval([ new_dict[feature] for feature in features])
            #test_eval = bdt.eval(data[data.columns.values.tolist()].values[0])
            print "test_eval = ", test_eval
            #bdt.setup_tmva(outputFile)
            #test_eval_tmva = bdt.eval_tmva([0.410943, 0.172003, 39.9027, 74.987, 0.177587, 0.864957, 31.3425])
            #test_eval_tmva = bdt.eval_tmva([ new_dict[feature] for feature in features])
            #print "test_eval_tmva = ", test_eval_tmva

            fileOpen.close()
    return result

if __name__ == "__main__":
    mul()
