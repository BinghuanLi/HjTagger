import itertools as it
import numpy as np
#from root_numpy import root2array, stretch
from numpy.lib.recfunctions import append_fields
from itertools import product
from ROOT.Math import PtEtaPhiEVector,VectorUtil
import ROOT
import math , array
from random import randint

def load_ttHGen() :
    procP1=glob.glob("/hdfs/cms/store/user/atiko/VHBBHeppyV25tthtautau/MC/ttHJetToNonbb_M125_13TeV_amcatnloFXFX_madspin_pythia8_mWCutfix/VHBB_HEPPY_V25tthtautau_ttHJetToNonbb_M125_13TeV_amcatnloFXFX_madspin_Py8_mWCutfix__RunIISummer16MAv2-PUMoriond17_80r2as_2016_TrancheIV_v6_ext1-v1/170207_122849/0000/tree_*.root")
    list=procP1
    for ii in range(0, len(list)) : #
    	#print (list[ii],inputTree)
    	try: tfile = ROOT.TFile(list[ii])
    	except :
    		#print "Doesn't exist"
    		#print ('file ', list[ii],' corrupt')
    		continue
    	try: tree = tfile.Get("tree")
    	except :
    		#print "Doesn't exist"
    		#print ('file ', list[ii],' corrupt')
    		continue
    	if tree is not None :
    		try:
    			chunk_arr = tree2array(tree) #,  start=start, stop = stop)
    		except :
    			#print "Doesn't exist"
    			#print ('file ', list[ii],' corrupt')
    			continue
    		else :
    			chunk_df = pandas.DataFrame(chunk_arr) #
    			#if ii ==0 : print (chunk_df.columns.values.tolist())
    			chunk_df['key']=folderName
    			chunk_df['target']=target
    			data=data.append(chunk_df, ignore_index=True)
    	else : print ("file "+list[ii]+"was empty")
    	tfile.Close()
        	#if len(data) == 0 : continue
        #print ("weigths", data.loc[data['target']==0]["totalWeight"].sum() , data.loc[data['target']==1]["totalWeight"].sum() )
        return data



def load_dataGen(inputPath,channelInTree,variables,criteria,testtruth,folderName) :
    print variables
    my_cols_list=variables+['key','target','file']+criteria #,'tau_frWeight','lep1_frWeight','lep1_frWeight' trainVars(False)
    # if channel=='2lss_1tau' : my_cols_list=my_cols_list+['tau_frWeight','lep1_frWeight','lep2_frWeight']
    # those last are only for channels where selection is relaxed (2lss_1tau) === solve later
    data = pandas.DataFrame(columns=my_cols_list)
    #if bdtType=="all" : keys=['ttHToNonbb','TTZToLLNuNu','TTWJetsToLNu','TTTo2L2Nu','TTToSemilepton']
    #if bdtType=="all" : keys=['TTToSemilepton']
    #for folderName in keys :
    if 1>0 :
    	print (folderName, channelInTree)
    	if 'TTT' in folderName :
    		sampleName='TT'
    		target=0
    	if folderName=='ttHToNonbb' :
    		sampleName='signal'
    		target=1
    	if 'TTW' in folderName :
    		sampleName='TTW'
    		target=0
    	if 'TTZ' in folderName :
    		sampleName='TTZ'
    		target=0
    	inputTree = channelInTree+'/sel/evtntupleGen/'+sampleName+'/evtTree'
        # inputTree = channelInTree+'/sel/evtntuple/'+sampleName+'/evtTree'
    	if ('TTT' in folderName) or folderName=='ttHToNonbb' :
            procP1=glob.glob(inputPath+"/"+folderName+"_fastsim_p1/"+folderName+"_fastsim_p1_forBDTtraining*OS_central_*.root")
            procP2=glob.glob(inputPath+"/"+folderName+"_fastsim_p2/"+folderName+"_fastsim_p2_forBDTtraining*OS_central_*.root")
            procP3=glob.glob(inputPath+"/"+folderName+"_fastsim_p3/"+folderName+"_fastsim_p3_forBDTtraining*OS_central_*.root")
            list=procP1+procP2+procP3
        else :
            procP1=glob.glob(inputPath+"/"+folderName+"_fastsim/"+folderName+"_fastsim_forBDTtraining*OS_central_*.root")
            list=procP1
    	#print ("Date: ", time.asctime( time.localtime(time.time()) ))
    	for ii in range(0, len(list)) : #
    		#print (list[ii],inputTree)
    		try: tfile = ROOT.TFile(list[ii])
    		except :
    			#print "Doesn't exist"
    			#print ('file ', list[ii],' corrupt')
    			continue
    		try: tree = tfile.Get(inputTree)
    		except :
    			#print "Doesn't exist"
    			#print ('file ', list[ii],' corrupt')
    			continue
    		if tree is not None :
    			try:
    				chunk_arr = tree2array(tree) #,  start=start, stop = stop)
    			except :
    				#print "Doesn't exist"
    				#print ('file ', list[ii],' corrupt')
    				continue
    			else :
    				chunk_df = pandas.DataFrame(chunk_arr) #
    				#if ii ==0 : print (chunk_df.columns.values.tolist())
    				chunk_df['key']=folderName
    				chunk_df['target']=target
    				data=data.append(chunk_df, ignore_index=True)
    		else : print ("file "+list[ii]+"was empty")
    		tfile.Close()
    	#if len(data) == 0 : continue
        data = data.ix[data.evtWeight.values <1]
    	nS = len(data.ix[(data.target.values == 0) & (data.key.values==folderName)])
    	nB = len(data.ix[(data.target.values == 1) & (data.key.values==folderName)])
    if folderName=='ttHToNonbb' : print (data.columns.values.tolist())
    nS = len(data.ix[data.target.values == 0])
    nB = len(data.ix[data.target.values == 1])
    print channelInTree," length of sig, bkg: ", nS, nB
    #print ("weigths", data.loc[data['target']==0]["totalWeight"].sum() , data.loc[data['target']==1]["totalWeight"].sum() )
    return data





def load_data_2017(inputPath,channelInTree,variables,criteria,bdtType) :
    #print variables
    print bdtType
    my_cols_list=variables+['proces', 'key', 'target', "totalWeight"]
    data = pandas.DataFrame(columns=my_cols_list)
    if bdtType=="evtLevelTT_TTH" : keys=['ttHToNonbb','TTTo2L2Nu', 'TTToHadronic','TTToSemiLeptonic']
    if bdtType=="evtLevelTTV_TTH" : keys=['ttHToNonbb','TTWJets','TTZJets']
    if bdtType=="evtLevelDY_TTH" : keys=['ttHToNonbb','DYJetsToLL', 'DY1JetsToLL', 'DY2JetsToLL', 'DY3JetsToLL', 'DY4JetsToLL']
    if "evtLevelSUM_TTH" in bdtType : keys=['ttHToNonbb','TTWJets','TTZJets','TTTo2L2Nu','TTToSemiLeptonic'] # 'TTToHadronic',
    if "evtLevelSUM_HH" in bdtType :
        keys=[
    'TTWJets','TTZJets','TTTo2L2Nu','TTToSemiLeptonic', 'TTToHadronic',
    'signal_ggf_spin0_400_hh_2v2t', 'signal_ggf_spin0_400_hh_4t',  'signal_ggf_spin0_400_hh_4v',
    'signal_ggf_spin0_700_hh_2v2t', 'signal_ggf_spin0_700_hh_4t',   'signal_ggf_spin0_700_hh_4v',
    ## missing: diboson/ singleH
        ]
        masses = [400, 700]
    if channel in ["0l_2tau"] : keys = keys + ["DYJetsToLL"] ## list of channels to process DY for training
    if bdtType=="all" : keys=['ttHToNonbb','TTWJets','TTZJets','TTTo2L2Nu', 'TTToHadronic', 'TTToSemiLeptonic']
    print keys
    for folderName in keys :
        print (folderName, channelInTree)
        if 'TTT' in folderName :
                sampleName='TT'
                target=0
        if "evtLevelSUM_HH" in bdtType :
            if 'signal_ggf_spin0' in folderName :
                sampleName='signal_ggf_spin0_'
                for mass in masses :
                    if str(mass) in folderName : sampleName=sampleName+str(mass)
                if '_4t' in folderName : sampleName=sampleName+'_hh_tttt'
                if '_4v' in folderName : sampleName=sampleName+'_hh_wwww'
                if '_2v2t' in folderName : sampleName=sampleName+'_hh_wwtt'
                target=1
        elif folderName=='ttHToNonbb' :
                sampleName='signal'
                target=1
        if 'TTW' in folderName :
                sampleName='TTW'
                target=0
        if 'TTZ' in folderName :
                sampleName='TTZ'
                target=0
        if 'DY' in folderName :
                sampleName='EWK'
                target=0
        inputTree = channelInTree+'/sel/evtntuple/'+sampleName+'/evtTree'
        if folderName=='ttHToNonbb' :
            procP1=glob.glob(inputPath+"/"+folderName+"_M125_powheg/"+folderName+"*.root")
            list=procP1
        elif 'signal_ggf_spin0' in folderName :
            procP1=glob.glob(inputPath+"/"+folderName+"*/"+folderName+"*.root")
            list=procP1
        elif ('TTT' in folderName):
            procP1=glob.glob(inputPath+"/"+folderName+"*/"+folderName+"*.root")
            list=procP1
        elif ('TTW' in folderName) or ('TTZ' in folderName):
            procP1=glob.glob(inputPath+"/"+folderName+"_LO/"+folderName+"*.root")
            list=procP1
        elif ('DY' in folderName):
            procP1=glob.glob(inputPath+"/"+folderName+"*/"+folderName+"*.root")
            list=procP1
        for ii in range(0, len(list)) :
            #print (list[ii], inputTree)
            try: tfile = ROOT.TFile(list[ii])
            except : continue
            try: tree = tfile.Get(inputTree)
            except : continue
            if tree is not None :
                try: chunk_arr = tree2array(tree) #,  start=start, stop = stop)
                except : continue
                else :
                    chunk_df = pandas.DataFrame(chunk_arr, columns=variables)
                    #print (len(chunk_df))
                    #print (chunk_df.columns.tolist())
                    chunk_df['proces']=sampleName
                    chunk_df['key']=folderName
                    chunk_df['target']=target
                    if channel=="2los_1tau": chunk_df["totalWeight"] = 1
                    else : chunk_df["totalWeight"] = chunk_df["evtWeight"]
                    #if channel=="0l_2tau" :
                    #    chunk_df["tau1_eta"]=abs(chunk_df["tau1_eta"])
                    #    chunk_df["tau2_eta"]=abs(chunk_df["tau2_eta"])
                    #    chunk_df["HadTop1_eta"]=abs(chunk_df["HadTop1_eta"])
                    #    chunk_df["HadTop2_eta"]=abs(chunk_df["HadTop2_eta"])
                    if channel in ["2l_2tau_HH", "0l_2tau"] :
                        chunk_df["tau1_eta"]=abs(chunk_df["tau1_eta"])
                        chunk_df["tau2_eta"]=abs(chunk_df["tau2_eta"])
                        chunk_df["max_tau_eta"]=chunk_df[["tau1_eta", "tau2_eta"]].max(axis=1)
                    if channel == "3l_0tau" :
                        chunk_df["lep1_eta"] = abs(chunk_df["lep1_eta"])
                        chunk_df["lep2_eta"] = abs(chunk_df["lep2_eta"])
                        chunk_df["lep3_eta"] = abs(chunk_df["lep3_eta"])
                        chunk_df["max_lep_eta"]=chunk_df[["lep1_eta","lep2_eta","lep3_eta"]].max(axis=1)
                        chunk_df["max_lep_mT"]=chunk_df[["mT_lep1", "mT_lep2", "mT_lep3"]].max(axis=1)
                        chunk_df["min_lep_mT"]=chunk_df[["mT_lep1", "mT_lep2", "mT_lep3"]].min(axis=1)
                        chunk_df["max_lep_dr_os"]=chunk_df[["dr_los1","dr_los2"]].max(axis=1)
                        chunk_df["min_lep_dr_os"]=chunk_df[["dr_los1","dr_los2"]].min(axis=1)
                    ### gen- level info
                    if "evtLevelSUM_HH_res" in bdtType :
                        foundMass = False
                        for mass in masses :
                            if str(mass) in folderName :
                                chunk_df["gen_mHH"]=mass
                                foundMass = True
                        if not foundMass : chunk_df["gen_mHH"]=masses[randint(0, len(masses)-1)]
                    data=data.append(chunk_df, ignore_index=True)
            else : print ("file "+list[ii]+"was empty")
            tfile.Close()
        if len(data) == 0 : continue
        nS = len(data.ix[(data.target.values == 1) & (data.key.values==folderName) ])
        nB = len(data.ix[(data.target.values == 0) & (data.key.values==folderName) ])
        print folderName,"length of sig, bkg: ", nS, nB , data.ix[ (data.key.values==folderName)]["totalWeight"].sum(), data.ix[(data.key.values==folderName)]["totalWeight"].sum()
        nNW = len(data.ix[(data["totalWeight"].values < 0) & (data.key.values==folderName) ])
        print folderName, "events with -ve weights", nNW
    print (data.columns.values.tolist())
    n = len(data)
    nS = len(data.ix[data.target.values == 1])
    nB = len(data.ix[data.target.values == 0])
    print channelInTree," length of sig, bkg: ", nS, nB
    return data


def load_data(inputPath,channelInTree,variables,criteria,testtruth,bdtType) :
    print variables
    my_cols_list=variables+['key','target',"totalWeight"]
    data = pandas.DataFrame(columns=my_cols_list)
    if bdtType=="evtLevelTT_TTH" : keys=['ttHToNonbb','TTTo2L2Nu','TTToSemilepton']
    if bdtType=="evtLevelTTV_TTH" : keys=['ttHToNonbb','TTZToLLNuNu','TTWJetsToLNu']
    if "evtLevelSUM_TTH" in bdtType : keys=['ttHToNonbb','TTZToLLNuNu','TTWJetsToLNu','TTTo2L2Nu','TTToSemilepton']
    if bdtType=="all" : keys=['ttHToNonbb','TTZToLLNuNu','TTWJetsToLNu','TTTo2L2Nu','TTToSemilepton']
    for folderName in keys :
        print (folderName, channelInTree)
        if 'TTT' in folderName :
        	sampleName='TT'
        	target=0
        if folderName=='ttHToNonbb' :
        	sampleName='signal'
        	target=1
        if 'TTW' in folderName :
        	sampleName='TTW'
        	target=0
        if 'TTZ' in folderName :
        	sampleName='TTZ'
        	target=0
        inputTree = channelInTree+'/sel/evtntuple/'+sampleName+'/evtTree'
        if ('TTT' in folderName) or folderName=='ttHToNonbb' :
             	procP1=glob.glob(inputPath+"/"+folderName+"_fastsim_p1/"+folderName+"*.root")
        	procP2=glob.glob(inputPath+"/"+folderName+"_fastsim_p2/"+folderName+"*.root")
        	procP3=glob.glob(inputPath+"/"+folderName+"_fastsim_p3/"+folderName+"*.root")
        	list=procP1+procP2+procP3
        else :
        	procP1=glob.glob(inputPath+"/"+folderName+"_fastsim/"+folderName+"*.root")
        	list=procP1
        for ii in range(0, len(list)) :
            try: tfile = ROOT.TFile(list[ii])
            except : continue
            try: tree = tfile.Get(inputTree)
            except : continue
            if tree is not None :
                try: chunk_arr = tree2array(tree) #,  start=start, stop = stop)
                except : continue
                else :
                    chunk_df = pandas.DataFrame(chunk_arr)
                    #print (len(chunk_df))
                    #print (chunk_df.columns.tolist())
                    chunk_df['proces']=sampleName
                    chunk_df['key']=folderName
                    chunk_df['target']=target
                    if channel=="1l_2tau" : chunk_df["totalWeight"] = chunk_df.evtWeight*chunk_df["prob_fake_lepton"]*chunk_df["tau_fake_prob_lead"]*chunk_df["tau_fake_prob_sublead"]
                    if channel=="2lss_1tau" :
                        chunk_df["totalWeight"] = chunk_df["evtWeight"]*chunk_df["lep1_fake_prob"]*chunk_df["lep2_fake_prob"]
                        chunk_df["max_eta_Lep"]=chunk_df[['tau_eta', 'lep1_eta', 'lep2_eta']].max(axis=1)
                    if channel=="2los_1tau" :
                        chunk_df["totalWeight"] = chunk_df["evtWeight"]*chunk_df["lep1_fake_prob"]*chunk_df["lep2_fake_prob"]*chunk_df["tau_fake_prob"]
                    if channel=="3l_1tau" :
                        chunk_df["lep1_eta"]=abs(chunk_df["lep1_eta"])
                        chunk_df["lep2_eta"]=abs(chunk_df["lep2_eta"])
                        chunk_df["lep3_eta"]=abs(chunk_df["lep3_eta"])
                        chunk_df["max_lep_eta"]=chunk_df[["lep1_eta","lep2_eta","lep3_eta"]].max(axis=1)
                        chunk_df["avr_lep_eta"]=(abs(chunk_df["lep2_eta"])+abs(chunk_df["lep1_eta"])+abs(chunk_df["lep3_eta"]))/2.
                        chunk_df["tau_eta"]=abs(chunk_df["tau_eta"])
                        WtoMultiply=chunk_df["lep1_fake_prob"]*chunk_df["lep2_fake_prob"]*chunk_df["lep3_fake_prob"] #*chunk_df["tau_fake_prob"]
                        chunk_df["totalWeight"] = chunk_df["evtWeight"]*WtoMultiply
                    if channel=="2l_2tau" :
                        chunk_df["min_dr_lep_tau"]=chunk_df[["dr_lep1_tau1", "dr_lep1_tau2", "dr_lep2_tau1", "dr_lep2_tau2"]].min(axis=1)
                        chunk_df["max_dr_lep_tau"]=chunk_df[["dr_lep1_tau1", "dr_lep1_tau2", "dr_lep2_tau1", "dr_lep2_tau2"]].max(axis=1)
                        chunk_df["min_dr_lep_jet"]=chunk_df[["mindr_lep1_jet", "mindr_lep2_jet"]].min(axis=1)
                        chunk_df["mindr_tau_jet"]=chunk_df[["mindr_tau1_jet","mindr_tau2_jet"]].min(axis=1)
                        chunk_df["avr_dr_lep_tau"]=(chunk_df['dr_lep1_tau1']+chunk_df['dr_lep1_tau2']+chunk_df['dr_lep2_tau1']+chunk_df['dr_lep2_tau2'])/4.
                        chunk_df["lep1_eta"]=abs(chunk_df["lep1_eta"])
                        chunk_df["lep2_eta"]=abs(chunk_df["lep2_eta"])
                        chunk_df["avr_lep_eta"]=(abs(chunk_df["lep2_eta"])+abs(chunk_df["lep1_eta"]))/2.
                        chunk_df["tau1_eta"]=abs(chunk_df["tau1_eta"])
                        chunk_df["tau2_eta"]=abs(chunk_df["tau2_eta"])
                        chunk_df["avr_tau_eta"]=(abs(chunk_df["tau2_eta"])+abs(chunk_df["tau1_eta"]))/2.
                        chunk_df["leptonPairCharge"]=abs(chunk_df["leptonPairCharge"])
                        chunk_df["hadTauPairCharge"]=abs(chunk_df["hadTauPairCharge"])
                        WtoMultiply=chunk_df["lep1_fake_prob"]*chunk_df["lep2_fake_prob"]*chunk_df["tau1_fake_prob"]*chunk_df["tau2_fake_prob"]
                        chunk_df["totalWeight"] = chunk_df["evtWeight"]*WtoMultiply
                    if channel=="0l_2tau" :
                        chunk_df["totalWeight"] = chunk_df["evtWeight"]
                        chunk_df["tau1_eta"]=abs(chunk_df["tau1_eta"])
                        chunk_df["tau2_eta"]=abs(chunk_df["tau2_eta"])
                        chunk_df["HadTop1_eta"]=abs(chunk_df["HadTop1_eta"])
                        chunk_df["HadTop2_eta"]=abs(chunk_df["HadTop2_eta"])
                    data=data.append(chunk_df, ignore_index=True)
            else : print ("file "+list[ii]+"was empty")
            tfile.Close()
        if len(data) == 0 : continue
        nS = len(data.ix[(data.target.values == 1) & (data.key.values==folderName) ])
        nB = len(data.ix[(data.target.values == 0) & (data.key.values==folderName) ])
        print folderName,"length of sig, bkg: ", nS, nB , data.ix[ (data.key.values==folderName)]["totalWeight"].sum(), data.ix[(data.key.values==folderName)]["totalWeight"].sum()
        if channel=="2l_2tau" :
            print "tau1 all | lep  ",sampleName , len(data.ix[(data["tau1_fake_prob"].values != 1) & (data.proces.values==sampleName)]), len(data.ix[(data["tau1_fake_test"].values != 1) & (data.proces.values==sampleName)])
            print "tau2 all | lep  ",sampleName , len(data.ix[(data["tau2_fake_prob"].values != 1) & (data.proces.values==sampleName)]), len(data.ix[(data["tau2_fake_test"].values != 1) & (data.proces.values==sampleName)])
            print "tau1&2 all | lep  ",sampleName ,len(data.ix[(data["tau1_fake_prob"].values != 1) & (data["tau2_fake_prob"].values != 1) & (data.proces.values==sampleName)]), len(data.ix[(data["tau1_fake_test"].values != 1) & (data['tau2_fake_test'].values != 1) & (data.proces.values==sampleName)])
            datatest=data["evtWeight"]*data["lep1_fake_prob"]*data["lep2_fake_prob"]*data["tau1_fake_test"]*data["tau2_fake_test"]
            print "sum of weights with/ without lepton in FR ",sampleName , datatest.ix[ (data.proces.values==sampleName)].sum(),data.ix[ (data.proces.values==sampleName)]["totalWeight"].sum()
        if (channel=="1l_2tau" or channel=="2lss_1tau" or channel=="2los_1tau") :
            nSthuth = len(data.ix[(data.target.values == 0) & (data.bWj1Wj2_isGenMatched.values==1) & (data.key.values==folderName)])
            nBtruth = len(data.ix[(data.target.values == 1) & (data.bWj1Wj2_isGenMatched.values==1) & (data.key.values==folderName)])
            nSthuthKin = len(data.ix[(data.target.values == 0) & (data.bWj1Wj2_isGenMatchedWithKinFit.values==1) & (data.key.values==folderName)])
            nBtruthKin = len(data.ix[(data.target.values == 1) & (data.bWj1Wj2_isGenMatchedWithKinFit.values==1) & (data.key.values==folderName)])
            nShadthuth = len(data.ix[(data.target.values == 0) & (data.hadtruth.values==1) & (data.key.values==folderName)])
            nBhadtruth = len(data.ix[(data.target.values == 1) & (data.hadtruth.values==1) & (data.key.values==folderName)])
            print "truth:              ", nSthuth, nBtruth
            print "truth Kin:          ", nSthuthKin, nBtruthKin
            print "hadtruth:           ", nShadthuth, nBhadtruth
    print (data.columns.values.tolist())
    n = len(data)
    nS = len(data.ix[data.target.values == 1])
    nB = len(data.ix[data.target.values == 0])
    print channelInTree," length of sig, bkg: ", nS, nB
    return data


def load_data_2l2t():
    keys=['ttH','ttbar','ttv']
    keystoDraw=['ttHToNonbb','TTToSemilepton','TTWJetsToLNu']
    treetoread="evtTree"
    weight="weight"
    sourceA="/home/arun/VHbbNtuples_8_0_x/CMSSW_8_1_0/src/Clusterization/"
    variables=["mva_ttv","mva_tt"]
    data = pandas.DataFrame(columns=variables+[weight,'key','target','totalWeight'])
    for nn, folderName in enumerate(keys) :
        tfile = ROOT.TFile(sourceA+folderName+"_2l2tau.root")
        tree = tfile.Get(treetoread)
        chunk_arr = tree2array(tree)
        chunk_df = pandas.DataFrame(chunk_arr)
        chunk_df['key']=keystoDraw[nn]
        chunk_df['totalWeight']=chunk_df[weight]
        chunk_df['target']=1 if folderName=='ttH' else 0
        data=data.append(chunk_df, ignore_index=True)
        print folderName,keystoDraw[nn]," length of sig, bkg: ", len(data.ix[data.key.values == keystoDraw[nn]])
    return data

def load_data_fullsim(inputPath,channelInTree,variables,criteria,testtruth,bdtType) :
    print variables
    my_cols_list=variables+['key','target','file']+criteria #,'tau_frWeight','lep1_frWeight','lep1_frWeight' trainVars(False)
    # if channel=='2lss_1tau' : my_cols_list=my_cols_list+['tau_frWeight','lep1_frWeight','lep2_frWeight']
    # those last are only for channels where selection is relaxed (2lss_1tau) === solve later
    sampleNames=['signal','TT','TTW','TTZ',"EWK","Rares"]
    samplesTT=['TTJets_DiLept',
    'TTJets_DiLept_ext1',
    'TTJets_SingleLeptFromT',
    'TTJets_SingleLeptFromT_ext1',
    'TTJets_SingleLeptFromTbar',
    'TTJets_SingleLeptFromTbar_ext1',
    #'ST_tW_antitop_5f_inclusiveDecays',
    #'ST_tW_top_5f_inclusiveDecays',
    #'ST_s-channel_4f_leptonDecays',
    #'ST_t-channel_antitop_4f_inclusiveDecays',
    #'ST_t-channel_top_4f_inclusiveDecays', # + tH
    #'THQ',
    #'THW'
    ]
    samplesTTW=['TTWJetsToLNu_ext1','TTWJetsToLNu_ext2']
    samplesTTZ=['TTZToLL_M10_ext2', 'TTZToLL_M10_ext1', 'TTZToLL_M-1to10']
    samplesEWK=['DYJetsToLL_M-10to50',
    'DYJetsToLL_M-50_ext1',
    'DYJetsToLL_M-50_ext2',
    'WJetsToLNu',
    'WWTo2L2Nu',
    'ZZTo4L',
    'WZTo3LNu']
    samplesRares=['WGToLNuG_ext2',
    'TGJets',
    'TGJets_ext1',
    'TTTT',
    'TTWW',
    'WZZ',
    'ZGTo2LG',
    'WGToLNuG_ext1',
    'WWTo2L2Nu_DoubleScattering',
    'WWW_4F',
    'ZZZ',
    'TTGJets'
    'TTGJets_ext1',
    'tZq_ll_4f',
    'WpWpJJ_EWK-QCD']
    dataloc = pandas.DataFrame(columns=my_cols_list)
    for sampleName in sampleNames :
        print (sampleName, channelInTree)
        if sampleName=='TT' :
        	folderNames=samplesTT
        	target=0
        if sampleName=='signal' :
        	folderNames=['ttHJetToNonbb_M125_amcatnlo']
        	target=1
        if sampleName=='TTW':
        	folderNames=samplesTTW
        	target=0
        if sampleName=='TTZ' :
        	folderNames=samplesTTZ
        	target=0
        if sampleName=='EWK':
        	folderNames=samplesEWK
        	target=0
        if sampleName=='Rares' :
        	folderNames=samplesRares
        	target=0
        inputTree = channelInTree+'/sel/evtntuple/'+sampleName+'/evtTree'
        list=[]
        for folderName in folderNames :
            procP1=glob.glob(inputPath+"/"+folderName+"/"+folderName+"*.root")
            list= list+procP1
        for ii in range(0, len(list)) :
            try: tfile = ROOT.TFile(list[ii])
            except : continue
            try: tree = tfile.Get(inputTree)
            except : continue
            if tree is not None :
                try: chunk_arr = tree2array(tree) #,  start=start, stop = stop)
                except : continue
                else :
                    chunk_df = pandas.DataFrame(chunk_arr)
                    if channel == "3l_1tau" :
                        chunk_df["lep1_eta"]=abs(chunk_df["lep1_eta"])
                        chunk_df["lep2_eta"]=abs(chunk_df["lep2_eta"])
                        chunk_df["lep3_eta"]=abs(chunk_df["lep3_eta"])
                        chunk_df["avr_lep_eta"]=(abs(chunk_df["lep2_eta"])+abs(chunk_df["lep1_eta"])+abs(chunk_df["lep3_eta"]))/2.
                        chunk_df["tau_eta"]=abs(chunk_df["tau_eta"])
                        chunk_df["max_lep_eta"]=chunk_df[["lep1_eta","lep2_eta","lep3_eta"]].max(axis=1)
                        WtoMultiply=chunk_df["lep1_fake_prob"]*chunk_df["lep2_fake_prob"]*chunk_df["lep3_fake_prob"] #*chunk_df["tau_fake_prob"]
                        chunk_df["totalWeight"] = chunk_df["evtWeight"]*WtoMultiply
                    if channel=="2l_2tau" :
                        chunk_df["min_dr_lep_tau"]=chunk_df[["dr_lep1_tau1", "dr_lep1_tau2", "dr_lep2_tau1", "dr_lep2_tau2"]].min(axis=1)
                        chunk_df["min_dr_lep_jet"]=chunk_df[["mindr_lep1_jet", "mindr_lep2_jet"]].min(axis=1)
                        chunk_df["max_dr_lep_tau"]=chunk_df[["dr_lep1_tau1", "dr_lep1_tau2", "dr_lep2_tau1", "dr_lep2_tau2"]].max(axis=1)
                        chunk_df["avr_dr_lep_tau"]=(chunk_df['dr_lep1_tau1']+chunk_df['dr_lep1_tau2']+chunk_df['dr_lep2_tau1']+chunk_df['dr_lep2_tau2'])/4.
                        chunk_df["mindr_tau_jet"]=chunk_df[["mindr_tau1_jet","mindr_tau2_jet"]].min(axis=1)
                        chunk_df["lep1_eta"]=abs(chunk_df["lep1_eta"])
                        chunk_df["lep2_eta"]=abs(chunk_df["lep2_eta"])
                        chunk_df["avr_lep_eta"]=(abs(chunk_df["lep2_eta"])+abs(chunk_df["lep1_eta"]))/2.
                        chunk_df["tau1_eta"]=abs(chunk_df["tau1_eta"])
                        chunk_df["tau2_eta"]=abs(chunk_df["tau2_eta"])
                        chunk_df["avr_tau_eta"]=(abs(chunk_df["tau2_eta"])+abs(chunk_df["tau1_eta"]))/2.
                        chunk_df["leptonPairCharge"]=abs(chunk_df["leptonPairCharge"])
                        chunk_df["hadTauPairCharge"]=abs(chunk_df["hadTauPairCharge"])
                        #WtoMultiply=chunk_df["weight_fakeRate"]
                        WtoMultiply=chunk_df["lep1_fake_prob"]*chunk_df["lep2_fake_prob"]*chunk_df["tau1_fake_prob"]*chunk_df["tau2_fake_prob"]
                        #WtoMultiply=chunk_df["tau1_fake_prob"]*chunk_df["tau2_fake_prob"]
                        chunk_df["totalWeight"] = chunk_df["evtWeight"]*WtoMultiply
                    if channel=="2lss_1tau" :
                        chunk_df["totalWeight"] = chunk_df["evtWeight"]*chunk_df["lep1_fake_prob"]*chunk_df["lep2_fake_prob"] #WtoMultiply
                        chunk_df["max_eta_Lep"]=chunk_df[['tau_eta', 'lep1_eta', 'lep2_eta']].max(axis=1)
                    if channel=="2los_1tau" :
                        chunk_df["totalWeight"] = chunk_df["evtWeight"]*chunk_df["lep1_fake_prob"]*chunk_df["lep2_fake_prob"]*chunk_df["tau_fake_prob"]
                    chunk_df['key']=folderName
                    chunk_df["target"]=target
                    chunk_df['proces']=sampleName
                    if channel=="1l_2tau" : chunk_df["totalWeight"] = chunk_df["evtWeight"]*chunk_df["prob_fake_lepton"]*chunk_df["tau_fake_prob_lead"]*chunk_df["tau_fake_prob_sublead"]
                    dataloc=dataloc.append(chunk_df, ignore_index=True)
            else : print ("file "+list[ii]+"was empty")
            tfile.Close()
        if len(dataloc) == 0 : continue
        nS = len(dataloc.ix[(dataloc.target.values == 0) & (dataloc.proces.values==sampleName)])
        nB = len(dataloc.ix[(dataloc.target.values == 1) & (dataloc.proces.values==sampleName)])
        print sampleName,"length of sig, bkg: ", nS, nB, dataloc.ix[(dataloc.proces.values==sampleName)]["totalWeight"].sum(), dataloc.ix[(dataloc.proces.values==sampleName)]["totalWeight"].sum()
        if channel=="2l_2tau" :
            print "tau1 all | lep  ",sampleName , len(dataloc.ix[(dataloc["tau1_fake_prob"].values != 1) & (dataloc.proces.values==sampleName)]), len(dataloc.ix[(dataloc["tau1_fake_test"].values != 1) & (dataloc.proces.values==sampleName)])
            print "tau2 all | lep  ",sampleName , len(dataloc.ix[(dataloc["tau2_fake_prob"].values != 1) & (dataloc.proces.values==sampleName)]), len(dataloc.ix[(dataloc["tau2_fake_test"].values != 1) & (dataloc.proces.values==sampleName)])
            print "tau1&2 all | lep  ",sampleName ,len(dataloc.ix[(dataloc["tau1_fake_prob"].values != 1) & (dataloc["tau2_fake_prob"].values != 1) & (dataloc.proces.values==sampleName)]), len(dataloc.ix[(dataloc["tau1_fake_test"].values != 1) & (dataloc['tau2_fake_test'].values != 1) & (dataloc.proces.values==sampleName)])
            datatest=dataloc["evtWeight"]*dataloc["lep1_fake_prob"]*dataloc["lep2_fake_prob"]*dataloc["tau1_fake_test"]*dataloc["tau2_fake_test"]
            print "sum of weights with/ without lepton in FR ",sampleName , datatest.ix[ (dataloc.proces.values==sampleName)].sum(),dataloc.ix[ (dataloc.proces.values==sampleName)]["totalWeight"].sum()
        if (channel=="1l_2tau" or channel=="2lss_1tau") :
            nSthuth = len(dataloc.ix[(dataloc.target.values == 0) & (dataloc.bWj1Wj2_isGenMatched.values==1) & (dataloc.proces.values==sampleName)])
            nBtruth = len(dataloc.ix[(dataloc.target.values == 1) & (dataloc.bWj1Wj2_isGenMatched.values==1) & (dataloc.proces.values==sampleName)])
            nSthuthKin = len(dataloc.ix[(dataloc.target.values == 0) & (dataloc.bWj1Wj2_isGenMatchedWithKinFit.values==1) & (dataloc.proces.values==sampleName)])
            nBtruthKin = len(dataloc.ix[(dataloc.target.values == 1) & (dataloc.bWj1Wj2_isGenMatchedWithKinFit.values==1) & (dataloc.proces.values==sampleName)])
            nShadthuth = len(dataloc.ix[(dataloc.target.values == 0) & (dataloc.hadtruth.values==1) & (dataloc.proces.values==sampleName)])
            nBhadtruth = len(dataloc.ix[(dataloc.target.values == 1) & (dataloc.hadtruth.values==1) & (dataloc.proces.values==sampleName)])
            print "truth:              ", nSthuth, nBtruth
            print "truth Kin:          ", nSthuthKin, nBtruthKin
            print "hadtruth:           ", nShadthuth, nBhadtruth
    if 'ttHToNonbb' in folderName : print (dataloc.columns.values.tolist())
    n = len(dataloc)
    nS = len(dataloc.ix[dataloc.target.values == 0])
    nB = len(dataloc.ix[dataloc.target.values == 1])
    print sampleName," length of sig, bkg: ", nS, nB
    return dataloc

def normalize(arr): return (arr-arr.min())/(arr.max()-arr.min())

def make_plots(
    featuresToPlot,nbin,
    data1,label1,color1,
    data2,label2,color2,
    plotname,
    printmin,
    plotResiduals
    ) :
    print (len(featuresToPlot), featuresToPlot)
    hist_params = {'normed': True, 'histtype': 'bar', 'fill': True , 'lw':3}
    sizeArray=int(math.sqrt(len(featuresToPlot))) if math.sqrt(len(featuresToPlot)) % int(math.sqrt(len(featuresToPlot))) == 0 else int(math.sqrt(len(featuresToPlot)))+1
    drawStatErr=True
    residuals=[]
    plt.figure(figsize=(4*sizeArray, 4*sizeArray))
    for n, feature in enumerate(featuresToPlot):
        # add sub plot on our figure
        plt.subplot(sizeArray, sizeArray, n+1)
        # define range for histograms by cutting 1% of data from both ends
        min_value, max_value = np.percentile(data1[feature], [0.0, 99])
        min_value2, max_value2 = np.percentile(data2[feature], [0.0, 99])
        if printmin : print (min_value, max_value,feature)
        values1, bins, _ = plt.hist(data1[feature].values, weights= data1[weights].values.astype(np.float64) ,
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
            values2, bins, _ = plt.hist(data2[feature].values, weights= data2[weights].values.astype(np.float64) ,
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
        if plotResiduals : residuals=residuals+[(plot1[0]-plot2[0])/(plot1[0])]
        plt.ylim(ymin=0.00001)
        if n == len(featuresToPlot)-1 : plt.legend(loc='best')
        plt.xlabel(feature)
        #plt.xscale('log')
        #plt.yscale('log')
    plt.ylim(ymin=0)
    plt.savefig(plotname)
    plt.clf()
    if plotResiduals :
        residuals=np.nan_to_num(residuals)
        for n, feature  in enumerate(trainVars(True)):
            (mu, sigma) = norm.fit(residualsSignal[n])
            plt.subplot(8, 8, n+1)
            residualsSignal[n]=np.nan_to_num(residualsSignal[n])
            n, bins, patches = plt.hist(residualsSignal[n], label='Residuals '+label1+'/'+label2)
            # add a 'best fit' line
            y = mlab.normpdf( bins, mu, sigma)
            l = plt.plot(bins, y, 'r--', linewidth=2)
            plt.ylim(ymin=0)
            plt.title(feature+' '+r'mu=%.3f, sig=%.3f$' %(mu, sigma))
            print feature+' '+r'mu=%.3f, sig=%.3f$' %(mu, sigma)
        plt.savefig(channel+"/"+bdtType+"_"+trainvar+"_Variables_Signal_fullsim_residuals.pdf")
        plt.clf()

def make_plots_gen(
    featuresToPlot,nbin,
    data1,label1,color1,
    plotname,
    printmin
    ) :
    hist_params = {'histtype': 'bar', 'fill': True , 'lw':3, 'alpha':0.3}
    sizeArray=int(math.sqrt(len(featuresToPlot))) if math.sqrt(len(featuresToPlot)) % int(math.sqrt(len(featuresToPlot))) == 0 else int(math.sqrt(len(featuresToPlot)))+1
    drawStatErr=True
    residuals=[]
    plt.figure(figsize=(4*sizeArray, 4*sizeArray))
    for n in range(0,3):
        # add sub plot on our figure
        plt.subplot(sizeArray, sizeArray, n+1)
        # define range for histograms by cutting 1% of data from both ends
        min_value, max_value = [50,250] #np.percentile(data1[feature], [0.0, 99])
        min_value2, max_value2 = [50,250] #np.percentile(data2[feature], [0.0, 99])
        #if printmin : print (min_value, max_value,feature)
        values1, bins, _ = plt.hist(
                                   [data1[featuresToPlot[n]].values, data1[featuresToPlot[n+3]].values],
                                   #weights= data1[weights].values.astype(np.float64) ,
                                   range=(max(min(min_value,min_value2),0),  max(max_value,max_value2)), #  0.5 ),#
                                   bins=nbin, #edgecolor=[color1,color1], color=[color1,color1],
                                   stacked=True,
                                   label="Top", **hist_params )
        plt.ylim(ymin=0.00001)
        if n == 2 : plt.legend(loc='best')
        plt.xlabel(featuresToPlot[n])
        #plt.xscale('log')
        #plt.yscale('log')
    plt.ylim(ymin=0)
    plt.savefig(plotname)
    plt.clf()

def make_plots_genpt(data1,label1,color1, plotname) :
    # plot eff / genPt bins
    datax = []
    datay = []
    binsGenPt = [0, 50, 100, 150, 200, 250, 300, 350 , 400 , 500 , 600 , 700 , 800]
    for ii in range(0,len(binsGenPt)-1) :
        numerator = float((data1.loc[(data["genTopPt"] > binsGenPt[ii]) & (data["genTopPt"] <= binsGenPt[ii+1]) & (data[target]==1)]['weights'].sum()))
        denominator = float((data1.loc[(data["genTopPt"] > binsGenPt[ii]) & (data["genTopPt"] <= binsGenPt[ii+1]) & (data["hadtruth"] > 0)]['weights'].sum())) if numerator > 0 else 1.
        print (ii,binsGenPt[ii],binsGenPt[ii+1], numerator, denominator, numerator/denominator)
        datay.append(numerator/denominator)
        datax.append(binsGenPt[ii])
    plt.step(datax, datay, lw = 3 , color = color1)
    plt.ylabel("Accuracy", fontsize=20)
    plt.xlabel('genTop '+r'$\mathbf{\mathrm{p_T}}$'+' (GeV)', fontsize=20)
    plt.ylim(0, 1.45)
    plt.text(25, 1.35, 'CMS', style='normal', fontsize=25, fontweight='bold')
    plt.text(25, 1.25, 'Preliminary', fontsize=23, style='italic')
    plt.text(575, 1.48, '(13 TeV)', fontsize=20) # 35.9 fb$^{-1}$
    plt.savefig(plotname)
    plt.clf()

def load_data_xml (dataIn) :
    data=dataIn
    data["oldTrainTMVA_tt"]=-2.
    data["oldTrainTMVA_ttV"]=-2.
    for ii,ss in data.iterrows():
    	#if ii > 20 : break
    	clsTTreader = TMVA.Reader("Silent")
    	clsTTVreader = TMVA.Reader("Silent")
    	for feature in trainVarsTT(BDTvar) :
    		var= feature
    		if feature == "TMath::Max(TMath::Abs(lep1_eta),TMath::Abs(lep2_eta))" : var=  "max_lep_eta"
    		varVal=array.array('f',[ss[var]])
    		clsTTreader.AddVariable(str(feature),  varVal);
    	for feature in trainVarsTTV(BDTvar) :
    		var= feature
    		if feature == "TMath::Max(TMath::Abs(lep1_eta),TMath::Abs(lep2_eta))" : var=  "max_lep_eta"
    		varVal=array.array('f',[ss[var]])
    		clsTTVreader.AddVariable(str(feature),  varVal);
    	clsTTreader.BookMVA("BDT", "arun_xml_2lss_1tau/2lss_1tau_ttbar_BDTG.weights.xml")
    	clsTTVreader.BookMVA("BDT", "arun_xml_2lss_1tau/2lss_1tau_ttV_BDTG.weights.xml")
    	#for ii,ss in data.iterrows():
    	if ii % 1000 == 0 : print ("result TMVA ",ii, clsTTreader.EvaluateMVA("BDT"),clsTTVreader.EvaluateMVA("BDT"),data.loc[data.index[ii],trainVarsTT("oldTrainN")].values)
    	data.loc[data.index[ii],"oldTrainTMVA_tt"]=clsTTreader.EvaluateMVA("BDT")
    	data.loc[data.index[ii],"oldTrainTMVA_ttV"]=clsTTVreader.EvaluateMVA("BDT")
    data.to_csv('arun_xml_2lss_1tau/arun_xml_2lss_1tau_FromAnalysis.csv')
    return data

def gauss(x, *p):
    A, mu, sigma = p
    return A*numpy.exp(-(x-mu)**2/(2.*sigma**2))

def reverse_colourmap(cmap, name = 'my_cmap_r'):
    reverse = []
    k = []
    for key in cmap._segmentdata:
        k.append(key)
        channel = cmap._segmentdata[key]
        data = []
        for t in channel:
            data.append((1-t[0],t[2],t[1]))
        reverse.append(sorted(data))
    LinearL = dict(zip(k,reverse))
    my_cmap_r = colors.LinearSegmentedColormap(name, LinearL)
    return my_cmap_r

def divisorGenerator(n):
    large_divisors = []
    for i in xrange(1, int(math.sqrt(n) + 1)):
        if n % i == 0:
            yield i
            if i*i != n:
                if n / i <26 : large_divisors.append(n / i)
    for divisor in reversed(large_divisors):
        yield divisor

def doStackPlot(hTT,hTTH,hTTW,hEWK,hRares,name,label):
    print ("hTT, hTTH, hTTW, hEWK")
    print (hTT.Integral(),hTTH.Integral(),hTTW.Integral(),hEWK.Integral())
    hTT.SetFillColor( 17 );
    hTTH.SetFillColor( ROOT.kRed );
    hTTW.SetFillColor( 8 );
    hEWK.SetFillColor( 6 );
    hRares.SetFillColor( 65 );
    mc  = ROOT.THStack("mc","");
    mc.Add(hRares);
    mc.Add(hEWK);
    mc.Add(hTTW);
    mc.Add(hTTH);
    mc.Add( hTT );
    c4 = ROOT.TCanvas("c5","",500,500);
    c4.cd();
    c4.Divide(1,2,0,0);
    c4.cd(1)
    ROOT.gPad.SetLogy()
    ROOT.gPad.SetLogx()
    ROOT.gPad.SetBottomMargin(0.001)
    ROOT.gPad.SetTopMargin(0.065)
    ROOT.gPad.SetRightMargin(0.01)
    ROOT.gPad.SetLeftMargin(0.12)
    mc.Draw("HIST");
    mc.SetMaximum(15* mc.GetMaximum());
    mc.SetMinimum(max(0.04* mc.GetMinimum(),0.1));
    mc.GetYaxis().SetRangeUser(0.01,110);
    mc.GetXaxis().SetRangeUser(0.0001,1.0);
    mc.GetHistogram().GetYaxis().SetTitle("Expected events/bin");
    mc.GetHistogram().GetXaxis().SetTitle("Bin in the bdt1#times bdt2 plane");
    mc.GetHistogram().GetXaxis().SetTitleSize(0.06);
    mc.GetHistogram().GetXaxis().SetLabelSize(.06);
    mc.GetHistogram().GetYaxis().SetTitleSize(0.06);
    mc.GetHistogram().GetYaxis().SetLabelSize(.06);
    l = ROOT.TLegend(0.16,0.6,0.3,0.9);
    l.AddEntry(hTTH  , "ttH", "f");
    l.AddEntry(hTTW  , "ttV"       , "f");
    l.AddEntry(hTT, "tt"        , "f");
    l.AddEntry(hRares, "rares"        , "f");
    l.AddEntry(hEWK, "EWK"        , "f");
    l.Draw();
    latex= ROOT.TLatex();
    latex.SetTextSize(0.065);
    latex.SetTextAlign(13);  #//align at top
    latex.SetTextFont(62);
    latex.DrawLatexNDC(.15,1.0,"CMS Simulation");
    latex.DrawLatexNDC(.8,1.0,"#it{36 fb^{-1}}");
    latex.DrawLatexNDC(.55,.8,label);
    #latex.DrawLatexNDC(.55,.9,BDTvar);
    c4.cd(2)
    #ROOT.gPad.SetLogx()
    ROOT.gStyle.SetHatchesSpacing(100)
    ROOT.gPad.SetLeftMargin(0.12)
    ROOT.gPad.SetBottomMargin(0.12)
    ROOT.gPad.SetTopMargin(0.001)
    ROOT.gPad.SetRightMargin(0.005)
    if not hTT.GetSumw2N() : hTT.Sumw2()
    h2=hTT.Clone()
    h2.Add(hTTW)
    hBKG1D=h2.Clone()
    h3=hTTH.Clone()
    h4=hTT.Clone()
    if not h2.GetSumw2N() : h2.Sumw2()
    if not h3.GetSumw2N() : h3.Sumw2()
    for binn in range(0,h2.GetNbinsX()+1) :
    	ratio=0
    	ratio3=0
    	if h2.GetBinContent(binn) >0 :
    		ratio=h2.GetBinError(binn)/h2.GetBinContent(binn)
    		h2.SetBinContent(binn,ratio)
    	if hBKG1D.GetBinContent(binn) > 0 :
    		ratio3=h3.GetBinContent(binn)/hBKG1D.GetBinContent(binn)
    		h3.SetBinContent(binn,ratio3)
    	if h4.GetBinContent(binn) > 0 : h4.SetBinContent(binn,h4.GetBinError(binn)/h4.GetBinContent(binn))
    	print (binn,hTT.GetBinContent(binn),ratio,ratio3)
    h2.SetLineWidth(3)
    h2.SetLineColor(2)
    h2.SetFillStyle(3690)
    h3.SetLineWidth(3)
    h3.SetFillStyle(3690)
    h3.SetLineColor(28)
    h4.SetLineWidth(3)
    h4.SetFillStyle(3690)
    h4.SetLineColor(6)
    h3.Draw("HIST")
    h3.GetYaxis().SetTitle("S/B");
    h3.GetXaxis().SetTitle("Bin in the bdt1#times bdt2 plane");
    h3.GetYaxis().SetTitleSize(0.06);
    h3.GetYaxis().SetLabelSize(.06)
    h3.GetXaxis().SetTitleSize(0.06);
    h3.GetXaxis().SetLabelSize(.06)
    l2 = ROOT.TLegend(0.16,0.77,0.4,0.98);
    l2.AddEntry(h3  , "S/B" , "l");
    l2.AddEntry(h2  , "ttV + tt err/cont", "l");
    l2.AddEntry(h4  , "tt err/cont", "l");
    l2.Draw("same");
    h2.Draw("HIST,SAME")
    h4.Draw("HIST,SAME")

    c4.Modified();
    c4.Update();
    print ("s/B in last bin (tight)", h3.GetNbinsX(), h3.GetBinContent(h3.GetNbinsX()), h3.GetBinContent(h3.GetNbinsX()-1), h2.GetBinContent(h3.GetNbinsX()))
    c4.SaveAs(name+".pdf")
    print ("saved",name+".pdf")

def finMaxMin(histSource) :
    file = TFile(histSource+".root","READ");
    file.cd()
    hSum = TH1F()
    for keyO in file.GetListOfKeys() :
       obj =  keyO.ReadObj()
       if type(obj) is not TH1F : continue
       hSumDumb = obj.Clone()
       if not hSum.Integral()>0 : hSum=hSumDumb
       else : hSum.Add(hSumDumb)
    return [[hSum.FindFirstBinAbove(0.0),  hSum.FindLastBinAbove (0.0)],
            [hSum.GetBinCenter(hSum.FindFirstBinAbove(0.0)),  hSum.GetBinCenter(hSum.FindLastBinAbove (0.0))]]

def getQuantiles(histoP,ntarget,xmax) :
    #c = ROOT.TCanvas("c1","",600,600)
    #histoP.Rebin(4)
    histoP.Scale(1./histoP.Integral());
    histoP.GetCumulative()#.Draw();
    #histoP.Draw();
    histoP.GetXaxis().SetRangeUser(0.,1.)
    histoP.GetYaxis().SetRangeUser(0.,1.)
    histoP.SetMinimum(0.0)
    xq= array.array('d', [0.] * (ntarget+1))
    yq= array.array('d', [0.] * (ntarget+1))
    yqbin= array.array('d', [0.] * (ntarget+1)) # +2 if firsrt is not zero
    for  ii in range(0,ntarget) : xq[ii]=(float(ii)/(ntarget))
    xq[ntarget]=0.999999999
    histoP.GetQuantiles(ntarget,yq,xq)
    line = [None for point in range(ntarget)]
    line2 = [None for point in range(ntarget)]
    #yq[ntarget]=xmax
    #c.Modified();
    #c.Update();
    #yqbin[0]=0.0
    for  ii in range(1,ntarget+1) : yqbin[ii]=yq[ii]
    yqbin[ntarget]=xmax # +1 if first is not 0
    print yqbin
    return yqbin

def getQuantilesWStat(histoP,nmin) :
    histogramBinning=[]
    xAxis = histogram.GetXaxis();
    histogramBinning = histogramBinning + [xAxis.GetBinLowEdge(1)]
    sumEvents = 0.;
    numBins = xAxis.GetNbins();
    for idxBin in range(1, numBins) :
        print ("bin #" , idxBin , " (x=" , xAxis.GetBinLowEdge(idxBin) ,  xAxis.GetBinUpEdge(idxBin) , "):" , " binContent = ",  histogram.GetBinContent(idxBin) , " +/- " << histogram.GetBinError(idxBin) )
        sumEvents = sumEvents + histogram.GetBinContent(idxBin);
        if ( sumEvents >= minEvents ) :
            histogramBinning.push_back(xAxis.GetBinUpEdge(idxBin));
            sumEvents = 0.;
    if ( abs(histogramBinning.back() - xAxis.GetBinUpEdge(numBins)) > 1.e-3 ) :
        if histogramBinning.size() >= 2 : histogramBinning = [xAxis.GetBinUpEdge(numBins)];
        else :  histogramBinning= histogramBinning+ [xAxis.GetBinUpEdge(numBins)];
    #assert(histogramBinning.size() >= 2);
    print "binning =  "
    for  bin in histogramBinning : print ( bin)
    return histogramBinning;

    """
    std::vector<double> compBinning(TH1* histogram, double minEvents) {
    std::cout << "<compBinning>:" << std::endl;
    std::vector<double> histogramBinning;
    const TAxis* xAxis = histogram->GetXaxis();
    histogramBinning.push_back(xAxis->GetBinLowEdge(1));
    double sumEvents = 0.; int numBins = xAxis->GetNbins();
    for ( int idxBin = 1; idxBin <= numBins; ++idxBin ) {
        std::cout << "bin #" << idxBin << " (x=" << xAxis->GetBinLowEdge(idxBin) << ".." << xAxis->GetBinUpEdge(idxBin) << "):" << " binContent = " << histogram->GetBinContent(idxBin) << " +/- " << histogram->GetBinError(idxBin) << std::endl; sumEvents += histogram->GetBinContent(idxBin);
        if ( sumEvents >= minEvents ) { histogramBinning.push_back(xAxis->GetBinUpEdge(idxBin)); sumEvents = 0.; }
    }
    if ( TMath::Abs(histogramBinning.back() - xAxis->GetBinUpEdge(numBins)) > 1.e-3 ) {
        if ( histogramBinning.size() >= 2 ) histogramBinning.back() = xAxis->GetBinUpEdge(numBins);
        else histogramBinning.push_back(xAxis->GetBinUpEdge(numBins));
    }
    assert(histogramBinning.size() >= 2);
    std::cout << "binning = { ";
    for ( std::vector<double>::const_iterator bin = histogramBinning.begin(); bin != histogramBinning.end(); ++bin ) {
        if ( bin != histogramBinning.begin() ) std::cout << ", "; std::cout << (*bin); } std::cout << " }" << std::endl; return histogramBinning;
        }
    """



def GetRatio(histSource,namepdf) :
    file = TFile(histSource,"READ");
    file.cd()
    hSum = TH1F()
    h2 = TH1F()
    ratiohSum=1.
    ratiohSumP=1.
    ratio=1.
    ratioP=1.
    hTTi = TH1F()
    hTTHi = TH1F()
    hEWKi = TH1F()
    hTTWi = TH1F()
    hRaresi = TH1F()
    for keyO in file.GetListOfKeys() :
       obj =  keyO.ReadObj()
       if type(obj) is not TH1F : continue
       h2=obj.Clone()
       factor=1.
       if  not obj.GetSumw2N() : obj.Sumw2()
       if keyO.GetName() == "fakes_data"  or keyO.GetName() =="TTZ" or keyO.GetName() =="TTW" or keyO.GetName() =="TTWW" or keyO.GetName() == "EWK" :
           if not hSum.Integral()>0 : hSum=obj.Clone()
           else :
               hSum.Add(obj)
               print (keyO.GetName(),hSum.Integral())
       if keyO.GetName() == "fakes_data" :
           print ("last bin",keyO.GetName(),h2.GetBinContent(obj.GetNbinsX()),h2.GetBinContent(obj.GetNbinsX()-1),h2.Integral())
           if h2.GetBinContent(h2.GetNbinsX()) >0 : ratio=h2.GetBinError(h2.GetNbinsX())/h2.GetBinContent(h2.GetNbinsX())
           if obj.GetBinContent(h2.GetNbinsX()-1) >0 : ratioP=h2.GetBinError(h2.GetNbinsX()-1)/h2.GetBinContent(h2.GetNbinsX()-1)
       if h2.GetName() =="TTZ" or h2.GetName() =="TTW" :
            if not hTTWi.Integral()>0 : hTTWi=h2.Clone()
            else : hTTWi.Add(h2.Clone())
       if h2.GetName() == "fakes_data" : hTTi=h2.Clone()
       if h2.GetName() =="Rares" : hRaresi=h2.Clone()
       if h2.GetName() == "EWK" : hEWKi=h2.Clone()
       if h2.GetName() == "ttH_hww" or h2.GetName() == "ttH_hzz" or h2.GetName() ==  "ttH_htt" :
            if not hTTHi.Integral()>0 : hTTHi=h2.Clone()
            else : hTTHi.Add(h2.Clone())
    #doStackPlot(hTTi,hTTHi,hTTWi,hEWKi,hRaresi,namepdf,"2D Map")
    #print (namepdf+" created")
    if  not hSum.GetSumw2N() : hSum.Sumw2()
    if hSum.GetBinContent(hSum.GetNbinsX()) >0 :
            ratiohSum=hSum.GetBinError(hSum.GetNbinsX())/hSum.GetBinContent(hSum.GetNbinsX())
    if hSum.GetBinContent(hSum.GetNbinsX()-1) >0 : ratiohSumP=hSum.GetBinError(hSum.GetNbinsX()-1)/hSum.GetBinContent(hSum.GetNbinsX()-1)
    print (ratio,ratioP,ratiohSum,ratiohSumP)
    print (hSum.GetBinContent(hSum.GetNbinsX()))
    return [ratio,ratioP,ratiohSum,ratiohSumP]


def rebinRegular(histSource,nbin, BINtype,originalBinning,doplots,variables,bdtType, withFolder=False) :
    minmax = finMaxMin(histSource)
    errOcontTTLast=[]
    errOcontTTPLast=[]
    errOcontSUMLast=[]
    errOcontSUMPLast=[]
    #
    errTTLast=[]
    contTTLast=[]
    errSUMLast=[]
    contSUMLast=[]
    #
    realbins=[]
    xminbin=[]
    xmaxbin=[]
    xmaxLbin=[]
    #
    lastQuant=[]
    xmaxQuant=[]
    xminQuant=[]
    #
    if BINtype=="ranged" :
        xmin=minmax[1][0]
        xmax=minmax[1][1]
        xmindef=minmax[1][0]
        xmaxdef=minmax[1][1]
    else :
        if minmax[1][0] < 0 and not withFolder: xmin=-1.0
        else : xmin=0.0
        xmax=1.0
        xmaxdef=minmax[1][1]
        xmindef=minmax[1][0]
    for nn,nbins in enumerate(nbin) :
        file = TFile(histSource+".root","READ");
        file.cd()
        histograms=[]
        histograms2=[]
        h2 = TH1F()
        hSum = TH1F()
        hFakes = TH1F()
        hSumAll = TH1F()
        ratiohSum=1.
        ratiohSumP=1.
        ###///
        ### rebin and  write the histograms
        if BINtype=="none" : name=histSource+"_"+str(nbins)+"bins_none.root"
        if BINtype=="regular" or options.BINtype == "mTauTauVis": name=histSource+"_"+str(nbins)+"bins.root"
        if BINtype=="ranged" : name=histSource+"_"+str(nbins)+"bins_ranged.root"
        if BINtype=="quantiles" :
            name=histSource+"_"+str(nbins)+"bins_quantiles.root"

        fileOut  = TFile(name, "recreate");
        #for folder in folders :
        if withFolder : folders_Loop = file.GetListOfKeys()
        else : folders_Loop = ["none"]
        for nkey, keyF in enumerate(folders_Loop) :
            if withFolder :
                obj =  keyF.ReadObj()
                loop_on = obj.GetListOfKeys()
                histograms=[]
                histograms2=[]
                h2 = TH1F()
                hSum = TH1F()
                hFakes = TH1F()
                hSumAll = TH1F()
                ratiohSum=1.
                ratiohSumP=1.
            else :
                loop_on = file.GetListOfKeys()
            for keyO in loop_on :
               # print ( keyF.GetName(), keyO.GetName() )
               #if type(obj) is not TH1F : continue
               if not withFolder :
                   print "got histogram"
                   obj = keyO.ReadObj()
                   h2  = obj.Clone();
                   print h2.GetName()
               else : h2 = file.Get(keyF.GetName()+"/"+str(keyO.GetName())).Clone()
               factor=1.
               if  not h2.GetSumw2N() : h2.Sumw2()
               if  not hSum.GetSumw2N() : hSum.Sumw2()
               #if withFolder :
               if withFolder : h2.SetName("x_"+str(h2.GetName()))
               histograms.append(h2.Clone())
               print ("h2.", h2.Integral())
               if "fakes_data" in h2.GetName() : hFakes=h2.Clone()
               if "fakes_data" in h2.GetName() or "TT" in h2.GetName() or "EWK" in h2.GetName() or "Rares" in h2.GetName() : #  or "tH" in keyO.GetName()
                   #hSumDumb2 = obj # h2_rebin #
                   if not hSumAll.Integral()>0 : hSumAll=h2.Clone()
                   else : hSumAll.Add(h2)
            #################################################
            if withFolder: fileOut.mkdir(keyF.GetName()+"/")
            hTTi = TH1F()
            hTTHi = TH1F()
            hEWKi = TH1F()
            hTTWi = TH1F()
            hRaresi = TH1F()
            histo = TH1F()
            for nn, histogram in enumerate(histograms) :
                #if BINtype=="quantiles" : ### fix that -- I do not want these written to the file
                histogramCopy=histogram.Clone()
                nameHisto=histogramCopy.GetName()
                histogram.SetName(histogramCopy.GetName()+"_"+str(nn)+BINtype)
                histogramCopy.SetName(histogramCopy.GetName()+"Copy_"+str(nn)+BINtype)
                #else : nameHisto=h2.GetName()
                #histogramCopy.SetBit(ROOT.TH1F.kCanRebin)
                #if histogramCopy.GetName() == "fakes_data" or histogramCopy.GetName() =="TTZ" or histogramCopy.GetName() =="TTW" or histogramCopy.GetName() =="TTWW" or histogramCopy.GetName() == "EWK" :
                #print ("not rebinned",histogramCopy.GetName(),histogramCopy.Integral())
                if BINtype=="none" :
                    histo=histogramCopy.Clone()
                    histo.SetName(nameHisto)
                elif BINtype=="ranged" or BINtype=="regular" :
                    histo= TH1F( nameHisto, nameHisto , nbins , xmin , xmax)
                elif BINtype=="quantiles" :
                    print ("hSumAll", hSumAll.Integral(), hFakes.Integral())
                    nbinsQuant= getQuantiles(hFakes,nbins,xmax) # getQuantiles(hSumAll,nbins,xmax) ## nbins+1 if first quantile is zero
                    print ("Bins by quantiles",nbins,nbinsQuant)
                    xmaxLbin=xmaxLbin+[nbinsQuant[nbins-2]]
                    histo=TH1F( nameHisto, nameHisto , nbins , nbinsQuant) # nbins+1 if first is zero
                elif BINtype=="mTauTauVis" :
                    histo= TH1F( nameHisto, nameHisto , nbins , 0. , 200.)
                histo.Sumw2()
                #if BINtype=="quantiles" : ### fix that -- I do not want these written to the file
                for place in range(0,histogramCopy.GetNbinsX() + 1) :
                    content =      histogramCopy.GetBinContent(place)
                    #if content < 0 : continue # print (content,place)
                    binErrorCopy = histogramCopy.GetBinError(place);
                    newbin =       histo.GetXaxis().FindBin(histogramCopy.GetXaxis().GetBinCenter(place))
                    binError =     histo.GetBinError(newbin);
                    contentNew =   histo.GetBinContent(newbin)
                    histo.SetBinContent(newbin, content+contentNew)
                    histo.SetBinError(newbin, sqrt(binError*binError+binErrorCopy*binErrorCopy))
                    #if histogramCopy.GetBinCenter(place) > 0.174 and  content>0 and bdtType=="1B" and nbins==20 : print ("overflow bin", histogramCopy.GetBinCenter(place),content,nameHisto)
                #if not histo.GetSumw2N() : histo.Sumw2()
                if "fakes_data" in histogramCopy.GetName() or "TTZ" in histogramCopy.GetName() or "TTW" in histogramCopy.GetName() or "EWK" in histogramCopy.GetName()  :
                    print ("rebinned",histo.GetName(),histo.Integral())
                if "fakes_data" in histo.GetName() and nkey == 0 :
                    ratio=1.
                    ratioP=1.
                    hTTi=histo.Clone()
                    hTTi.SetName(histo.GetName()+"toplot_"+str(nn)+BINtype)
                    if histo.GetBinContent(histo.GetNbinsX()) >0 : ratio=histo.GetBinError(histo.GetNbinsX())/histo.GetBinContent(histo.GetNbinsX())
                    if histo.GetBinContent(histo.GetNbinsX()-1) >0 : ratioP=histo.GetBinError(histo.GetNbinsX()-1)/histo.GetBinContent(histo.GetNbinsX()-1)
                    errOcontTTLast=errOcontTTLast+[ratio] if ratio<1.01 else errOcontTTLast+[1.0]
                    errOcontTTPLast=errOcontTTPLast+[ratioP] if ratioP<1.01 else errOcontTTPLast+[1.0]
                    errTTLast=errTTLast+[histo.GetBinError(histo.GetNbinsX())]
                    contTTLast=contTTLast+[histo.GetBinContent(histo.GetNbinsX())]
                if "TTZ" in histo.GetName() or "TTW" in histo.GetName()  :
                    if not hTTWi.Integral()>0 :
                        hTTWi=histo.Clone()
                        hTTWi.SetName(histo.GetName()+"toplot_"+str(nn)+BINtype)
                    else : hTTWi.Add(histo.Clone())
                if "Rares" in histo.GetName()  :
                    hRaresi=histo.Clone()
                    hRaresi.SetName(histo.GetName()+"toplot_"+str(nn)+BINtype)
                if "EWK" in histo.GetName()  :
                    hEWKi=histo.Clone()
                    hEWKi.SetName(histo.GetName()+"toplot_"+str(nn)+BINtype)
                if "ttH_" in histo.GetName() :
                    if not hTTHi.Integral()>0 :
                        hTTHi=histo.Clone()
                        hTTHi.SetName(histo.GetName()+"toplot_"+str(nn)+BINtype)
                    else : hTTHi.Add(histo.Clone())
                if withFolder :
                    #print (histo.GetName(),histo.Integral(), BINtype)
                    fileOut.cd("/"+keyF.GetName()+"/")
                    histo.Write("",TObject.kOverwrite)
                    fileOut.cd()
                else : histo.Write("",TObject.kOverwrite)
            print (name+" created")
            if nkey == 0 :
                if doplots and bdtType=="1B_VT":
                    if nbins==4 : # nbins==6
                        if BINtype=="none" : namepdf=histSource
                        if BINtype=="regular" : namepdf=histSource+"_"+str(nbins)+"bins"
                        if BINtype=="ranged" : namepdf=histSource+"_"+str(nbins)+"bins_ranged"
                        if BINtype=="quantiles" :
                            namepdf=histSource+"_"+str(nbins)+"bins_quantiles"
                            label=str(nbins)+" bins "+BINtype+" "+variables+"  "+bdtType ## nbins+1 if it starts with 0
                        else : label=str(nbins)+" bins "+BINtype+" "+variables+" "+bdtType
                        doStackPlot(hTTi,hTTHi,hTTWi,hEWKi,hRaresi,namepdf,label)
                        print (namepdf+" created")
                hSumCopy=hSum.Clone()
                hSumi = TH1F()
                if BINtype=="ranged" or BINtype=="regular" : hSumi = TH1F( "hSum", "hSum" , nbins , xmin , xmax)
                elif BINtype=="quantiles" : hSumi = TH1F( "hSum", "hSum" , nbins , nbinsQuant)
                elif BINtype=="mTauTauVis" : hSumi = TH1F( "hSum", "hSum" , nbins , 0. , 200.)
                if not hSumi.GetSumw2N() : hSumi.Sumw2()
                for place in range(1,hSumCopy.GetNbinsX() + 2) :
                    content=hSumCopy.GetBinContent(place)
                    newbin=hSumi.FindBin(hSumCopy.GetBinCenter(place))
                    binErrorCopy = hSumCopy.GetBinError(place);
                    binError = hSumi.GetBinError(newbin);
                    hSumi.SetBinContent(newbin, hSumi.GetBinContent(newbin)+content)
                    hSumi.SetBinError(newbin,sqrt(binError*binError+ binErrorCopy*binErrorCopy))
                hSumi.SetBinErrorOption(1)
                #if not hSum.GetSumw2N() : hSum.Sumw2()
                if hSumi.GetBinContent(hSumi.GetNbinsX()) >0 :
                    ratiohSum=hSumi.GetBinError(hSumi.GetNbinsX())/hSumi.GetBinContent(hSumi.GetNbinsX())
                if hSumi.GetBinContent(hSumi.GetNbinsX()-1) >0 : ratiohSumP=hSumi.GetBinError(hSumi.GetNbinsX()-1)/hSumi.GetBinContent(hSumi.GetNbinsX()-1)
                errOcontSUMLast=errOcontSUMLast+[ratiohSum] if ratiohSum<1.001 else errOcontSUMLast+[1.0]
                errOcontSUMPLast=errOcontSUMPLast+[ratiohSumP] if ratiohSumP<1.001 else errOcontSUMPLast+[1.0]
                errSUMLast=errSUMLast+[hSumi.GetBinError(hSumi.GetNbinsX())]
                contSUMLast=contSUMLast+[hSumi.GetBinContent(hSumi.GetNbinsX())]
                if BINtype=="quantiles" :
                    lastQuant=lastQuant+[nbinsQuant[nbins]]
                    xmaxQuant=xmaxQuant+[xmaxdef]
                    xminQuant=xminQuant+[xmindef]
                print ("it should be only one ",  nkey, errOcontTTLast)
    print ("min",xmindef,xmin)
    print ("max",xmaxdef,xmax)
    return [errOcontTTLast,errOcontTTPLast,errOcontSUMLast,errOcontSUMPLast,lastQuant,xmaxQuant,xminQuant]

def ReadLimits(bdtType,nbin, BINtype,channel,local,nstart,ntarget):
    central=[]
    do1=[]
    do2=[]
    up1=[]
    up2=[]
    for nn,nbins in enumerate(nbin) :
        # ttH_2lss_1taumvaOutput_2lss_MEM_1D_nbin_9.log
        if nstart==-1 : shapeVariable=bdtType
        elif nstart==0 :
            if channel == "2l_2tau" :
                shapeVariable=options.variables+'_'+bdtType+'_nbin_'+str(nbins)
            else : shapeVariable=options.variables+'_'+bdtType+'_nbin_'+str(nbins)
        elif nstart==1 : shapeVariable=options.variables+'_'+str(nbins)+'bins'
        else : shapeVariable=options.variables+'_from'+str(nstart)+'_to_'+str(nbins)
        if BINtype=="ranged" : shapeVariable=shapeVariable+"_ranged"
        if BINtype=="quantiles" : shapeVariable=shapeVariable+"_quantiles"
        datacardFile_output = os.path.join(local, "ttH_%s.log" % shapeVariable)
        #if nn==0 :print  shapeVariable
        if nn==0 : print ("reading ", datacardFile_output)
        f = open(datacardFile_output, 'r+')
        lines = f.readlines() # get all lines as a list (array)
        for line in  lines:
          l = []
          tokens = line.split()
          if "Expected  2.5%"  in line : do2=do2+[float(tokens[4])]
          if "Expected 16.0%:" in line : do1=do1+[float(tokens[4])]
          if "Expected 50.0%:" in line : central=central+[float(tokens[4])]
          if "Expected 84.0%:" in line : up1=up1+[float(tokens[4])]
          if "Expected 97.5%:" in line : up2=up2+[float(tokens[4])]
    #print (shapeVariable,nbin)
    #print (shapeVariable,central)
    #print do1
    return [central,do1,do2,up1,up2]

def evaluateFOM(clf,keys,features,tag,train,test,nBdeplet,nB,nS,f_score_dicts,datatest):
    #datatest=pandas.read_csv('structured/'+process+'_Structured_from_20000sig_1.csv')
    #datatest['pT_b_o_kinFit_pT_b']=datatest['pT_b']/datatest['kinFit_pT_b']
    #datatest['pT_Wj2_o_kinFit_pT_Wj2']=datatest['pT_Wj2']/datatest['kinFit_pT_Wj2']
    #datatest['pT_Wj1_o_kinFit_pT_Wj1']=datatest['pT_Wj1']/datatest['kinFit_pT_Wj1']
    #datatest['cosTheta_leadEWj_restTop'] = datatest['cosTheta_leadEWj_restTop'].abs()
    # make angles abs
    countTruth=0
    countEvt=0
    countHadTruth=0
    print ("events raw: ",int(datatest["counter"].min()),int(datatest["counter"].max()))
    for ii in  np.unique(data["counter"].values):   #range(int(datatest["counter"].min(axis=0)),int(datatest["counter"].max())) :
        if countEvt > 20000 : continue
        #print ii
        row=datatest.loc[datatest["counter"].values == ii]
        if len(row)>0 :
            countEvt=countEvt+1
            row=datatest.loc[datatest["counter"].values == ii]
            proba = clf.predict_proba(row[features].values)
            if proba[:,1].sum() > 0 : countHadTruth = countHadTruth + 1
            max= np.argmax(proba[:,1] )
            if row["bWj1Wj2_isGenMatched"].iloc[max] == 1 : countTruth=countTruth+1
    print ("process"+\
    			" truthRatio(%)"+\
    			" hyp Nfeat"+\
    			" trainROC testROC ratioROC"+\
    			" nB nBdeplet nS"+\
    			" variables"+\
    			" totEvt"+\
    			" EvtThruth")
    print (str(process)+\
        #" "+str(round(100*float(countTruth)/float(countEvt), 2))+\
        #" "+trainvar+" "+str(len(features))+\
        #" "+str(train)+" "+str(test)+" "+str(round(100.0*float(test)/train,2))+\
        #" "+str(nB)+" "+str(nBdeplet)+" "+str(nS)+\
        #" "+str(f_score_dicts)+\
        " "+str(countEvt)+\
        " "+str(countHadTruth)+\
        " "+str(countTruth))
    file = open(channel+'/'+keys+'_in_'+'_tag_'+tag+'_XGB_FOM'+'_nvar'+str(len(features))+'.txt',"w")
    file.write(
    			" "+str(round(100*float(countTruth)/float(countEvt), 2))+\
    			" "+trainvar+" "+str(len(features))+\
    			" "+str(train)+" "+str(test)+" "+str(round(100.0*float(test)/train,2))+\
    			" "+str(nB)+" "+str(nBdeplet)+" "+str(nS)+\
    			" "+str(f_score_dicts)+\
    			" "+str(countEvt)+\
                " "+str(countHadTruth)+\
    			" "+str(countTruth)
    			)
    file.close()
    print ("Date: ", time.asctime( time.localtime(time.time()) ))
####################################################################################################

def run_cmd(command):
  print "executing command = '%s'" % command
  p = subprocess.Popen(command, shell = True, stdout = subprocess.PIPE, stderr = subprocess.PIPE)
  stdout, stderr = p.communicate()
  print stderr
  return stdout

###########################################################
# doYields

def AddSystQuad(list):
    ell = []
    for element in list : ell = ell + [math.pow(element, 2.)]
    quad =  math.sqrt(sum(ell))
    return quad

def AddSystQuad2(a,b):
    a2 = math.pow(a, 2.)
    b2 = math.pow(b, 2.)
    x  = a2 + b2
    quad =  math.sqrt(x)
    return quad

def AddSystQuad4(a,b,c,d):
    a2 = math.pow(a, 2.)
    b2 = math.pow(b, 2.)
    c2 = math.pow(c, 2.)
    d2 = math.pow(d, 2.)
    x  = a2 + b2 + c2 + d2
    quad =  math.sqrt(x)
    return quad

def AddSystQuad7(a,b,c,d,e,f,g):
    a2 = math.pow(a, 2.)
    b2 = math.pow(b, 2.)
    c2 = math.pow(c, 2.)
    d2 = math.pow(d, 2.)
    e2 = math.pow(e, 2.)
    f2 = math.pow(f, 2.)
    g2 = math.pow(g, 2.)
    x  = a2 + b2 + c2 + d2 + e2 + f2 + g2
    quad =  math.sqrt(x)
    return quad

def PrintTables(cmb, uargs, label, filey, uni, channel, blinded):
    c_cat = cmb.cp().bin([label])

    filey.write(r"""
\begin{tabular}{|l|r@{$ \,\,\pm\,\, $}l|}
\hline
Process & \multicolumn{2}{c|}{channel} \\
\hline
\hline"""+"\n")
    if channel == "2lss_1tau" or channel == "3l_1tau":
        print (channel, "2lss 3l")
        filey.write(r'ttH,H$\rightarrow$ZZ  & $%.2f$ & $%.2f$ \\' % (c_cat.cp().process(['ttH_hzz_faketau']).GetRate()+c_cat.cp().process(['ttH_hzz_gentau']).GetRate(), AddSystQuad2(c_cat.cp().process(['ttH_hzz_faketau']).GetUncertainty(*uargs),c_cat.cp().process(['ttH_hzz_gentau']).GetUncertainty(*uargs)))+"\n")
        filey.write(r'ttH,H$\rightarrow$WW  & $%.2f$ & $%.2f$ \\' % (c_cat.cp().process(['ttH_hww_faketau']).GetRate()+c_cat.cp().process(['ttH_hww_gentau']).GetRate(), AddSystQuad2(c_cat.cp().process(['ttH_hww_faketau']).GetUncertainty(*uargs),c_cat.cp().process(['ttH_hww_gentau']).GetUncertainty(*uargs)))+"\n")
        filey.write(r'ttH,H$\rightarrow\tau \tau$  & $%.2f$ & $%.2f$ \\' % (c_cat.cp().process(['ttH_htt_faketau']).GetRate()+c_cat.cp().process(['ttH_htt_gentau']).GetRate(), AddSystQuad2(c_cat.cp().process(['ttH_htt_faketau']).GetUncertainty(*uargs),c_cat.cp().process(['ttH_tt_gentau']).GetUncertainty(*uargs)))+"\n")
        filey.write(r'EWK  & $%.2f$ & $%.2f$ \\' % (c_cat.cp().process(['EWK_faketau']).GetRate()+c_cat.cp().process(['EWK_gentau']).GetRate(), AddSystQuad2(c_cat.cp().process(['EWK_faketau']).GetUncertainty(*uargs),c_cat.cp().process(['EWK_gentau']).GetUncertainty(*uargs)))+"\n")
        filey.write(r'TTZ  & $%.2f$ & $%.2f$ \\' % (c_cat.cp().process(['TTZ_faketau']).GetRate()+c_cat.cp().process(['TTZ_gentau']).GetRate(), AddSystQuad2(c_cat.cp().process(['TTZ_faketau']).GetUncertainty(*uargs),c_cat.cp().process(['TTZ_gentau']).GetUncertainty(*uargs)))+"\n")
        filey.write(r'TTW  & $%.2f$ & $%.2f$ \\' % (c_cat.cp().process(['TTW_faketau']).GetRate()+c_cat.cp().process(['TTW_gentau']).GetRate(), AddSystQuad2(c_cat.cp().process(['TTW_faketau']).GetUncertainty(*uargs),c_cat.cp().process(['TTW_gentau']).GetUncertainty(*uargs)))+"\n")
        filey.write(r'TTWW  & $%.2f$ & $%.2f$ \\' % (c_cat.cp().process(['TTWW_faketau']).GetRate()+c_cat.cp().process(['TTWW_gentau']).GetRate(), AddSystQuad2(c_cat.cp().process(['TTWW_faketau']).GetUncertainty(*uargs),c_cat.cp().process(['TTWW_gentau']).GetUncertainty(*uargs)))+"\n")
        filey.write(r'Rares  & $%.2f$ & $%.2f$ \\' % (c_cat.cp().process(['Rares_faketau']).GetRate()+c_cat.cp().process(['Rares_gentau']).GetRate(), AddSystQuad2(c_cat.cp().process(['Rares_faketau']).GetUncertainty(*uargs),c_cat.cp().process(['Rares_gentau']).GetUncertainty(*uargs)))+"\n")
        filey.write(r'tH  & $%.2f$ & $%.2f$ \\' % (c_cat.cp().process(['tH_faketau']).GetRate()+c_cat.cp().process(['tH_gentau']).GetRate(), AddSystQuad2(c_cat.cp().process(['tH_faketau']).GetUncertainty(*uargs),c_cat.cp().process(['tH_gentau']).GetUncertainty(*uargs)))+"\n")
    else :
        print channel
        filey.write(r'ttH,H$\rightarrow$ZZ  & $%.2f$ & $%.2f$ \\' % (c_cat.cp().process(['ttH_hzz']).GetRate(), c_cat.cp().process(['ttH_hzz']).GetUncertainty(*uargs))+"\n")
        filey.write(r'tt$\rightarrow$WW  & $%.2f$ & $%.2f$ \\' % (c_cat.cp().process(['ttH_hww']).GetRate(), c_cat.cp().process(['ttH_hww']).GetUncertainty(*uargs))+"\n")
        filey.write(r'tt$\rightarrow \tau \tau$  & $%.2f$ & $%.2f$ \\' % (c_cat.cp().process(['ttH_htt']).GetRate(), c_cat.cp().process(['ttH_htt']).GetUncertainty(*uargs))+"\n")
        filey.write(r'EWK  & $%.2f$ & $%.2f$ \\' % (c_cat.cp().process(['EWK']).GetRate(), c_cat.cp().process(['EWK']).GetUncertainty(*uargs))+"\n")
        filey.write(r'TTZ  & $%.2f$ & $%.2f$ \\' % (c_cat.cp().process(['TTZ']).GetRate(), c_cat.cp().process(['TTZ']).GetUncertainty(*uargs))+"\n")
        filey.write(r'TTW  & $%.2f$ & $%.2f$ \\' % (c_cat.cp().process(['TTW']).GetRate(), c_cat.cp().process(['TTW']).GetUncertainty(*uargs))+"\n")
        filey.write(r'TTWW  & $%.2f$ & $%.2f$ \\' % (c_cat.cp().process(['TTWW']).GetRate(), c_cat.cp().process(['TTWW']).GetUncertainty(*uargs))+"\n")
        filey.write(r'Rares  & $%.2f$ & $%.2f$ \\' % (c_cat.cp().process(['Rares']).GetRate(), c_cat.cp().process(['tH']).GetUncertainty(*uargs))+"\n")
        filey.write(r'tH  & $%.2f$ & $%.2f$ \\' % (c_cat.cp().process(['tH']).GetRate(), c_cat.cp().process(['tH']).GetUncertainty(*uargs))+"\n")
    filey.write(r'Fakes  & $%.2f$ & $%.2f$ \\' % (c_cat.cp().process(['fakes_data']).GetRate(), c_cat.cp().process(['fakes_data']).GetUncertainty(*uargs))+"\n")
    if uni == "Tallinn" :
        filey.write(r'Conversions  & $%.2f$ & $%.2f$ \\' % (c_cat.cp().process(['conversions']).GetRate(), c_cat.cp().process(['conversions']).GetUncertainty(*uargs))+"\n")
    if uni == "Cornell" :
        filey.write(r'Conversions  & $%.2f$ & $%.2f$ \\' % (c_cat.cp().process(['Conversion']).GetRate(), c_cat.cp().process(['Conversion']).GetUncertainty(*uargs))+"\n")
    if channel == "2lss_1tau" :
        filey.write(r'Flips  & $%.2f$ & $%.2f$ \\' % (c_cat.cp().process(['flips_data']).GetRate(), c_cat.cp().process(['flips_data']).GetUncertainty(*uargs))+"\n")
    filey.write(r'\hline Total Expected background    & $%.2f$ & $%.2f$ \\' % (c_cat.cp().backgrounds().GetRate(), c_cat.cp().backgrounds().GetUncertainty(*uargs))+"\n")
    filey.write(r'\hline SM expectation                        & $%.2f$ & $%.2f$ \\' % (
        c_cat.cp().backgrounds().GetRate() + c_cat.cp().process(['ttH_htt']).GetRate() + c_cat.cp().process(['ttH_hww']).GetRate() + c_cat.cp().process(['ttH_hzz']).GetRate() , AddSystQuad4( c_cat.cp().backgrounds().GetUncertainty(*uargs), c_cat.cp().process(['ttH_htt']).GetUncertainty(*uargs), c_cat.cp().process(['ttH_hww']).GetUncertainty(*uargs),c_cat.cp().process(['ttH_hzz']).GetUncertainty(*uargs)) )+"\n")
    filey.write(r'\hline'+"\n")
    if blinded : filey.write(r'Observed data & \multicolumn{2}{c|}{$-$} \\'+"\n")
    else : filey.write(r'Observed data & \multicolumn{2}{c|}{$%g$} \\' % (c_cat.cp().GetObservedRate())+"\n")
    filey.write(r"""\hline
\end{tabular}"""+"\n")


def PrintTables_Tau(cmb, uargs, filey, blinded, labels, type, ColapseCat = []):

    c_cat = []
    sum_proc = []
    err_sum_proc = []
    for label in labels :
        c_cat = c_cat  + [cmb.cp().bin(['ttH_'+label])]
        sum_proc = sum_proc + [0]
        err_sum_proc = err_sum_proc + [0]

    header = r'\begin{tabular}{|l|'
    bottom = r'Observed data & '
    for ll in xrange(len(labels)) :
        header = header + r'r@{$ \,\,\pm\,\, $}l|'
        if blinded : bottom = bottom + r' \multicolumn{2}{c|}{$-$} '
        else : bottom = bottom + r' \multicolumn{2}{c|}{$%g$} ' % (c_cat[ll].cp().GetObservedRate())
        if ll == len(labels) - 1 : bottom = bottom + r' \\'
        else : bottom = bottom + ' &'
    header = header +"} \n"
    bottom = bottom +"\n"
    filey.write(header)

    if type == 'tau' :
        conversions = "conversions"
        flips = 'flips'
        fakes_data = 'fakes_data'

        filey.write(r"""
        \hline
        Process & \multicolumn{2}{c|}{$1\Plepton + 2\tauh$} & \multicolumn{2}{c|}{$2\Plepton + 2\tauh$} & \multicolumn{2}{c|}{$3\Plepton + 1\tauh$} & \multicolumn{2}{c|}{$2\Plepton ss + 1\tauh$} \\
        \hline
        \hline"""+"\n")

    if type == 'multilep2lss' :
        conversions = "Convs"
        flips = 'data_flips'
        fakes_data = 'data_fakes'

        filey.write(r"""
        \hline
        Process & \multicolumn{20}{c|}{$2\Plepton ss$}  \\ \hline
        B-tag  & \multicolumn{4}{c|}{no req.}  & \multicolumn{8}{c|}{Loose}  & \multicolumn{8}{c|}{Tight}   \\ \hline
        Leptons  & \multicolumn{4}{c|}{$ee$} & \multicolumn{4}{c|}{$em$} & \multicolumn{4}{c|}{$mm$} & \multicolumn{4}{c|}{$em$} & \multicolumn{4}{c|}{$mm$}  \\ \hline
        Signal & \multicolumn{2}{c|}{$-$} & \multicolumn{2}{c|}{$+$} & \multicolumn{2}{c|}{$-$} & \multicolumn{2}{c|}{$+$} & \multicolumn{2}{c|}{$-$} & \multicolumn{2}{c|}{$+$} & \multicolumn{2}{c|}{$-$} & \multicolumn{2}{c|}{$+$} & \multicolumn{2}{c|}{$-$} & \multicolumn{2}{c|}{$+$} \\ \hline
        \hline
        \hline"""+"\n")

    if type == 'multilepCR2lss' :
        conversions = "Convs"
        flips = 'flips_data'
        fakes_data = 'data_fakes'

        filey.write(r"""
        \hline
        Process & \multicolumn{20}{c|}{$2\Plepton ss$}  \\ \hline
        B-tag   & \multicolumn{4}{c|}{no req.} & \multicolumn{8}{c|}{Loose}  & \multicolumn{8}{c|}{Tight}  \\ \hline
        Leptons  & \multicolumn{4}{c|}{$ee$} & \multicolumn{4}{c|}{$em$}  & \multicolumn{4}{c|}{$mm$} & \multicolumn{4}{c|}{$em$}  & \multicolumn{4}{c|}{$mm$} \\ \hline
        Signal & \multicolumn{2}{c|}{$-$} & \multicolumn{2}{c|}{$+$} & \multicolumn{2}{c|}{$-$} & \multicolumn{2}{c|}{$+$} & \multicolumn{2}{c|}{$-$} & \multicolumn{2}{c|}{$+$} & \multicolumn{2}{c|}{$-$} & \multicolumn{2}{c|}{$-$} & \multicolumn{2}{c|}{$+$} & \multicolumn{2}{c|}{$-$} & \multicolumn{2}{c|}{$+$} \\ \hline
        \hline
        \hline"""+"\n")

    if type == 'multilepCR3l4l' :
        conversions = "Convs"
        flips = 'flips_data'
        fakes_data = 'data_fakes'

        filey.write(r"""
        \hline
        Process & \multicolumn{10}{c|}{$3\Plepton$} & \multicolumn{2}{c|}{$4\Plepton$}  \\ \hline
        CR & \multicolumn{8}{c|}{$\PcZ$-peak} & \multicolumn{2}{c|}{$WZ$ enrich.}  & \multicolumn{2}{c|}{$ZZ$ enrich.} \\ \hline
        B-tag  & \multicolumn{4}{c|}{Loose}  & \multicolumn{4}{c|}{Tight}  & \multicolumn{4}{c|}{no req.}   \\ \hline
        Signal & \multicolumn{2}{c|}{$-$} & \multicolumn{2}{c|}{$+$} & \multicolumn{2}{c|}{$-$} & \multicolumn{2}{c|}{$+$} & & \multicolumn{4}{c|}{no req.} \\ \hline
        \hline
        \hline"""+"\n")

    if type == 'multilep3l4l' :
        conversions = "Convs"
        flips = 'flips_data'
        fakes_data = 'data_fakes'

        filey.write(r"""
        \hline
        Process &  \multicolumn{8}{c|}{$3\Plepton$} & \multicolumn{2}{c|}{$4\Plepton + 1\tauh$}  \\ \hline
        B-tag  & \multicolumn{4}{c|}{no req.}  & \multicolumn{8}{c|}{Loose}  & \multicolumn{8}{c|}{Tight}  & \multicolumn{4}{c|}{Loose}  & \multicolumn{4}{c|}{Tight} & \multicolumn{2}{c|}{no req.}  \\ \hline
        Signal & \multicolumn{2}{c|}{$-$} & \multicolumn{2}{c|}{$+$} & \multicolumn{2}{c|}{$-$} & \multicolumn{2}{c|}{$+$} & \multicolumn{2}{c|}{$-$} & \multicolumn{2}{c|}{$+$} \\ \hline
        \hline
        \hline"""+"\n")

    signals = [
        'ttH_hzz',
        'ttH_hww',
        'ttH_htt',
        'ttH_hmm',
        'ttH_hzg'
        ]

    TTWX = [
        'TTW',
        'TTWW'
        ]

    if 'multilep' in type :
        tH = [
        'tHW_htt',
        'tHq_htt',
        'tHW_hww',
        'tHq_hww',
        'tHW_hzz',
        'tHq_hzz'
        ]
        signalslabel_tH = [
            r'$\cPqt\PHiggs q$ $\PHiggs \to \Pgt\Pgt$& ',
            r'$\cPqt\PHiggs\PW$ $\PHiggs \to \Pgt\Pgt$& ',
            r'$\cPqt\PHiggs q$ $\PHiggs \to \PW\PW$ & ',
            r'$\cPqt\PHiggs\PW$ $\PHiggs \to \PW\PW$ & ',
            r'$\cPqt\PHiggs q$ $\PHiggs \to \cPZ\cPZ$  & ',
            r'$\cPqt\PHiggs\PW$ $\PHiggs \to \cPZ\cPZ$  & '
            ]

    if type == 'tau' :
        tH = [
        'tHq',
        'tHW'
        ]
        signalslabel_tH = [
            r'$\cPqt\PHiggs q$ & ',
            r'$\cPqt\PHiggs\PW$ & '
            ]

    EWK = [
        'ZZ',
        'WZ'
    ]

    singleCompMC = []
    if type == 'tau' : singleCompMC = singleCompMC + ['EWK']
    singleCompMC = singleCompMC + [
        'TTZ',
        fakes_data,
        conversions,
        flips,
        'Rares'
    ]

    singleCompMClabels = []
    if type == 'tau' : singleCompMClabels = singleCompMClabels + ['$\PW\cPZ + \cPZ\cPZ$']
    singleCompMClabels = singleCompMClabels + [
        '$\cPqt\cPaqt\cPZ$',
        'Misidentified',
        'Conversions',
        'signal flip',
        'Other'
    ]

    if type == 'tau' : listTosum = [signals, TTWX, tH]
    if 'multilep' in type : listTosum = [signals, TTWX, tH, EWK]
    for todo in listTosum :

        sigsum = [0.0 for i in xrange(len(labels))]
        sigsumErr = [0.0 for i in xrange(len(labels))]

        if todo == signals :
            linesigsum = 'ttH (sum) &'
            signalslabel = [
                r'$\cPqt\cPaqt\PHiggs$, $\PHiggs \to \cPZ\cPZ$ & ',
                r'$\cPqt\cPaqt\PHiggs$, $\PHiggs \to \PW\PW$ & ',
                r'$\cPqt\cPaqt\PHiggs$, $\PHiggs \to \Pgt\Pgt$ & ',
                r'$\cPqt\cPaqt\PHiggs$, $\PHiggs \to \mu\mu$ & ',
                r'$\cPqt\cPaqt\PHiggs$, $\PHiggs \to \cPZ\gamma$& ',
                ]
        elif todo == TTWX :
            linesigsum = 'ttW + ttWW &'
            signalslabel = [
                r'$\cPqt\cPaqt\PW$ & ',
                r'$\cPqt\cPaqt\PW\PW$ & '
                ]
        elif todo == tH :
            linesigsum = '$\cPqt\PHiggs$ (sum) &'
            signalslabel = signalslabel_tH
        if todo == EWK :
            linesigsum = '$\PW\cPZ + \cPZ\cPZ$ &'
            signalslabel = [
                r'$\cPZ\cPZ$ & ',
                r'$\PW\cPZ$ & '
                ]

        for ss, signal in enumerate(todo) :
            linesig = signalslabel[ss]
            for ll, label in enumerate(labels) :
                if "2lss_1tau" in label or  "3l_1tau" in label :
                    thissig = c_cat[ll].cp().process([signal+'_faketau']).GetRate() + c_cat[ll].cp().process([signal+'_gentau']).GetRate()
                    thissigErr = AddSystQuad({c_cat[ll].cp().process([signal+'_faketau']).GetUncertainty(*uargs), c_cat[ll].cp().process([signal+'_gentau']).GetUncertainty(*uargs)})
                else :
                    thissig = c_cat[ll].cp().process([signal]).GetRate()
                    thissigErr = c_cat[ll].cp().process([signal]).GetUncertainty(*uargs)
                if not thissig + thissigErr < 0.05:
                    linesig = linesig + ' $%.2f$ & $%.2f$ ' % (thissig, thissigErr)
                else : linesig = linesig + r' \multicolumn{2}{c|}{$< 0.05$} '
                if ll == len(labels) - 1 : linesig = linesig + r' \\'
                else : linesig = linesig + ' &'
                sigsum[ll] = sigsum[ll] + thissig
                sigsumErr[ll] = AddSystQuad({sigsumErr[ll], thissigErr})
                sum_proc[ll] = sum_proc[ll] + thissig
                err_sum_proc[ll] = AddSystQuad({err_sum_proc[ll], thissigErr})
            filey.write(linesig+"\n")
        filey.write(r'\hline'+"\n")

        for ll, label in enumerate(labels) :
            if not sigsum[ll] +  sigsumErr[ll] < 0.05:
                linesigsum = linesigsum + ' $%.2f$ & $%.2f$ ' % (sigsum[ll], sigsumErr[ll])
            else :  linesigsum = linesigsum + r' \multicolumn{2}{c|}{$< 0.05$} '
            if ll == len(labels) - 1 : linesigsum = linesigsum + r' \\'
            else : linesigsum = linesigsum + ' &'
        filey.write(linesigsum+"\n")
        filey.write(r'\hline'+"\n")

    for ss, signal in enumerate(singleCompMC) :
        lineTTZ = singleCompMClabels[ss]+' & '
        for ll, label in enumerate(labels) :
            if ("2lss_1tau" in label or  "3l_1tau" in label ) and signal not in ['fakes_data', 'flips']:
                thissig = c_cat[ll].cp().process([signal+'_faketau']).GetRate() + c_cat[ll].cp().process([signal+'_gentau']).GetRate()
                thissigErr = AddSystQuad({c_cat[ll].cp().process([signal+'_faketau']).GetUncertainty(*uargs), c_cat[ll].cp().process([signal+'_gentau']).GetUncertainty(*uargs)})
            else :
                thissig = c_cat[ll].cp().process([signal]).GetRate()
                thissigErr = c_cat[ll].cp().process([signal]).GetUncertainty(*uargs)
            if not thissig + thissigErr < 0.05:
                lineTTZ = lineTTZ + ' $%.2f$ & $%.2f$ ' % (thissig, thissigErr)
            else : lineTTZ = lineTTZ + r' \multicolumn{2}{c|}{$< 0.05$} '
            sum_proc[ll] = sum_proc[ll] + thissig
            err_sum_proc[ll] = AddSystQuad({err_sum_proc[ll], thissigErr})
            if ll == len(labels) - 1 : lineTTZ = lineTTZ + r' \\ '+"\n"
            else : lineTTZ = lineTTZ + ' &'
        filey.write(lineTTZ+"\n")

    lineSUM = r'\hline\hline'+"\n"+' SM expectation & '
    for ll, label in enumerate(labels) :
        if not sum_proc[ll] + err_sum_proc[ll] < 0.05:
            lineSUM = lineSUM + ' $%.2f$ & $%.2f$ ' % (sum_proc[ll] , err_sum_proc[ll] )
        else : lineSUM = lineSUM + r' \multicolumn{2}{c|}{$< 0.05$} '
        if ll == len(labels) - 1 : lineSUM = lineSUM + r' \\ '+"\n"
        else : lineSUM = lineSUM + ' &'
    filey.write(lineSUM+"\n")

    filey.write(r'\hline'+"\n")
    filey.write(bottom)
    filey.write(r"""\hline
    \end{tabular}"""+"\n")

def val_tune_rf(estimator,x_train,y_train,x_val,y_val,params, fileToWrite):
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
        estimator.fit(x_train,y_train)
        preds_prob = estimator.predict_proba(x_val)
        preds_prob_train = estimator.predict_proba(x_train)
        # print preds_prob[:,1]
        result = roc_auc_score(y_val,preds_prob[:,1])
        print 'roc_auc_score : %f'%result
        results.append((param,result))
        fileToWrite.write(str(param)+"\n")
        fileToWrite.write(str(roc_auc_score(y_val,preds_prob[:,1]))+" "+str(roc_auc_score(y_train,preds_prob_train[:,1])))
        fileToWrite.write("\n")
        print ("Date: ", time.asctime( time.localtime(time.time()) ))
    results.sort(key=lambda k: k[1])
    #print results
    #print results[-1]
    return results
