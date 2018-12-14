/**
This Macro
pick the jet saved in vector<float> and resaved as Double_t

Need to specify
0. See Declare constants
*/
/////
//   To select higgs jet run: root -l -b -q MVASkim.cc+'("TTW_2L.root","TTW_NJet.root")'      
//   To select non higgs jet run: root -l -b -q MVASkim.cc+'("TTH_hww_2L.root","TTH_hww_HJet.root")'
/////
/////
//   Prepare Root and Roofit
/////
#include "TFile.h"
#include "TObject.h"
#include "TTree.h"
#include "TTreePlayer.h"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <iostream>
#include <TLorentzVector.h>
#include <cmath>
#include "TMath.h"
#include "TMVA/Reader.h"
using namespace std;
////
//   Declare constants
/////
const int nentries = -1;//-1 is all entries  51 for first DiMuSR
TString InputPath = "/eos/user/b/binghuan/Rootplas/Rootplas_20181106/2L/";
TString OutputPath = "../data/";
/////
//   Declare analysis functions 
/////
//Variable handleling
void rSetBranchAddress(TTree* readingtree);
void wSetBranchAddress(TTree* newtree);
void wClearInitialization();
void rGetEntry(Long64_t tentry);
void wBranch(int curr_en);
/////
//   Variables to be read
/////
   Long64_t        nEvent;
   Float_t         EventWeight;
   Int_t           Jet_numLoose;
   vector<double>  *Jet25_isToptag;
   vector<double>  *Jet25_axis2;
   vector<double>  *Jet25_bDiscriminator;
   vector<double>  *Jet25_bjdeta;
   vector<double>  *Jet25_bjdphi;
   vector<double>  *Jet25_bjdr;
   vector<double>  *Jet25_bjptratio;
   vector<double>  *Jet25_chargedEmEnergyFraction;
   vector<double>  *Jet25_chargedHadronEnergyFraction;
   vector<double>  *Jet25_chargedMultiplicity;
   vector<double>  *Jet25_dilepdeta;
   vector<double>  *Jet25_dilepdphi;
   vector<double>  *Jet25_dilepdr;
   vector<double>  *Jet25_dilepmetptratio;
   vector<double>  *Jet25_dilepptratio;
   vector<double>  *Jet25_electronEnergy;
   vector<double>  *Jet25_energy;
   vector<double>  *Jet25_eta;
   vector<double>  *Jet25_isFromH;
   vector<double>  *Jet25_isFromTop;
   vector<double>  *Jet25_lepdetamax;
   vector<double>  *Jet25_lepdetamin;
   vector<double>  *Jet25_lepdphimax;
   vector<double>  *Jet25_lepdphimin;
   vector<double>  *Jet25_lepdrmax;
   vector<double>  *Jet25_lepdrmin;
   vector<double>  *Jet25_lepptratiomax;
   vector<double>  *Jet25_lepptratiomin;
   vector<double>  *Jet25_mass;
   vector<double>  *Jet25_matchId;
   vector<double>  *Jet25_metptratio;
   vector<double>  *Jet25_mult;
   vector<double>  *Jet25_muonEnergyFraction;
   vector<double>  *Jet25_neutralHadEnergyFraction;
   vector<double>  *Jet25_nonjdeta;
   vector<double>  *Jet25_nonjdilepdeta;
   vector<double>  *Jet25_nonjdilepdphi;
   vector<double>  *Jet25_nonjdilepdr;
   vector<double>  *Jet25_nonjdilepptratio;
   vector<double>  *Jet25_nonjdphi;
   vector<double>  *Jet25_nonjdr;
   vector<double>  *Jet25_nonjptratio;
   vector<double>  *Jet25_numberOfConstituents;
   vector<double>  *Jet25_pfCombinedCvsBJetTags;
   vector<double>  *Jet25_pfCombinedCvsLJetTags;
   vector<double>  *Jet25_pfCombinedInclusiveSecondaryVertexV2BJetTags;
   vector<double>  *Jet25_pfCombinedMVAV2BJetTags;
   vector<double>  *Jet25_pfDeepCSVCvsBJetTags;
   vector<double>  *Jet25_pfDeepCSVCvsLJetTags;
   vector<double>  *Jet25_pfJetProbabilityBJetTags;
   vector<double>  *Jet25_phi;
   vector<double>  *Jet25_photonEnergy;
   vector<double>  *Jet25_pt;
   vector<double>  *Jet25_ptD;
   vector<double>  *Jet25_px;
   vector<double>  *Jet25_py;
   vector<double>  *Jet25_pz;
   vector<double>  *Jet25_qg;

   TBranch        *b_nEvent;   //!
   TBranch        *b_EventWeight;   //!
   TBranch        *b_Jet_numLoose;   //!
   TBranch        *b_Jet25_isToptag;   //!
   TBranch        *b_Jet25_axis2;   //!
   TBranch        *b_Jet25_bDiscriminator;   //!
   TBranch        *b_Jet25_bjdeta;   //!
   TBranch        *b_Jet25_bjdphi;   //!
   TBranch        *b_Jet25_bjdr;   //!
   TBranch        *b_Jet25_bjptratio;   //!
   TBranch        *b_Jet25_chargedEmEnergyFraction;   //!
   TBranch        *b_Jet25_chargedHadronEnergyFraction;   //!
   TBranch        *b_Jet25_chargedMultiplicity;   //!
   TBranch        *b_Jet25_dilepdeta;   //!
   TBranch        *b_Jet25_dilepdphi;   //!
   TBranch        *b_Jet25_dilepdr;   //!
   TBranch        *b_Jet25_dilepmetptratio;   //!
   TBranch        *b_Jet25_dilepptratio;   //!
   TBranch        *b_Jet25_electronEnergy;   //!
   TBranch        *b_Jet25_energy;   //!
   TBranch        *b_Jet25_eta;   //!
   TBranch        *b_Jet25_isFromH;   //!
   TBranch        *b_Jet25_isFromTop;   //!
   TBranch        *b_Jet25_lepdetamax;   //!
   TBranch        *b_Jet25_lepdetamin;   //!
   TBranch        *b_Jet25_lepdphimax;   //!
   TBranch        *b_Jet25_lepdphimin;   //!
   TBranch        *b_Jet25_lepdrmax;   //!
   TBranch        *b_Jet25_lepdrmin;   //!
   TBranch        *b_Jet25_lepptratiomax;   //!
   TBranch        *b_Jet25_lepptratiomin;   //!
   TBranch        *b_Jet25_mass;   //!
   TBranch        *b_Jet25_matchId;   //!
   TBranch        *b_Jet25_metptratio;   //!
   TBranch        *b_Jet25_mult;   //!
   TBranch        *b_Jet25_muonEnergyFraction;   //!
   TBranch        *b_Jet25_neutralHadEnergyFraction;   //!
   TBranch        *b_Jet25_nonjdeta;   //!
   TBranch        *b_Jet25_nonjdilepdeta;   //!
   TBranch        *b_Jet25_nonjdilepdphi;   //!
   TBranch        *b_Jet25_nonjdilepdr;   //!
   TBranch        *b_Jet25_nonjdilepptratio;   //!
   TBranch        *b_Jet25_nonjdphi;   //!
   TBranch        *b_Jet25_nonjdr;   //!
   TBranch        *b_Jet25_nonjptratio;   //!
   TBranch        *b_Jet25_numberOfConstituents;   //!
   TBranch        *b_Jet25_pfCombinedCvsBJetTags;   //!
   TBranch        *b_Jet25_pfCombinedCvsLJetTags;   //!
   TBranch        *b_Jet25_pfCombinedInclusiveSecondaryVertexV2BJetTags;   //!
   TBranch        *b_Jet25_pfCombinedMVAV2BJetTags;   //!
   TBranch        *b_Jet25_pfDeepCSVCvsBJetTags;   //!
   TBranch        *b_Jet25_pfDeepCSVCvsLJetTags;   //!
   TBranch        *b_Jet25_pfJetProbabilityBJetTags;   //!
   TBranch        *b_Jet25_phi;   //!
   TBranch        *b_Jet25_photonEnergy;   //!
   TBranch        *b_Jet25_pt;   //!
   TBranch        *b_Jet25_ptD;   //!
   TBranch        *b_Jet25_px;   //!
   TBranch        *b_Jet25_py;   //!
   TBranch        *b_Jet25_pz;   //!
   TBranch        *b_Jet25_qg;   //!


//variables to be written
   Long64_t        EventNumber;
   Double_t        EvtWeight;
   Int_t           n_presel_jet;
   Double_t        EvtWgtOVnJet;
   Double_t        Jet_isToptag;
   Double_t        Jet_axis2;
   Double_t        Jet_bDiscriminator;
   Double_t        Jet_bjdeta;
   Double_t        Jet_bjdphi;
   Double_t        Jet_bjdr;
   Double_t        Jet_bjptratio;
   Double_t        Jet_chargedEmEnergyFraction;
   Double_t        Jet_chargedHadronEnergyFraction;
   Double_t        Jet_chargedMultiplicity;
   Double_t        Jet_dilepdeta;
   Double_t        Jet_dilepdphi;
   Double_t        Jet_dilepdr;
   Double_t        Jet_dilepmetptratio;
   Double_t        Jet_dilepptratio;
   Double_t        Jet_electronEnergy;
   Double_t        Jet_energy;
   Double_t        Jet_eta;
   Double_t        Jet_isFromH;
   Double_t        Jet_isFromTop;
   Double_t        Jet_lepdetamax;
   Double_t        Jet_lepdetamin;
   Double_t        Jet_lepdphimax;
   Double_t        Jet_lepdphimin;
   Double_t        Jet_lepdrmax;
   Double_t        Jet_lepdrmin;
   Double_t        Jet_lepptratiomax;
   Double_t        Jet_lepptratiomin;
   Double_t        Jet_mass;
   Double_t        Jet_matchId;
   Double_t        Jet_metptratio;
   Double_t        Jet_mult;
   Double_t        Jet_muonEnergyFraction;
   Double_t        Jet_neutralHadEnergyFraction;
   Double_t        Jet_nonjdeta;
   Double_t        Jet_nonjdilepdeta;
   Double_t        Jet_nonjdilepdphi;
   Double_t        Jet_nonjdilepdr;
   Double_t        Jet_nonjdilepptratio;
   Double_t        Jet_nonjdphi;
   Double_t        Jet_nonjdr;
   Double_t        Jet_nonjptratio;
   Double_t        Jet_numberOfConstituents;
   Double_t        Jet_pfCombinedCvsBJetTags;
   Double_t        Jet_pfCombinedCvsLJetTags;
   Double_t        Jet_pfCombinedInclusiveSecondaryVertexV2BJetTags;
   Double_t        Jet_pfCombinedMVAV2BJetTags;
   Double_t        Jet_pfDeepCSVCvsBJetTags;
   Double_t        Jet_pfDeepCSVCvsLJetTags;
   Double_t        Jet_pfJetProbabilityBJetTags;
   Double_t        Jet_phi;
   Double_t        Jet_photonEnergy;
   Double_t        Jet_pt;
   Double_t        Jet_ptD;
   Double_t        Jet_px;
   Double_t        Jet_py;
   Double_t        Jet_pz;
   Double_t        Jet_qg;

