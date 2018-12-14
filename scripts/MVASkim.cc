#include "MVASkim.h"
/////
//   Main function
/////
void MVASkim(const char * Input = "", const char * Output =""){
 // define signal or bkg
 string sample = "HJet";
 if(string(Output).find("NJet") != std::string::npos)sample = "NJet";
 //Call input file
 TFile *inputfile = TFile::Open(InputPath+Input);
 TTree *readingtree = new TTree("readingtree","readingtree"); readingtree = (TTree*) inputfile->Get("syncTree");
 rSetBranchAddress(readingtree);
 //Define output file
 TFile *newfile = new TFile(OutputPath+Output,"recreate");
 TTree* newtree = new TTree("syncTree","syncTree");
 newtree->SetMaxTreeSize(99000000000);
 wSetBranchAddress(newtree);
 //Fill new branches
 int nen = nentries; if(nentries==-1) nen = readingtree->GetEntries();
 for(Int_t en=0; en<nen; en++){
  //Ini
  wClearInitialization();
  Long64_t tentry = readingtree->LoadTree(en); 
  rGetEntry(tentry);
  for(uint jet_en=0; jet_en<Jet25_pt->size(); jet_en++){
   if(Jet25_isFromH->at(jet_en)==1&&sample=="NJet")continue;//select NonHiggsJet
   if(Jet25_isFromH->at(jet_en)!=1&&sample=="HJet")continue;//select HiggsJet
   wBranch(jet_en);
   //Fill tree
   newtree->Fill();
  }
 }
 //Save new file
 newfile->cd();
 newfile->Write();
 newfile->Close();
} 
/////
//   Analysis functions
/////
//Fill Branches
void rSetBranchAddress(TTree* readingtree){
   // Set branch addresses and branch pointers
   readingtree->SetBranchAddress("nEvent", &nEvent, &b_nEvent);
   readingtree->SetBranchAddress("Jet_numLoose", &Jet_numLoose, &b_Jet_numLoose);
   readingtree->SetBranchAddress("EventWeight", &EventWeight, &b_EventWeight);
   readingtree->SetBranchAddress("Jet25_isToptag", &Jet25_isToptag, &b_Jet25_isToptag);
   readingtree->SetBranchAddress("Jet25_axis2", &Jet25_axis2, &b_Jet25_axis2);
   readingtree->SetBranchAddress("Jet25_bDiscriminator", &Jet25_bDiscriminator, &b_Jet25_bDiscriminator);
   readingtree->SetBranchAddress("Jet25_bjdeta", &Jet25_bjdeta, &b_Jet25_bjdeta);
   readingtree->SetBranchAddress("Jet25_bjdphi", &Jet25_bjdphi, &b_Jet25_bjdphi);
   readingtree->SetBranchAddress("Jet25_bjdr", &Jet25_bjdr, &b_Jet25_bjdr);
   readingtree->SetBranchAddress("Jet25_bjptratio", &Jet25_bjptratio, &b_Jet25_bjptratio);
   readingtree->SetBranchAddress("Jet25_chargedEmEnergyFraction", &Jet25_chargedEmEnergyFraction, &b_Jet25_chargedEmEnergyFraction);
   readingtree->SetBranchAddress("Jet25_chargedHadronEnergyFraction", &Jet25_chargedHadronEnergyFraction, &b_Jet25_chargedHadronEnergyFraction);
   readingtree->SetBranchAddress("Jet25_chargedMultiplicity", &Jet25_chargedMultiplicity, &b_Jet25_chargedMultiplicity);
   readingtree->SetBranchAddress("Jet25_dilepdeta", &Jet25_dilepdeta, &b_Jet25_dilepdeta);
   readingtree->SetBranchAddress("Jet25_dilepdphi", &Jet25_dilepdphi, &b_Jet25_dilepdphi);
   readingtree->SetBranchAddress("Jet25_dilepdr", &Jet25_dilepdr, &b_Jet25_dilepdr);
   readingtree->SetBranchAddress("Jet25_dilepmetptratio", &Jet25_dilepmetptratio, &b_Jet25_dilepmetptratio);
   readingtree->SetBranchAddress("Jet25_dilepptratio", &Jet25_dilepptratio, &b_Jet25_dilepptratio);
   readingtree->SetBranchAddress("Jet25_electronEnergy", &Jet25_electronEnergy, &b_Jet25_electronEnergy);
   readingtree->SetBranchAddress("Jet25_energy", &Jet25_energy, &b_Jet25_energy);
   readingtree->SetBranchAddress("Jet25_eta", &Jet25_eta, &b_Jet25_eta);
   readingtree->SetBranchAddress("Jet25_isFromH", &Jet25_isFromH, &b_Jet25_isFromH);
   readingtree->SetBranchAddress("Jet25_isFromTop", &Jet25_isFromTop, &b_Jet25_isFromTop);
   readingtree->SetBranchAddress("Jet25_lepdetamax", &Jet25_lepdetamax, &b_Jet25_lepdetamax);
   readingtree->SetBranchAddress("Jet25_lepdetamin", &Jet25_lepdetamin, &b_Jet25_lepdetamin);
   readingtree->SetBranchAddress("Jet25_lepdphimax", &Jet25_lepdphimax, &b_Jet25_lepdphimax);
   readingtree->SetBranchAddress("Jet25_lepdphimin", &Jet25_lepdphimin, &b_Jet25_lepdphimin);
   readingtree->SetBranchAddress("Jet25_lepdrmax", &Jet25_lepdrmax, &b_Jet25_lepdrmax);
   readingtree->SetBranchAddress("Jet25_lepdrmin", &Jet25_lepdrmin, &b_Jet25_lepdrmin);
   readingtree->SetBranchAddress("Jet25_lepptratiomax", &Jet25_lepptratiomax, &b_Jet25_lepptratiomax);
   readingtree->SetBranchAddress("Jet25_lepptratiomin", &Jet25_lepptratiomin, &b_Jet25_lepptratiomin);
   readingtree->SetBranchAddress("Jet25_mass", &Jet25_mass, &b_Jet25_mass);
   readingtree->SetBranchAddress("Jet25_matchId", &Jet25_matchId, &b_Jet25_matchId);
   readingtree->SetBranchAddress("Jet25_metptratio", &Jet25_metptratio, &b_Jet25_metptratio);
   readingtree->SetBranchAddress("Jet25_mult", &Jet25_mult, &b_Jet25_mult);
   readingtree->SetBranchAddress("Jet25_muonEnergyFraction", &Jet25_muonEnergyFraction, &b_Jet25_muonEnergyFraction);
   readingtree->SetBranchAddress("Jet25_neutralHadEnergyFraction", &Jet25_neutralHadEnergyFraction, &b_Jet25_neutralHadEnergyFraction);
   readingtree->SetBranchAddress("Jet25_nonjdeta", &Jet25_nonjdeta, &b_Jet25_nonjdeta);
   readingtree->SetBranchAddress("Jet25_nonjdilepdeta", &Jet25_nonjdilepdeta, &b_Jet25_nonjdilepdeta);
   readingtree->SetBranchAddress("Jet25_nonjdilepdphi", &Jet25_nonjdilepdphi, &b_Jet25_nonjdilepdphi);
   readingtree->SetBranchAddress("Jet25_nonjdilepdr", &Jet25_nonjdilepdr, &b_Jet25_nonjdilepdr);
   readingtree->SetBranchAddress("Jet25_nonjdilepptratio", &Jet25_nonjdilepptratio, &b_Jet25_nonjdilepptratio);
   readingtree->SetBranchAddress("Jet25_nonjdphi", &Jet25_nonjdphi, &b_Jet25_nonjdphi);
   readingtree->SetBranchAddress("Jet25_nonjdr", &Jet25_nonjdr, &b_Jet25_nonjdr);
   readingtree->SetBranchAddress("Jet25_nonjptratio", &Jet25_nonjptratio, &b_Jet25_nonjptratio);
   readingtree->SetBranchAddress("Jet25_numberOfConstituents", &Jet25_numberOfConstituents, &b_Jet25_numberOfConstituents);
   readingtree->SetBranchAddress("Jet25_pfCombinedCvsBJetTags", &Jet25_pfCombinedCvsBJetTags, &b_Jet25_pfCombinedCvsBJetTags);
   readingtree->SetBranchAddress("Jet25_pfCombinedCvsLJetTags", &Jet25_pfCombinedCvsLJetTags, &b_Jet25_pfCombinedCvsLJetTags);
   readingtree->SetBranchAddress("Jet25_pfCombinedInclusiveSecondaryVertexV2BJetTags", &Jet25_pfCombinedInclusiveSecondaryVertexV2BJetTags, &b_Jet25_pfCombinedInclusiveSecondaryVertexV2BJetTags);
   readingtree->SetBranchAddress("Jet25_pfCombinedMVAV2BJetTags", &Jet25_pfCombinedMVAV2BJetTags, &b_Jet25_pfCombinedMVAV2BJetTags);
   readingtree->SetBranchAddress("Jet25_pfDeepCSVCvsBJetTags", &Jet25_pfDeepCSVCvsBJetTags, &b_Jet25_pfDeepCSVCvsBJetTags);
   readingtree->SetBranchAddress("Jet25_pfDeepCSVCvsLJetTags", &Jet25_pfDeepCSVCvsLJetTags, &b_Jet25_pfDeepCSVCvsLJetTags);
   readingtree->SetBranchAddress("Jet25_pfJetProbabilityBJetTags", &Jet25_pfJetProbabilityBJetTags, &b_Jet25_pfJetProbabilityBJetTags);
   readingtree->SetBranchAddress("Jet25_phi", &Jet25_phi, &b_Jet25_phi);
   readingtree->SetBranchAddress("Jet25_photonEnergy", &Jet25_photonEnergy, &b_Jet25_photonEnergy);
   readingtree->SetBranchAddress("Jet25_pt", &Jet25_pt, &b_Jet25_pt);
   readingtree->SetBranchAddress("Jet25_ptD", &Jet25_ptD, &b_Jet25_ptD);
   readingtree->SetBranchAddress("Jet25_px", &Jet25_px, &b_Jet25_px);
   readingtree->SetBranchAddress("Jet25_py", &Jet25_py, &b_Jet25_py);
   readingtree->SetBranchAddress("Jet25_pz", &Jet25_pz, &b_Jet25_pz);
   readingtree->SetBranchAddress("Jet25_qg", &Jet25_qg, &b_Jet25_qg);
};
void wSetBranchAddress(TTree* newtree){
   newtree->Branch("EventNumber", &EventNumber );
   newtree->Branch("n_presel_jet", &n_presel_jet );
   newtree->Branch("EvtWeight", &EvtWeight);
   newtree->Branch("EvtWgtOVnJet", &EvtWgtOVnJet);
   newtree->Branch("Jet_isToptag", &Jet_isToptag);
   newtree->Branch("Jet_axis2", &Jet_axis2); 
   newtree->Branch("Jet_bDiscriminator", &Jet_bDiscriminator); 
   newtree->Branch("Jet_bjdeta", &Jet_bjdeta); 
   newtree->Branch("Jet_bjdphi", &Jet_bjdphi); 
   newtree->Branch("Jet_bjdr", &Jet_bjdr); 
   newtree->Branch("Jet_bjptratio", &Jet_bjptratio); 
   newtree->Branch("Jet_chargedEmEnergyFraction", &Jet_chargedEmEnergyFraction); 
   newtree->Branch("Jet_chargedHadronEnergyFraction", &Jet_chargedHadronEnergyFraction); 
   newtree->Branch("Jet_chargedMultiplicity", &Jet_chargedMultiplicity); 
   newtree->Branch("Jet_dilepdeta", &Jet_dilepdeta); 
   newtree->Branch("Jet_dilepdphi", &Jet_dilepdphi); 
   newtree->Branch("Jet_dilepdr", &Jet_dilepdr); 
   newtree->Branch("Jet_dilepmetptratio", &Jet_dilepmetptratio); 
   newtree->Branch("Jet_dilepptratio", &Jet_dilepptratio); 
   newtree->Branch("Jet_electronEnergy", &Jet_electronEnergy); 
   newtree->Branch("Jet_energy", &Jet_energy); 
   newtree->Branch("Jet_eta", &Jet_eta); 
   newtree->Branch("Jet_isFromH", &Jet_isFromH); 
   newtree->Branch("Jet_isFromTop", &Jet_isFromTop); 
   newtree->Branch("Jet_lepdetamax", &Jet_lepdetamax); 
   newtree->Branch("Jet_lepdetamin", &Jet_lepdetamin); 
   newtree->Branch("Jet_lepdphimax", &Jet_lepdphimax); 
   newtree->Branch("Jet_lepdphimin", &Jet_lepdphimin); 
   newtree->Branch("Jet_lepdrmax", &Jet_lepdrmax); 
   newtree->Branch("Jet_lepdrmin", &Jet_lepdrmin); 
   newtree->Branch("Jet_lepptratiomax", &Jet_lepptratiomax); 
   newtree->Branch("Jet_lepptratiomin", &Jet_lepptratiomin); 
   newtree->Branch("Jet_mass", &Jet_mass); 
   newtree->Branch("Jet_matchId", &Jet_matchId); 
   newtree->Branch("Jet_metptratio", &Jet_metptratio); 
   newtree->Branch("Jet_mult", &Jet_mult); 
   newtree->Branch("Jet_muonEnergyFraction", &Jet_muonEnergyFraction); 
   newtree->Branch("Jet_neutralHadEnergyFraction", &Jet_neutralHadEnergyFraction); 
   newtree->Branch("Jet_nonjdeta", &Jet_nonjdeta); 
   newtree->Branch("Jet_nonjdilepdeta", &Jet_nonjdilepdeta); 
   newtree->Branch("Jet_nonjdilepdphi", &Jet_nonjdilepdphi); 
   newtree->Branch("Jet_nonjdilepdr", &Jet_nonjdilepdr); 
   newtree->Branch("Jet_nonjdilepptratio", &Jet_nonjdilepptratio); 
   newtree->Branch("Jet_nonjdphi", &Jet_nonjdphi); 
   newtree->Branch("Jet_nonjdr", &Jet_nonjdr); 
   newtree->Branch("Jet_nonjptratio", &Jet_nonjptratio); 
   newtree->Branch("Jet_numberOfConstituents", &Jet_numberOfConstituents); 
   newtree->Branch("Jet_pfCombinedCvsBJetTags", &Jet_pfCombinedCvsBJetTags); 
   newtree->Branch("Jet_pfCombinedCvsLJetTags", &Jet_pfCombinedCvsLJetTags); 
   newtree->Branch("Jet_pfCombinedInclusiveSecondaryVertexV2BJetTags", &Jet_pfCombinedInclusiveSecondaryVertexV2BJetTags); 
   newtree->Branch("Jet_pfCombinedMVAV2BJetTags", &Jet_pfCombinedMVAV2BJetTags); 
   newtree->Branch("Jet_pfDeepCSVCvsBJetTags", &Jet_pfDeepCSVCvsBJetTags); 
   newtree->Branch("Jet_pfDeepCSVCvsLJetTags", &Jet_pfDeepCSVCvsLJetTags); 
   newtree->Branch("Jet_pfJetProbabilityBJetTags", &Jet_pfJetProbabilityBJetTags); 
   newtree->Branch("Jet_phi", &Jet_phi); 
   newtree->Branch("Jet_photonEnergy", &Jet_photonEnergy); 
   newtree->Branch("Jet_pt", &Jet_pt); 
   newtree->Branch("Jet_ptD", &Jet_ptD); 
   newtree->Branch("Jet_px", &Jet_px); 
   newtree->Branch("Jet_py", &Jet_py); 
   newtree->Branch("Jet_pz", &Jet_pz); 
   newtree->Branch("Jet_qg", &Jet_qg); 
};
void wClearInitialization(){
   EventNumber = -999;
   EvtWeight = -999;
   n_presel_jet = -999;
   EvtWgtOVnJet = -999;
   Jet_isToptag = -999;
   Jet_axis2 = -999;
   Jet_bDiscriminator = -999;
   Jet_bjdeta = -999;
   Jet_bjdphi = -999;
   Jet_bjdr = -999;
   Jet_bjptratio = -999;
   Jet_chargedEmEnergyFraction = -999;
   Jet_chargedHadronEnergyFraction = -999;
   Jet_chargedMultiplicity = -999;
   Jet_dilepdeta = -999;
   Jet_dilepdphi = -999;
   Jet_dilepdr = -999;
   Jet_dilepmetptratio = -999;
   Jet_dilepptratio = -999;
   Jet_electronEnergy = -999;
   Jet_energy = -999;
   Jet_eta = -999;
   Jet_isFromH = -999;
   Jet_isFromTop = -999;
   Jet_lepdetamax = -999;
   Jet_lepdetamin = -999;
   Jet_lepdphimax = -999;
   Jet_lepdphimin = -999;
   Jet_lepdrmax = -999;
   Jet_lepdrmin = -999;
   Jet_lepptratiomax = -999;
   Jet_lepptratiomin = -999;
   Jet_mass = -999;
   Jet_matchId = -999;
   Jet_metptratio = -999;
   Jet_mult = -999;
   Jet_muonEnergyFraction = -999;
   Jet_neutralHadEnergyFraction = -999;
   Jet_nonjdeta = -999;
   Jet_nonjdilepdeta = -999;
   Jet_nonjdilepdphi = -999;
   Jet_nonjdilepdr = -999;
   Jet_nonjdilepptratio = -999;
   Jet_nonjdphi = -999;
   Jet_nonjdr = -999;
   Jet_nonjptratio = -999;
   Jet_numberOfConstituents = -999;
   Jet_pfCombinedCvsBJetTags = -999;
   Jet_pfCombinedCvsLJetTags = -999;
   Jet_pfCombinedInclusiveSecondaryVertexV2BJetTags = -999;
   Jet_pfCombinedMVAV2BJetTags = -999;
   Jet_pfDeepCSVCvsBJetTags = -999;
   Jet_pfDeepCSVCvsLJetTags = -999;
   Jet_pfJetProbabilityBJetTags = -999;
   Jet_phi = -999;
   Jet_photonEnergy = -999;
   Jet_pt = -999;
   Jet_ptD = -999;
   Jet_px = -999;
   Jet_py = -999;
   Jet_pz = -999;
   Jet_qg = -999;
};
void rGetEntry(Long64_t tentry){
   b_nEvent->GetEntry(tentry);   //!
   b_EventWeight->GetEntry(tentry);   //!
   b_Jet_numLoose->GetEntry(tentry);   //!
   b_Jet25_isToptag->GetEntry(tentry);   //!
   b_Jet25_axis2->GetEntry(tentry);   //!
   b_Jet25_bDiscriminator->GetEntry(tentry);   //!
   b_Jet25_bjdeta->GetEntry(tentry);   //!
   b_Jet25_bjdphi->GetEntry(tentry);   //!
   b_Jet25_bjdr->GetEntry(tentry);   //!
   b_Jet25_bjptratio->GetEntry(tentry);   //!
   b_Jet25_chargedEmEnergyFraction->GetEntry(tentry);   //!
   b_Jet25_chargedHadronEnergyFraction->GetEntry(tentry);   //!
   b_Jet25_chargedMultiplicity->GetEntry(tentry);   //!
   b_Jet25_dilepdeta->GetEntry(tentry);   //!
   b_Jet25_dilepdphi->GetEntry(tentry);   //!
   b_Jet25_dilepdr->GetEntry(tentry);   //!
   b_Jet25_dilepmetptratio->GetEntry(tentry);   //!
   b_Jet25_dilepptratio->GetEntry(tentry);   //!
   b_Jet25_electronEnergy->GetEntry(tentry);   //!
   b_Jet25_energy->GetEntry(tentry);   //!
   b_Jet25_eta->GetEntry(tentry);   //!
   b_Jet25_isFromH->GetEntry(tentry);   //!
   b_Jet25_isFromTop->GetEntry(tentry);   //!
   b_Jet25_lepdetamax->GetEntry(tentry);   //!
   b_Jet25_lepdetamin->GetEntry(tentry);   //!
   b_Jet25_lepdphimax->GetEntry(tentry);   //!
   b_Jet25_lepdphimin->GetEntry(tentry);   //!
   b_Jet25_lepdrmax->GetEntry(tentry);   //!
   b_Jet25_lepdrmin->GetEntry(tentry);   //!
   b_Jet25_lepptratiomax->GetEntry(tentry);   //!
   b_Jet25_lepptratiomin->GetEntry(tentry);   //!
   b_Jet25_mass->GetEntry(tentry);   //!
   b_Jet25_matchId->GetEntry(tentry);   //!
   b_Jet25_metptratio->GetEntry(tentry);   //!
   b_Jet25_mult->GetEntry(tentry);   //!
   b_Jet25_muonEnergyFraction->GetEntry(tentry);   //!
   b_Jet25_neutralHadEnergyFraction->GetEntry(tentry);   //!
   b_Jet25_nonjdeta->GetEntry(tentry);   //!
   b_Jet25_nonjdilepdeta->GetEntry(tentry);   //!
   b_Jet25_nonjdilepdphi->GetEntry(tentry);   //!
   b_Jet25_nonjdilepdr->GetEntry(tentry);   //!
   b_Jet25_nonjdilepptratio->GetEntry(tentry);   //!
   b_Jet25_nonjdphi->GetEntry(tentry);   //!
   b_Jet25_nonjdr->GetEntry(tentry);   //!
   b_Jet25_nonjptratio->GetEntry(tentry);   //!
   b_Jet25_numberOfConstituents->GetEntry(tentry);   //!
   b_Jet25_pfCombinedCvsBJetTags->GetEntry(tentry);   //!
   b_Jet25_pfCombinedCvsLJetTags->GetEntry(tentry);   //!
   b_Jet25_pfCombinedInclusiveSecondaryVertexV2BJetTags->GetEntry(tentry);   //!
   b_Jet25_pfCombinedMVAV2BJetTags->GetEntry(tentry);   //!
   b_Jet25_pfDeepCSVCvsBJetTags->GetEntry(tentry);   //!
   b_Jet25_pfDeepCSVCvsLJetTags->GetEntry(tentry);   //!
   b_Jet25_pfJetProbabilityBJetTags->GetEntry(tentry);   //!
   b_Jet25_phi->GetEntry(tentry);   //!
   b_Jet25_photonEnergy->GetEntry(tentry);   //!
   b_Jet25_pt->GetEntry(tentry);   //!
   b_Jet25_ptD->GetEntry(tentry);   //!
   b_Jet25_px->GetEntry(tentry);   //!
   b_Jet25_py->GetEntry(tentry);   //!
   b_Jet25_pz->GetEntry(tentry);   //!
   b_Jet25_qg->GetEntry(tentry);   //!
};
void wBranch(int curr_en){
   EventNumber = nEvent;
   EvtWeight = EventWeight;
   n_presel_jet = Jet_numLoose;
   EvtWgtOVnJet = Jet_numLoose ==0? 0 : EventWeight/Jet_numLoose ;
   Jet_isToptag=Jet25_isToptag->at(curr_en);
   Jet_axis2=Jet25_axis2->at(curr_en);
   Jet_bDiscriminator=Jet25_bDiscriminator->at(curr_en);
   Jet_isFromH=Jet25_isFromH->at(curr_en);
   Jet_isFromTop=Jet25_isFromTop->at(curr_en);
   Jet_matchId=Jet25_matchId->at(curr_en);
   Jet_lepdrmin=Jet25_lepdrmin->at(curr_en);
   Jet_qg=Jet25_qg->at(curr_en);
   Jet_ptD=Jet25_ptD->at(curr_en);
   Jet_pfDeepCSVCvsLJetTags=Jet25_pfDeepCSVCvsLJetTags->at(curr_en);
   Jet_pfDeepCSVCvsBJetTags=Jet25_pfDeepCSVCvsBJetTags->at(curr_en);
   Jet_pfCombinedCvsLJetTags=Jet25_pfCombinedCvsLJetTags->at(curr_en);
   Jet_pfCombinedCvsBJetTags=Jet25_pfCombinedCvsBJetTags->at(curr_en);
   Jet_mult=Jet25_mult->at(curr_en);
   Jet_pt=Jet25_pt->at(curr_en);
   Jet_eta=Jet25_eta->at(curr_en);
   Jet_phi=Jet25_phi->at(curr_en);
   Jet_energy=Jet25_energy->at(curr_en);
   Jet_px=Jet25_px->at(curr_en);
   Jet_py=Jet25_py->at(curr_en);
   Jet_pz=Jet25_pz->at(curr_en);
   Jet_mass=Jet25_mass->at(curr_en);
   Jet_pfCombinedInclusiveSecondaryVertexV2BJetTags=Jet25_pfCombinedInclusiveSecondaryVertexV2BJetTags->at(curr_en);
   Jet_pfCombinedMVAV2BJetTags=Jet25_pfCombinedMVAV2BJetTags->at(curr_en);
   Jet_pfJetProbabilityBJetTags=Jet25_pfJetProbabilityBJetTags->at(curr_en);
   Jet_neutralHadEnergyFraction=Jet25_neutralHadEnergyFraction->at(curr_en);
   Jet_chargedHadronEnergyFraction=Jet25_chargedHadronEnergyFraction->at(curr_en);
   Jet_chargedEmEnergyFraction=Jet25_chargedEmEnergyFraction->at(curr_en);
   Jet_muonEnergyFraction=Jet25_muonEnergyFraction->at(curr_en);
   Jet_electronEnergy=Jet25_electronEnergy->at(curr_en);
   Jet_photonEnergy=Jet25_photonEnergy->at(curr_en);
   Jet_numberOfConstituents=Jet25_numberOfConstituents->at(curr_en);
   Jet_chargedMultiplicity=Jet25_chargedMultiplicity->at(curr_en);
   Jet_metptratio=Jet25_metptratio->at(curr_en);
   Jet_dilepmetptratio=Jet25_dilepmetptratio->at(curr_en);
   Jet_nonjdr=Jet25_nonjdr->at(curr_en);
   Jet_nonjdilepdr=Jet25_nonjdilepdr->at(curr_en);
   Jet_lepdrmax=Jet25_lepdrmax->at(curr_en);
   Jet_dilepdr=Jet25_dilepdr->at(curr_en);
   Jet_bjdr=Jet25_bjdr->at(curr_en);
   Jet_nonjdeta=Jet25_nonjdeta->at(curr_en);
   Jet_nonjdilepdeta=Jet25_nonjdilepdeta->at(curr_en);
   Jet_lepdetamin=Jet25_lepdetamin->at(curr_en);
   Jet_lepdetamax=Jet25_lepdetamax->at(curr_en);
   Jet_dilepdeta=Jet25_dilepdeta->at(curr_en);
   Jet_bjdeta=Jet25_bjdeta->at(curr_en);
   Jet_nonjdphi=Jet25_nonjdphi->at(curr_en);
   Jet_nonjdilepdphi=Jet25_nonjdilepdphi->at(curr_en);
   Jet_lepdphimin=Jet25_lepdphimin->at(curr_en);
   Jet_lepdphimax=Jet25_lepdphimax->at(curr_en);
   Jet_dilepdphi=Jet25_dilepdphi->at(curr_en);
   Jet_bjdphi=Jet25_bjdphi->at(curr_en);
   Jet_nonjptratio=Jet25_nonjptratio->at(curr_en);
   Jet_nonjdilepptratio=Jet25_nonjdilepptratio->at(curr_en);
   Jet_lepptratiomin=Jet25_lepptratiomin->at(curr_en);
   Jet_lepptratiomax=Jet25_lepptratiomax->at(curr_en);
   Jet_dilepptratio=Jet25_dilepptratio->at(curr_en);
   Jet_bjptratio=Jet25_bjptratio->at(curr_en);
};
