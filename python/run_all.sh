#mkdir -p ../outputs/ttWbkg_alljets/logs
#python sklearn_xgboost.py -y 2016 -o ../outputs/ttWbkg_alljets > ../outputs/ttWbkg_alljets/logs/train_printout_2016.log 
#python sklearn_xgboost.py -y 2017 -o ../outputs/ttWbkg_alljets > ../outputs/ttWbkg_alljets/logs/train_printout_2017.log 
#python sklearn_xgboost.py -y 2018 -o ../outputs/ttWbkg_alljets > ../outputs/ttWbkg_alljets/logs/train_printout_2018.log 
#python apply_test.py -y 2016 -o ../outputs/ttWbkg_alljets > ../outputs/ttWbkg_alljets/logs/apply_printout_2016.log 
python apply_test.py -y 2017 -o ../outputs/ttWbkg_alljets > ../outputs/ttWbkg_alljets/logs/apply_printout_2017.log 
python apply_test.py -y 2018 -o ../outputs/ttWbkg_alljets > ../outputs/ttWbkg_alljets/logs/apply_printout_2018.log 
