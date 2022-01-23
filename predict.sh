echo "struc2vec"
python3 predict.py --dataset Cora --embed emb/Cora-struc2vec.emb
echo "HPE"
python3 predict.py --dataset Cora --embed emb/Cora-HPE.emb
echo "struc2vec+HPE"
python3 predict.py --dataset Cora --embed emb/Cora-struc2vec-HPE.emb