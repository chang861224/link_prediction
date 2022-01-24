for dataset in "Cora" "CiteSeer" "facebook348" "brazil-airports" "europe-airports" "usa-airports"
do
    echo "--------------------"
    echo $dataset
    echo 
    echo ">> struc2vec"
    python3 predict.py --dataset $dataset --embed emb/$dataset-struc2vec.emb
    echo
    echo ">> HPE"
    python3 predict.py --dataset $dataset --embed emb/$dataset-HPE.emb
    echo
    echo ">> struc2vec+HPE"
    python3 predict.py --dataset $dataset --embed emb/$dataset-struc2vec-HPE.emb
done
