for dataset in "Cora" "CiteSeer" "facebook348" "brazil-airports" "europe-airports" "usa-airports"
do
    #python3 split.py --dataset $dataset
    #python3 struc2vec/src/main.py --input split/train_$dataset.edgelist --output emb/$dataset-struc2vec.emb
    python3 genNetwork.py --dataset $dataset --weighted
    ./smore/cli/hpe -train split/train_$dataset\_weighted.edgelist -save emb/$dataset-HPE.emb -dimensions 128 -undirected 1 -sample_times 2000 -walk_steps 5 -threads 8
    ./smore/cli/hpe -train split/train_$dataset\_weighted.edgelist -save emb/$dataset-struc2vec-HPE.emb -load_v emb/$dataset-struc2vec.emb -dimensions 128 -undirected 1 -sample_times 2000 -walk_steps 5 -threads 8
done
