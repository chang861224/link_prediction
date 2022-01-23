python3 split.py --dataset Cora
python3 struc2vec/src/main.py --input split/train_Cora.edgelist --output emb/Cora-struc2vec.emb
./smore/cli/hpe -train split/train_Cora_weighted.edgelist -save emb/Cora-HPE.emb -dimensions 128 -undirected 1 -sample_times 1200 -walk_steps 5 -threads 8
./smore/cli/hpe -train split/train_Cora_weighted.edgelist -save emb/Cora-struc2vec-HPE.emb -load_v emb/Cora-struc2vec.emb -dimensions 128 -undirected 1 -sample_times 1200 -walk_steps 5 -threads 8
