# combrot
Expand, combine, evaluate multiple machine translation systems output


### Build word representations

    fasttext supervised -dim 600 -minn 1 -maxn 6 -input corpus.tok.src -output model.src
    fasttext supervised -dim 600 -minn 1 -maxn 6 -input corpus.tok.tgt -output model.tgt

The models can be built from monolingual corpuses.

### Train translation matrix

    python train_tm.py model.src model.tgt parallel.tok.src parallel.tok.tgt tm_model.pickle

### Preparing input data

    perl paste-files.pl source.txt output_system_1.txt output_system_2.txt ... > combined.txt

