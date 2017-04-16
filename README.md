# combrot
Expand, combine, evaluate multiple machine translation systems output

## Neural

### Build word representations

    fasttext supervised -dim 600 -minn 1 -maxn 6 -input corpus.tok.src -output model.src
    fasttext supervised -dim 600 -minn 1 -maxn 6 -input corpus.tok.tgt -output model.tgt

The models can be built from monolingual corpuses.

### Train translation matrix

    python train_tm.py model.src model.tgt parallel.tok.src parallel.tok.tgt tm_model.pickle

### Preparing input data

    paste source.txt output_system_1.txt output_system_2.txt ... | perl -pe 's/\t/ ||| /g' > combined.txt

### Similarity scoring

    python combrot_sim.py model.src model.tgt tm_model.pickle combined.txt > output.sim.txt

### LM scoring

    lamtram --operation nbest --src_in output.sim.txt ... > output.ll.txt
    paste output.sim.txt output.ll.txt | perl -pe 's/\t/ /g' > output.score.txt

## Statistical

### Build LM

    lmplz -o 5 < corpus.tok.tgt > klm_o5.arpa
    build_binary klm_o5.arpa klm_o5.bin

### LM scoring

    python combrot_lm.py klm_o5.bin combined.txt > output.lm.score
