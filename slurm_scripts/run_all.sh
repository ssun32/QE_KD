for l in {0..1}; do
    for h in {0..16}; do
        sbatch eval_t_prune.sh $l $h
    done
done
