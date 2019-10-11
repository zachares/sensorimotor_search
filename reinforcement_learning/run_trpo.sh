logdir=${PWD}
for GAMMA in 0.99 0.995
do
    for TAU in 0.97 0.99
    do
        for BATCHSIZE in 512 1024
        do
            python main.py --gamma $GAMMA --tau $TAU --batch-size $BATCHSIZE --env-name 'peg1-multimodal-v0' --log-dir "${logdir}/trpo_tune" &> outs/gamma-$GAMMA,tau-$TAU,bs-$BATCHSIZE.out
            sleep 100
        done
    done
done
