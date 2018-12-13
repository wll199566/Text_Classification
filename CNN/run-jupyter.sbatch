#!/bin/bash

#SBATCH --job-name=jupyter
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu
#SBATCH --mem=10GB
#SBATCH --time=12:00:00

module purge
module load python3/intel/3.6.3 cudnn/9.0v7.0.5
module swap cuda  cuda/9.2.88
source activate vision

## Please don't modify the part after this line

port=$(shuf -i 6000-9999 -n 1)

/usr/bin/ssh -N -f -R $port:localhost:$port log-0
/usr/bin/ssh -N -f -R $port:localhost:$port log-1

cat<<EOF

Jupyter server is running on: $(hostname)
Job starts at: $(date)

Step 1 :

If you are working in NYU campus, please open an iTerm window, run command

ssh -L $port:localhost:$port $USER@prince.hpc.nyu.edu

If you are working off campus, you should already have ssh tunneling setup through HPC bastion host, 
that you can directly login to prince with command

ssh $USER@prince

Please open an iTerm window, run command

ssh -L $port:localhost:$port $USER@prince

Step 2:

Keep the iTerm windows in the previouse step open. Now open browser, find the line with

The Jupyter Notebook is running at: $(hostname)

the URL is something: http://localhost:${port}/?token=XXXXXXXX (see your token below)

you should be able to connect to jupyter notebook running remotly on prince compute node with above url

EOF

unset XDG_RUNTIME_DIR
if [ "$SLURM_JOBTMP" != "" ]; then
    export XDG_RUNTIME_DIR=$SLURM_JOBTMP
fi

jupyter notebook --no-browser --port $port --notebook-dir=$(pwd)




