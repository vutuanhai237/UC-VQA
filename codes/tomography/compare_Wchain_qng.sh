#!/bin/bash
#$ -N compare_Wchain_qng
#$ -M vutuanhai237@gmail.com
#$ -q all.q
#$ -m ea
#$ -S /bin/bash
#$ -cwd
#$ -pe smp 8
#$ -o compare_Wchain_qng_$JOB_ID.out
#$ -e compare_Wchain_qng_$JOB_ID.err
python compare_Wchain_qng.py
