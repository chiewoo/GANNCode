#!/bin/bash

for i in {1..100};
do
  python ensembleROC.py --output-dir=Aux_NMIFS35_3rd --tag=Aux_NMIFS35_3rd &
done
