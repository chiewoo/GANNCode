echo Computing 100 GANN processes....
{
for j in {0..99}; do
GANN_Aux_100_Oneclick.sh $j &
done
} 
wait
echo All round-robin jobs done.
#echo Combining 
#ROCComb_GRB_NMIFS8_100Runs.sh
echo All processes has been done.

