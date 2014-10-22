### GAnnRun ###                                                                                                           
input_path=/home/johnoh/MLA/Aux/new_norm/
e_file=ALL_S6_959126400_hveto_channels_signif_dt_combined_norm_reduced_nmifs35_rr10_
e_file_tail=_evaluation_reduced.ann
t_file=ALL_S6_959126400_hveto_channels_signif_dt_combined_norm_reduced_nmifs35_rr10_
t_file_tail=_training_reduced.ann
tag_name=ALL_S6_959126400_hveto_channels_signif_dt_combined_norm_reduced_nmifs35_RR
nlayer=35,15,1
conn_rate=1.0
gen=5
pops=1000
mut_rate=0.8
min_range=-100.0
max_range=100.0
g_sigma=1.0
inc_rate=1.221
max_epoch=6000
output_path=Aux_NMIFS35_100_Out$1
### combined_dat_ROC ###                                                                                                  
file_dat_RR0=ALL_S6_959126400_hveto_channels_signif_dt_combined_norm_reduced_nmifs35_RR0_n35n15n1_c10_mr08_p1000_rmin-100.0_rmax100.0_gsigma1.0_w01x01_y1221_f05g09_m6000.dat
num_RR=10
nLayer=n35n15n1
tags=ALL_S6_959126400_hveto_channels_signif_dt_combined_norm_reduced_nmifs35_100
min_weights=-1.0
max_weights=1.0
steep_hidden=0.5
steep_out=0.9

echo Making output directories....
mkdir "$output_path"

echo Computing GAnnRun process of $tag_name data for $num_RR Round-Robin sets....
for i in {0..9}; do
GAnnRun.py -t "$input_path"$t_file"$i"$t_file_tail -e "$input_path"$e_file"$i"$e_file_tail --tag $tag_name"$i" -n $nlayer --connection-rate $conn_rate --generations $gen --population $pops --mutation-rate $mut_rate --range-min $min_range --range-max $max_range --gauss-sigma $g_sigma -y $inc_rate -m $max_epoch --output-dir "$output_path"
done
#echo Combining data files and plotting a combined ROC curve...
#echo The plot was saved: $output_path
#python combined_dat_ROC.py -i "$output_path" -f $file_dat_RR0 -o "$output_path" -l "$output_path"/logs -N $num_RR -n $nLayer -C $conn_rate -m $mut_rate -p $pops -g $gen -r $min_range -R $max_range -U $g_sigma -w $min_weights -W $max_weights -y $inc_rate -s $steep_hidden -S $steep_out -E $max_epoch -T $tags


