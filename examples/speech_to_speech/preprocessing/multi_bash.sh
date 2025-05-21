y=( 14 15 16 17 18 19 20 )
for year in ${y[@]};
do
    bash bash.sh 20$year &
done
wait
