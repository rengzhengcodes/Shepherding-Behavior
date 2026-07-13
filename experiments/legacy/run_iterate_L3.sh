count=0
Repetition=25
for N_sheep in $(seq 100 100 100)
do
  for N_shepherd in $(seq 1 1 5)
  do
    for Iterations in 200000
    do
      for L3 in 0
      do
        nohup python "main_interface_metric.py" $N_sheep $N_shepherd $Iterations $L3 $Repetition >/dev/null 2>&1 &
        echo "N_sheep: $N_sheep";
        echo "N_shepherd: $N_shepherd";
        echo "Iterations: $Iterations";
        echo "L3: $L3";
        echo "Repetition: $Repetition";
        count=$((count + 1));
      done
    done
  done
done

echo "Number of total runs: $count"
