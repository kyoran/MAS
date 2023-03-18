#!/bin/bash

#density_type="200"
#density_type="400"
#density_type="600"
#density_type="800"
density_type="1000"

#topology_types=("001" "002" "003" "004" "005" "006" "007" "008" "009" "010")
topology_types=("001" "006" "007" "009" "010")
#topology_types=("009")

algorithm_type="SAN"
#algorithm_type="CA"
#algorithm_type="SDB_DSG"
#algorithm_type="RSRSP"
#algorithm_type="RNCHCA"
#algorithm_type="HSBMAS"
#algorithm_type="HSBMAS_no_CS"
#algorithm_type="Motif"

for ((i=0; i<${#topology_types[@]}; i++))
do

  topology_type=${topology_types[i]}

  echo ${density_type}, ${topology_type}, ${algorithm_type}

  LOGFILE=./${density_type}-${topology_type}-${algorithm_type}.log

  echo "LOGFILE is" ${LOGFILE}
  echo -e "\n"

  python -u run_evolve.py \
      --density_type ${density_type} \
      --topology_type ${topology_type} \
      --algorithm_type ${algorithm_type} \
      > ${LOGFILE} 2>&1 &

done


