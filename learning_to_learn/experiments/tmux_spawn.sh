#!bin/bash

spawnrec() {
  dirs=${args[$1 - 1]}
  dirs=${dirs//,/ }
  #echo "dirs: $dirs"
  for dir in $dirs; do
    new_path="$2"/"$dir"
    if [ $1 -lt $depth ]; then
      let "new_index = $1 + 1"
      spawnrec $new_index $new_path
    else
      new_path="${new_path:1}"
      #echo "$new_path"
      for ((i=0; i<$num_exps; i++)); do
        #command="cd $basedir;cd $new_path;read"
        #echo command: $command
        command="cd $basedir;cd $new_path;mkdir \"${resdirs[$i]}\";python3 run.py ${prepared[$i]};read"
        if [[ $i -eq 0 ]]; then
          echo i: $i
          echo "$command"
          tmux new-window -n "$new_path" "$command"
        else
          echo i: $i
          echo "$command"
          tmux split-window -t "$new_path"."${pane_indices[$i]}" "$command"
        fi
      done
    fi
  done
}

spawn() {
  export depth=$(($# - 1))
  #echo "depth: $depth"
  args=("$@")
  #echo ${args[@]}
  export args=(${args[@]:1}) 
  #echo ${args[@]}
  export counter=0
  run_args=${1//:/ }
  prepared=()
  resdirs=()
  for ra in $run_args
  do
    nra=${ra//,/ }
    splitted=($nra)
    conf=${splitted[-1]}
    resdir="${conf%.*}"
    resdirs+=("$resdir")
    prep="$ra"",|,tee,$resdir/log.txt"
    prep=${prep//,/ }
    prepared+=("$prep")
  done
  #echo ${prepared[@]}
  #echo ${resdirs[@]}
  num_exps=${#resdirs[@]}
  basedir=$PWD
  export basedir
  export num_exps resdirs prepared basedir
  export pane_indices=(0 0 1 0 1 2 0 1 2 3 0 1 2 3 4 5 6 7)
  #export pane_counter=0
  spawnrec 1 ""

  unset basedir depth args counter prepared num_exps
}

