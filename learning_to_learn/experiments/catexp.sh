#!bin/bash

catrec() {
  dirs=${args[$1 - 1]}
  dirs=${dirs//,/ }
  #echo "dirs: $dirs"
  for dir in $dirs
  do
    new_path="$2"/"$dir"
    if [ $1 -lt $depth ]
    then
      let "new_index = $1 + 1"
      catrec $new_index $new_path
    else
      new_path="${new_path:1}"
      #echo "$new_path"
      for f in "$new_path"/*.txt
      do
        #echo "$f"
        name="${f%.*}"
        #cp "$f" "$name"_pdf_descr.txt
        #echo -e "\n$new_path" >> "$name"_pdf_descr.txt
        for_joining_name="$new_path"/join_"$counter".pdf
        ((counter++))
        descr_size=$(stat -c%s $f)
        if [[ $descr -eq 0 ]]
        then
          descr=""
        else
          descr=$(head -n 1 $f)
        fi
        convert -size 900x60 xc:white -pointsize 36 -fill red -draw "text 5,30 \"$new_path  $descr\" " "$name"_descr.png
        convert "$name"_descr.png "$name"_descr.pdf
        pdfjam "$name".pdf "$name"_descr.pdf --nup 1x2 --outfile "$for_joining_name"
        #rm "$name"*_descr*.png "$name"*_descr*.pdf "$name"*_descr*.txt
      done
    fi
  done
}

export depth=$(($# - 1))
#echo "depth: $depth"
args=("$@")
#echo ${args[@]}
export args=(${args[@]:1})
#echo ${args[@]}
export counter=0
catrec 1 ""
shopt -s globstar
pdftk **/join_*.pdf cat output "concatenated_plots/$1.pdf"
#rm **/join_*.pdf
shopt -u globstar
unset depth
unset args
unset counter
