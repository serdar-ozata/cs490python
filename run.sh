# variables
interpreter="./venv/bin/python"
declare -a dnames=("Flickr" "pattern1")
cores=7
alpha=0.3
cores_per_node=1 # default
remove_reduced_mtx=1 # *.reduced.mtx files are not used by the c program, set to 0 if you'll use it for debugging purposes
# END variables

for dname in "${dnames[@]}"; do
    # Construct folder name
    folder_name="$dname-$cores-$alpha-cpn$cores_per_node"
    folder_path="./folders/$folder_name"

    base_command="$interpreter ./solution.py run $cores $dname --noconstructive --alpha $alpha --node_core_count $cores_per_node --convertbin 'python3 ./ConvertBin1D.py'"
    eval $base_command

    # Add --noreduce argument, also output onephase
    command_with_noreduce="$base_command --noreduce --onephase"
    eval $command_with_noreduce

    if [ $remove_reduced_mtx -eq 1 ]; then
        rm "$folder_path/$dname.reduced.mtx"
    fi

    # Move one phase bin file to the folder
    onephase_fname="$dname.phases.$cores.one.bin"
    mv "./out/$onephase_fname" "$folder_path/$onephase_fname"

    # Move inpart file to the folder
    inpart_name="$dname.inpart.$cores"
    bin_inpart_name="$inpart_name.bin"
    cp "./mmdsets/schemes/$bin_inpart_name" "$folder_path/$bin_inpart_name"
    cp "./mmdsets/schemes/$inpart_name" "$folder_path/$inpart_name"
done
