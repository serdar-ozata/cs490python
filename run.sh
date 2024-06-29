# variables
interpreter="./venv/bin/python"
dname="Yelp"
cores=63
alpha=0.3
node_per_core=1 # default

folder_name=$dname"-"$cores"-"$alpha
$interpreter ./solution.py run $cores $dname --onephase --noconstructive --alpha $alpha --node_core_count $node_per_core --convertbin "python3 ./ConvertBin1D.py"

