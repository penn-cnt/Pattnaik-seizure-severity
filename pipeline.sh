#!/bin/zsh
find_in_conda_env(){
    conda env list | grep "${@}" >/dev/null 2>/dev/null
}
eval "$(conda shell.bash hook)"
if ! find_in_conda_env ".*szsev.*" ; then
    conda env create --file szsev-env.txt
fi
conda activate szsev

cd code

# echo "Starting script 00"
# time python 00-electrode_selection.py
# echo "Script 00 is done"

# echo
# echo "Starting script 01"
# time python 01-get_ictal_clips.py
# echo "Script 01 is done"

# echo
# echo "Starting script 02"
# time python 02-schindler_recruited_channels.py
# echo "Script 02 is done"

# echo
# echo "Starting script 03"
# time python 03-load_and_format_atlas_matlab_files.py
# echo "Script 03 is done"

# echo
# echo "Starting script 04"
# time python 04-atlas_spread.py
# echo "Script 04 is done"

# echo
# echo "Starting script 05"
# time python 05-nhs3.py
# echo "Script 05 is done"

# echo
# echo "Starting script 06"
# time python 06-seizure_severity.py
# echo "Script 06 is done"

# echo
# echo "Starting script 07"
# time python 07-preictal_features.py
# echo "Script 07 is done"

# echo
# echo "Starting script 08"
# time python 08-identify_interictal_clip_times.py
# echo "Script 08 is done"

echo
echo "Starting script 09"
time python 09-interictal_features.py
echo "Script 09 is done"
