#!/bin/bash
#
# get_pcoord.sh
#
# This script is run when calculating initial progress coordinates for new 
# initial states (istates).  This script is NOT run for calculating the progress
# coordinates of most trajectory segments; that is instead the job of runseg.sh.

# If we are debugging, output a lot of extra information.
if [ -n "$SEG_DEBUG" ] ; then
  set -x
  env | sort
fi

# Make sure we are in the correct directory
cd $WEST_SIM_ROOT

TEMP=$(mktemp -d)
cd $TEMP
echo $TEMP

# Symbolic link necessary files in TEMP
ln -sv $WEST_SIM_ROOT/common_files/reload_model_bstate.py ./reload_model.py
ln -sv $WEST_STRUCT_DATA_REF ./seg.gro
ln -sv $WEST_SIM_ROOT/common_files/dmin.npy .
ln -sv $WEST_SIM_ROOT/common_files/dmax.npy .
ln -sv $WEST_SIM_ROOT/common_files/model000250.pth .

# Calculate the committor value
python reload_model.py
cat nn_output.dat > $WEST_PCOORD_RETURN

# Copy files for HDF5 Framework
#cp $WEST_SIM_ROOT/common_files/topol.top $WEST_TRAJECTORY_RETURN
cp $WEST_STRUCT_DATA_REF $WEST_TRAJECTORY_RETURN
#cp $WEST_SIM_ROOT/bstates/bstate.edr $WEST_TRAJECTORY_RETURN
#cp $WEST_SIM_ROOT/bstates/bstate.trr $WEST_TRAJECTORY_RETURN

cp $WEST_SIM_ROOT/common_files/topol.top $WEST_RESTART_RETURN
cp $WEST_SIM_ROOT/bstates/bstate.gro $WEST_RESTART_RETURN/parent.gro
cp $WEST_SIM_ROOT/bstates/bstate.edr $WEST_RESTART_RETURN/parent.edr
cp $WEST_SIM_ROOT/bstates/bstate.trr $WEST_RESTART_RETURN/parent.trr

# Cleanup
cd $WEST_SIM_ROOT
#rm -r $TEMP

# if [ -n "$SEG_DEBUG" ] ; then
#   head -v $WEST_PCOORD_RETURN
# fi
