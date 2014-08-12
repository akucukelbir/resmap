'''
ResMap_toChimera: module containing interface functions between ResMap and
                  UCSF Chimera (Alp Kucukelbir, 2013)

Description of functions:
 createChimeraScript: produces a Chimera CMD script that loads
                      and surface colors the volume
	  try_alternatives: tries to run a command that might be in
                      any one of multiple locations

Requirements:
			numpy

Please see individual functions for attributions.
'''

import os
import subprocess as sp
from numpy import linspace

def createChimeraScript(inputFileName, Mbegin, Mmax, N, animated=False):

	(fname,ext)       = os.path.splitext(inputFileName)
	basename          = os.path.basename(inputFileName)
	(basenameRAW,ext) = os.path.splitext(basename)

	newFileName = fname + "_resmap_chimera.cmd"

	f = open(newFileName, 'w')

	f.write('# Color a map by local resolution computed by ResMap\n')
	f.write('#   Initial version of script courtesy of Tom Goddard, UCSF.\n\n')

	f.write('# Open both volumes and hide the ResMap volume.\n')
	f.write('set bg_color white\n')
	f.write('open #0 {}\n'.format(basename))
	f.write('open #1 {}_resmap{}\n'.format(basenameRAW,ext))
	f.write('volume #1 hide\n\n')

	# Define color mapping
	colorsChimera = ['blue','cyan','green','yellow','orange','red']
	valuesChimera = linspace(Mbegin,Mmax,len(colorsChimera))

	# Create a string to input to Chimera's scolor command
	colorStr = ''
	for i in range(len(colorsChimera)):
		colorStr += '%.2f' % valuesChimera[i] + ',' + colorsChimera[i] + ':'
	colorStr += '%.2f' % (valuesChimera[i] + 0.01) + ',' + 'gray'

	f.write('# Color the original map with the values from the ResMap output.\n')
	f.write('scolor #0 volume #1 cmap ' + colorStr + '\n\n\n\n')

	f.write('# OPTIONAL: ResMap Slice Animation.\n\n')

	f.write('# Show midway slice of the original map with contour level below the minimum map value.\n')
	if animated == False:
		f.write('# ')
	f.write('volume #0 planes z,' + str(N/2) + ' step 1 level -1 style surface\n\n');

	f.write('# Show a smooth transparent contour surface indicating the structure boundaries.\n')
	if animated == False:
		f.write('# ')
	f.write('vop gaussian #0 sDev 5 model #2\n');
	if animated == False:
		f.write('# ')
	f.write('volume #2 level 0.02 step 1 color .9,.7,.7,.5\n\n');

	f.write('# Zoom out a bit and tilt to a nice viewing angle.\n')
	if animated == False:
		f.write('# ')
	f.write('turn x -45\n');
	if animated == False:
		f.write('# ')
	f.write('turn y -30\n');
	if animated == False:
		f.write('# ')
	f.write('turn z -30\n');
	if animated == False:
		f.write('# ')
	f.write('scale 0.5\n\n');

	f.write('# Cycle through planes from N/2 to 4N/5 up to N/5 and back to N/2.\n')
	if animated == False:
		f.write('# ')
	f.write('volume #0 planes z,' + str(N/2) + ',' + str(4*N/5) + ',0.25\n');
	if animated == False:
		f.write('# ')
	f.write('wait ' + str(4*int(4*N/5 - N/2)) + '\n');
	if animated == False:
		f.write('# ')
	f.write('volume #0 planes z,' + str(4*N/5) + ',' + str(N/5) + ',0.25\n');
	if animated == False:
		f.write('# ')
	f.write('wait ' + str(4*int(4*N/5 - N/5)) + '\n');
	if animated == False:
		f.write('# ')
	f.write('volume #0 planes z,' + str(N/5) + ',' + str(N/2) + ',0.25\n');

	f.close()

	return newFileName

def try_alternatives(cmd, locations, args):
    """
    Try to run a command that might be in any one of multiple locations.

    Takes a single string argument for the command to run, a sequence
    of locations, and a sequence of arguments to the command.  Tries
    to run the command in each location, in order, until the command
    is found (does not raise OSError on the attempt).

    Courtesy of stackoverflew user steveha.

	  LINK: http://stackoverflow.com/a/14106775

    """
    # build a list to pass to subprocess
    lst_cmd = [None]  # dummy arg to reserve position 0 in the list
    lst_cmd.extend(args)  # arguments come after position 0

    for path in locations:
        # It's legal to put a callable in the list of locations.
        # When this happens, we should call it and use its return
        # value for the path.  It should always return a string.
        if callable(path):
            path = path()

        # put full pathname of cmd into position 0 of list
        lst_cmd[0] = os.path.join(path, cmd)
        try:
            return sp.call(lst_cmd)
        except OSError:
            pass
    raise OSError('command "{}" not found in locations list'.format(cmd))
