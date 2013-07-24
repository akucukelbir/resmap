"""
ResMap is a Python (NumPy/SciPy) application with a Tkinter GUI. It is a software package for computing the 
local resolution of 3D density maps studied in structural biology, primarily electron cryo-microscopy (cryo-EM).

Please find the manual at https://sourceforge.net/projects/resmap/

If you use ResMap in your work, please consider citing us:

A. Kucukelbir, F.J. Sigworth, and H.D. Tagare, The Local Resolution of Cryo-EM Density Maps, In Review, 2013.

This package is released under the Creative Commons Attribution-NonCommercial-NoDerivs 
CC BY-NC-ND License (http://creativecommons.org/licenses/by-nc-nd/3.0/)

Usage: 
  ResMap.py [(--nogui INPUT)] [--vxSize=VXSIZE] 
            [--pVal=PVAL] 
            [--minRes=MINRES] [--maxRes=MAXRES] [--stepRes=STEPRES]
            [--variance=VAR]
            [--maskVol=MASKVOL]
            [--vis2D] [--launchChimera]

NOTE: INPUT and --vxSize are mandatory inputs to ResMap 

Arguments:
  INPUT               Input volume in MRC format

Options:
  --nogui             Run ResMap in command-line mode
  --vxSize=VXSIZE     Voxel size of input map (A) [default: 0.0]
  --pVal=PVAL         P-value for likelihood ratio test [default: 0.05]
  --minRes=MINRES     Minimum resolution (A) [default: 0.0]      -> algorithm will start at just above 2*vxSize
  --maxRes=MAXRES     Maximum resolution (A) [default: 0.0]      -> algorithm will stop at around 4*vxSize
  --stepRes=STEPRES   Step size (A) [default: 1.0]               -> min 0.25A 
  --variance=VAR      Noise Variance of input map [default: 0.0] -> [NOT RECOMMENDED]
  --maskVol=MASKVOL   Mask volume                                -> ResMap will automatically compute a mask
  --vis2D             Output 2D visualization
  --launchChimera     Attempt to launch Chimera after execution
  -h --help           Show this help message and exit
  --version           Show version. 

"""

import Tkinter as tk
from tkFileDialog import askopenfilename
from tkMessageBox import showerror
from tkMessageBox import showinfo

import os
from docopt import docopt
from sys import exit

from ResMap_fileIO import *
from ResMap_algorithm import ResMap_algorithm

from scipy import ndimage

class ResMapApp(object):

	"""GUI Tkinter class for ResMap"""

	def __init__(self, parent):

		self.parent = parent
		self.parent.title("Local Resolution Map (ResMap) v" + version)

		# Create frame widget that holds everything 
		self.mainframe = tk.Frame(parent)
		self.mainframe.grid(column=0, row=0, sticky=(tk.N, tk.W, tk.E, tk.S))
		self.mainframe.columnconfigure(0, weight=1)
		self.mainframe.rowconfigure(   0, weight=1)

		# Create menubar
		self.menubar = tk.Menu(parent)
		self.parent.config(menu = self.menubar)

		# Create help menu
		self.helpMenu = tk.Menu(self.menubar)
		self.helpMenu.add_command(label="Documentation", command=self.showDocumentation)
		self.helpMenu.add_command(label="About ResMap",  command=self.showAbout)
		self.menubar.add_cascade(label="Help", menu=self.helpMenu)

		# Create Tk variables
		self.graphicalOutput = tk.IntVar()
		self.chimeraLaunch   = tk.IntVar()
		self.volFileName     = tk.StringVar()
		self.voxelSize       = tk.StringVar()
		self.alphaValue      = tk.StringVar(value="0.05")
		self.minRes          = tk.StringVar(value="0.0")
		self.maxRes          = tk.StringVar(value="0.0")
		self.stepRes         = tk.StringVar(value="1.0")
		self.variance        = tk.StringVar(value="0.0")
		self.maskFileName    = tk.StringVar(value="None; ResMap will automatically compute a mask. Load File to override.")

		# ROW 0
		tk.Label(self.mainframe, text="Required Inputs", font = "Helvetica 12 bold").grid(column=1, row=0, columnspan=10, sticky=tk.W)

		# ROW 1
		tk.Label(self.mainframe, text="Volume:").grid(column=1, row=1, sticky=tk.E)

		volFileName_entry = tk.Entry(self.mainframe, width=100, textvariable=self.volFileName)
		volFileName_entry.grid(column=2, columnspan=10, row=1, sticky=(tk.W, tk.E))

		tk.Button(self.mainframe, text="Load File", command=(lambda: self.load_file(self.volFileName))).grid(column=12, row=1, sticky=tk.W)

		# ROW 2
		tk.Label(self.mainframe, text="Voxel Size:").grid(column=1, row=2, sticky=tk.E)

		voxelSize_entry = tk.Entry(self.mainframe, width=5, textvariable=self.voxelSize)
		voxelSize_entry.grid(column=2, row=2, sticky=tk.W)

		tk.Label(self.mainframe, text="in Angstroms (A/voxel)").grid(column=3, row=2, sticky=tk.W)

		# ROW 3
		tk.Label(self.mainframe, text="Optional Inputs", font = "Helvetica 12 bold").grid(column=1, row=3, columnspan=8, sticky=tk.W)

		# ROW 4
		tk.Label(self.mainframe, text="Confidence Level:").grid(column=1, row=4, sticky=tk.E)

		alphaValue_entry = tk.Entry(self.mainframe, width=5, textvariable=self.alphaValue)
		alphaValue_entry.grid(column=2, row=4, sticky=tk.W)

		tk.Label(self.mainframe, text="usually between [0.01, 0.05]").grid(column=3, row=4, sticky=tk.W)

		# ROW 5
		tk.Label(self.mainframe, text="Min Resolution:").grid(column=1, row=5, sticky=tk.E)

		minRes_entry = tk.Entry(self.mainframe, width=5, textvariable=self.minRes)
		minRes_entry.grid(column=2, row=5, sticky=tk.W)

		tk.Label(self.mainframe, text="in Angstroms (default: algorithm will start at just above 2*voxelSize)").grid(column=3, row=5, sticky=tk.W)

		# ROW 6
		tk.Label(self.mainframe, text="Max Resolution:").grid(column=1, row=6, sticky=tk.E)

		maxRes_entry = tk.Entry(self.mainframe, width=5, textvariable=self.maxRes)
		maxRes_entry.grid(column=2, row=6, sticky=tk.W)

		tk.Label(self.mainframe, text="in Angstroms (default: algorithm will stop at around 4*voxelSize)").grid(column=3, row=6, sticky=tk.W)

		# ROW 7
		tk.Label(self.mainframe, text="Step Size:").grid(column=1, row=7, sticky=tk.E)

		stepRes_entry = tk.Entry(self.mainframe, width=6, textvariable=self.stepRes)
		stepRes_entry.grid(column=2, row=7, sticky=tk.W)

		tk.Label(self.mainframe, text="in Angstroms (min: 0.25, default: 1.0)").grid(column=3, row=7, sticky=tk.W)

		# ROW 8
		tk.Label(self.mainframe, text="Noise Variance:").grid(column=1, row=8, sticky=tk.E)

		variance_entry = tk.Entry(self.mainframe, width=6, textvariable=self.variance)
		variance_entry.grid(column=2, row=8, sticky=tk.W)

		tk.Label(self.mainframe, text="noise variance of input map [NOT RECOMMENDED: ResMap will assume the input map has already been pre-whitened]\n(default: ResMap will pre-whiten and automatically compute the variance from the background)").grid(column=3, row=8, sticky=tk.W)		

		# ROW 9
		tk.Label(self.mainframe, text="Mask Volume:").grid(column=1, row=9, sticky=tk.E)

		maskFileName_entry = tk.Entry(self.mainframe, width=100, textvariable=self.maskFileName, fg="gray")
		maskFileName_entry.grid(column=2, columnspan=10, row=9, sticky=(tk.W, tk.E))

		tk.Button(self.mainframe, text="Load File", command=(lambda: self.load_file(self.maskFileName))).grid(column=12, row=9, sticky=tk.W)

		# ROW 10
		tk.Label(self.mainframe, text="Visualization Options:", font = "Helvetica 10 bold").grid(column=1, row=10, sticky=tk.E)
		tk.Checkbutton(self.mainframe, text="2D Graphical Result Visualization (ResMap)", variable=self.graphicalOutput).grid(column=2, row=10, columnspan=4, sticky=tk.W)

		# ROW 11
		tk.Checkbutton(self.mainframe, text="3D Graphical Result Visualization (UCSF Chimera)", variable=self.chimeraLaunch).grid(column=2, row=11, columnspan=4, sticky=tk.W)
		tk.Button(self.mainframe, text="Check Inputs and RUN", font = "Helvetica 12 bold",command=self.checkInputsAndRun).grid(column=9, columnspan=4, row=11, sticky=tk.E)

		self.parent.bind("<Return>",self.checkInputsAndRun)

		# Setup grid with padding
		for child in self.mainframe.winfo_children(): child.grid_configure(padx=5, pady=10)
		volFileName_entry.focus()		

	def load_file(self, fileNameStringVar):
		options =  {}
		# options['filetypes'] = [ ("All files", ".*"), ("MRC map", ".map,.mrc") ]		# THIS SEEMS BUGGY...
		options['title'] = "ResMap - Select data file"
		fname = askopenfilename(**options)
		if fname:
			try:
				fileNameStringVar.set(fname)
			except:                     # <- naked except is a bad idea
				showerror("Open Source File", "Failed to read file\n'%s'" % fname)
			return 

	def checkInputsAndRun(self,*args):

		# Check volume file name and try loading MRC file
		if self.volFileName.get() == "":
			showerror("Check Inputs", "'volFileName' is not set. Please select a MRC volume to analyze.")
			return
		else:
			try:
				inputFileName = self.volFileName.get()
				data = MRC_Data(inputFileName,'ccp4')
			except:
				showerror("Check Inputs", "The MRC volume could not be read.")
				return

		# Check voxel size
		if self.voxelSize.get() == "":
			showerror("Check Inputs", "'voxelSize' is not set. Please input a voxel size in Angstroms.")
			return
		else:
			try:
				vxSize = float(self.voxelSize.get())
			except ValueError:
				showerror("Check Inputs", "'voxelSize' is not a valid number. Please input a valid voxel size in Angstroms.")
				return

			if vxSize <= 0:
				showerror("Check Inputs", "'voxelSize' is not a positive number. Please input a positive voxel size in Angstroms.")
				return

		# Check confidence level
		if self.alphaValue.get() == "":
			showerror("Check Inputs", "'alphaValue' is not set. Please input a valid confidence level.")
			return
		else:
			try:
				pValue = float(self.alphaValue.get())
			except ValueError:
				showerror("Check Inputs", "'alphaValue' is not a valid number. Please input a valid confidence level.")
				return

			if pValue <= 0 or pValue > 0.05:
				showerror("Check Inputs", "'alphaValue' is outside of (0, 0.05]. Please input a valid confidence level.")
				return

		# Check min resolution
		if self.minRes.get() == "":
			showerror("Check Inputs", "'minRes' is not set. Please input a valid minimum resolution in Angstroms.")
			return
		else:
			try:
				minRes = float(self.minRes.get())
			except ValueError:
				showerror("Check Inputs", "'minRes' is not a valid number. Please input a valid minimum resolution in Angstroms.")
				return

			if minRes < 0.0:
				showerror("Check Inputs", "'minRes' is not a positive number. Please input a positive minimum resolution in Angstroms.")
				return

		# Check max resolution
		if self.maxRes.get() == "":
			showerror("Check Inputs", "'maxRes' is not set. Please input a valid maximum resolution in Angstroms.")
			return
		else:
			try:
				maxRes = float(self.maxRes.get())
			except ValueError:
				showerror("Check Inputs", "'maxRes' is not a valid number. Please input a valid maximum resolution in Angstroms.")
				return

			if maxRes < 0.0:
				showerror("Check Inputs", "'maxRes' is not a positive number. Please input a positive maximum resolution in Angstroms.")
				return	

		# Check step size
		if self.stepRes.get() == "":
			showerror("Check Inputs", "'stepRes' is not set. Please input a valid step size in Angstroms.")
			return
		else:
			try:
				stepRes = float(self.stepRes.get())
			except ValueError:
				showerror("Check Inputs", "'stepRes' is not a valid number. Please input a valid step size in Angstroms.")
				return

			if stepRes < 0.25:
				showerror("Check Inputs", "'stepRes' is too small. Please input a step size greater than 0.25 in Angstroms.")
				return	

		# Check noise variance
		if self.variance.get() == "":
			showerror("Check Inputs", "'variance' is not set. Please input a valid noise variance value.")
			return
		else:
			try:
				variance = float(self.variance.get())
			except ValueError:
				showerror("Check Inputs", "'variance' is not a valid number. Please input a valid nois variance value.")
				return

			if variance < 0.0:
				showerror("Check Inputs", "'variance' is too small. Please input a noise variance greater than 0.0.")
				return					

		# Check mask file name and try loading MRC file
		if self.maskFileName.get() == "":
			showerror("Check Inputs", "'maskFileName' is not set. Please type (None;) without the parantheses.")
			return
		elif self.maskFileName.get().split(';',1)[0] == "None":
			dataMask = 0
		else:
			try:
				maskVolFileName = self.maskFileName.get()
				dataMask = MRC_Data(maskVolFileName,'ccp4')
			except:
				showerror("Check Inputs", "The MRC mask file could not be read.")
				return

		showinfo("ResMap","Inputs are all valid!\n\nPress OK to close GUI and RUN.\n\nCheck console for progress.")

		root.destroy()

		# Call ResMap
		ResMap_algorithm(
				inputFileName   = inputFileName,
				data            = data,
				vxSize          = vxSize,
				pValue          = pValue,
				minRes          = minRes,
				maxRes          = maxRes,
				stepRes         = stepRes,
				dataMask        = dataMask,
				variance        = variance,
				graphicalOutput = self.graphicalOutput.get(),
				chimeraLaunch   = self.chimeraLaunch.get(),
			 )

		raw_input("\n== DONE! ==\n\nPress any key or close window to EXIT.\n\n")

		return

	def showAbout(self):
		showinfo("About ResMap",
		("This is ResMap v"+version+".\n\n"
		 "If you use ResMap in your work, please cite the following paper:\n\n" 
		 "A. Kucukelbir, F.J. Sigworth, H.D. Tagare, The Local Resolution of Cryo-EM Density Maps, In Review, 2013.\n\n"
		 "This package is released under the Creative Commons Attribution-NonCommercial-NoDerivs CC BY-NC-ND License (http://creativecommons.org/licenses/by-nc-nd/3.0/)\n\n"
		 "Please send comments, suggestions, and bug reports to alp.kucukelbir@yale.edu or hemant.tagare@yale.edu"))

	def showDocumentation(self):
		showinfo("ResMap Documentation","Please go to http://sf.net/p/resmap to download the manual.")

if __name__ == '__main__':
	
	global version
	version = "1.0.7"

	args = docopt(__doc__, version=version)

	if args['--nogui'] == False:

		# Create root window and initiate GUI interface
		root = tk.Tk()
		resmapapp = ResMapApp(root)
		root.mainloop()

	else:

		# INPUT
		try:
			data = MRC_Data(args['INPUT'],'ccp4')
		except:
			exit("The input volume (MRC/CCP4 format) could not be read.")

		# inputFileName
		inputFileName = os.path.normpath(os.path.join(os.getcwd(),args['INPUT']))

		# --vxSize
		if args['--vxSize'] == '0.0':
			exit("A voxel size (--vxSize) was not defined.")
		else:
			try:
				vxSize = float(args['--vxSize'])
			except:
				exit("The voxel size (--vxSize) is not a valid floating point number.")

			if vxSize <= 0.0:
				exit("The voxel size (--vxSize) is not a positive number.")

		# --pVal
		try:
			pValue = float(args['--pVal'])
		except:
			exit("The confidence level (--pVal) is not a valid floating point number.")

		if pValue <= 0.0 or pValue > 0.05:
			exit("The confidence level (--pVal) is outside of the [0, 0.05] range.")		

		# --minRes
		try:
			minRes = float(args['--minRes'])
		except:
			exit("The minimum resolution (--minRes) is not a valid floating point number.")

		if minRes < 0.0:
			exit("The minimum resolution (--minRes) is not a positive number.")			

		# --maxRes
		try:
			maxRes = float(args['--maxRes'])
		except:
			exit("The maximum resolution (--maxRes) is not a valid floating point number.")

		if maxRes < 0.0:
			exit("The maximum resolution (--maxRes) is not a positive number.")						

		# --stepRes
		try:
			stepRes = float(args['--stepRes'])
		except:
			exit("The step size (--stepRes) is not a valid floating point number.")

		if stepRes < 0.25:
			exit("The step size (--stepRes) is too small.")			

		# --variance
		try:
			variance = float(args['--variance'])
		except:
			exit("The noise variance (--variance) is not a valid floating point number.")

		if variance < 0.0:
			exit("The noise variance (--variance) is a negative number.")		

		# --maskVol
		if args['--maskVol'] == None:
			dataMask = 0
		else:
			try:
				dataMask = MRC_Data(args['--maskVol'],'ccp4')
			except:
				exit("The mask volume (MRC/CCP4 format) could not be read.")

		# Call ResMap
		ResMap_algorithm(
				inputFileName   = inputFileName,
				data            = data,
				vxSize          = vxSize,
				pValue          = pValue,
				minRes          = minRes,
				maxRes          = maxRes,
				stepRes         = stepRes,
				dataMask        = dataMask,
				variance        = variance,
				graphicalOutput = args['--vis2D'],
				chimeraLaunch   = args['--launchChimera']
			 )
