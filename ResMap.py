"""
ResMap is a Python (NumPy/SciPy) application with a Tkinter GUI. It is a software package for computing the 
local resolution of 3D density maps studied in structural biology, primarily electron cryo-microscopy (cryo-EM).

Please find the manual at http://resmap.sourceforge.net

If you use ResMap in your work, please consider citing us:

A. Kucukelbir, F.J. Sigworth, and H.D. Tagare, Quantifying the Local Resolution of Cryo-EM Density Maps, Nature Methods, In Press, 2013.

This package is released under the Creative Commons Attribution-NonCommercial-NoDerivs 
CC BY-NC-ND License (http://creativecommons.org/licenses/by-nc-nd/3.0/)

Usage: 
  ResMap.py [(--noguiSingle INPUT)] [--vxSize=VXSIZE]
            [--pVal=PVAL]
            [--minRes=MINRES] [--maxRes=MAXRES] [--stepRes=STEPRES]
            [--maskVol=MASKVOL]
            [--vis2D] [--launchChimera] [--noiseDiag]
  ResMap.py [(--noguiSplit INPUT1 INPUT2)] [--vxSize=VXSIZE]
            [--pVal=PVAL]
            [--minRes=MINRES] [--maxRes=MAXRES] [--stepRes=STEPRES]
            [--maskVol=MASKVOL]
            [--vis2D] [--launchChimera] [--noiseDiag]

NOTE: INPUT(s) is/are mandatory

Arguments:
  INPUT(s)            Input volume(s) in MRC format

Options:
  --noguiSingle       Run ResMap for Single Volumes in command-line mode
  --noguiSplit        Run ResMap for Split Volumes in command-line mode
  --vxSize=VXSIZE     Voxel size of input map (A) [default: 0.0]
  --pVal=PVAL         P-value for likelihood ratio test [default: 0.05]
  --minRes=MINRES     Minimum resolution (A) [default: 0.0]       -> algorithm will start at just above 2*vxSize
  --maxRes=MAXRES     Maximum resolution (A) [default: 0.0]       -> algorithm will stop at around 4*vxSize
  --stepRes=STEPRES   Step size (A) [default: 1.0]                -> min 0.25A 
  --maskVol=MASKVOL   Mask volume                                 -> ResMap will automatically compute a mask
  --vis2D             Output 2D visualization
  --launchChimera     Attempt to launch Chimera after execution
  --noiseDiag         Run and show noise diagnostics
  -h --help           Show this help message and exit
  --version           Show version. 

"""

import Tkinter as tk
from tkFileDialog import askopenfilename
from tkFileDialog import askopenfilenames
from tkMessageBox import showerror
from tkMessageBox import showinfo

import ttk

import os
from docopt import docopt
from sys import exit

from ResMap_fileIO import *
from ResMap_algorithm import ResMap_algorithm

from scipy import ndimage

class ResMapApp(object):

	"""GUI Tkinter class for ResMap"""

	def __init__(self, parent):

		# General Settings
		self.parent = parent
		self.parent.title("ResMap (Local Resolution) v" + version)
		self.parent.option_add('*tearOff', False)

		self.myStyle = ttk.Style()
		self.myStyle.configure('ResMap.TButton', foreground='red4', font='Helvetica 16 bold')


		## MENUBAR
		# Create menubar frame
		self.menubar = ttk.Frame(parent);
		self.menubar.pack(side = tk.TOP, fill = tk.X)

		# Create Help menubutton"path":
		self.mb_help = ttk.Menubutton(self.menubar, text="Help")
		self.mb_help.pack(side = tk.RIGHT)

		# Create Help menu
		self.helpMenu = tk.Menu(self.mb_help)
		self.helpMenu.add_command(label="Documentation", command=self.showDocumentation)
		self.helpMenu.add_command(label="About ResMap",  command=self.showAbout)

		# Attach the Help menu to the Help menubutton
		self.mb_help.config(menu = self.helpMenu)


		## MASTER FRAME
		# Create master frame that holds everything 
		self.masterframe = ttk.Frame(parent)
		self.masterframe.pack()

		# Create notebook
		self.nb = ttk.Notebook(self.masterframe, style='ResMap.TNotebook')  # create the ttk.Notebook widget
		self.nb.enable_traversal() # allow for keyboard bindings
		self.nb.pack()


		## GLOBAL VARIABLES FOR BOTH NOTEBOOK FRAMES
		# Create Tk variables
		self.graphicalOutput  = tk.IntVar(value=1)
		self.chimeraLaunch    = tk.IntVar()
		self.noiseDiagnostics = tk.IntVar()
		self.volFileName      = tk.StringVar()
		self.volFileName1     = tk.StringVar()
		self.volFileName2     = tk.StringVar()				
		self.voxelSize        = tk.StringVar(value="Auto")
		self.alphaValue       = tk.StringVar(value="0.05")
		self.minRes           = tk.StringVar(value="0.0")
		self.maxRes           = tk.StringVar(value="0.0")
		self.stepRes          = tk.StringVar(value="1.0")
		self.maskFileName     = tk.StringVar(value="None; ResMap will automatically compute a mask. Load File to override.")

		## SINGLE VOLUME FRAME
		# Create single volume input frame 
		self.mainframe = ttk.Frame(self.nb)

		self.mainframe.grid(column=0, row=0, sticky=(tk.N, tk.W, tk.E, tk.S))
		self.mainframe.columnconfigure(0, weight=1)
		self.mainframe.rowconfigure(   0, weight=1)

		self.nb.add(self.mainframe, text='Single Volume Input', underline=0, padding=10)

		# ROW 0
		ttk.Label(self.mainframe, text="Required Inputs", font = "Helvetica 14 bold").grid(column=1, row=0, columnspan=10, sticky=tk.W)

		# ROW 1
		ttk.Label(self.mainframe, text="Volume:").grid(column=1, row=1, sticky=tk.E)
		ttk.Entry(self.mainframe, width=100, textvariable=self.volFileName).grid(column=2, columnspan=10, row=1, sticky=(tk.W, tk.E))
		ttk.Button(self.mainframe, text="Load", command=(lambda: self.load_file(self.volFileName))).grid(column=12, row=1, sticky=tk.W)

		# ROW 2
		ttk.Label(self.mainframe, text="Voxel Size:").grid(column=1, row=2, sticky=tk.E)
		ttk.Entry(self.mainframe, width=5, textvariable=self.voxelSize, foreground="gray").grid(column=2, row=2, sticky=tk.W)		
		ttk.Label(self.mainframe, text="Angstrom/voxel (default: algorithm will read value from MRC header)").grid(column=3, row=2, sticky=tk.W)

		# ROW 3
		ttk.Label(self.mainframe, text="Optional Inputs", font = "Helvetica 14 bold").grid(column=1, row=3, columnspan=8, sticky=tk.W)

		# ROW 4
		ttk.Label(self.mainframe, text="Step Size:").grid(column=1, row=4, sticky=tk.E)
		ttk.Entry(self.mainframe, width=5, textvariable=self.stepRes).grid(column=2, row=4, sticky=tk.W)
		ttk.Label(self.mainframe, text="in Angstroms (min: 0.25, default: 1.0)").grid(column=3, row=4, sticky=tk.W)

		# ROW 5
		ttk.Label(self.mainframe, text="Min Resolution:").grid(column=1, row=5, sticky=tk.E)
		ttk.Entry(self.mainframe, width=5, textvariable=self.minRes).grid(column=2, row=5, sticky=tk.W)		
		ttk.Label(self.mainframe, text="in Angstroms (default: algorithm will start at just above 2*voxelSize)").grid(column=3, row=5, sticky=tk.W)

		# ROW 6
		ttk.Label(self.mainframe, text="Max Resolution:").grid(column=1, row=6, sticky=tk.E)
		ttk.Entry(self.mainframe, width=5, textvariable=self.maxRes).grid(column=2, row=6, sticky=tk.W)
		ttk.Label(self.mainframe, text="in Angstroms (default: algorithm will stop at around 4*voxelSize)").grid(column=3, row=6, sticky=tk.W)

		# ROW 7
		ttk.Label(self.mainframe, text="Confidence Level:").grid(column=1, row=7, sticky=tk.E)
		ttk.Entry(self.mainframe, width=5, textvariable=self.alphaValue).grid(column=2, row=7, sticky=tk.W)
		ttk.Label(self.mainframe, text="usually between [0.01, 0.05]").grid(column=3, row=7, sticky=tk.W)		

		# ROW 8
		ttk.Label(self.mainframe, text="Mask Volume:").grid(column=1, row=8, sticky=tk.E)
		ttk.Entry(self.mainframe, width=100, textvariable=self.maskFileName, foreground="gray").grid(column=2, columnspan=10, row=8, sticky=(tk.W, tk.E))
		ttk.Button(self.mainframe, text="Load", command=(lambda: self.load_file(self.maskFileName))).grid(column=12, row=8, sticky=tk.W)

		# ROW 9
		ttk.Label(self.mainframe, text="Visualization Options:").grid(column=1, row=9, sticky=tk.E)
		ttk.Checkbutton(self.mainframe, text="2D Result Visualization (ResMap)", variable=self.graphicalOutput).grid(column=2, row=9, columnspan=2, sticky=tk.W)

		# ROW 10
		ttk.Checkbutton(self.mainframe, text="3D Result Visualization (UCSF Chimera)", variable=self.chimeraLaunch).grid(column=2, row=10, columnspan=2 , sticky=tk.W)

		# # ROW 12
		# ttk.Label(self.mainframe, text="Diagnostics Options:").grid(column=1, row=12, sticky=tk.E)
		# ttk.Checkbutton(self.mainframe, text="Display Noise Diagnostics (ResMap)", variable=self.noiseDiagnostics).grid(column=2, row=12, columnspan=2, sticky=tk.W)

		# ROW 13
		ttk.Button(self.mainframe, text="Check Inputs and RUN", style='ResMap.TButton', command=self.checkInputsAndRun).grid(column=9, columnspan=4, row=13, sticky=tk.E)

		# Setup grid with padding
		for child in self.mainframe.winfo_children(): child.grid_configure(padx=5, pady=5)

		self.parent.bind("<Return>",self.checkInputsAndRun)

		## SPLIT VOLUME FRAME
		# Create split volume input frame 
		self.splitframe = ttk.Frame(self.nb)

		self.splitframe.grid(column=0, row=0, sticky=(tk.N, tk.W, tk.E, tk.S))
		self.splitframe.columnconfigure(0, weight=1)
		self.splitframe.rowconfigure(   0, weight=1)

		self.nb.add(self.splitframe, text='Split Volume Input', underline=0, padding=10)

		# ROW 0
		ttk.Label(self.splitframe, text="Required Inputs", font = "Helvetica 14 bold").grid(column=1, row=0, columnspan=10, sticky=tk.W)

		# ROW 1
		ttk.Label(self.splitframe, text="Split Volume 1:").grid(column=1, row=1, sticky=tk.E)
		ttk.Entry(self.splitframe, width=100, textvariable=self.volFileName1).grid(column=2, columnspan=10, row=1, sticky=(tk.W, tk.E))
		
		# ROW 2
		ttk.Label(self.splitframe, text="Split Volume 2:").grid(column=1, row=2, sticky=tk.E)
		ttk.Entry(self.splitframe, width=100, textvariable=self.volFileName2).grid(column=2, columnspan=10, row=2, sticky=(tk.W, tk.E))
		ttk.Button(self.splitframe, text="Load Both", command=(lambda: self.load_files(self.volFileName1, self.volFileName2))).grid(column=12, row=2, sticky=tk.W)

		# ROW 3
		ttk.Label(self.splitframe, text="Voxel Size:").grid(column=1, row=3, sticky=tk.E)
		ttk.Entry(self.splitframe, width=5, textvariable=self.voxelSize, foreground="gray").grid(column=2, row=3, sticky=tk.W)
		ttk.Label(self.splitframe, text="Angstrom/voxel (default: algorithm will read value from MRC header)").grid(column=3, row=3, sticky=tk.W)

		# ROW 4
		ttk.Label(self.splitframe, text="Optional Inputs", font = "Helvetica 14 bold").grid(column=1, row=4, columnspan=8, sticky=tk.W)

		# ROW 5
		ttk.Label(self.splitframe, text="Step Size:").grid(column=1, row=5, sticky=tk.E)
		ttk.Entry(self.splitframe, width=5, textvariable=self.stepRes).grid(column=2, row=5, sticky=tk.W)
		ttk.Label(self.splitframe, text="in Angstroms (min: 0.25, default: 1.0)").grid(column=3, row=5, sticky=tk.W)

		# ROW 6
		ttk.Label(self.splitframe, text="Min Resolution:").grid(column=1, row=6, sticky=tk.E)
		ttk.Entry(self.splitframe, width=5, textvariable=self.minRes).grid(column=2, row=6, sticky=tk.W)		
		ttk.Label(self.splitframe, text="in Angstroms (default: algorithm will start at just above 2*voxelSize)").grid(column=3, row=6, sticky=tk.W)

		# ROW 7
		ttk.Label(self.splitframe, text="Max Resolution:").grid(column=1, row=7, sticky=tk.E)
		ttk.Entry(self.splitframe, width=5, textvariable=self.maxRes).grid(column=2, row=7, sticky=tk.W)
		ttk.Label(self.splitframe, text="in Angstroms (default: algorithm will stop at around 4*voxelSize)").grid(column=3, row=7, sticky=tk.W)

		# ROW 8
		ttk.Label(self.splitframe, text="Confidence Level:").grid(column=1, row=8, sticky=tk.E)
		ttk.Entry(self.splitframe, width=5, textvariable=self.alphaValue).grid(column=2, row=8, sticky=tk.W)
		ttk.Label(self.splitframe, text="usually between [0.01, 0.05]").grid(column=3, row=8, sticky=tk.W)

		# ROW 9
		ttk.Label(self.splitframe, text="Mask Volume:").grid(column=1, row=9, sticky=tk.E)
		ttk.Entry(self.splitframe, width=100, textvariable=self.maskFileName, foreground="gray").grid(column=2, columnspan=10, row=9, sticky=(tk.W, tk.E))
		ttk.Button(self.splitframe, text="Load", command=(lambda: self.load_file(self.maskFileName))).grid(column=12, row=9, sticky=tk.W)

		# ROW 10
		ttk.Label(self.splitframe, text="Visualization Options:").grid(column=1, row=10, sticky=tk.E)
		ttk.Checkbutton(self.splitframe, text="2D Result Visualization (ResMap)", variable=self.graphicalOutput).grid(column=2, row=10, columnspan=2, sticky=tk.W)

		# ROW 11
		ttk.Checkbutton(self.splitframe, text="3D Result Visualization (UCSF Chimera)", variable=self.chimeraLaunch).grid(column=2, row=11, columnspan=2, sticky=tk.W)

		# # ROW 12
		# ttk.Label(self.splitframe, text="Diagnostics Options:").grid(column=1, row=12, sticky=tk.E)
		# ttk.Checkbutton(self.splitframe, text="Display Noise Diagnostics (ResMap)", variable=self.noiseDiagnostics).grid(column=2, row=12, columnspan=2, sticky=tk.W)

		# ROW 13
		ttk.Button(self.splitframe, text="Check Inputs and RUN", style='ResMap.TButton', command=self.checkInputsAndRun).grid(column=9, columnspan=4, row=13, sticky=tk.E)

		# Setup grid with padding
		for child in self.splitframe.winfo_children(): child.grid_configure(padx=5, pady=5)	

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

	def load_files(self, fileNameStringVar1, fileNameStringVar2):
		options =  {}
		options['title'] = "ResMap - Select data file"
		fname = askopenfilenames(**options)
		if isinstance( fname, tuple ):
			try:
				fileNameStringVar1.set(fname[0])
				fileNameStringVar2.set(fname[1])
			except:                     # <- naked except is a bad idea
				showerror("Open Source Files", "Failed to read files\n'%s'" % fname)
			return 
		if isinstance( fname, unicode ):
			try:
				fileNameStringVar1.set(fname.partition(' ')[0])
				fileNameStringVar2.set(fname.partition(' ')[2])
			except:                     # <- naked except is a bad idea
				showerror("Open Source Files", "Failed to read files\n'%s'" % fname)
			return 

	def checkInputsAndRun(self,*args):

		# Find which tab the button was pressed from
		if self.nb.index(self.nb.select()) == 0:
			singleVolumeTab = True
		else:
			singleVolumeTab = False

		if singleVolumeTab:
			# Check single volume file name and try loading MRC file
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
		else:
			# Check split volume file names and try loading MRC files
			if self.volFileName1.get() == "":
				showerror("Check Inputs", "'volFileName1' is not set. Please select a MRC volume to analyze.")
				return
			else:
				try:
					inputFileName1 = self.volFileName1.get()
					data1 = MRC_Data(inputFileName1,'ccp4')
				except:
					showerror("Check Inputs", "The MRC volume could not be read.")
					return

			if self.volFileName2.get() == "":
				showerror("Check Inputs", "'volFileName2' is not set. Please select a MRC volume to analyze.")
				return
			else:
				try:
					inputFileName2 = self.volFileName2.get()
					data2 = MRC_Data(inputFileName2,'ccp4')
				except:
					showerror("Check Inputs", "The MRC volume could not be read.")
					return					

		# Check voxel size
		if self.voxelSize.get() == "":
			showerror("Check Inputs", "'voxelSize' is not set. Please input a voxel size in Angstroms.")
			return
		elif self.voxelSize.get().split(';',1)[0] == "Auto":
			try:
				if singleVolumeTab:
					vxSize = float(data.data_step[0])
				else:
					vxSize = float(data1.data_step[0])
			except ValueError:
				showerror("Check Inputs", "'voxelSize' cannot be read from the MRC file. Please input a valid voxel size in Angstroms.")
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
		if singleVolumeTab:
			ResMap_algorithm(
					inputFileName    = inputFileName,
					data             = data,
					vxSize           = vxSize,
					pValue           = pValue,
					minRes           = minRes,
					maxRes           = maxRes,
					stepRes          = stepRes,
					dataMask         = dataMask,
					graphicalOutput  = self.graphicalOutput.get(),
					chimeraLaunch    = self.chimeraLaunch.get(),
					noiseDiagnostics = self.noiseDiagnostics.get(),
				 )
		else:
			ResMap_algorithm(
					inputFileName1   = inputFileName1,
					inputFileName2   = inputFileName2,
					data1            = data1,
					data2            = data2,
					vxSize           = vxSize,
					pValue           = pValue,
					minRes           = minRes,
					maxRes           = maxRes,
					stepRes          = stepRes,
					dataMask         = dataMask,
					graphicalOutput  = self.graphicalOutput.get(),
					chimeraLaunch    = self.chimeraLaunch.get(),
					noiseDiagnostics = self.noiseDiagnostics.get(),
				 )

		raw_input("\n== DONE! ==\n\nPress any key or close window to EXIT.\n\n")

		return

	def showAbout(self):
		showinfo("About ResMap",
		("This is ResMap v"+version+".\n\n"
		 "If you use ResMap in your work, please cite the following paper:\n\n" 
		 "A. Kucukelbir, F.J. Sigworth, H.D. Tagare, Quantifying the Local Resolution of Cryo-EM Density Maps, Nature Methods, In Press, 2013.\n\n"
		 "This package is released under the Creative Commons Attribution-NonCommercial-NoDerivs CC BY-NC-ND License (http://creativecommons.org/licenses/by-nc-nd/3.0/)\n\n"
		 "Please send comments, suggestions, and bug reports to alp.kucukelbir@yale.edu or hemant.tagare@yale.edu"))

	def showDocumentation(self):
		showinfo("ResMap Documentation","Please go to http://resmap.sourceforge.net to download the manual.")

if __name__ == '__main__':
	
	global version
	version = "1.1.3"

	args = docopt(__doc__, version=version)

	if args['--noguiSingle'] == False and args['--noguiSplit'] == False:

		# Create root window and initiate GUI interface
		root = tk.Tk()
		resmapapp = ResMapApp(root)
		root.mainloop()

	else:

		if args['--noguiSingle'] == True:
			# INPUT
			try:
				data = MRC_Data(args['INPUT'],'ccp4')
				vxSize = float(data.data_step[0])
			except:
				exit("The input volume (MRC/CCP4 format) could not be read.")

			# inputFileName
			inputFileName = os.path.normpath(os.path.join(os.getcwd(),args['INPUT']))
		elif args['--noguiSplit'] == True:
			# INPUT
			try:
				data1 = MRC_Data(args['INPUT1'],'ccp4')
				data2 = MRC_Data(args['INPUT2'],'ccp4')
				vxSize = float(data1.data_step[0])
			except:
				exit("The input volumes (MRC/CCP4 format) could not be read.")

			# inputFileName
			inputFileName1 = os.path.normpath(os.path.join(os.getcwd(),args['INPUT1']))			
			inputFileName2 = os.path.normpath(os.path.join(os.getcwd(),args['INPUT2']))	

		# --vxSize
		if args['--vxSize'] != 0.0:	# Only grab the vxSize if it is inputted
			try:
				vxSize = float(args['--vxSize'])
			except:
				exit("The voxel size (--vxSize) is not a valid floating point number.")

		if vxSize <= 0.0:
			exit("The voxel size is not a positive number.")

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

		# --maskVol
		if args['--maskVol'] == None:
			dataMask = 0
		else:
			try:
				dataMask = MRC_Data(args['--maskVol'],'ccp4')
			except:
				exit("The mask volume (MRC/CCP4 format) could not be read.")

		# Call ResMap
		if args['--noguiSingle'] == True:
			ResMap_algorithm(
					inputFileName    = inputFileName,
					data             = data,
					vxSize           = vxSize,
					pValue           = pValue,
					minRes           = minRes,
					maxRes           = maxRes,
					stepRes          = stepRes,
					dataMask         = dataMask,
					graphicalOutput  = args['--vis2D'],
					chimeraLaunch    = args['--launchChimera'],
					noiseDiagnostics = args['--noiseDiag']
				 )
		elif args['--noguiSplit'] == True:
			ResMap_algorithm(
					inputFileName1   = inputFileName1,
					inputFileName2   = inputFileName2,
					data1            = data1,
					data2            = data2,
					vxSize           = vxSize,
					pValue           = pValue,
					minRes           = minRes,
					maxRes           = maxRes,
					stepRes          = stepRes,
					dataMask         = dataMask,
					graphicalOutput  = args['--vis2D'],
					chimeraLaunch    = args['--launchChimera'],
					noiseDiagnostics = args['--noiseDiag']
				 )
		else:
			print "\n\nSomething somewhere went terribly wrong.\n\n"