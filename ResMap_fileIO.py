'''
ResMap_fileIO: module containing file input/output functions. All functions courtesy of the Rubinstein Group.
             LINK: https://sites.google.com/site/rubinsteingroup

Citation:
	Zhao J, Brubaker MA, Rubinstein JL (2013). TMaCS: A hybrid template matching and
	classification system for partially-automated particle selection. J Struct Biol 181, 231-42.

Description of classes:
    mrc_image: Read, process, and write MRC volumes.

Requirements:
	struct
    numpy
'''
import struct,numpy

class mrc_image:
	def __init__(self,filename):

		# Reading and writing MRC files

		# Header items
		# 0-11    = nx,ny,nz (i)
		# 12-15   = mode (i)
		# 16-27   = nxstart,nystart,nzstart (i)
		# 28-39   = mx,my,mz (i)
		# 40-51   = cell size (x,y,z) (f)
		# 52-63   = cell angles (alpha,beta,gamma) (f)
		# 64-75   = mapc,mapr,maps (1,2,3) (i)
		# 76-87   = amin,amax,amean (f)
		# 88-95   = ispg,nsymbt (i)
		# 96-207  = 0 (i)
		# 208-215 = cmap,stamp (c)
		# 216-219 = rms (f)
		# 220-223 = nlabels (i)
		# 224-1023 = 10 lines of 80 char titles (c)

		self.filename=filename
		self.byte_pattern1='=' + 'i'*10 + 'f'*6 + 'i'*3 + 'f'*3 + 'i'*30 + '4s'*2 + 'fi'
		self.byte_pattern2='=' + '800s'

	def change_filename(self,filename):
		self.filename=filename

	def read_head(self):
		self.input_image=open(self.filename,'rb')
		self.header1=self.input_image.read(224)
		self.header2=self.input_image.read(800)
		self.dim=struct.unpack(self.byte_pattern1,self.header1)[:3]   #(dimx,dimy,dimz)
		self.input_image.close()

	def read(self,asBool):
		self.input_image=open(self.filename,'rb')
		self.header1=self.input_image.read(224)
		self.header2=self.input_image.read(800)
		self.header = struct.unpack(self.byte_pattern1,self.header1)
		self.dim=self.header[:3]   #(dimx,dimy,dimz)
		self.imagetype=self.header[3]  
		#0: 8-bit signed, 1:16-bit signed, 2: 32-bit float, 6: unsigned 16-bit (non-std)
		if (self.imagetype == 0):
			self.imtype=numpy.uint8
		elif (self.imagetype ==1):
			self.imtype='h'
		elif (self.imagetype ==2):
			self.imtype='f4'
		elif (self.imagetype ==6):
			self.imtype='H'
		else:
			self.imtype='unknown'   #should put a fail here
		self.input_image_dimension=(self.dim[2],self.dim[1],self.dim[0]) 
		self.image_data=numpy.fromfile(file=self.input_image,dtype=self.imtype,
									   count=self.dim[0]*self.dim[1]*self.dim[2]).reshape(self.input_image_dimension)
		if asBool:
			self.image_data=self.image_data.astype(numpy.bool)
		else:	
			self.image_data=self.image_data.astype(numpy.float32)
		self.input_image.close()
	
	def write(self,image,output=numpy.ones(1)):
		output_image=open(self.filename,'w')

		if output.shape[0] == 1:
			dum = image.shape # Get dimensions of image/stk
			dim = [dum[1],dum[2],1]
			if len(dum) == 3:
				dim[2] = dum[0]
			amin  = numpy.min(image)
			amax  = numpy.max(image)
			amean = numpy.mean(image)
			# header1 = struct.pack(self.byte_pattern1,dim[0],dim[1],dim[2],2,dim[0],dim[1],dim[2],dim[0],dim[1],dim[2],
			# 					  dim[0],dim[1],dim[2],90,90,90,1,2,3,amin,amax,amean,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
			# 					  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,'MAP','DA\x00\x00',0.0,1)
			# header2 = struct.pack(self.byte_pattern2,' ')
			output_image.write(self.header1)
			output_image.write(self.header2)
			image.tofile(output_image)
			output_image.close()

		else:
			nparts = output.shape[0]
			loc = numpy.where(output==1)[0]
			ntrue = loc.shape[0]
			if ntrue == 0:
				print 'ERROR: no labeled images'
			else:
				dum = image.shape # Get dimensions of image/stk
				dim = [dum[1],dum[2],ntrue]
				image2 = numpy.zeros((dim[2],dim[0],dim[1]),dtype='f4')
				for i in range(ntrue):
					image2[i,:,:] = image[int(loc[i]),:,:] # Take true particles only
				amin = min(image2.reshape(dim[0]*dim[1]*dim[2]))
				amax = max(image2.reshape(dim[0]*dim[1]*dim[2]))
				amean = sum(sum(sum(image2)))/(dim[0]*dim[1]*dim[2])
				header1 = struct.pack(self.byte_pattern1,dim[0],dim[1],dim[2],2,dim[0],dim[1],dim[2],dim[0],dim[1],dim[2],
									  dim[0],dim[1],dim[2],90,90,90,1,2,3,amin,amax,amean,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
									  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,'MAP','DA\x00\x00',0.0,1)
				header2 = struct.pack(self.byte_pattern2,'File created by TMaCS View')
				output_image.write(header1)
				output_image.write(header2)
				image2.tofile(output_image)
				output_image.close()
