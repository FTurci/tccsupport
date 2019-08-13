import pathlib
import os
import sys
import subprocess
import numpy as np
import shutil
import tqdm

def autocorrelation(y):
	yunbiased = y-np.mean(y)
	ynorm = np.sum(yunbiased**2)
	acor = np.correlate(yunbiased, yunbiased, "same")/ynorm
	# use only second half
	acor = acor[int(len(acor)/2):]
	return acor

class TccSupporter:
	def __init__(self, filepath, foldername=None,tcc_binary=None):
		
		if tcc_binary==None:
			self.tcc = "tcc" 
		else :
			self.tcc = str(pathlib.Path(tcc_binary).resolve())
			if not os.path.exists(self.tcc):
				print(":!: No TCC binary at",self.tcc)
				sys.exit(1)
			else:
				print (":: TCC binary set to",self.tcc)


		self.path = pathlib.Path(filepath).resolve()
		self.filepath=str(self.path)
		self.filename=self.path.parts[-1]
		self.parent = str(self.path.parent)

		if foldername==None:
			foldername = "/"+self.path.parts[-1]+".tcc_results"

		
		self.directory = self.parent+foldername
		print (":: Results are stored in", self.directory)

		if not os.path.exists(self.directory):		
			os.makedirs(self.directory)
		else:
			print(":!: Folder", self.directory,"already exists.")
			test = "something"
			while test!="yes" and test!="no":
				test=input(":!: Continue anyway?")
				print("   Answer \'yes\' or \'no\'.")
			if test=="yes":
				pass
			else:
				print(":!: Quitting program.")
				sys.exit(0)
	
		# move the xyz to the results directory
		shutil.copy(self.filepath, self.directory)

		self.structs="sp3a","sp3b","sp3c","sp4a","sp4b","sp4c","sp5a","sp5b","sp5c","6A","6Z","7K","7T_a","7T_s","8A","8B","8K","9A","9B","9K","10A","10B","10K","10W","11A","11B","11C","11E","11F","11W","12A","12B","12D","12E","12K","13A","13B","13K","FCC","HCP","BCC_9","BCC_15"

		self.flags = {}
		for key in self.structs:
			self.flags[key] = 1

	def write_box(self, boxes):
		boxes = np.array(boxes)
		if boxes.ndim==1:
			boxes = [boxes]
		else:
			pass

		with open(self.directory+"/box.txt",'w') as fout:
			fout.write("#iter Lx Ly Lz\n")
			for i,b in enumerate(boxes):
				fout.write("%d %g %g %g\n"%(i,b[0],b[1],b[2]))

	def write_clusters_to_analyse(self):
		with open(self.directory+"/clusters_to_analyse.ini",'w') as fout:
			fout.write("""# Select clusters to analyse. All clusters are analysed by default unless "analyse_all_clusters" in "input_parameters.ini" is set to 0.
# Cluster dependecies will be considered. If a large cluster is selected for analysis, all required smaller clusters will be analysed regardless of their setting here
""")			
			fout.write("[Clusters]\n")
			for key in self.structs:
				fout.write("%s = %d\n"%(key,self.flags[key]))


	def write_inputparameters(self,frames,boxtype,rcutaa,rcutbb,rcutab,bondtype, pbc,fc,nbonds,output_bonds,output_clusts,output_raw,output_xyz, output_11a,output_13a, output_pop):
		inputs = f"""[Box]	
; Specifies how to read simulation box
box_type			= {boxtype}				; 1 if NVT, 2 if NPT, 3 if triclinc with tilt
box_name			= box.txt		; name of parameters file for box size

[Run]	
; Run specific settings - these depend on your xyz file
xyzfilename			= {self.filename}	; File name of the xyz file to be analysed.
frames				= {frames}			; Frames to process

[Simulation]	
; Simulation specific settings - these depend on the type of system you are analysing
rcutAA				= {rcutaa}	; maximum A-A bond lengths  // The cutoff is always applied whether Voronoi bonds are used or not
rcutAB				= {rcutab}	; maximum A-B bond lengths
rcutBB				= {rcutbb}	; maximum B-B bond lengths
min_cutAA           = 0.0   ; minimum A-A bond length. Good for excluding overlapping particles in ideal gases.
bond_type			= {bondtype}		; 0 simple bond length, 1 Voronoi bond detection
PBCs				= {pbc}     ; 0 Dont use periodic boundary conditions, 1 Use PBCs,
voronoi_parameter	= {fc}   ; Modified Voronoi Fc parameter
num_bonds			= {nbonds}	; max number of bonds to one particle
cell_list			= 0		; use Cell List to calculate bond network
analyse_all_clusters = 0    ; If set to zero, read clusters to analyse from clusters_to_analyse.ini

[Output]		
; Determines what the TCC will output
bonds 				= {output_bonds}		; write out bonds file
clusts 				= {output_clusts}		; write clusts_** files containing all clusters - USES LOTS OF HDD SPACE
raw 				= {output_raw}		; write raw_** xmol cluster files
do_XYZ              = {output_xyz}     ; write clusters to xyz files
11a 				= {output_11a}		; write centres of 11A
13a 				= {output_13a}		; write centres of 13A
pop_per_frame 		= {output_pop}		; write particle fraction of each cluster per frame
"""
		with open(self.directory+"/inputparameters.ini",'w') as fout:
			fout.write(inputs)


	def run(self,frames,box,boxtype=1,rcutaa=5.,rcutbb=5.,rcutab=5.,bondtype=1, pbc=1,fc=0.9,nbonds=50,output_bonds=0,output_clusts=0,output_raw=0,output_xyz=0, output_11a=0,output_13a=0, output_pop=1):
		
		cwd = os.getcwd()
		print(":: Current working directory",cwd)
		os.chdir(self.directory)
		self.write_box(box)
		self.write_clusters_to_analyse()
		self.write_inputparameters(frames,boxtype,rcutaa,rcutbb,rcutab,bondtype, pbc,fc,nbonds,output_bonds,output_clusts,output_raw,output_xyz, output_11a,output_13a, output_pop)
		subprocess.run(self.tcc)
		os.chdir(cwd)


class ClusterReader:
	def __init__(self,filename):
		self.filename = filename
		self.valid=True

	def read(self):
		print(":: Reading",self.filename)
		self.frames=[]
		self.unique_clusters =set()
		self.presence={}
		self.signals={}
		self.autocors={}

		frameid=-1
		with open(self.filename, 'r') as f:
			while True:
				line = f.readline()
				
				if len(line) == 0:
					# print("end file")
				# 	# End of file
					break

				if line.split()[0]=="Frame":
					# print (line)
					frame = []
					self.frames.append(frame)
					frameid+=1
					continue
				cluster = tuple(np.sort(np.array(line.split(),dtype=int)).tolist())
				self.frames[-1].append(cluster)
				self.unique_clusters.add(cluster)
				try:
					self.presence[cluster].append(frameid)
				except Exception:
					self.presence[cluster]=[frameid]

		if len(self.unique_clusters)<1:
			print(":!: No clusters present")
			self.valid=False
			return
		# construct signals

		for cluster in self.unique_clusters:
			signal = np.zeros(len(self.frames))

			signal[np.array(self.presence[cluster])]=1
			self.signals[cluster]=signal
			self.autocors[cluster]=autocorrelation(signal)

		self.len_autocor=len(self.autocors[list(self.unique_clusters)[0]])

	def plot_signals(self):
		if self.valid:
			import matplotlib.pyplot as pl
			for key,value in self.autocors.items():
				pl.plot(value)


	def average_autocor(self):
		if self.valid:
			avg = np.zeros(self.len_autocor)
			for value in self.autocors.values():
				avg+=value
			avg/=len(self.autocors)
			self.avg_autocor=avg
			return avg