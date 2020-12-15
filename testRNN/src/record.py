import os
import time

class record: 

    def __init__(self,filename,startTime): 

        self.startTime = startTime

        directory = os.path.dirname(filename)
        try:
            os.stat(directory)
        except:
            os.mkdir(directory) 
        self.file = open(filename,"w+") 
        
    def write(self,text): 
        self.file.write(text) 
        
    def close(self): 
        self.file.close()

    def resetTime(self): 
        self.write("reset time at %s\n\n"%(time.time() - self.startTime))
        self.startTime = time.time()

def writeInfo(r, numSamples, numAdv, perturbations, nc_coverage, kmnc_coverage, nbc_coverage, sanc_coverage, SC_coverage,BC_coverage, TC_coverage, unique_adv):
    r.write("time:%s\n" % (time.time() - r.startTime))
    r.write("samples: %s\n" % (numSamples))
    r.write("neuron coverage: %.3f\n" % (nc_coverage))
    r.write("k-multisection neuron coverage: %.3f\n" % (kmnc_coverage))
    r.write("neuron boundary coverage: %.3f\n" % (nbc_coverage))
    r.write("strong activation neuron coverage: %.3f\n" % (sanc_coverage))
    r.write("cell coverage: %.3f\n" % (SC_coverage))
    r.write("gate coverage: %.3f\n" % (BC_coverage))
    r.write("sequence coverage: %.3f\n" % (TC_coverage))
    r.write("adv. examples: %s\n" % (numAdv))
    r.write("adv. rate: %.3f\n" % (numAdv / numSamples))
    r.write("unique adv. examples: %s\n" % (unique_adv))
    if numAdv > 0 :
        r.write("average perturbation: %.3f\n" % (sum(perturbations) / len(perturbations)))
        r.write("minimum perturbation: %.3f\n\n" % (min(perturbations)))
    else :
        r.write("average perturbation: 0\n")
        r.write("minimum perturbation: 0\n\n")


