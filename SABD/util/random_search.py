#!/usr/bin/env python
# -*- coding: utf-8 -*-


import codecs
import json
import os
import re
import uuid
from collections import OrderedDict
from subprocess import check_output

import numpy
import scipy.stats
import subprocess
import sys
import time

from numpy import random

"""
This script automatically launches experiments on german servers.
Each experiment has your parameters vary using random search and can have more than one round.
A example of a valid json:
{
	"basic_param": 
	    {
	    "bug_dataset": "HOME_DIR/dataset/sun_2011/small_eclipse_2004_v2/eclipse_clear_v2.json", 
	    "training": "HOME_DIR/dataset/sun_2011/small_eclipse_2004_v2/training_split_eclipse_pairs_random_5.txt",
	    "validation": "HOME_DIR/dataset/sun_2011/small_eclipse_2004_v2/validation_eclipse_pairs_random_5.txt",
	    },	
	"batch_basic": "#!/bin/bash
                  #SBATCH --account=def-aloise
                  #SBATCH --gres=gpu:1              # Number of GPUs (per node)
                  #SBATCH --cpus-per-task=1
                  #SBATCH --time=1:00:00
                  #SBATCH --mem=30000M
                  #SBATCH --output=HOME_DIR/logs/dl_duplicate_eclipse_v2_%x-%j.out"
	"parameters" : [
		{
			"name": "lr",
			"type": "float",
			"dist" "uniform",
			"round": 4,
			"a": 3,
			"b": 2
		},
		{
			"name": "window",
			"type": "list",
			"values": [3,5,7]
		},
		{
			"name": "hidden_size",
			"type": "int",
			"min": 25,
			"max": 200
		},
		{
			"name": "cuda",
			"type": "bool",
		}
	],
	"nm": 5,
	"nrep": 1,
	"save_model_path": "test",
	'model_name': 'model_',
	"append_file": "/home/irving/aux/append",
	"scrip_path": "/home/irving/workspace/nlp-deeplearning/experiments/postag/wnn.py",
	"temp_folder": "temp",
}
"""


def executeSub(scriptPath, batchBasic, parameters, tempFolder):
    fName = os.path.join(tempFolder,"%s.sh" % uuid.uuid4().hex)
    f = codecs.open(fName, 'w')

    f.write(batchBasic)
    f.write('\n')

    progPar = "python %s" % scriptPath

    for k, v in parameters.items():
        if isinstance(v, list):
            for i in range(len(v)):
                v[i] = str(v[i])

            progPar += ' --%s %s' % (k, ' '.join(v))
        elif isinstance(v, (int, numpy.int64)):
            progPar += ' --%s %d' % (k, v)
        elif isinstance(v, float):
            progPar += ' --%s %f' % (k, v)
        elif v is None or isinstance(v, bool):
            progPar += ' --%s' % (k)
        else:
            progPar += ' --%s %s' % (k, v)

    f.write(progPar)
    f.close()

    out = check_output(["sbatch", fName]).decode('utf-8')

    if out:
        print("O job " + out + " foi criado")

    return out


def getNumbersOfJobs():
    sp = subprocess.Popen("bjobs", stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    out, err = sp.communicate()

    return len(re.findall("\n[0-9]+", out))


def uniform(min, max, round):
    return round(scipy.stats.uniform.rvs(), round)


class UniformDistribution:

    def __init__(self, min, max, round):
        self.round = round
        self.dist = scipy.stats.uniform(loc=min, scale=max - min)

    def draw(self):
        return round(self.dist.rvs(), self.round)


def main():
    # Get json with parameters of this script
    f = codecs.open(sys.argv[1], "r", "utf-8")
    js = json.load(f)

    # The parameters that will be the same for all experiments
    basicParameters = js["basic_param"]

    # Parameter will be varied
    parameters = js["parameters"]

    # A basic header to run the job in Slurm
    batchBasic = js['batch_basic']

    # Number of jobs to be launched
    nm = js["nm"]

    # Python script
    scriptPath = js["scrip_path"]

    # The file which will be store the jobId and parameters of each job
    appendFile = js["append_file"]

    # Number repetition per hyperparatemer set
    nRep = js["nrep"]

    # Temporary folder where the sbatch acript will stored
    tempFolder = js['temp_folder']

    # Save model path
    if "save_model_path" in js:
        saveModelPath = js["save_model_path"]
        # Basic name of the model name
        model_name = js.get('model_name', '')
    else:
        saveModelPath = None

    log = codecs.open(appendFile, "a", encoding="utf-8")

    # Create parameters that vary
    parametersValue = {}

    for param in parameters:
        name = param["name"]

        if param["type"] == "float":
            r = param["round"]
            a = param["a"]
            b = param["b"]

            """
            If dim is 1, so the scripts generates numbers. Otherwise, it generates vectors with "dim"
            dimensions. By default, dim = 1.
            """
            dimension = param.get("dim", 1)

            parametersValue[name] = []
            distName = param.get('distribution', 'uniform')
            if distName == 'uniform':
                distribution = UniformDistribution(a, b, r)
            else:
                distribution = UniformDistribution(a, b, r)

            for _ in range(nm):
                aux = []

                for _ in range(dimension):
                    aux.append(distribution.draw())

                if dimension == 1:
                    aux = aux[-1]

                parametersValue[name].append(aux)

        elif param["type"] == "int":
            min = param["min"]
            max = param["max"]
            parametersValue[name] = [random.randint(min, max) for _ in range(nm)]
        elif param["type"] == "list":
            values = param["values"]

            parametersValue[name] = [values[random.randint(0, len(values) - 1)] for _ in range(nm)]
        elif param["type"] == "bool":
            choices = [False, True]
            parametersValue[name] = [random.choice(choices).item() for _ in range(nm)]

    # Launch the 'nm' jobs
    for launchNum in range(nm):
        name = model_name
        parIteration = {}
        submissionJson = OrderedDict()

        # Transfer the parameters values of a launch to a dictionary
        for parName, paramValues in parametersValue.items():
            name += parName
            name += "_"
            if isinstance(paramValues[launchNum], list):
                name += "_".join(map(lambda k: str(k), paramValues[launchNum]))
            else:
                name += str(paramValues[launchNum])
            name += "_"

            parIteration[parName] = paramValues[launchNum]

        submissionJson['parameters'] = parIteration

        print("#######################################")
        print(parIteration)
        print("\t")

        finalPar = dict(basicParameters)

        for parName, parValue in parIteration.items():
            if isinstance(parValue, (bool, numpy.bool)):
                if parValue:
                    finalPar[parName] = None
                else:
                    try:
                        finalPar.pop(parName)
                    except:
                        pass
            else:
                finalPar[parName] = parValue

        jobIds = []

        print("Repete the experiments %d times" % nRep)
        for repIdx in range(nRep):
            if saveModelPath:
                saveModel = os.path.join(saveModelPath, name)

                if nRep > 1:
                    saveModel += "_%d" % (repIdx)

                finalPar["save"] = saveModel

            # Launch job
            print("Run %d" % launchNum)

            out = executeSub(scriptPath, batchBasic, finalPar, tempFolder)

            # Get job id
            r = re.findall("[0-9]+", out)

            if len(r) > 1:
                print("There are more than 1 nunmber in the output.")
                sys.exit()

            jobIds.append(r[0])

            print(finalPar)
            print("\n\n")
            launchNum += 1
            time.sleep(1)

        submissionJson['job_ids'] = jobIds
        st = json.dumps(submissionJson)
        print(st)
        log.write(st)
        log.write('\n')


if __name__ == '__main__':
    main()
