import subprocess
import sys
import random
import os

def addLineToFile(oFile, line):
    with open("oFile", "a+") as file_object:
        # Move read cursor to the start of file.
        file_object.seek(0)
        # If file is not empty then append '\n'
        data = file_object.read(100)
        if len(data) > 0 :
            file_object.write("\n")
        # Append text at the end of file
        file_object.write(line)

def onPprocessors(command,p) :
  if os.environ.has_key("OPENMP"):
    return "OMP_NUM_THREADS="+repr(p)+" " + command
    return command  
  elif os.environ.has_key("CILK"):
    return "CILK_NWORKERS="+repr(p)+" " + command
  else:
    return "PARLAY_NUM_THREADS="+repr(p)+" " + command
  
def shellGetOutput(str) :
  process = subprocess.Popen(str,shell=True,stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
  output, err = process.communicate()
  
  if (len(err) > 0):
      raise NameError(str+"\n"+output+err)
  return output

def stripFloat(val) :
  trunc = float(int(val*1000))/1000
  return str(trunc).rstrip('0')    

def runSingle(runProgram, options, ifile, procs, oFile) :
  comString = "./"+runProgram+" "+options+" "+ifile
  if (procs > 0) :
    comString = onPprocessors(comString,procs)
  out = shellGetOutput(comString)
  nonCommentLines = [s for s in out.split('\n') if len(s)>0]
  for i in nonCommentLines:
    addLineToFile(oFile, i)
  try:
    times = [float(str[str.index(':')+2:]) for str in out.split('\n') if str.startswith("Parlay time: ")]
    return times
  except (ValueError,IndexError):
    raise NameError(comString+"\n"+out)

def runTest(runProgram, checkProgram, dataDir, test, rounds, procs, noOutput, oFile) :
    random.seed()
    outFile="/tmp/ofile%d_%d" %(random.randint(0, 1000000), random.randint(0, 1000000)) 
    [weight, gFileName, qFileName, iFileName, runOptions, checkOptions] = test
    if type(gFileName) is str :
      gFileName = [gFileName]
    shortgFileName = " ".join(gFileName)
    if len(dataDir)>0:
      out = shellGetOutput("cd " + dataDir + "; make " + shortgFileName)
    longgFileName = " ".join(dataDir + "/" + name for name in gFileName)
    if type(qFileName) is str :
      qFileName = [qFileName]
    shortqFileName = " ".join(qFileName)
    if len(dataDir)>0:
      out = shellGetOutput("cd " + dataDir + "; make " + shortqFileName)
    longqFileName = " ".join(dataDir + "/" + name for name in qFileName)
    if type(iFileName) is str :
      iFileName = [iFileName]
    shortiFileName = " ".join(iFileName)
    if len(dataDir)>0:
      out = shellGetOutput("cd " + dataDir + "; make " + shortiFileName)
    longiFileName = " ".join(dataDir + "/" + name for name in iFileName)
    runOptions = runOptions + " -q " + longqFileName
    runOptions = runOptions + " -r " + repr(rounds)
    if (noOutput == 0) :
      runOptions = runOptions + " -o " + outFile
    times = runSingle(runProgram, runOptions, longgFileName, procs, oFile)
    if (noOutput == 0) :
      checkString = ("./" + checkProgram + " " + checkOptions + " "
                     + longiFileName + " " + outFile)
      checkOut = shellGetOutput(checkString)
      nonCommentLines = [s for s in checkOut.split('\n') if len(s)>0]
      for line in nonCommentLines:
        print(line)
        addLineToFile(oFile, line)
      os.remove(outFile)
    ptimes = str([stripFloat(time)
                  for time in times])[1:-1]
    outputStr = ""
    if (len(runOptions) > 0) :
      outputStr = " : " + runOptions
    outStr = repr(weight) + outputStr + " : " + ptimes
    print(outStr)
    addLineToFile(oFile, outStr)
    return [weight,times]
    
def averageTime(times) :
    return sum(times)/len(times)
    

def timeAll(name, runProgram, checkProgram, dataDir, tests, rounds, procs, noOutput,
            problem, oFile) :
  totalTime = 0
  totalWeight = 0
  try:
    results = [runTest(runProgram, checkProgram, dataDir, test, rounds, procs,
                       noOutput, oFile)
               for test in tests]
    totalTimeMean = 0
    totalTimeMin = 0
    totalTimeMedian = 0
    totalWeight = 0
    j = 0
    for (weight,times) in results:
      l = len(times)
      if (l == 0):
        print("Warning, no timed results for", tests[j])
        continue
      times = sorted(times)
      totalTimeMean = totalTimeMean + weight*sum(times)/l
      totalTimeMin = totalTimeMin + weight*times[0]
      totalTimeMedian = totalTimeMedian + weight*times[(l-1)/2]
      totalWeight = totalWeight + weight
      j += 1
    print(name + " : " + repr(procs) +" : " +
          "weighted time, min=" + stripFloat(totalTimeMin/totalWeight) +
          " median=" + stripFloat(totalTimeMedian/totalWeight) +
          " mean=" + stripFloat(totalTimeMean/totalWeight))
    # return 0
  except NameError as x:
    print("TEST TERMINATED ABNORMALLY:\n["+str(x) + "]")
    return 1
  except KeyboardInterrupt:
    return 1


def getOption(str) :
  a = sys.argv
  l = len(a)
  for i in range(1, l) :
    if (a[i] == str) :
      return True
  return False

def getArg(str, default) :
  a = sys.argv
  l = len(a)
  for i in range(1, l) :
    if (a[i] == str and  (i+1 != l)) :
        return sys.argv[i+1]
  return default

def getArgs() :
  noOutput = getOption("-x")
  processors = int(getArg("-p", 0))
  rounds = int(getArg("-r", 1))
  return (noOutput, rounds, processors)

def timeAllArgs(runProgram, problem, checkProgram, dataDir, tests, oFile) :
    (noOutput, rounds, procs) = getArgs()
    name = os.path.basename(os.getcwd())
    timeAll(name, runProgram, checkProgram, dataDir, tests, rounds, procs, noOutput, problem, oFile)

