import sys
import csv
import os.path
import datetime
from git import Repo

def write_file(comment, outputDir, commandLine):
    with open('archive.csv', 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                                quotechar='"', quoting=csv.QUOTE_MINIMAL)
        time = datetime.datetime.now().isoformat()
        writer.writerow([time, comment,"", outputDir, commandLine])

def push_current(comment):
    repo = Repo("/home/vdeschai/deepMaterials/Sources/Multi_inputs/")
    assert not repo.bare
    origin = repo.remotes.origin

    repo.git.pull('originSSH', 'master')
    #origin.pull()
    repo.git.add("Pixes2Material/*")
    #repo.git.add("Training/runATrainingCluster.sh")

    repo.git.commit("-m", comment, "--allow-empty")

    repo.git.push('originSSH', 'master')

comment = sys.argv[1]
outputDir = sys.argv[2]
commandLine = ""

write_file(comment, outputDir, commandLine)
#push_current(comment)
