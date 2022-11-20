import sys

ONE_TWO_TREE_PATH = "/bioseq/oneTwoTree/"
DAILY_TESTS_PATH = "/bioseq/bioSequence_scripts_and_constants/daily_tests"

import os
import shutil


# Create one job file for list of genera:
def create_job_file(job_name, command_align, email_start_cmd, email_end_cmd, file_name, output_path, queue):

    with open(file_name, "w") as handle:
        handle.write("#!/bin/bash\n\n")  # #!/bin/bash
        handle.write("#PBS -N " + job_name + "\n")
        # handle.write("#PBS -j oe\n")
        handle.write("#PBS -r y\n")
        # handle.write("#PBS -q itaym\n")
        handle.write(f"#PBS -q {queue}\n")
        handle.write("#PBS -l select=1:ncpus=2:mem=1gb\n")
        handle.write("#PBS -v PBS_O_SHELL=bash,PBS_ENVIRONMENT=PBS_BATCH\n")
        handle.write("#PBS -e " + output_path + "\n")
        handle.write("#PBS -o " + output_path + "\n")
        handle.write("cd " + output_path + "\n")
        # handle.write("module load python/python-3.3.0\n")
        handle.write("module load python/python-anaconda3.6.5-michaldrori\n")
        handle.write("module load blast/blast230\n")
        handle.write("module load mrbayes/mrbayes_3.2.2\n")
        handle.write("module load Gblocks_0.91b\n")
        handle.write("module load R/3.5.1\n")
        handle.write("module load treePL-1\n")
        handle.write("module load pll-dppdiv-master\n")
        # handle.write("module load perl/perl-5.26\n")
        # handle.write("module unload gcc/gcc480\n")
        handle.write("module load gcc/gcc620\n")  # Jekyl
        # handle.write("module load gcc/gcc-7.3.0\n")
        # handle.write("module load mafft/mafft-7149-new\n") #Jekyl
        handle.write("module load mafft/mafft7149\n")  # Jekyl
        # handle.write("module load mafft/mafft-7.407\n")
        handle.write("module load java/java-1.8\n")
        handle.write("export LANG=aa_DJ.utf8\n")
        handle.write("module load raXML\n")
        handle.write("module load ExaML/examl-2018\n")
        handle.write("module load cdhit/cd-hit-4.7\n")
        handle.write("module load mpi/openmpi-1.10.4\n")
        handle.write("module load rocks-openmpi\n")
        handle.write("export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/opt/openmpi/lib\n")
        handle.write("export PYTHONPATH=/bioseq/oneTwoTree\n")
        handle.write("module load rocks-openmpi\n")

        handle.write("hostname;\n")

        # Alignment code:
        handle.write(email_start_cmd + "\n")
        handle.write(command_align + "\n")  # Alignment script
        handle.write(email_end_cmd + "\n")
    return file_name


# Create one job file for list of Taxon:
def create_Spesific_job_file(job_name, command, file_name, error_files_path):

    with open(file_name, "w") as handle:
        handle.write("#!/bin/bash\n\n")  # #!/bin/bash
        handle.write("#PBS -N " + job_name + "\n")
        # handle.write("#PBS -j oe\n")
        handle.write("#PBS -r y\n")
        handle.write("#PBS -q lifesciweb\n")
        handle.write("#PBS -l select=1:ncpus=2:mem=1gb\n")
        handle.write("#PBS -v PBS_O_SHELL=bash,PBS_ENVIRONMENT=PBS_BATCH\n")
        handle.write("#PBS -e " + error_files_path + " \n")
        handle.write("#PBS -o " + error_files_path + "\n")
        handle.write("cd " + error_files_path + "\n")
        # handle.write("module load python/python-3.3.0\n")
        handle.write("module load python/python-anaconda3.6.5-michaldrori\n")
        handle.write("module load blast/blast230\n")
        handle.write("module load mrbayes/mrbayes_3.2.2\n")
        handle.write("module load Gblocks_0.91b\n")
        handle.write("module load R/3.5.1\n")
        handle.write("module load treePL-1\n")
        handle.write("module load pll-dppdiv-master\n")
        # handle.write("module unload gcc/gcc480\n")
        handle.write("module load gcc/gcc620\n")  # Jekyl
        # handle.write("module load gcc/gcc-7.3.0\n")
        # handle.write("module load perl/perl-5.26\n")
        # handle.write("module load mafft/mafft-7149-new\n") #Jekyl
        handle.write("module load mafft/mafft7149\n")  # Jekyl
        # handle.write("module load mafft/mafft-7.407\n")
        # handle.write("module load mafft/mafft-7.407\n")
        handle.write("module load java/java-1.8\n")
        handle.write("export LANG aa_DJ.utf8\n")
        handle.write("module load raXML\n")
        handle.write("module load ExaML/examl-2018\n")
        handle.write("module load rocks-openmpi\n")
        handle.write("module load mpi/openmpi-1.10.4\n")
        handle.write("module load openmpi-x86_64\n")
        handle.write("module load cdhit/cd-hit-4.7\n")
        handle.write("export LD_LIBRARY_PATH ${LD_LIBRARY_PATH}:/opt/openmpi/lib\n")
        handle.write("export PYTHONPATH=/bioseq/oneTwoTree\n")
        handle.write("hostname;\n")
        handle.write(command + "\n")
    return file_name


# Create the file of genera / genus to be under each folder:
# Case 1 - each genus is running with all known species under it: the file will specify only the genus name
# Case 2 - may run on a number of genera and different species under each genus

# def Create_taxa_list_file(case_num):
#   if(case_num == 0)


def remove_directory(folder):

    # folder = '/path/to/folder'
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as exce:
            print(exce)
            print("Failed to remove dir:" + folder)
    return


###############################################################################################################
#
#                   This is the Main Tree Generation Jobs handler
#               ---------------------------------------------------------------
#               It will create the Directories for each genus in the input file
#               And send a job file that will run the Alignment and MrBays code
#
###############################################################################################################

# Running options:
# --------------------------------------------------------------------------------------------------------------

file_name = sys.argv[1]  # "generaMD.txt"
# Create jobs in loop
email_var = None
cmd = "none"
OUTPUT_OTT_PATH = sys.argv[2]
JobName = sys.argv[3]
queue = sys.argv[4]


# This case is for special runs - genera list includes all species and genera for 1 alignment:
# Dir name will be as the input file and the list will be copied into the taxa_list.txt input for the alignment
# species_list_file = os.path.basename(file_name)
species_list_file = file_name
# special_dir = species_list_file.replace(".txt", "")
print(OUTPUT_OTT_PATH)
# Dir_name=(OUTPUT_OTT_PATH + special_dir + "/")
Dir_name = OUTPUT_OTT_PATH + "/"
if not os.path.exists(Dir_name):
    os.makedirs(Dir_name)
f_taxa_list_file = open(OUTPUT_OTT_PATH + "/taxa_list.txt", "w")
shutil.copyfile(species_list_file, OUTPUT_OTT_PATH + "/taxa_list.txt")
align_script = ONE_TWO_TREE_PATH + "buildTaxaTree.py"
align_ini = ONE_TWO_TREE_PATH + "OneTwoTree.ini"

cmd = (
    "python "
    + align_script
    + " --taxa-list-file "
    + OUTPUT_OTT_PATH
    + "/taxa_list.txt --working-dir "
    + OUTPUT_OTT_PATH
    + "/ --config-filename "
    + align_ini
    + " --id "
    + JobName
)
errorFilePath = OUTPUT_OTT_PATH + "/"
GenusDir = OUTPUT_OTT_PATH + "/"
FileNameAndPath = OUTPUT_OTT_PATH + "/OTT_" + JobName + ".sh"

# send email at the end of the command:
email_file = "/bioseq/data/results/oneTwoTree/" + JobName + "/email.txt"
if os.path.exists(email_file):
    with open(email_file, "r") as f_email:
        first_line = f_email.readline()
        email_var = first_line.rstrip()
    params_file = OUTPUT_OTT_PATH + "/params.txt"
    if os.path.exists(params_file):
        with open(params_file, "r") as f_params:
            for line in f_params:
                if "jobTitle" in line:
                    line = line.strip()
                    jobTitle = line.split(":")[1]
        # email_cmd = "perl /bioseq/oneTwoTree/sendLastEmail.pl --toEmail " + email_var + " --id " + JobName
        if jobTitle != "daily test":
            email_start_cmd = (
                "perl "
                + ONE_TWO_TREE_PATH
                + "webServer_files/sendFirstEmail.pl --toEmail "
                + email_var
                + " --id "
                + JobName
                + " --jobTitle "
                + '"'
                + jobTitle
                + '"'
            )
            email_end_cmd = (
                "perl  "
                + ONE_TWO_TREE_PATH
                + "webServer_files/sendLastEmail.pl --toEmail "
                + email_var
                + " --id "
                + JobName
                + " --jobTitle "
                + '"'
                + jobTitle
                + '"'
            )
        else:
            email_start_cmd = ""
            final_pass_file = os.path.join("/bioseq/data/results/oneTwoTree/", JobName, "JOB_PASS.txt")
            email_end_cmd = f'python {ONE_TWO_TREE_PATH}webServer_files/write_daily_test.py {DAILY_TESTS_PATH} oneTwoTree "http://onetwotree.tau.ac.il/results.html?jobId={JobName}&jobTitle={jobTitle.replace(" ", "%20")}" {final_pass_file}'
else:
    email_start_cmd = ""
    email_end_cmd = ""

job_filename = create_job_file(
    "-".join(["OTT", JobName]), cmd, email_start_cmd, email_end_cmd, FileNameAndPath, errorFilePath, queue
)
os.chdir(OUTPUT_OTT_PATH + "/")
# log_file_name = OUTPUT_OTT_PATH + 'OTT_log.txt'
# f_log = open(log_file_name, "w")
# f_log.write("---------------------------------------- START ---------------------------------------------------\n")
os.system("qsub " + job_filename)
# f_log.write("---------------------------------------- END ---------------------------------------------------\n")
# f_log.write (cmd)
